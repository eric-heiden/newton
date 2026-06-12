# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Screen-space fluid renderer for ViewerGL, ported from the NVIDIA Flex demo.

The pipeline follows Flex's ``demo/opengl/shadersGL.cpp``:

1. Splat anisotropic ellipsoids (quadric ray-cast) into a linear eye-depth
   buffer.
2. Accumulate optical thickness from enlarged point sprites, depth-tested
   against the scene.
3. Smooth the depth buffer with a single 2D bilateral filter whose radius
   adapts to the projected particle size (range falloff 5.5).
4. Composite the surface over the scene: one-sided finite-difference normals,
   Schlick Fresnel, screen-space refraction and reflection taps, shadow-mapped
   wrap-diffuse lighting, and a tight specular highlight. The pass writes real
   depth so the scene depth test resolves occlusion.
5. Draw diffuse spray/foam particles as velocity-stretched billboards with
   premultiplied over-blending, depth-tested against the composited surface.

Fluid particles are also splatted into the scene shadow map so water shadows
the scene through the regular shadow path.
"""

import ctypes

import numpy as np
import warp as wp

FLUID_VERTEX_STRIDE = 16  # floats: position.xyz, radius, q1.xyzw, q2.xyzw, q3.xyzw


@wp.kernel
def _pack_fluid_vertices(
    points: wp.array[wp.vec3],
    radii: wp.array[float],
    use_radii: int,
    uniform_radius: float,
    radius_scale: float,
    anisotropy: wp.array[wp.vec4],
    anisotropy_secondary: wp.array[wp.vec4],
    anisotropy_tertiary: wp.array[wp.vec4],
    use_anisotropy: int,
    dest: wp.array[float],
):
    tid = wp.tid()
    p = points[tid]
    r = uniform_radius
    if use_radii != 0:
        r = radii[tid]
    r *= radius_scale

    a1 = wp.vec4(1.0, 0.0, 0.0, 1.0)
    a2 = wp.vec4(0.0, 1.0, 0.0, 1.0)
    a3 = wp.vec4(0.0, 0.0, 1.0, 1.0)
    if use_anisotropy != 0:
        a1 = anisotropy[tid]
        a2 = anisotropy_secondary[tid]
        a3 = anisotropy_tertiary[tid]
        if a1[3] <= 0.0:
            # Inactive particle: zero radii collapse the quadric so the vertex
            # shader emits a degenerate quad.
            r = 0.0

    base = tid * 16
    dest[base + 0] = p[0]
    dest[base + 1] = p[1]
    dest[base + 2] = p[2]
    dest[base + 3] = r
    dest[base + 4] = a1[0]
    dest[base + 5] = a1[1]
    dest[base + 6] = a1[2]
    dest[base + 7] = a1[3] * r
    dest[base + 8] = a2[0]
    dest[base + 9] = a2[1]
    dest[base + 10] = a2[2]
    dest[base + 11] = a2[3] * r
    dest[base + 12] = a3[0]
    dest[base + 13] = a3[1]
    dest[base + 14] = a3[2]
    dest[base + 15] = a3[3] * r


# --------------------------------------------------------------------------
# Ellipsoid depth pass (Flex vertex/geometry/fragmentEllipsoidDepthShader)

ELLIPSOID_DEPTH_VS = """
#version 330 core
layout (location = 0) in vec4 aPositionRadius;
layout (location = 1) in vec4 aQ1;
layout (location = 2) in vec4 aQ2;
layout (location = 3) in vec4 aQ3;

uniform mat4 view;
uniform mat4 projection;
uniform mat4 inv_view;

out vec4 Bounds;
out vec4 InvQ0;
out vec4 InvQ1;
out vec4 InvQ2;
out vec4 InvQ3;
out vec4 NdcPos;

float Sign(float x) { return x < 0.0 ? -1.0 : 1.0; }

bool solveQuadratic(float a, float b, float c, out float minT, out float maxT)
{
    if (a == 0.0 && b == 0.0) {
        minT = 0.0;
        maxT = 0.0;
        return false;
    }

    float discriminant = b * b - 4.0 * a * c;
    if (discriminant < 0.0) {
        return false;
    }

    float t = -0.5 * (b + Sign(b) * sqrt(discriminant));
    minT = t / a;
    maxT = c / t;
    if (minT > maxT) {
        float tmp = minT;
        minT = maxT;
        maxT = tmp;
    }
    return true;
}

float DotInvW(vec4 a, vec4 b) { return a.x * b.x + a.y * b.y + a.z * b.z - a.w * b.w; }

void main()
{
    vec3 worldPos = aPositionRadius.xyz;

    // quadric matrix in world space
    mat4 q;
    q[0] = vec4(aQ1.xyz * aQ1.w, 0.0);
    q[1] = vec4(aQ2.xyz * aQ2.w, 0.0);
    q[2] = vec4(aQ3.xyz * aQ3.w, 0.0);
    q[3] = vec4(worldPos, 1.0);

    // solve the screen-space bounds of the projected quadric
    mat4 invClip = transpose(projection * view * q);

    float xmin = 0.0;
    float xmax = 0.0;
    float a1 = DotInvW(invClip[3], invClip[3]);
    float b1 = -2.0 * DotInvW(invClip[0], invClip[3]);
    float c1 = DotInvW(invClip[0], invClip[0]);
    solveQuadratic(a1, b1, c1, xmin, xmax);

    float ymin = 0.0;
    float ymax = 0.0;
    float b2 = -2.0 * DotInvW(invClip[1], invClip[3]);
    float c2 = DotInvW(invClip[1], invClip[1]);
    solveQuadratic(a1, b2, c2, ymin, ymax);

    Bounds = vec4(xmin, xmax, ymin, ymax);

    // inverse quadric: transforms view space to the unit-sphere parameter space
    mat4 invq;
    invq[0] = vec4(aQ1.xyz / max(aQ1.w, 1.0e-8), 0.0);
    invq[1] = vec4(aQ2.xyz / max(aQ2.w, 1.0e-8), 0.0);
    invq[2] = vec4(aQ3.xyz / max(aQ3.w, 1.0e-8), 0.0);
    invq[3] = vec4(0.0, 0.0, 0.0, 1.0);

    invq = transpose(invq);
    invq[3] = -(invq * vec4(worldPos, 1.0));
    invq = invq * inv_view;

    InvQ0 = invq[0];
    InvQ1 = invq[1];
    InvQ2 = invq[2];
    InvQ3 = invq[3];

    vec4 ndcPos = projection * view * vec4(worldPos, 1.0);
    NdcPos = ndcPos / max(abs(ndcPos.w), 1.0e-8);
    gl_Position = vec4(worldPos, 1.0);
}
"""

ELLIPSOID_DEPTH_GS = """
#version 330 core
layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

in vec4 Bounds[];
in vec4 InvQ0[];
in vec4 InvQ1[];
in vec4 InvQ2[];
in vec4 InvQ3[];
in vec4 NdcPos[];

flat out vec4 FragInvQ0;
flat out vec4 FragInvQ1;
flat out vec4 FragInvQ2;
flat out vec4 FragInvQ3;

void main()
{
    vec4 ndcPos = NdcPos[0];
    const float ndcBound = 1.0;
    if (ndcPos.x < -ndcBound || ndcPos.x > ndcBound) return;
    if (ndcPos.y < -ndcBound || ndcPos.y > ndcBound) return;

    vec4 bounds = Bounds[0];
    if (bounds.x == bounds.y || bounds.z == bounds.w) return;

    FragInvQ0 = InvQ0[0];
    FragInvQ1 = InvQ1[0];
    FragInvQ2 = InvQ2[0];
    FragInvQ3 = InvQ3[0];

    gl_Position = vec4(bounds.x, bounds.w, 0.0, 1.0);
    EmitVertex();
    gl_Position = vec4(bounds.x, bounds.z, 0.0, 1.0);
    EmitVertex();
    gl_Position = vec4(bounds.y, bounds.w, 0.0, 1.0);
    EmitVertex();
    gl_Position = vec4(bounds.y, bounds.z, 0.0, 1.0);
    EmitVertex();
    EndPrimitive();
}
"""

ELLIPSOID_DEPTH_FS = """
#version 330 core
flat in vec4 FragInvQ0;
flat in vec4 FragInvQ1;
flat in vec4 FragInvQ2;
flat in vec4 FragInvQ3;

out float FragEyeZ;

uniform mat4 projection;
uniform mat4 inv_projection;
uniform vec2 inv_viewport;

float Sign(float x) { return x < 0.0 ? -1.0 : 1.0; }
float sqr(float x) { return x * x; }

bool solveQuadratic(float a, float b, float c, out float minT, out float maxT)
{
    if (a == 0.0 && b == 0.0) {
        minT = 0.0;
        maxT = 0.0;
        return true;
    }

    float discriminant = b * b - 4.0 * a * c;
    if (discriminant < 0.0) {
        return false;
    }

    float t = -0.5 * (b + Sign(b) * sqrt(discriminant));
    minT = t / a;
    maxT = c / t;
    if (minT > maxT) {
        float tmp = minT;
        minT = maxT;
        maxT = tmp;
    }
    return true;
}

void main()
{
    mat4 invQuadric;
    invQuadric[0] = FragInvQ0;
    invQuadric[1] = FragInvQ1;
    invQuadric[2] = FragInvQ2;
    invQuadric[3] = FragInvQ3;

    vec4 ndcPos = vec4(gl_FragCoord.xy * inv_viewport * 2.0 - vec2(1.0), -1.0, 1.0);
    vec4 viewDir = inv_projection * ndcPos;

    // ray in parameter space
    vec4 dir = invQuadric * vec4(viewDir.xyz, 0.0);
    vec4 origin = invQuadric[3];

    float a = sqr(dir.x) + sqr(dir.y) + sqr(dir.z);
    float b = dir.x * origin.x + dir.y * origin.y + dir.z * origin.z - dir.w * origin.w;
    float c = sqr(origin.x) + sqr(origin.y) + sqr(origin.z) - sqr(origin.w);

    float minT;
    float maxT;
    if (solveQuadratic(a, 2.0 * b, c, minT, maxT)) {
        vec3 eyePos = viewDir.xyz * minT;
        vec4 clipPos = projection * vec4(eyePos, 1.0);
        clipPos.z /= clipPos.w;
        FragEyeZ = eyePos.z;
        gl_FragDepth = clipPos.z * 0.5 + 0.5;
        return;
    }
    discard;
}
"""

# --------------------------------------------------------------------------
# Thickness pass (Flex vertexPointDepthShader/fragmentPointThicknessShader)

THICKNESS_VS = """
#version 330 core
layout (location = 0) in vec4 aPositionRadius;

uniform mat4 view;
uniform mat4 projection;
uniform float thickness_scale;
uniform float point_scale;

void main()
{
    vec4 eyePos = view * vec4(aPositionRadius.xyz, 1.0);
    gl_Position = projection * eyePos;
    gl_PointSize = aPositionRadius.w * thickness_scale * (point_scale / max(gl_Position.w, 1.0e-6));
}
"""

THICKNESS_FS = """
#version 330 core
out float FragThickness;

uniform float thickness_gain;

void main()
{
    vec2 d = gl_PointCoord.xy * 2.0 - vec2(1.0);
    float mag = dot(d, d);
    if (mag > 1.0) discard;
    FragThickness = sqrt(1.0 - mag) * thickness_gain;
}
"""

# --------------------------------------------------------------------------
# Bilateral depth blur (Flex fragmentBlurDepthShader)

BLUR_VS = """
#version 330 core
layout (location = 0) in vec2 aPos;
out vec2 TexCoord;
void main()
{
    TexCoord = aPos * 0.5 + vec2(0.5);
    gl_Position = vec4(aPos, 0.0, 1.0);
}
"""

BLUR_FS = """
#version 330 core
in vec2 TexCoord;
out float FragEyeZ;

uniform sampler2D depth_tex;
uniform float blur_radius_world;
uniform float blur_scale;
uniform float max_blur_radius;

float sqr(float x) { return x * x; }

void main()
{
    float depth = texelFetch(depth_tex, ivec2(gl_FragCoord.xy), 0).x;
    if (depth == 0.0) {
        FragEyeZ = 0.0;
        return;
    }

    const float blurDepthFalloff = 5.5;

    // fractional tap contributions avoid visible steps between tap counts
    float radius = min(max_blur_radius, blur_scale * (blur_radius_world / -depth));
    float radiusInv = 1.0 / max(radius, 1.0e-6);
    float taps = ceil(radius);
    float frac = taps - radius;

    float sum = 0.0;
    float wsum = 0.0;
    float count = 0.0;

    for (float y = -taps; y <= taps; y += 1.0) {
        for (float x = -taps; x <= taps; x += 1.0) {
            float s = texelFetch(depth_tex, ivec2(gl_FragCoord.xy) + ivec2(int(x), int(y)), 0).x;

            // spatial domain
            float r1 = length(vec2(x, y)) * radiusInv;
            float w = exp(-(r1 * r1));

            // range domain on the depth difference preserves silhouettes
            float r2 = (s - depth) * blurDepthFalloff;
            float g = exp(-(r2 * r2));

            float wBoundary = step(radius, max(abs(x), abs(y)));
            float wFrac = 1.0 - wBoundary * frac;

            sum += s * w * g * wFrac;
            wsum += w * g * wFrac;
            count += g * wFrac;
        }
    }

    if (wsum > 0.0) {
        sum /= wsum;
    }

    float blend = count / sqr(2.0 * radius + 1.0);
    FragEyeZ = mix(depth, sum, blend);
}
"""

# --------------------------------------------------------------------------
# Composite (Flex fragmentCompositeShader)

COMPOSITE_FS = """
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D depth_tex;
uniform sampler2D thickness_tex;
uniform sampler2D scene_tex;
uniform sampler2D shadow_tex;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 inv_view;
uniform mat4 light_transform;
uniform vec3 light_dir;
uniform vec2 inv_tex_scale;
uniform vec2 clip_pos_to_eye;
uniform vec2 tex_scale;
uniform vec4 color;
uniform float ior;
uniform vec3 sky_color;
uniform vec3 ground_color;
uniform vec3 up_vec;

float sqr(float x) { return x * x; }
float cube(float x) { return x * x * x; }

float shadowSample(vec3 worldPos)
{
    vec4 pos = light_transform * vec4(worldPos + light_dir * 0.15, 1.0);
    pos /= pos.w;
    vec3 uvw = pos.xyz * 0.5 + vec3(0.5);

    if (uvw.x < 0.0 || uvw.x > 1.0) return 1.0;
    if (uvw.y < 0.0 || uvw.y > 1.0) return 1.0;
    if (uvw.z > 1.0) return 1.0;

    vec2 shadowTaps[8] = vec2[](
        vec2(-0.326212, -0.40581), vec2(-0.840144, -0.07358),
        vec2(-0.695914,  0.457137), vec2(-0.203345, 0.620716),
        vec2( 0.96234,  -0.194983), vec2( 0.473434, -0.480026),
        vec2( 0.519456,  0.767022), vec2( 0.185461, -0.893124)
    );

    float biased = uvw.z - 0.001;
    float s = 0.0;
    const float radius = 0.002;
    for (int i = 0; i < 8; ++i) {
        float mapDepth = texture(shadow_tex, uvw.xy + shadowTaps[i] * radius).r;
        s += mapDepth < biased ? 0.0 : 1.0;
    }
    return s / 8.0;
}

vec3 viewportToEyeSpace(vec2 coord, float eyeZ)
{
    vec2 uv = (coord * 2.0 - vec2(1.0)) * clip_pos_to_eye;
    return vec3(-uv * eyeZ, eyeZ);
}

float upCoord(vec3 p) { return dot(p, up_vec); }

void main()
{
    float eyeZ = texture(depth_tex, TexCoord).x;
    if (eyeZ == 0.0)
        discard;

    // reconstruct eye space position from linear depth
    vec3 eyePos = viewportToEyeSpace(TexCoord, eyeZ);

    // one-sided finite differences: pick the side with the smaller depth jump
    // so silhouettes stay crisp
    vec3 zl = eyePos - viewportToEyeSpace(TexCoord - vec2(inv_tex_scale.x, 0.0),
        texture(depth_tex, TexCoord - vec2(inv_tex_scale.x, 0.0)).x);
    vec3 zr = viewportToEyeSpace(TexCoord + vec2(inv_tex_scale.x, 0.0),
        texture(depth_tex, TexCoord + vec2(inv_tex_scale.x, 0.0)).x) - eyePos;
    vec3 zt = viewportToEyeSpace(TexCoord + vec2(0.0, inv_tex_scale.y),
        texture(depth_tex, TexCoord + vec2(0.0, inv_tex_scale.y)).x) - eyePos;
    vec3 zb = eyePos - viewportToEyeSpace(TexCoord - vec2(0.0, inv_tex_scale.y),
        texture(depth_tex, TexCoord - vec2(0.0, inv_tex_scale.y)).x);

    vec3 dx = zl;
    vec3 dy = zt;
    if (abs(zr.z) < abs(zl.z)) dx = zr;
    if (abs(zb.z) < abs(zt.z)) dy = zb;

    vec4 worldPos = inv_view * vec4(eyePos, 1.0);

    float shadow = shadowSample(worldPos.xyz);

    vec3 l = (view * vec4(light_dir, 0.0)).xyz;
    vec3 v = -normalize(eyePos);

    vec3 n = normalize(cross(dx, dy));
    vec3 h = normalize(v + l);

    float fresnel = 0.1 + (1.0 - 0.1) * cube(1.0 - max(dot(n, v), 0.0));
    float ln = dot(l, n);

    vec3 rEye = reflect(-v, n).xyz;
    vec3 rWorld = (inv_view * vec4(rEye, 0.0)).xyz;

    float refractScale = ior * 0.025;
    float reflectScale = ior * 0.1;

    // attenuate refraction near the ground to avoid sampling under the floor
    refractScale *= smoothstep(0.1, 0.4, upCoord(worldPos.xyz));

    vec2 refractCoord = TexCoord + n.xy * refractScale * tex_scale;

    // read thickness from the refracted coordinate to avoid halos
    float thickness = max(texture(thickness_tex, refractCoord).x, 0.3);
    vec3 transmission = (1.0 - (1.0 - color.xyz) * thickness * 0.8) * color.w;
    vec3 refractCol = texture(scene_tex, refractCoord).xyz * transmission;

    vec2 sceneReflectCoord = TexCoord - rEye.xy * tex_scale * reflectScale / eyeZ;
    vec3 sceneReflect = texture(scene_tex, sceneReflectCoord).xyz * shadow;

    vec3 reflectCol = sceneReflect
        + mix(ground_color, sky_color, smoothstep(0.15, 0.25, upCoord(rWorld)) * shadow);

    vec3 diffuse = color.xyz
        * mix(vec3(0.29, 0.379, 0.59), vec3(1.0), (ln * 0.5 + 0.5) * max(shadow, 0.4))
        * (1.0 - color.w);
    vec3 specular = vec3(1.2 * pow(max(dot(h, n), 0.0), 400.0));

    FragColor.xyz = diffuse + (mix(refractCol, reflectCol, fresnel) + specular) * color.w;
    FragColor.w = 1.0;

    vec4 clipPos = projection * vec4(0.0, 0.0, eyeZ, 1.0);
    clipPos.z /= clipPos.w;
    gl_FragDepth = clipPos.z * 0.5 + 0.5;
}
"""

# --------------------------------------------------------------------------
# Diffuse spray/foam (Flex vertex/geometry/fragmentDiffuseShader)

DIFFUSE_VS = """
#version 330 core
layout (location = 0) in vec4 aPositionLife;
layout (location = 1) in vec4 aVelocity;

uniform mat4 view;
uniform mat4 projection;

out vec4 WorldPosLife;
out vec4 EyePos;
out vec3 EyeVel;
out vec4 NdcPos;

void main()
{
    WorldPosLife = aPositionLife;
    EyePos = view * vec4(aPositionLife.xyz, 1.0);
    EyeVel = (view * vec4(aVelocity.xyz, 0.0)).xyz;
    vec4 ndcPos = projection * EyePos;
    NdcPos = ndcPos / max(abs(ndcPos.w), 1.0e-8);
    gl_Position = ndcPos;
}
"""

DIFFUSE_GS = """
#version 330 core
layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

in vec4 WorldPosLife[];
in vec4 EyePos[];
in vec3 EyeVel[];
in vec4 NdcPos[];

uniform mat4 projection;
uniform float point_scale;
uniform float motion_blur_scale;
uniform float diffusion;

out vec2 TexCoord;
flat out float LifeFade;
flat out float VelocityFade;

void main()
{
    float life = WorldPosLife[0].w;
    if (life <= 0.0) return;

    vec4 ndcPos = NdcPos[0];
    const float ndcBound = 1.0;
    if (ndcPos.x < -ndcBound || ndcPos.x > ndcBound) return;
    if (ndcPos.y < -ndcBound || ndcPos.y > ndcBound) return;

    vec3 v = EyeVel[0];
    vec3 p = EyePos[0].xyz;

    // billboard in eye space
    vec3 u = vec3(0.0, point_scale, 0.0);
    vec3 l = vec3(point_scale, 0.0, 0.0);

    // sprites grow as they age (life counts down from 1)
    float sizeFade = mix(1.0 + diffusion, 1.0, min(1.0, life * 4.0));
    u *= sizeFade;
    l *= sizeFade;

    float fade = 1.0 / (sizeFade * sizeFade);
    float vlen = length(v) * motion_blur_scale;

    if (vlen > 0.5) {
        // stretch along the velocity direction (assume 60 Hz like Flex)
        float len = max(point_scale, vlen * 0.016);
        fade = min(1.0, 2.0 / (len / point_scale));
        u = normalize(v) * len;
        l = normalize(cross(u, vec3(0.0, 0.0, -1.0))) * point_scale;
    }

    LifeFade = min(1.0, life * 2.0);
    VelocityFade = fade;

    TexCoord = vec2(0.0, 1.0);
    gl_Position = projection * vec4(p + u - l, 1.0);
    EmitVertex();
    TexCoord = vec2(0.0, 0.0);
    gl_Position = projection * vec4(p - u - l, 1.0);
    EmitVertex();
    TexCoord = vec2(1.0, 1.0);
    gl_Position = projection * vec4(p + u + l, 1.0);
    EmitVertex();
    TexCoord = vec2(1.0, 0.0);
    gl_Position = projection * vec4(p - u + l, 1.0);
    EmitVertex();
    EndPrimitive();
}
"""

DIFFUSE_FS = """
#version 330 core
in vec2 TexCoord;
flat in float LifeFade;
flat in float VelocityFade;

out vec4 FragColor;

uniform vec4 color;

float sqr(float x) { return x * x; }

void main()
{
    vec2 d = TexCoord * 2.0 - vec2(1.0);
    float mag = dot(d, d);
    if (mag > 1.0) discard;

    float alpha = LifeFade * VelocityFade * sqr(1.0 - mag) * color.w;
    FragColor = vec4(color.xyz * alpha, alpha);
}
"""

# --------------------------------------------------------------------------
# Fluid splats for the scene shadow map

SHADOW_SPLAT_VS = """
#version 330 core
layout (location = 0) in vec4 aPositionRadius;

uniform mat4 light_view;
uniform mat4 light_projection;
uniform float point_scale;

void main()
{
    gl_Position = light_projection * light_view * vec4(aPositionRadius.xyz, 1.0);
    gl_PointSize = aPositionRadius.w * 2.0 * point_scale;
}
"""

SHADOW_SPLAT_FS = """
#version 330 core
void main()
{
    vec2 d = gl_PointCoord.xy * 2.0 - vec2(1.0);
    if (dot(d, d) > 1.0) discard;
}
"""


def _std_mat(flat_matrix) -> np.ndarray:
    """Convert a pyglet column-major flat matrix to a standard numpy matrix."""
    return np.array(flat_matrix, dtype=np.float32).reshape(4, 4).T


class _Program:
    """Small wrapper around a pyglet ShaderProgram with uniform helpers."""

    def __init__(self, gl, vertex: str, fragment: str, geometry: str | None = None):
        from pyglet.graphics.shader import Shader, ShaderProgram

        self._gl = gl
        shaders = [Shader(vertex, "vertex"), Shader(fragment, "fragment")]
        if geometry is not None:
            shaders.append(Shader(geometry, "geometry"))
        self.program = ShaderProgram(*shaders)

    def __enter__(self):
        self._gl.glUseProgram(self.program.id)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._gl.glUseProgram(0)

    def _loc(self, name: str) -> int:
        return self._gl.glGetUniformLocation(self.program.id, ctypes.c_char_p(name.encode()))

    def set_mat4(self, name: str, m: np.ndarray):
        data = np.ascontiguousarray(m, dtype=np.float32)
        self._gl.glUniformMatrix4fv(
            self._loc(name), 1, self._gl.GL_TRUE, data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )

    def set_float(self, name: str, v: float):
        self._gl.glUniform1f(self._loc(name), float(v))

    def set_int(self, name: str, v: int):
        self._gl.glUniform1i(self._loc(name), int(v))

    def set_vec2(self, name: str, v):
        self._gl.glUniform2f(self._loc(name), float(v[0]), float(v[1]))

    def set_vec3(self, name: str, v):
        self._gl.glUniform3f(self._loc(name), float(v[0]), float(v[1]), float(v[2]))

    def set_vec4(self, name: str, v):
        self._gl.glUniform4f(self._loc(name), float(v[0]), float(v[1]), float(v[2]), float(v[3]))


class FluidBatch:
    """GPU vertex data for one logged fluid (positions plus anisotropy)."""

    def __init__(self, gl, capacity: int):
        self._gl = gl
        self.capacity = max(int(capacity), 1)
        self.count = 0
        self.hidden = False

        # material (Flex demo defaults)
        self.color = (0.113, 0.425, 0.55, 0.8)
        self.ior = 1.0
        self.blur_radius_world = 0.06
        self.max_blur_radius = 14.0
        self.thickness_scale = 4.0
        self.thickness_gain = 0.005

        self.vao = gl.GLuint()
        self.vbo = gl.GLuint()
        gl.glGenVertexArrays(1, self.vao)
        gl.glGenBuffers(1, self.vbo)

        stride = FLUID_VERTEX_STRIDE * 4
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.capacity * stride, None, gl.GL_DYNAMIC_DRAW)
        for i in range(4):
            gl.glVertexAttribPointer(i, 4, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(i * 16))
            gl.glEnableVertexAttribArray(i)
        gl.glBindVertexArray(0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

        self._packed_gpu = None
        self._dummy_radii = None

    def destroy(self):
        gl = self._gl
        if getattr(self, "vao", None) is not None:
            gl.glDeleteVertexArrays(1, self.vao)
            gl.glDeleteBuffers(1, self.vbo)
            self.vao = None
            self.vbo = None
        self._packed_gpu = None
        self._dummy_radii = None

    def _ensure_capacity(self, count: int):
        if count <= self.capacity:
            return
        material = (
            self.color,
            self.ior,
            self.blur_radius_world,
            self.max_blur_radius,
            self.thickness_scale,
            self.thickness_gain,
        )
        self.destroy()
        self.__init__(self._gl, max(count, self.capacity * 2))
        (
            self.color,
            self.ior,
            self.blur_radius_world,
            self.max_blur_radius,
            self.thickness_scale,
            self.thickness_gain,
        ) = material

    def update(
        self, points, radii, radius_scale=1.0, anisotropy=None, anisotropy_secondary=None, anisotropy_tertiary=None
    ):
        """Pack particle data into the vertex buffer (device path when possible)."""
        gl = self._gl
        if points is None:
            self.count = 0
            return

        count = int(len(points))
        self._ensure_capacity(count)
        self.count = count
        if count == 0:
            return

        scalar_radius = radii is None or isinstance(radii, (int, float, np.integer, np.floating))
        use_aniso = anisotropy is not None and anisotropy_secondary is not None and anisotropy_tertiary is not None

        if isinstance(points, wp.array) and points.device.is_cuda:
            device = points.device
            if self._packed_gpu is None or len(self._packed_gpu) < self.capacity * 16:
                self._packed_gpu = wp.empty(self.capacity * 16, dtype=float, device=device)
                self._dummy_radii = wp.zeros(1, dtype=float, device=device)
            if scalar_radius:
                radii_array = self._dummy_radii
                uniform_radius = 0.1 if radii is None else float(radii)
                use_radii = 0
            else:
                radii_array = radii
                uniform_radius = 0.0
                use_radii = 1
            dummy4 = anisotropy if use_aniso else wp.zeros(1, dtype=wp.vec4, device=device)
            wp.launch(
                _pack_fluid_vertices,
                dim=count,
                inputs=[
                    points,
                    radii_array,
                    use_radii,
                    uniform_radius,
                    float(radius_scale),
                    dummy4,
                    anisotropy_secondary if use_aniso else dummy4,
                    anisotropy_tertiary if use_aniso else dummy4,
                    1 if use_aniso else 0,
                    self._packed_gpu,
                ],
                device=device,
            )
            host = self._packed_gpu[: count * 16].numpy()
        else:
            host_points = points.numpy() if isinstance(points, wp.array) else np.asarray(points)
            host_points = host_points.astype(np.float32, copy=False)
            if scalar_radius:
                r = np.full(count, 0.1 if radii is None else float(radii), dtype=np.float32)
            else:
                r = (
                    radii.numpy().astype(np.float32, copy=False)
                    if isinstance(radii, wp.array)
                    else np.asarray(radii, dtype=np.float32)
                )
            r = r * float(radius_scale)
            data = np.zeros((count, 16), dtype=np.float32)
            data[:, :3] = host_points
            data[:, 3] = r
            if use_aniso:
                q1 = anisotropy.numpy().astype(np.float32, copy=False)
                q2 = anisotropy_secondary.numpy().astype(np.float32, copy=False)
                q3 = anisotropy_tertiary.numpy().astype(np.float32, copy=False)
                inactive = q1[:, 3] <= 0.0
                data[inactive, 3] = 0.0
                data[:, 4:7] = q1[:, :3]
                data[:, 7] = q1[:, 3] * data[:, 3]
                data[:, 8:11] = q2[:, :3]
                data[:, 11] = q2[:, 3] * data[:, 3]
                data[:, 12:15] = q3[:, :3]
                data[:, 15] = q3[:, 3] * data[:, 3]
            else:
                data[:, 4] = 1.0
                data[:, 7] = data[:, 3]
                data[:, 9] = 1.0
                data[:, 11] = data[:, 3]
                data[:, 14] = 1.0
                data[:, 15] = data[:, 3]
            host = np.ascontiguousarray(data.reshape(-1))

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, host.nbytes, host.ctypes.data)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

    def draw(self):
        if self.hidden or self.count == 0:
            return
        gl = self._gl
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_POINTS, 0, self.count)
        gl.glBindVertexArray(0)


class DiffuseBatch:
    """GPU vertex data for diffuse spray/foam particles."""

    def __init__(self, gl, capacity: int):
        self._gl = gl
        self.capacity = max(int(capacity), 1)
        self.count = 0
        self.hidden = False

        self.radius = 0.02
        self.color = (0.9, 0.95, 1.0, 0.8)
        self.motion_blur_scale = 1.0
        self.diffusion = 1.0

        self._host_positions = np.zeros((0, 4), dtype=np.float32)
        self._host_velocities = np.zeros((0, 4), dtype=np.float32)

        self.vao = gl.GLuint()
        self.position_vbo = gl.GLuint()
        self.velocity_vbo = gl.GLuint()
        gl.glGenVertexArrays(1, self.vao)
        gl.glGenBuffers(1, self.position_vbo)
        gl.glGenBuffers(1, self.velocity_vbo)

        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.position_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.capacity * 16, None, gl.GL_DYNAMIC_DRAW)
        gl.glVertexAttribPointer(0, 4, gl.GL_FLOAT, gl.GL_FALSE, 16, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.velocity_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.capacity * 16, None, gl.GL_DYNAMIC_DRAW)
        gl.glVertexAttribPointer(1, 4, gl.GL_FLOAT, gl.GL_FALSE, 16, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)
        gl.glBindVertexArray(0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

    def destroy(self):
        gl = self._gl
        if getattr(self, "vao", None) is not None:
            gl.glDeleteVertexArrays(1, self.vao)
            gl.glDeleteBuffers(1, self.position_vbo)
            gl.glDeleteBuffers(1, self.velocity_vbo)
            self.vao = None

    def _ensure_capacity(self, count: int):
        if count <= self.capacity:
            return
        material = (self.radius, self.color, self.motion_blur_scale, self.diffusion)
        self.destroy()
        self.__init__(self._gl, max(count, self.capacity * 2))
        self.radius, self.color, self.motion_blur_scale, self.diffusion = material

    def update(self, positions, velocities):
        if positions is None:
            self.count = 0
            return

        host_positions = positions.numpy() if isinstance(positions, wp.array) else np.asarray(positions)
        host_positions = host_positions.astype(np.float32, copy=False)
        if velocities is None:
            host_velocities = np.zeros_like(host_positions)
        else:
            host_velocities = velocities.numpy() if isinstance(velocities, wp.array) else np.asarray(velocities)
            host_velocities = host_velocities.astype(np.float32, copy=False)

        live = host_positions[:, 3] > 0.0
        self._host_positions = np.ascontiguousarray(host_positions[live])
        self._host_velocities = np.ascontiguousarray(host_velocities[live])
        count = int(self._host_positions.shape[0])
        self._ensure_capacity(count)
        self.count = count
        self._upload()

    def _upload(self):
        if self.count == 0:
            return
        gl = self._gl
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.position_vbo)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, self._host_positions.nbytes, self._host_positions.ctypes.data)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.velocity_vbo)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, self._host_velocities.nbytes, self._host_velocities.ctypes.data)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

    def sort_for_view(self, view_std: np.ndarray):
        """Back-to-front sort so premultiplied over-blending composites correctly."""
        if self.count <= 1:
            return
        rot = view_std[:3, :3]
        trans = view_std[:3, 3]
        eye_z = self._host_positions[:, :3] @ rot[2] + trans[2]
        order = np.argsort(eye_z, kind="mergesort")
        self._host_positions = np.ascontiguousarray(self._host_positions[order])
        self._host_velocities = np.ascontiguousarray(self._host_velocities[order])
        self._upload()

    def draw(self):
        if self.hidden or self.count == 0:
            return
        gl = self._gl
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_POINTS, 0, self.count)
        gl.glBindVertexArray(0)


class FluidRenderer:
    """Owns the fluid render targets and programs; driven by RendererGL."""

    def __init__(self, gl):
        self._gl = gl
        self._width = 0
        self._height = 0

        self._depth_tex = None
        self._depth_smooth_tex = None
        self._thickness_tex = None
        self._scene_tex = None
        self._depth_buffer = None
        self._fbo = None

        self._ellipsoid_prog = _Program(gl, ELLIPSOID_DEPTH_VS, ELLIPSOID_DEPTH_FS, ELLIPSOID_DEPTH_GS)
        self._thickness_prog = _Program(gl, THICKNESS_VS, THICKNESS_FS)
        self._blur_prog = _Program(gl, BLUR_VS, BLUR_FS)
        self._composite_prog = _Program(gl, BLUR_VS, COMPOSITE_FS)
        self._diffuse_prog = _Program(gl, DIFFUSE_VS, DIFFUSE_FS, DIFFUSE_GS)
        self._shadow_prog = _Program(gl, SHADOW_SPLAT_VS, SHADOW_SPLAT_FS)

        # fullscreen triangle-strip quad
        self._quad_vao = gl.GLuint()
        self._quad_vbo = gl.GLuint()
        gl.glGenVertexArrays(1, self._quad_vao)
        gl.glGenBuffers(1, self._quad_vbo)
        quad = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        gl.glBindVertexArray(self._quad_vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._quad_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, quad.nbytes, quad.ctypes.data, gl.GL_STATIC_DRAW)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 8, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)
        gl.glBindVertexArray(0)

    def _make_texture(self, internal_format, fmt, dtype):
        gl = self._gl
        tex = gl.GLuint()
        gl.glGenTextures(1, tex)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, internal_format, self._width, self._height, 0, fmt, dtype, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        return tex

    def _ensure_targets(self, width: int, height: int):
        gl = self._gl
        if width == self._width and height == self._height and self._fbo is not None:
            return
        for tex in (self._depth_tex, self._depth_smooth_tex, self._thickness_tex, self._scene_tex):
            if tex is not None:
                gl.glDeleteTextures(1, tex)
        if self._depth_buffer is not None:
            gl.glDeleteRenderbuffers(1, self._depth_buffer)
        if self._fbo is not None:
            gl.glDeleteFramebuffers(1, self._fbo)

        self._width = width
        self._height = height
        self._depth_tex = self._make_texture(gl.GL_R32F, gl.GL_RED, gl.GL_FLOAT)
        self._depth_smooth_tex = self._make_texture(gl.GL_R32F, gl.GL_RED, gl.GL_FLOAT)
        self._thickness_tex = self._make_texture(gl.GL_R32F, gl.GL_RED, gl.GL_FLOAT)
        self._scene_tex = self._make_texture(gl.GL_RGBA8, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)

        self._depth_buffer = gl.GLuint()
        gl.glGenRenderbuffers(1, self._depth_buffer)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self._depth_buffer)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT24, width, height)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, 0)

        self._fbo = gl.GLuint()
        gl.glGenFramebuffers(1, self._fbo)

    def _attach(self, color_tex, with_depth_buffer: bool, depth_tex=None):
        gl = self._gl
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._fbo)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, color_tex, 0)
        if depth_tex is not None:
            gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, 0)
            gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_TEXTURE_2D, depth_tex, 0)
        elif with_depth_buffer:
            gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_TEXTURE_2D, 0, 0)
            gl.glFramebufferRenderbuffer(
                gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, self._depth_buffer
            )
        else:
            gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_TEXTURE_2D, 0, 0)
            gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, 0)
        gl.glDrawBuffer(gl.GL_COLOR_ATTACHMENT0)

    def _draw_quad(self):
        gl = self._gl
        gl.glBindVertexArray(self._quad_vao)
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
        gl.glBindVertexArray(0)

    def render_shadow(self, host, fluids):
        """Splat fluid particles into the currently bound scene shadow map."""
        batches = [b for b in fluids.values() if not b.hidden and b.count > 0]
        if not batches:
            return
        gl = self._gl
        light_view = _std_mat(getattr(host, "_light_view_matrix", np.eye(4, dtype=np.float32).flatten()))
        light_proj = _std_mat(getattr(host, "_light_projection_matrix", np.eye(4, dtype=np.float32).flatten()))
        # orthographic projection: pixels per world unit
        point_scale = 0.5 * host._shadow_width * light_proj[0, 0]

        if hasattr(gl, "GL_PROGRAM_POINT_SIZE"):
            gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthMask(True)
        with self._shadow_prog as prog:
            prog.set_mat4("light_view", light_view)
            prog.set_mat4("light_projection", light_proj)
            prog.set_float("point_scale", point_scale)
            for batch in batches:
                batch.draw()

    def render(self, host, fluids, diffuse):
        """Run the fluid passes over the resolved scene in ``host._frame_fbo``."""
        gl = self._gl
        batches = [b for b in fluids.values() if not b.hidden and b.count > 0] if fluids else []
        diffuse_batches = [b for b in diffuse.values() if not b.hidden and b.count > 0] if diffuse else []
        if not batches and not diffuse_batches:
            return

        width, height = host._screen_width, host._screen_height
        self._ensure_targets(width, height)

        view = _std_mat(host._view_matrix)
        projection = _std_mat(host._projection_matrix)
        inv_view = np.linalg.inv(view)
        inv_projection = np.linalg.inv(projection)
        light_transform = np.asarray(host._light_space_matrix, dtype=np.float32).reshape(4, 4).T

        tan_half_fov = 1.0 / projection[1, 1]
        aspect = projection[1, 1] / projection[0, 0]
        point_scale = height / (2.0 * tan_half_fov)  # Flex: pixels = r * point_scale / w (diameter)
        clip_pos_to_eye = (tan_half_fov * aspect, tan_half_fov)
        inv_viewport = (1.0 / width, 1.0 / height)

        sun = np.asarray(host._sun_direction, dtype=np.float32)
        light_dir = -sun / max(np.linalg.norm(sun), 1.0e-6)

        up_vec = np.zeros(3, dtype=np.float32)
        up_vec[int(host.camera.up_axis)] = 1.0

        material = batches[0] if batches else None

        if batches:
            # 1. copy the resolved scene color for refraction/reflection taps
            gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, host._frame_fbo)
            gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
            self._attach(self._scene_tex, with_depth_buffer=False)
            gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, host._frame_fbo)
            gl.glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, gl.GL_COLOR_BUFFER_BIT, gl.GL_NEAREST)

            # 2. ellipsoid depth
            self._attach(self._depth_tex, with_depth_buffer=True)
            gl.glViewport(0, 0, width, height)
            gl.glClearBufferfv(gl.GL_COLOR, 0, (gl.GLfloat * 4)(0.0, 0.0, 0.0, 0.0))
            gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glDepthFunc(gl.GL_LESS)
            gl.glDepthMask(True)
            gl.glDisable(gl.GL_BLEND)
            with self._ellipsoid_prog as prog:
                prog.set_mat4("view", view)
                prog.set_mat4("projection", projection)
                prog.set_mat4("inv_view", inv_view)
                prog.set_mat4("inv_projection", inv_projection)
                prog.set_vec2("inv_viewport", inv_viewport)
                for batch in batches:
                    batch.draw()

            # 3. thickness (additive point sprites, depth-tested against the scene)
            self._attach(self._thickness_tex, with_depth_buffer=False, depth_tex=host._frame_depth_texture)
            gl.glClearBufferfv(gl.GL_COLOR, 0, (gl.GLfloat * 4)(0.0, 0.0, 0.0, 0.0))
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glDepthMask(False)
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE)
            if hasattr(gl, "GL_PROGRAM_POINT_SIZE"):
                gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
            with self._thickness_prog as prog:
                prog.set_mat4("view", view)
                prog.set_mat4("projection", projection)
                prog.set_float("point_scale", point_scale)
                for batch in batches:
                    prog.set_float("thickness_scale", batch.thickness_scale)
                    prog.set_float("thickness_gain", batch.thickness_gain)
                    batch.draw()
            gl.glDisable(gl.GL_BLEND)
            gl.glDepthMask(True)

            # 4. bilateral depth blur (single 2D pass, adaptive radius)
            self._attach(self._depth_smooth_tex, with_depth_buffer=False)
            gl.glDisable(gl.GL_DEPTH_TEST)
            with self._blur_prog as prog:
                gl.glActiveTexture(gl.GL_TEXTURE0)
                gl.glBindTexture(gl.GL_TEXTURE_2D, self._depth_tex)
                prog.set_int("depth_tex", 0)
                prog.set_float("blur_radius_world", material.blur_radius_world)
                prog.set_float("blur_scale", point_scale)
                prog.set_float("max_blur_radius", material.max_blur_radius)
                self._draw_quad()

            # 5. composite into the frame buffer; the fragment shader writes the
            # fluid depth, so the scene depth test resolves occlusion
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, host._frame_fbo)
            gl.glDrawBuffer(gl.GL_COLOR_ATTACHMENT0)
            gl.glViewport(0, 0, width, height)
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glDepthFunc(gl.GL_LESS)
            gl.glDepthMask(True)
            gl.glDisable(gl.GL_BLEND)
            with self._composite_prog as prog:
                gl.glActiveTexture(gl.GL_TEXTURE0)
                gl.glBindTexture(gl.GL_TEXTURE_2D, self._depth_smooth_tex)
                gl.glActiveTexture(gl.GL_TEXTURE1)
                gl.glBindTexture(gl.GL_TEXTURE_2D, self._thickness_tex)
                gl.glActiveTexture(gl.GL_TEXTURE2)
                gl.glBindTexture(gl.GL_TEXTURE_2D, self._scene_tex)
                gl.glActiveTexture(gl.GL_TEXTURE3)
                if host._shadow_texture is not None:
                    gl.glBindTexture(gl.GL_TEXTURE_2D, host._shadow_texture)
                else:
                    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
                prog.set_int("depth_tex", 0)
                prog.set_int("thickness_tex", 1)
                prog.set_int("scene_tex", 2)
                prog.set_int("shadow_tex", 3)
                prog.set_mat4("projection", projection)
                prog.set_mat4("view", view)
                prog.set_mat4("inv_view", inv_view)
                prog.set_mat4("light_transform", light_transform)
                prog.set_vec3("light_dir", light_dir)
                prog.set_vec2("inv_tex_scale", inv_viewport)
                prog.set_vec2("clip_pos_to_eye", clip_pos_to_eye)
                prog.set_vec2("tex_scale", (1.0 / aspect, 1.0))
                prog.set_vec4("color", material.color)
                prog.set_float("ior", material.ior)
                prog.set_vec3("sky_color", host.sky_upper)
                prog.set_vec3("ground_color", host.ambient_ground)
                prog.set_vec3("up_vec", up_vec)
                self._draw_quad()

        # 6. diffuse spray/foam over the composited result
        if diffuse_batches:
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, host._frame_fbo)
            gl.glDrawBuffer(gl.GL_COLOR_ATTACHMENT0)
            gl.glViewport(0, 0, width, height)
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glDepthFunc(gl.GL_LESS)
            gl.glDepthMask(False)
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE_MINUS_SRC_ALPHA)
            with self._diffuse_prog as prog:
                prog.set_mat4("view", view)
                prog.set_mat4("projection", projection)
                for batch in diffuse_batches:
                    batch.sort_for_view(view)
                    prog.set_float("point_scale", batch.radius)
                    prog.set_float("motion_blur_scale", batch.motion_blur_scale)
                    prog.set_float("diffusion", batch.diffusion)
                    prog.set_vec4("color", batch.color)
                    batch.draw()
            gl.glDisable(gl.GL_BLEND)
            gl.glDepthMask(True)

        gl.glDepthFunc(gl.GL_LESS)
