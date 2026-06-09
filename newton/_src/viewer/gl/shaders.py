# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import ctypes

import numpy as np

shadow_vertex_shader = """
#version 330 core
layout (location = 0) in vec3 aPos;

// column vectors of the instance transform matrix
layout (location = 3) in vec4 aInstanceTransform0;
layout (location = 4) in vec4 aInstanceTransform1;
layout (location = 5) in vec4 aInstanceTransform2;
layout (location = 6) in vec4 aInstanceTransform3;

uniform mat4 light_space_matrix;

void main()
{
    mat4 transform = mat4(aInstanceTransform0, aInstanceTransform1, aInstanceTransform2, aInstanceTransform3);
    gl_Position = light_space_matrix * transform * vec4(aPos, 1.0);
}
"""

shadow_fragment_shader = """
#version 330 core

void main() { }
"""


shape_vertex_shader = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;

// column vectors of the instance transform matrix
layout (location = 3) in vec4 aInstanceTransform0;
layout (location = 4) in vec4 aInstanceTransform1;
layout (location = 5) in vec4 aInstanceTransform2;
layout (location = 6) in vec4 aInstanceTransform3;

// colors to use for the checker_enable pattern
layout (location = 7) in vec3 aObjectColor;

// material properties
layout (location = 8) in vec4 aMaterial;

uniform mat4 view;
uniform mat4 projection;
uniform mat4 light_space_matrix;

out vec3 Normal;
out vec3 FragPos;
out vec3 LocalPos;
out vec2 TexCoord;
out vec3 ObjectColor;
out vec4 FragPosLightSpace;
out vec4 Material;

void main()
{
    mat4 transform = mat4(aInstanceTransform0, aInstanceTransform1, aInstanceTransform2, aInstanceTransform3);

    vec4 worldPos = transform * vec4(aPos, 1.0);
    gl_Position = projection * view * worldPos;
    FragPos = vec3(worldPos);
    LocalPos = aPos;

    mat3 rotation = mat3(transform);
    // transpose(inverse(...)) handles non-uniform scale. The extra sign flip for
    // det < 0 keeps shading normals outward when the viewer caches a winding-
    // flipped variant of the source mesh for mirrored instances: the winding
    // swap exposes the originally-back side of the mesh as front-facing, and
    // negating here restores the outward-pointing normal in world space.
    mat3 normalMatrix = transpose(inverse(rotation));
    if (determinant(rotation) < 0.0) normalMatrix = -normalMatrix;
    Normal = normalMatrix * aNormal;
    TexCoord = aTexCoord;
    ObjectColor = aObjectColor;
    FragPosLightSpace = light_space_matrix * worldPos;
    Material = aMaterial;
}
"""

shape_fragment_shader = """
#version 330 core
out vec4 FragColor;

in vec3 Normal;
in vec3 FragPos;
in vec3 LocalPos;
in vec2 TexCoord;
in vec3 ObjectColor; // used as albedo
in vec4 FragPosLightSpace;
in vec4 Material;

uniform vec3 view_pos;
uniform vec3 light_color;
uniform vec3 sky_color;
uniform vec3 ground_color;
uniform vec3 sun_direction;
uniform sampler2D shadow_map;
uniform sampler2D env_map;
uniform float env_intensity;
uniform sampler2D albedo_map;

uniform vec3 fogColor;
uniform int up_axis;

uniform mat4 light_space_matrix;

uniform float shadow_radius;
uniform float diffuse_scale;
uniform float specular_scale;
uniform bool spotlight_enabled;
uniform float shadow_extents;
uniform float exposure;

const float PI = 3.14159265359;

float rand(vec2 co){
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

// Analytic filtering helpers for smooth checker_enable pattern
float filterwidth(vec2 v)
{
    vec2 fw = max(abs(dFdx(v)), abs(dFdy(v)));
    return max(fw.x, fw.y);
}

vec2 bump(vec2 x)
{
    return (floor(x / 2.0) + 2.0 * max(x / 2.0 - floor(x / 2.0) - 0.5, 0.0));
}

float checker(vec2 uv)
{
    float width = filterwidth(uv);
    vec2 p0 = uv - 0.5 * width;
    vec2 p1 = uv + 0.5 * width;

    vec2 i = (bump(p1) - bump(p0)) / width;
    return i.x * i.y + (1.0 - i.x) * (1.0 - i.y);
}

vec2 poissonDisk[16] = vec2[](
   vec2( -0.94201624, -0.39906216 ),
   vec2( 0.94558609, -0.76890725 ),
   vec2( -0.094184101, -0.92938870 ),
   vec2( 0.34495938, 0.29387760 ),
   vec2( -0.91588581, 0.45771432 ),
   vec2( -0.81544232, -0.87912464 ),
   vec2( -0.38277543, 0.27676845 ),
   vec2( 0.97484398, 0.75648379 ),
   vec2( 0.44323325, -0.97511554 ),
   vec2( 0.53742981, -0.47373420 ),
   vec2( -0.26496911, -0.41893023 ),
   vec2( 0.79197514, 0.19090188 ),
   vec2( -0.24188840, 0.99706507 ),
   vec2( -0.81409955, 0.91437590 ),
   vec2( 0.19984126, 0.78641367 ),
   vec2( 0.14383161, -0.14100790 )
);

float ShadowCalculation()
{
    vec3 normal = normalize(Normal);

    if (!gl_FrontFacing)
        normal = -normal;

    vec3 lightDir = normalize(sun_direction);

    // bias in normal dir - adjust for backfacing triangles
    float worldTexel = (shadow_extents * 2.0) / float(4096); // world extent / shadow map resolution
    float normalBias = 2.0 * worldTexel;   // tune ~1-3

    // For backfacing triangles, we might need different bias handling
    vec4 light_space_pos;
    light_space_pos = light_space_matrix * vec4(FragPos + normal * normalBias, 1.0);
    vec3 projCoords = light_space_pos.xyz/light_space_pos.w;

    // map to [0,1]
    projCoords = projCoords * 0.5 + 0.5;
    if (projCoords.z > 1.0)
        return 0.0;
    float frag_depth = projCoords.z;

    // Fade shadow to zero near edges of the shadow map to avoid hard rectangle
    float fade = 1.0;
    float margin = 0.15;
    fade *= smoothstep(0.0, margin, projCoords.x);
    fade *= smoothstep(0.0, margin, 1.0 - projCoords.x);
    fade *= smoothstep(0.0, margin, projCoords.y);
    fade *= smoothstep(0.0, margin, 1.0 - projCoords.y);

    // Slope-scaled depth bias: more bias when surface is nearly parallel to light
    // (where self-shadowing from float precision is worst), minimal when facing light.
    float NdotL_bias = max(dot(normal, lightDir), 0.0);
    float depthBias = mix(0.0003, 0.00002, NdotL_bias);
    float biased_depth = frag_depth - depthBias;

    float shadow = 0.0;
    float radius = shadow_radius;
    vec2 texelSize = 1.0 / textureSize(shadow_map, 0);
    float angle = rand(gl_FragCoord.xy) * 2.0 * PI;
    float s = sin(angle);
    float c = cos(angle);
    mat2 rotationMatrix = mat2(c, -s, s, c);
    for(int i = 0; i < 16; i++)
    {
        vec2 offset = rotationMatrix * poissonDisk[i];
        float pcf_depth = texture(shadow_map, projCoords.xy + offset * radius * texelSize).r;
        if(pcf_depth < biased_depth)
            shadow += 1.0;
    }
    shadow /= 16.0;
    return shadow * fade;
}

float SpotlightAttenuation()
{
    if (!spotlight_enabled)
        return 1.0;

    // Calculate spotlight position as 20 units from the camera in sun direction
    vec3 spotlight_pos = view_pos + sun_direction * 20.0;

    // Vector from fragment to spotlight
    vec3 fragToLight = normalize(spotlight_pos - FragPos);

    // Angle between spotlight direction (towards origin) and vector from light to fragment
    float cosAngle = dot(normalize(sun_direction), fragToLight);

    // Fixed cone angles (inner: 30 degrees, outer: 45 degrees)
    float cosInnerAngle = cos(radians(30.0));
    float cosOuterAngle = cos(radians(45.0));

    // Smooth falloff between inner and outer cone
    float intensity = smoothstep(cosOuterAngle, cosInnerAngle, cosAngle);

    return intensity;
}

vec3 sample_env_map(vec3 dir, float lod)
{
    // dir assumed normalized
    // Convert to a Y-up reference frame before equirect sampling.
    vec3 dir_up = dir;
    if (up_axis == 0) {
        dir_up = vec3(-dir.y, dir.x, dir.z); // X-up -> Y-up
    } else if (up_axis == 2) {
        dir_up = vec3(dir.x, dir.z, -dir.y); // Z-up -> Y-up
    }
    float u = atan(dir_up.z, dir_up.x) / (2.0 * PI) + 0.5;
    float v = asin(clamp(dir_up.y, -1.0, 1.0)) / PI + 0.5;
    return textureLod(env_map, vec2(fract(u), clamp(v, 0.001, 0.999)), lod).rgb;
}

void main()
{
    // material properties from vertex shader
    float roughness = clamp(Material.x, 0.0, 1.0);
    float metallic = clamp(Material.y, 0.0, 1.0);
    float checker_enable = Material.z;
    float texture_enable = Material.w;
    float checker_scale = 1.0;

    // convert to linear space
    vec3 albedo = pow(ObjectColor, vec3(2.2));
    if (texture_enable > 0.5)
    {
        vec3 tex_color = texture(albedo_map, TexCoord).rgb;
        albedo *= pow(tex_color, vec3(2.2));
    }

    // Optional checker pattern in object-space so it follows instance transforms
    if (checker_enable > 0.0)
    {
        vec2 uv = LocalPos.xy * checker_scale;
        float cb = checker(uv);
        vec3 albedo2 = albedo*0.7;
        // pick between the two colors
        albedo = mix(albedo, albedo2, cb);
    }

    // Specular color: dielectrics ~0.04, metals use albedo.
    // Computed before desaturation so F0 reflects true material reflectance.
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    // Metals appear paler/desaturated because their look is dominated by
    // bright specular reflections.  Without full IBL we approximate this by
    // lifting the albedo toward a brighter, less saturated version.
    float luma = dot(albedo, vec3(0.2126, 0.7152, 0.0722));
    albedo = mix(albedo, vec3(luma * 1.4), metallic * 0.45);

    // surface vectors
    vec3 N = normalize(Normal);
    vec3 V = normalize(view_pos - FragPos);
    // Flip normal for backfacing triangles
    if (!gl_FrontFacing) N = -N;
    vec3 L = normalize(sun_direction);
    vec3 H = normalize(V + L);

    // Cook-Torrance PBR
    float NdotL = max(dot(N, L), 0.0);
    float NdotH = max(dot(N, H), 0.0);
    float NdotV = max(dot(N, V), 0.001);
    float HdotV = max(dot(H, V), 0.0);

    // GGX/Trowbridge-Reitz normal distribution
    float a = roughness * roughness;
    float a2 = a * a;
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    float D = a2 / (PI * denom * denom);

    // Schlick-GGX geometry function (Smith method for both view and light)
    float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    float G1_V = NdotV / (NdotV * (1.0 - k) + k);
    float G1_L = NdotL / (NdotL * (1.0 - k) + k);
    float G = G1_V * G1_L;

    // Schlick Fresnel, dampened by roughness to reduce edge aliasing
    vec3 F_max = mix(F0, vec3(1.0), 1.0 - roughness);
    vec3 F = F0 + (F_max - F0) * pow(1.0 - HdotV, 5.0);

    // Cook-Torrance specular BRDF
    vec3 spec = (D * G * F) / (4.0 * NdotV * NdotL + 0.0001);

    // Diffuse uses remaining energy not reflected
    vec3 kD = (1.0 - F) * (1.0 - metallic);
    vec3 diffuse = kD * albedo / PI;

    // Direct lighting
    vec3 Lo = (diffuse * diffuse_scale + spec * specular_scale) * light_color * NdotL * 3.0;

    // Hemispherical ambient (kept subtle for depth)
    vec3 up = vec3(0.0, 1.0, 0.0);
    if (up_axis == 0) up = vec3(1.0, 0.0, 0.0);
    if (up_axis == 2) up = vec3(0.0, 0.0, 1.0);
    float sky_fac = dot(N, up) * 0.5 + 0.5;
    vec3 ambient = mix(ground_color, sky_color, sky_fac) * albedo * 0.7;
    // Fresnel-weighted ambient specular — only significant for metals
    // (dielectrics need a prefiltered IBL for correct ambient specular)
    vec3 F_ambient = F0 + (F_max - F0) * pow(1.0 - NdotV, 5.0);
    vec3 kD_ambient = (1.0 - F_ambient) * (1.0 - metallic);
    vec3 ambient_spec = F_ambient * mix(ground_color, sky_color, sky_fac) * 0.35;
    ambient = kD_ambient * ambient + ambient_spec * metallic;

    // shadows
    float shadow = ShadowCalculation();

    float spotAttenuation = SpotlightAttenuation();
    vec3 color = ambient + (1.0 - shadow) * spotAttenuation * Lo;

    // Environment / image-based lighting for metals
    vec3 R = reflect(-V, N);
    float env_lod = roughness * 8.0;
    vec3 env_color = pow(sample_env_map(R, env_lod), vec3(2.2));
    vec3 env_F = F0 + (F_max - F0) * pow(1.0 - NdotV, 5.0);
    vec3 env_spec = env_color * env_F * env_intensity;
    color += env_spec * metallic;

    // fog
    float dist = length(FragPos - view_pos);
    float fog_start = 20.0;
    float fog_end   = 200.0;
    float fog_factor = clamp((dist - fog_start) / (fog_end - fog_start), 0.0, 1.0);
    color = mix(color, pow(fogColor, vec3(2.2)), fog_factor);

    // ACES filmic tone mapping
    color = color * exposure;
    vec3 x = color;
    color = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
    color = clamp(color, 0.0, 1.0);

    // gamma correction (sRGB)
    color = pow(color, vec3(1.0 / 2.2));

    FragColor = vec4(color, 1.0);
}
"""


sky_vertex_shader = """
#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;

uniform mat4 view;
uniform mat4 projection;
uniform vec3 view_pos;

uniform float far_plane;

out vec3 FragPos;
out vec2 TexCoord;

void main()
{
    vec4 worldPos = vec4(aPos * far_plane + view_pos, 1.0);
    gl_Position = projection * view * worldPos;

    FragPos = vec3(worldPos);
    TexCoord = aTexCoord;
}
"""

sky_fragment_shader = """
#version 330 core

out vec4 FragColor;

in vec3 FragPos;
in vec2 TexCoord;

uniform vec3 sky_upper;
uniform vec3 sky_lower;
uniform float far_plane;

uniform vec3 sun_direction;
uniform int up_axis;

void main()
{
    float h = up_axis == 0 ? FragPos.x : (up_axis == 1 ? FragPos.y : FragPos.z);
    float height = max(0.0, h / far_plane);
    vec3 sky = mix(sky_lower, sky_upper, height);

    float diff = max(dot(sun_direction, normalize(FragPos)), 0.0);
    vec3 sun = pow(diff, 32) * vec3(1.0, 0.8, 0.6) * 0.5;

    FragColor = vec4(sky + sun, 1.0);
}
"""

frame_vertex_shader = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

void main() {
    gl_Position = vec4(aPos, 1.0);
    TexCoord = aTexCoord;
}
"""

frame_fragment_shader = """
#version 330 core
in vec2 TexCoord;

out vec4 FragColor;

uniform sampler2D texture_sampler;

void main() {
    FragColor = texture(texture_sampler, TexCoord);
}
"""

fluid_particle_vertex_shader = """
#version 330 core
layout (location = 0) in vec4 aParticle;
layout (location = 1) in vec4 aAnisotropy;
layout (location = 2) in vec4 aAnisotropySecondary;
layout (location = 3) in vec4 aAnisotropyTertiary;

uniform mat4 view;
uniform mat4 projection;

out vec3 CenterEye;
out float Radius;
out vec3 MajorAxisEye;
out vec3 SecondaryAxisEye;
out vec3 TertiaryAxisEye;
out float Stretch;
out float MinorScale;
out float DepthScale;

void main()
{
    vec4 center_eye4 = view * vec4(aParticle.xyz, 1.0);
    gl_Position = projection * center_eye4;
    CenterEye = center_eye4.xyz;
    Radius = max(aParticle.w, 1.0e-6);
    vec3 major_axis = aAnisotropy.xyz;
    if (dot(major_axis, major_axis) < 1.0e-8)
        major_axis = vec3(1.0, 0.0, 0.0);
    MajorAxisEye = mat3(view) * normalize(major_axis);
    vec3 secondary_axis = aAnisotropySecondary.xyz;
    if (dot(secondary_axis, secondary_axis) < 1.0e-8)
        secondary_axis = vec3(0.0, 1.0, 0.0);
    SecondaryAxisEye = mat3(view) * normalize(secondary_axis);
    vec3 tertiary_axis = aAnisotropyTertiary.xyz;
    if (dot(tertiary_axis, tertiary_axis) < 1.0e-8)
        tertiary_axis = vec3(0.0, 0.0, 1.0);
    TertiaryAxisEye = mat3(view) * normalize(tertiary_axis);
    Stretch = clamp(aAnisotropy.w, 1.0, 2.0);
    MinorScale = clamp(aAnisotropySecondary.w, 0.05, 2.0);
    DepthScale = clamp(aAnisotropyTertiary.w, 0.05, 2.0);
}
"""

fluid_particle_geometry_shader = """
#version 330 core
layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

in vec3 CenterEye[];
in float Radius[];
in vec3 MajorAxisEye[];
in vec3 SecondaryAxisEye[];
in vec3 TertiaryAxisEye[];
in float Stretch[];
in float MinorScale[];
in float DepthScale[];

uniform mat4 projection;

flat out vec3 ParticleCenterEye;
flat out vec3 ParticleMajorDir;
flat out vec3 ParticleSideDir;
flat out vec3 ParticleDepthDir;
flat out vec3 ParticleBillboardMajorDir;
flat out vec3 ParticleBillboardSideDir;
flat out float ParticleMajorRadius;
flat out float ParticleSideRadius;
flat out float ParticleDepthRadius;
flat out float ParticleBillboardMajorRadius;
flat out float ParticleBillboardSideRadius;
out vec2 LocalCoord;

void emit_corner(vec2 local)
{
    vec3 axis_eye = normalize(MajorAxisEye[0]);
    vec2 major_xy = axis_eye.xy;
    float axis_screen_len = clamp(length(major_xy), 0.0, 1.0);
    if (axis_screen_len < 1.0e-4)
        major_xy = vec2(1.0, 0.0);
    else
        major_xy /= axis_screen_len;

    float stretch = clamp(Stretch[0], 1.0, 2.0);
    float ellipsoid_major_radius = Radius[0] * stretch;
    float ellipsoid_minor_radius = Radius[0] * clamp(MinorScale[0], 0.05, 2.0);
    float ellipsoid_depth_radius = Radius[0] * clamp(DepthScale[0], 0.05, 2.0);
    float billboard_side_radius = max(ellipsoid_minor_radius, ellipsoid_depth_radius);
    float projected_major_radius = sqrt(
        ellipsoid_minor_radius * ellipsoid_minor_radius
            + (ellipsoid_major_radius * ellipsoid_major_radius - ellipsoid_minor_radius * ellipsoid_minor_radius)
                * axis_screen_len * axis_screen_len
    );
    vec3 billboard_major_dir = vec3(major_xy, 0.0);
    vec3 billboard_side_dir = vec3(-major_xy.y, major_xy.x, 0.0);
    vec3 ellipsoid_side_dir = SecondaryAxisEye[0] - axis_eye * dot(SecondaryAxisEye[0], axis_eye);
    if (dot(ellipsoid_side_dir, ellipsoid_side_dir) < 1.0e-8) {
        vec3 side_seed = abs(axis_eye.z) < 0.96 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
        ellipsoid_side_dir = cross(side_seed, axis_eye);
    }
    ellipsoid_side_dir = normalize(ellipsoid_side_dir);
    vec3 ellipsoid_depth_dir = TertiaryAxisEye[0]
        - axis_eye * dot(TertiaryAxisEye[0], axis_eye)
        - ellipsoid_side_dir * dot(TertiaryAxisEye[0], ellipsoid_side_dir);
    if (dot(ellipsoid_depth_dir, ellipsoid_depth_dir) < 1.0e-8)
        ellipsoid_depth_dir = cross(axis_eye, ellipsoid_side_dir);
    ellipsoid_depth_dir = normalize(ellipsoid_depth_dir);

    vec3 eye_pos = CenterEye[0]
        + billboard_major_dir * (local.x * projected_major_radius)
        + billboard_side_dir * (local.y * billboard_side_radius);
    gl_Position = projection * vec4(eye_pos, 1.0);
    ParticleCenterEye = CenterEye[0];
    ParticleMajorDir = axis_eye;
    ParticleSideDir = ellipsoid_side_dir;
    ParticleDepthDir = ellipsoid_depth_dir;
    ParticleBillboardMajorDir = billboard_major_dir;
    ParticleBillboardSideDir = billboard_side_dir;
    ParticleMajorRadius = ellipsoid_major_radius;
    ParticleSideRadius = ellipsoid_minor_radius;
    ParticleDepthRadius = ellipsoid_depth_radius;
    ParticleBillboardMajorRadius = projected_major_radius;
    ParticleBillboardSideRadius = billboard_side_radius;
    LocalCoord = local;
    EmitVertex();
}

void main()
{
    emit_corner(vec2(-1.0, -1.0));
    emit_corner(vec2( 1.0, -1.0));
    emit_corner(vec2(-1.0,  1.0));
    emit_corner(vec2( 1.0,  1.0));
    EndPrimitive();
}
"""

fluid_particle_fragment_shader = """
#version 330 core
flat in vec3 ParticleCenterEye;
flat in vec3 ParticleMajorDir;
flat in vec3 ParticleSideDir;
flat in vec3 ParticleDepthDir;
flat in vec3 ParticleBillboardMajorDir;
flat in vec3 ParticleBillboardSideDir;
flat in float ParticleMajorRadius;
flat in float ParticleSideRadius;
flat in float ParticleDepthRadius;
flat in float ParticleBillboardMajorRadius;
flat in float ParticleBillboardSideRadius;
in vec2 LocalCoord;

out float FragValue;

uniform mat4 projection;
uniform mat4 inv_projection;
uniform vec2 texel_size;
uniform int output_thickness;
uniform float thickness_scale;

void reconstruct_view_ray(vec2 uv, out vec3 ray_origin, out vec3 ray_dir)
{
    vec2 ndc_xy = uv * 2.0 - vec2(1.0);
    vec4 near_view = inv_projection * vec4(ndc_xy, -1.0, 1.0);
    vec4 far_view = inv_projection * vec4(ndc_xy, 1.0, 1.0);
    near_view /= max(abs(near_view.w), 1.0e-6);
    far_view /= max(abs(far_view.w), 1.0e-6);
    ray_origin = near_view.xyz;
    ray_dir = normalize(far_view.xyz - near_view.xyz);
}

bool intersect_ellipsoid(
    vec3 ray_origin,
    vec3 ray_dir,
    out float t_entry,
    out float t_exit
)
{
    vec3 major_dir = normalize(ParticleMajorDir);
    vec3 side_dir = normalize(ParticleSideDir);
    vec3 depth_dir = normalize(ParticleDepthDir);
    vec3 offset = ray_origin - ParticleCenterEye;
    vec3 inv_radii = 1.0 / max(vec3(ParticleMajorRadius, ParticleSideRadius, ParticleDepthRadius), vec3(1.0e-6));

    vec3 d = vec3(dot(ray_dir, major_dir), dot(ray_dir, side_dir), dot(ray_dir, depth_dir)) * inv_radii;
    vec3 o = vec3(dot(offset, major_dir), dot(offset, side_dir), dot(offset, depth_dir)) * inv_radii;
    float a = dot(d, d);
    float b = 2.0 * dot(o, d);
    float c = dot(o, o) - 1.0;
    float discriminant = b * b - 4.0 * a * c;
    if (a <= 1.0e-8 || discriminant < 0.0) {
        return false;
    }

    float root = sqrt(discriminant);
    float inv_denom = 0.5 / a;
    t_entry = (-b - root) * inv_denom;
    t_exit = (-b + root) * inv_denom;
    return t_exit >= 0.0;
}

void main()
{
    vec3 ray_origin;
    vec3 ray_dir;
    reconstruct_view_ray(gl_FragCoord.xy * texel_size, ray_origin, ray_dir);

    float t_entry;
    float t_exit;
    float thickness = 0.0;
    vec3 eye_pos;
    if (intersect_ellipsoid(ray_origin, ray_dir, t_entry, t_exit)) {
        t_entry = max(t_entry, 0.0);
        if (t_exit <= t_entry)
            discard;
        eye_pos = ray_origin + ray_dir * t_entry;
        thickness = t_exit - t_entry;
    } else {
        float local_r2 = dot(LocalCoord, LocalCoord);
        if (local_r2 > 1.0)
            discard;
        float sphere_z = sqrt(max(1.0 - local_r2, 0.0)) * ParticleDepthRadius;
        eye_pos = ParticleCenterEye
            + normalize(ParticleBillboardMajorDir) * (LocalCoord.x * ParticleBillboardMajorRadius)
            + normalize(ParticleBillboardSideDir) * (LocalCoord.y * ParticleBillboardSideRadius);
        eye_pos.z += sphere_z;
        thickness = 2.0 * sphere_z;
    }

    vec4 clip = projection * vec4(eye_pos, 1.0);
    float ndc_depth = clip.z / clip.w;
    gl_FragDepth = ndc_depth * 0.5 + 0.5;

    if (output_thickness != 0)
        FragValue = thickness * thickness_scale;
    else
        FragValue = -eye_pos.z;
}
"""

fluid_diffuse_vertex_shader = """
#version 330 core
layout (location = 0) in vec4 aPositionLife;
layout (location = 1) in vec4 aVelocity;

uniform mat4 view;
uniform mat4 projection;

out vec4 WorldPosLife;
out vec3 ViewPos;
out vec3 ViewVel;
out float NeighborCount;

void main()
{
    WorldPosLife = aPositionLife;
    ViewPos = (view * vec4(aPositionLife.xyz, 1.0)).xyz;
    ViewVel = (view * vec4(aVelocity.xyz, 0.0)).xyz;
    NeighborCount = max(aVelocity.w, 0.0);
    gl_Position = projection * vec4(ViewPos, 1.0);
}
"""

fluid_diffuse_geometry_shader = """
#version 330 core
layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

in vec4 WorldPosLife[];
in vec3 ViewPos[];
in vec3 ViewVel[];
in float NeighborCount[];

uniform mat4 projection;
uniform float radius;
uniform float motion_blur_scale;
uniform float diffuse_expansion;

out vec2 TexCoord;
out float ParticleAlpha;
flat out float ParticleFoamDensity;
flat out float ParticleViewDepth;
flat out float ParticleViewRadius;
flat out float ParticleNeighbors;
flat out vec3 ParticleWorldPos;

void emit_corner(vec3 p, vec2 uv)
{
    TexCoord = uv;
    gl_Position = projection * vec4(p, 1.0);
    EmitVertex();
}

void main()
{
    float life = WorldPosLife[0].w;
    if (life <= 0.0) {
        return;
    }

    vec3 p = ViewPos[0];
    vec3 v = ViewVel[0];
    float neighbors = max(NeighborCount[0], 0.0);
    float spray_weight = 1.0 - smoothstep(1.5, 6.0, neighbors);
    float foam_weight = smoothstep(1.0, 5.0, neighbors) * (1.0 - smoothstep(13.0, 24.0, neighbors));
    float bubble_weight = smoothstep(8.0, 18.0, neighbors);
    float type_weight = max(spray_weight + foam_weight + bubble_weight, 1.0e-4);
    float foam_density = smoothstep(2.5, 11.0, neighbors) * (1.0 - smoothstep(22.0, 30.0, neighbors));
    float bubble_density = smoothstep(13.0, 26.0, neighbors);
    float type_radius = (0.72 * spray_weight + 1.10 * foam_weight + 0.82 * bubble_weight) / type_weight;
    type_radius *= mix(0.92, 1.20, foam_density) * mix(1.0, 0.88, bubble_density);
    float motion_blur_gain = (1.08 * spray_weight + 0.28 * foam_weight + 0.10 * bubble_weight) / type_weight;

    float base_radius = max(radius * type_radius, 0.0001);
    float age = 1.0 - life;
    float expansion_gain = (0.88 * spray_weight + 1.18 * foam_weight + 0.64 * bubble_weight) / type_weight;
    float death_expansion = 1.0
        + max(diffuse_expansion, 0.0) * expansion_gain * (1.0 - smoothstep(0.08, 0.42, life));
    float expansion = death_expansion;
    vec3 up = vec3(0.0, base_radius * expansion, 0.0);
    vec3 right = vec3(base_radius * expansion, 0.0, 0.0);
    float stretch_fade = 1.0 / max(death_expansion * death_expansion, 1.0);

    float speed = length(v) * motion_blur_scale * motion_blur_gain;
    if (speed > 0.8) {
        float max_stretch = base_radius * mix(2.15, 1.12, foam_density) * mix(1.0, 0.72, bubble_density);
        float stretch = clamp(max(base_radius, speed * 0.0045), base_radius, max_stretch);
        up = normalize(v) * stretch;
        vec3 side = cross(up, vec3(0.0, 0.0, -1.0));
        if (length(side) < 1.0e-5) {
            side = vec3(1.0, 0.0, 0.0);
        }
        right = normalize(side) * base_radius * expansion;
        stretch_fade *= min(1.0, 2.0 / max(stretch / base_radius, 1.0));
    }

    vec4 center_clip = projection * vec4(p, 1.0);
    if (center_clip.w <= 0.0) {
        return;
    }
    vec2 center_ndc = center_clip.xy / center_clip.w;
    float view_extent = max(length(up), length(right));
    vec2 ndc_margin = abs(vec2(projection[0][0], projection[1][1])) * view_extent / max(center_clip.w, 1.0e-6);
    ndc_margin = max(ndc_margin, vec2(0.035));
    if (center_ndc.x < -1.0 - ndc_margin.x || center_ndc.x > 1.0 + ndc_margin.x ||
        center_ndc.y < -1.0 - ndc_margin.y || center_ndc.y > 1.0 + ndc_margin.y) {
        return;
    }

    float fade_in = smoothstep(0.0, 0.045, age);
    float fade_out = smoothstep(0.0, 0.18, life);
    ParticleAlpha = fade_in * fade_out * stretch_fade;
    ParticleFoamDensity = clamp(foam_density + bubble_density * 0.35, 0.0, 1.0);
    ParticleViewDepth = -p.z;
    ParticleViewRadius = max(base_radius * expansion, abs(up.z) + abs(right.z));
    ParticleNeighbors = NeighborCount[0];
    ParticleWorldPos = WorldPosLife[0].xyz;

    emit_corner(p + up - right, vec2(0.0, 1.0));
    emit_corner(p - up - right, vec2(0.0, 0.0));
    emit_corner(p + up + right, vec2(1.0, 1.0));
    emit_corner(p - up + right, vec2(1.0, 0.0));
    EndPrimitive();
}
"""

fluid_diffuse_fragment_shader = """
#version 330 core
out vec4 FragColor;

in vec2 TexCoord;
in float ParticleAlpha;
flat in float ParticleFoamDensity;
flat in float ParticleViewDepth;
flat in float ParticleViewRadius;
flat in float ParticleNeighbors;

uniform vec3 diffuse_color;
uniform float alpha;
uniform sampler2D scene_depth_texture;
uniform sampler2D fluid_depth_texture;
uniform sampler2D shadow_map;
uniform mat4 inv_projection;
uniform mat4 light_space_matrix;
uniform vec3 sun_direction_view;
uniform vec2 texel_size;
uniform int depth_mode;
uniform float inscatter;
uniform float outscatter;
uniform float shadow_strength;
uniform bool shadow_enabled;

flat in vec3 ParticleWorldPos;

float diffuse_particle_shadow(vec3 world_pos)
{
    if (!shadow_enabled || shadow_strength <= 0.0) {
        return 0.0;
    }

    vec4 light_clip = light_space_matrix * vec4(world_pos, 1.0);
    vec3 proj = light_clip.xyz / max(abs(light_clip.w), 1.0e-6);
    proj = proj * 0.5 + vec3(0.5);
    if (proj.z > 1.0 || any(lessThan(proj.xy, vec2(0.0))) || any(greaterThan(proj.xy, vec2(1.0)))) {
        return 0.0;
    }

    vec2 texel = 1.0 / vec2(textureSize(shadow_map, 0));
    vec2 taps[8] = vec2[](
        vec2(-0.75, -0.25),
        vec2(-0.25,  0.65),
        vec2( 0.45, -0.55),
        vec2( 0.80,  0.35),
        vec2(-0.55,  0.85),
        vec2( 0.15, -0.90),
        vec2( 0.95, -0.05),
        vec2(-0.95, -0.70)
    );

    float biased_depth = proj.z - 0.00025;
    float shadow = 0.0;
    for (int i = 0; i < 8; ++i) {
        float map_depth = texture(shadow_map, proj.xy + taps[i] * texel * 2.25).r;
        shadow += map_depth < biased_depth ? 1.0 : 0.0;
    }
    return shadow * 0.125;
}

float diffuse_hash(vec2 p)
{
    p = fract(p * vec2(127.1, 311.7));
    p += dot(p, p + 37.1);
    return fract(p.x * p.y);
}

float diffuse_noise(vec2 p)
{
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    float a = diffuse_hash(i);
    float b = diffuse_hash(i + vec2(1.0, 0.0));
    float c = diffuse_hash(i + vec2(0.0, 1.0));
    float d = diffuse_hash(i + vec2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

float diffuse_breakup(vec2 disk, vec3 world_pos, float neighbors)
{
    vec2 floor_p = world_pos.xy * 37.0 + world_pos.zy * 9.0;
    float angle = diffuse_hash(floor_p) * 6.2831853;
    mat2 rot = mat2(cos(angle), -sin(angle), sin(angle), cos(angle));
    vec2 p = floor_p + rot * disk * 3.35;
    float coarse = diffuse_noise(p * 0.55);
    float fine = diffuse_noise(p * 2.35 + vec2(11.7, -5.3));
    float edge = smoothstep(0.42, 0.96, dot(disk, disk));
    float spray = 1.0 - smoothstep(4.0, 13.0, neighbors);
    return mix(0.74, 1.12, coarse) * mix(0.82, 1.08, fine) * mix(1.0, 0.62 + 0.38 * fine, edge * spray);
}

float scene_view_depth(vec2 uv, float device_depth)
{
    vec4 clip = vec4(uv * 2.0 - vec2(1.0), device_depth * 2.0 - 1.0, 1.0);
    vec4 view = inv_projection * clip;
    view /= max(view.w, 1.0e-6);
    return -view.z;
}

void main()
{
    vec2 disk = TexCoord * 2.0 - vec2(1.0);
    float r2 = dot(disk, disk);
    if (r2 > 1.0) {
        discard;
    }

    float sphere_z = sqrt(max(1.0 - r2, 0.0));
    float diffuse_view_depth = max(ParticleViewDepth - sphere_z * ParticleViewRadius, 0.0);

    vec2 uv = gl_FragCoord.xy * texel_size;
    float scene_depth = texture(scene_depth_texture, uv).r;
    float scene_surface_fade = 1.0;
    if (scene_depth < 1.0) {
        float scene_depth_overlap = diffuse_view_depth - scene_view_depth(uv, scene_depth);
        float scene_fade_width = max(ParticleViewRadius * 1.2, 0.010);
        scene_surface_fade = 1.0 - smoothstep(0.0, scene_fade_width, scene_depth_overlap);
        if (scene_surface_fade <= 0.001)
            discard;
    }

    float fluid_depth = texture(fluid_depth_texture, uv).r;
    float fluid_surface_fade = 1.0;
    if (depth_mode == 1) {
        if (fluid_depth <= 0.0) {
            discard;
        }
        float fluid_gap = diffuse_view_depth - fluid_depth;
        float fluid_fade_width = max(ParticleViewRadius * 1.4, 0.018);
        fluid_surface_fade = smoothstep(-ParticleViewRadius * 0.25, fluid_fade_width, fluid_gap);
        if (fluid_surface_fade <= 0.001)
            discard;
    } else if (depth_mode == 2) {
        if (fluid_depth > 0.0) {
            float fluid_gap = diffuse_view_depth - fluid_depth;
            float fluid_fade_width = max(ParticleViewRadius * 1.4, 0.018);
            fluid_surface_fade = 1.0 - smoothstep(-ParticleViewRadius * 0.25, fluid_fade_width, fluid_gap);
            if (fluid_surface_fade <= 0.001)
                discard;
        }
    }

    float z = 1.0 - r2;
    float foam_density = clamp(ParticleFoamDensity, 0.0, 1.0);
    float soft = mix(z * z, smoothstep(0.0, 1.0, z), foam_density * 0.45);
    float breakup = diffuse_breakup(disk, ParticleWorldPos, ParticleNeighbors);
    breakup = mix(breakup, max(breakup, 0.82), foam_density * 0.65);
    vec3 normal_view = normalize(vec3(disk * 0.42, max(z, 0.0)));
    vec3 light_view = normalize(sun_direction_view);
    float ndotl = clamp(dot(normal_view, light_view), 0.0, 1.0);
    float backscatter = pow(1.0 - max(normal_view.z, 0.0), 2.0);
    float spray = 1.0 - smoothstep(1.5, 6.0, ParticleNeighbors);
    float foam = smoothstep(1.0, 5.0, ParticleNeighbors) * (1.0 - smoothstep(13.0, 24.0, ParticleNeighbors));
    float bubble = smoothstep(8.0, 18.0, ParticleNeighbors);
    float type_sum = max(spray + foam + bubble, 1.0e-4);

    vec3 spray_color = mix(vec3(0.74, 0.90, 1.0), diffuse_color, 0.82);
    vec3 foam_color = mix(vec3(0.92, 0.98, 1.0), diffuse_color, 0.72);
    vec3 bubble_color = vec3(0.38, 0.78, 1.0);
    vec3 color = (spray_color * spray + foam_color * foam + bubble_color * bubble) / type_sum;

    float core = pow(z, mix(3.0, 1.35, foam_density));
    color += foam_color * core * (0.10 + 0.16 * max(spray, foam));
    color += foam_color * foam_density * core * 0.18;
    color *= mix(vec3(1.0), vec3(0.86, 0.94, 1.0), clamp((1.0 - breakup) * spray, 0.0, 0.35));
    float scatter = inscatter * (0.24 + 0.76 * ndotl) * (0.45 + 0.55 * core);
    scatter += inscatter * backscatter * (0.10 + 0.18 * spray);
    color += vec3(0.52, 0.84, 1.0) * scatter;
    color *= exp(-outscatter * (1.0 - soft) * (0.45 + 0.80 * bubble));
    float shadow_amount = diffuse_particle_shadow(ParticleWorldPos) * clamp(shadow_strength, 0.0, 1.0);
    color *= mix(vec3(1.0), vec3(0.60, 0.74, 0.92), shadow_amount);

    float type_alpha = (0.78 * spray + 1.04 * foam + 0.44 * bubble) / type_sum;
    type_alpha *= mix(0.80, 1.28, foam_density) * mix(1.0, 0.78, bubble);
    if (depth_mode == 1) {
        type_alpha *= mix(0.18, 0.36, bubble);
        color = mix(color, bubble_color, 0.45 + 0.25 * bubble);
    } else if (depth_mode == 2) {
        type_alpha *= 1.0 - 0.62 * bubble;
    }

    float a = clamp(alpha, 0.0, 1.0) * ParticleAlpha * soft * type_alpha * breakup * fluid_surface_fade * scene_surface_fade;
    FragColor = vec4(color * a, a);
}
"""

fluid_blur_fragment_shader = """
#version 330 core
in vec2 TexCoord;

out float FragValue;

uniform sampler2D depth_texture;
uniform sampler2D guide_texture;
uniform vec2 texel_size;
uniform vec2 direction;
uniform float filter_radius;
uniform float max_depth_delta;
uniform float depth_edge_falloff;
uniform int max_radial_samples;
uniform int use_guide_texture;

float edge_weight(float sample_guide, float center_guide)
{
    if (sample_guide <= 0.0 || center_guide <= 0.0 || depth_edge_falloff <= 0.0) {
        return 0.0;
    }

    float delta = abs(sample_guide - center_guide);
    float edge_width = max(max_depth_delta * depth_edge_falloff, 1.0e-6);
    return 1.0 - smoothstep(edge_width * 0.35, edge_width, delta);
}

void main()
{
    float center = texture(depth_texture, TexCoord).r;
    float center_guide = center;
    if (use_guide_texture != 0) {
        center_guide = texture(guide_texture, TexCoord).r;
    }
    if (center <= 0.0 || center_guide <= 0.0) {
        FragValue = 0.0;
        return;
    }

    float weights[5] = float[](0.204164, 0.304005, 0.093913, 0.010381, 0.000873);
    float sum = center * weights[0];
    float weight_sum = weights[0];
    int sample_count = clamp(max_radial_samples, 0, 4);

    for (int i = 1; i < 5; ++i) {
        if (i > sample_count) {
            continue;
        }
        vec2 offset = direction * texel_size * filter_radius * float(i);
        float d0 = texture(depth_texture, TexCoord + offset).r;
        float d1 = texture(depth_texture, TexCoord - offset).r;
        float g0 = d0;
        float g1 = d1;
        if (use_guide_texture != 0) {
            g0 = texture(guide_texture, TexCoord + offset).r;
            g1 = texture(guide_texture, TexCoord - offset).r;
        }
        float w = weights[i];
        float e0 = edge_weight(g0, center_guide);
        float e1 = edge_weight(g1, center_guide);
        if (e0 > 0.0) {
            sum += d0 * w * e0;
            weight_sum += w * e0;
        }
        if (e1 > 0.0) {
            sum += d1 * w * e1;
            weight_sum += w * e1;
        }
    }

    FragValue = sum / max(weight_sum, 1.0e-6);
}
"""

fluid_shadow_fragment_shader = """
#version 330 core
in vec2 TexCoord;

out vec4 FragColor;

uniform sampler2D scene_texture;
uniform sampler2D scene_depth_texture;
uniform sampler2D fluid_shadow_depth_texture;
uniform sampler2D fluid_shadow_thickness_texture;
uniform mat4 inv_projection;
uniform mat4 inv_view;
uniform mat4 light_projection;
uniform mat4 light_view;
uniform mat4 light_space_matrix;
uniform vec3 sun_direction_world;
uniform vec3 fluid_bounds_lower;
uniform vec3 fluid_bounds_upper;
uniform float caustic_scale;
uniform float floor_caustic_strength;
uniform float surface_shadow_strength;
uniform int up_axis;

vec3 reconstruct_device_depth_view_pos(vec2 uv, float depth)
{
    vec4 clip = vec4(uv * 2.0 - vec2(1.0), depth * 2.0 - 1.0, 1.0);
    vec4 view = clip * inv_projection;
    view /= max(abs(view.w), 1.0e-6);
    return view.xyz;
}

vec3 view_to_world(vec3 view_pos)
{
    vec4 world = vec4(view_pos, 1.0) * inv_view;
    return world.xyz / max(abs(world.w), 1.0e-6);
}

vec2 world_floor_coord(vec3 world_pos)
{
    if (up_axis == 0)
        return world_pos.yz;
    if (up_axis == 1)
        return world_pos.xz;
    return world_pos.xy;
}

float world_up_coord(vec3 world_pos)
{
    if (up_axis == 0)
        return world_pos.x;
    if (up_axis == 1)
        return world_pos.y;
    return world_pos.z;
}

float fluid_bounds_floor_density(vec3 receiver_world, out float bounds_depth)
{
    vec2 receiver_floor = world_floor_coord(receiver_world);
    vec2 floor_min = min(world_floor_coord(fluid_bounds_lower), world_floor_coord(fluid_bounds_upper));
    vec2 floor_max = max(world_floor_coord(fluid_bounds_lower), world_floor_coord(fluid_bounds_upper));
    vec2 extent = max(floor_max - floor_min, vec2(1.0e-4));
    float feather = clamp(max(extent.x, extent.y) * 0.045, 0.035, 0.18);

    float mask_x = smoothstep(floor_min.x - feather, floor_min.x + feather, receiver_floor.x)
        * (1.0 - smoothstep(floor_max.x - feather, floor_max.x + feather, receiver_floor.x));
    float mask_y = smoothstep(floor_min.y - feather, floor_min.y + feather, receiver_floor.y)
        * (1.0 - smoothstep(floor_max.y - feather, floor_max.y + feather, receiver_floor.y));

    float water_top = max(world_up_coord(fluid_bounds_lower), world_up_coord(fluid_bounds_upper));
    float water_bottom = min(world_up_coord(fluid_bounds_lower), world_up_coord(fluid_bounds_upper));
    bounds_depth = max(water_top - max(world_up_coord(receiver_world), water_bottom), 0.0);
    float vertical_mask = smoothstep(0.015, 0.18, bounds_depth);
    return mask_x * mask_y * vertical_mask;
}

float fluid_projected_bounds_density(vec3 receiver_world, out float bounds_depth)
{
    vec3 lower = min(fluid_bounds_lower, fluid_bounds_upper);
    vec3 upper = max(fluid_bounds_lower, fluid_bounds_upper);
    vec3 dir = normalize(sun_direction_world);
    vec3 inv_dir = vec3(
        abs(dir.x) > 1.0e-5 ? 1.0 / dir.x : 1.0e5,
        abs(dir.y) > 1.0e-5 ? 1.0 / dir.y : 1.0e5,
        abs(dir.z) > 1.0e-5 ? 1.0 / dir.z : 1.0e5
    );
    vec3 t0 = (lower - receiver_world) * inv_dir;
    vec3 t1 = (upper - receiver_world) * inv_dir;
    vec3 t_min = min(t0, t1);
    vec3 t_max = max(t0, t1);
    float t_enter = max(max(t_min.x, t_min.y), t_min.z);
    float t_exit = min(min(t_max.x, t_max.y), t_max.z);
    float entry = max(t_enter, 0.0);
    bounds_depth = max(t_exit - entry, 0.0);
    float hit = step(entry, t_exit) * step(0.0, t_exit);
    return hit * smoothstep(0.015, 0.30, bounds_depth);
}

float stable_floor_caustics(vec2 floor_pos, float water_depth, float water_thickness)
{
    vec2 p = floor_pos * vec2(0.86, 1.14) * caustic_scale * 0.36;
    float a = sin(p.x + 0.72 * sin(p.y * 1.43));
    float b = sin(p.y * 1.21 + 0.60 * sin(p.x * 1.09));
    float c = sin((p.x - p.y) * 1.67 + 0.42 * sin(p.x + p.y));
    float d = sin((p.x + p.y) * 0.93 + 0.55 * sin(p.y));
    float la = 1.0 - smoothstep(0.025, 0.18, abs(a));
    float lb = 1.0 - smoothstep(0.025, 0.18, abs(b));
    float lc = 1.0 - smoothstep(0.030, 0.22, abs(c));
    float ld = 1.0 - smoothstep(0.030, 0.22, abs(d));
    float lines = max(max(la, lb), max(lc, ld));
    float intersections = clamp(la * lb + lc * ld + max(la, lc) * max(lb, ld) * 0.35, 0.0, 1.0);
    float web = pow(lines, 2.8) * 0.55 + pow(intersections, 1.6) * 0.80;
    float focus = smoothstep(0.03, 0.20, water_depth) * (1.0 - smoothstep(3.2, 7.5, water_depth));
    float volume = smoothstep(0.002, 0.065, water_thickness);
    return web * focus * volume;
}

vec2 refracted_floor_caustic_coord(
    vec3 receiver_world,
    vec2 shadow_uv,
    float fluid_depth,
    float fluid_thickness,
    float water_depth
)
{
    vec2 floor_pos = world_floor_coord(receiver_world);
    vec2 texel = 1.0 / vec2(textureSize(fluid_shadow_depth_texture, 0));
    vec2 uv_l = clamp(shadow_uv - vec2(texel.x, 0.0), vec2(0.0), vec2(1.0));
    vec2 uv_r = clamp(shadow_uv + vec2(texel.x, 0.0), vec2(0.0), vec2(1.0));
    vec2 uv_d = clamp(shadow_uv - vec2(0.0, texel.y), vec2(0.0), vec2(1.0));
    vec2 uv_u = clamp(shadow_uv + vec2(0.0, texel.y), vec2(0.0), vec2(1.0));

    float d_l = texture(fluid_shadow_depth_texture, uv_l).r;
    float d_r = texture(fluid_shadow_depth_texture, uv_r).r;
    float d_d = texture(fluid_shadow_depth_texture, uv_d).r;
    float d_u = texture(fluid_shadow_depth_texture, uv_u).r;
    float t_l = texture(fluid_shadow_thickness_texture, uv_l).r;
    float t_r = texture(fluid_shadow_thickness_texture, uv_r).r;
    float t_d = texture(fluid_shadow_thickness_texture, uv_d).r;
    float t_u = texture(fluid_shadow_thickness_texture, uv_u).r;

    d_l = mix(fluid_depth, d_l, step(1.0e-5, d_l) * step(1.0e-5, t_l));
    d_r = mix(fluid_depth, d_r, step(1.0e-5, d_r) * step(1.0e-5, t_r));
    d_d = mix(fluid_depth, d_d, step(1.0e-5, d_d) * step(1.0e-5, t_d));
    d_u = mix(fluid_depth, d_u, step(1.0e-5, d_u) * step(1.0e-5, t_u));
    t_l = mix(fluid_thickness, t_l, step(1.0e-5, t_l));
    t_r = mix(fluid_thickness, t_r, step(1.0e-5, t_r));
    t_d = mix(fluid_thickness, t_d, step(1.0e-5, t_d));
    t_u = mix(fluid_thickness, t_u, step(1.0e-5, t_u));

    vec2 depth_gradient = vec2(d_r - d_l, d_u - d_d);
    vec2 thickness_gradient = vec2(t_r - t_l, t_u - t_d);
    vec2 lens_gradient = depth_gradient * 0.58 + thickness_gradient * 0.26;
    float lens_strength = smoothstep(0.004, 0.10, fluid_thickness)
        * smoothstep(0.02, 0.36, water_depth)
        * (1.0 - smoothstep(5.0, 9.0, water_depth));
    vec2 refracted_floor_offset = lens_gradient * lens_strength * clamp(water_depth * 0.38, 0.0, 1.15);
    return floor_pos + refracted_floor_offset;
}

void sample_fluid_volume(vec2 shadow_uv, out float fluid_depth, out float fluid_thickness, out float fluid_coverage)
{
    vec2 texel = 1.0 / vec2(textureSize(fluid_shadow_thickness_texture, 0));
    vec2 taps[9] = vec2[](
        vec2( 0.0,  0.0),
        vec2( 1.0,  0.0),
        vec2(-1.0,  0.0),
        vec2( 0.0,  1.0),
        vec2( 0.0, -1.0),
        vec2( 0.75,  0.75),
        vec2(-0.75,  0.75),
        vec2( 0.75, -0.75),
        vec2(-0.75, -0.75)
    );
    float weights[9] = float[](1.0, 0.64, 0.64, 0.64, 0.64, 0.42, 0.42, 0.42, 0.42);

    fluid_depth = 0.0;
    fluid_thickness = 0.0;
    fluid_coverage = 0.0;
    float depth_weight = 0.0;
    float total_weight = 0.0;
    for (int i = 0; i < 9; ++i) {
        vec2 uv = shadow_uv + taps[i] * texel * 1.65;
        if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) {
            continue;
        }
        float w = weights[i];
        float d = texture(fluid_shadow_depth_texture, uv).r;
        float t = texture(fluid_shadow_thickness_texture, uv).r;
        float valid = step(1.0e-5, d) * step(1.0e-5, t);
        fluid_depth += d * w * valid;
        depth_weight += w * valid;
        fluid_thickness += t * w;
        fluid_coverage += w * valid;
        total_weight += w;
    }

    fluid_depth /= max(depth_weight, 1.0e-6);
    fluid_thickness /= max(total_weight, 1.0e-6);
    fluid_coverage /= max(total_weight, 1.0e-6);
}

float light_space_caustic_focus(vec2 shadow_uv, float fluid_depth, float fluid_thickness, float fluid_coverage)
{
    vec2 texel = 1.0 / vec2(textureSize(fluid_shadow_depth_texture, 0));
    vec2 uv_l = clamp(shadow_uv - vec2(texel.x, 0.0), vec2(0.0), vec2(1.0));
    vec2 uv_r = clamp(shadow_uv + vec2(texel.x, 0.0), vec2(0.0), vec2(1.0));
    vec2 uv_d = clamp(shadow_uv - vec2(0.0, texel.y), vec2(0.0), vec2(1.0));
    vec2 uv_u = clamp(shadow_uv + vec2(0.0, texel.y), vec2(0.0), vec2(1.0));

    float d_l = texture(fluid_shadow_depth_texture, uv_l).r;
    float d_r = texture(fluid_shadow_depth_texture, uv_r).r;
    float d_d = texture(fluid_shadow_depth_texture, uv_d).r;
    float d_u = texture(fluid_shadow_depth_texture, uv_u).r;
    float t_l = texture(fluid_shadow_thickness_texture, uv_l).r;
    float t_r = texture(fluid_shadow_thickness_texture, uv_r).r;
    float t_d = texture(fluid_shadow_thickness_texture, uv_d).r;
    float t_u = texture(fluid_shadow_thickness_texture, uv_u).r;

    d_l = mix(fluid_depth, d_l, step(1.0e-5, d_l) * step(1.0e-5, t_l));
    d_r = mix(fluid_depth, d_r, step(1.0e-5, d_r) * step(1.0e-5, t_r));
    d_d = mix(fluid_depth, d_d, step(1.0e-5, d_d) * step(1.0e-5, t_d));
    d_u = mix(fluid_depth, d_u, step(1.0e-5, d_u) * step(1.0e-5, t_u));
    t_l = mix(fluid_thickness, t_l, step(1.0e-5, t_l));
    t_r = mix(fluid_thickness, t_r, step(1.0e-5, t_r));
    t_d = mix(fluid_thickness, t_d, step(1.0e-5, t_d));
    t_u = mix(fluid_thickness, t_u, step(1.0e-5, t_u));

    vec2 depth_gradient = vec2(d_r - d_l, d_u - d_d);
    vec2 thickness_gradient = vec2(t_r - t_l, t_u - t_d);
    float depth_curvature = abs((d_l + d_r + d_d + d_u) - 4.0 * fluid_depth);
    float thickness_ridge = 1.0 - smoothstep(0.035, 0.22, length(thickness_gradient));
    float focus = smoothstep(0.0008, 0.026, depth_curvature + length(depth_gradient) * 0.42);
    focus *= smoothstep(0.004, 0.085, fluid_thickness) * smoothstep(0.12, 0.72, fluid_coverage);
    return focus * mix(0.72, 1.18, thickness_ridge);
}

void main()
{
    vec3 scene = texture(scene_texture, TexCoord).rgb;
    float raw_scene_depth = texture(scene_depth_texture, TexCoord).r;
    if (raw_scene_depth >= 0.9999) {
        FragColor = vec4(scene, 1.0);
        return;
    }

    vec3 view_pos = reconstruct_device_depth_view_pos(TexCoord, raw_scene_depth);
    vec3 world_pos = view_to_world(view_pos);

    vec4 light_clip = light_space_matrix * vec4(world_pos, 1.0);
    float valid_light_projection = step(1.0e-6, light_clip.w);
    vec3 light_ndc = light_clip.xyz / max(light_clip.w, 1.0e-6);
    vec2 shadow_uv = light_ndc.xy * 0.5 + vec2(0.5);
    valid_light_projection *= step(0.0, shadow_uv.x) * step(shadow_uv.x, 1.0);
    valid_light_projection *= step(0.0, shadow_uv.y) * step(shadow_uv.y, 1.0);

    float fluid_light_depth = 0.0;
    float fluid_thickness = 0.0;
    float fluid_coverage = 0.0;
    if (valid_light_projection > 0.0) {
        sample_fluid_volume(shadow_uv, fluid_light_depth, fluid_thickness, fluid_coverage);
    }

    float projected_bounds_depth = 0.0;
    float projected_bounds_density = fluid_projected_bounds_density(world_pos, projected_bounds_depth);
    float coverage_density = smoothstep(0.06, 0.55, fluid_coverage);
    float has_light_volume = valid_light_projection * projected_bounds_density
        * step(0.035, fluid_coverage) * step(1.0e-4, fluid_thickness);
    vec4 light_view_pos = light_view * vec4(world_pos, 1.0);
    float receiver_light_depth = -light_view_pos.z;
    // Light-space water column between the fluid front and this floor point. The fluid-front
    // depth can be missing where the light-space splats are sparse, so clamp it to zero rather
    // than culling on it: the shadow is driven by optical density (thickness/coverage), and the
    // old `fluid_light_depth <= 0` / `water_depth <= 0.015` gates rejected nearly all of the floor.
    float water_depth = max(receiver_light_depth - fluid_light_depth, 0.0);

    float bounds_depth = 0.0;
    float bounds_density = fluid_bounds_floor_density(world_pos, bounds_depth);
    float receiver_light_separation = abs(receiver_light_depth - fluid_light_depth);
    float receiver_behind_fluid = max(
        smoothstep(0.010, 0.12, receiver_light_separation) * has_light_volume,
        projected_bounds_density
    );

    if (projected_bounds_density <= 0.0 || receiver_behind_fluid <= 0.0) {
        FragColor = vec4(scene, 1.0);
        return;
    }

    // Optical density of the water column between the light and this receiver.
    float effective_thickness = max(
        fluid_thickness * has_light_volume,
        max(bounds_depth * bounds_density, projected_bounds_depth * projected_bounds_density) * 0.18
    );
    float density = max(
        smoothstep(0.0015, 0.060, fluid_thickness) * coverage_density,
        max(
            bounds_density * smoothstep(0.03, 0.36, bounds_depth) * 0.28,
            projected_bounds_density * smoothstep(0.04, 0.72, projected_bounds_depth) * 0.56
        )
    ) * receiver_behind_fluid;
    // Caustics are a near-floor refractive effect, so keep them depth-windowed (but relaxed).
    float caustic_depth = max(water_depth, 0.05);
    caustic_depth = max(caustic_depth, max(bounds_depth, projected_bounds_depth));
    float caustic_volume = density
        * smoothstep(0.02, 0.30, caustic_depth)
        * (1.0 - smoothstep(6.0, 13.0, caustic_depth));

    float light_focus = 0.0;
    vec2 refracted_floor_coord = world_floor_coord(world_pos);
    if (has_light_volume > 0.0) {
        light_focus = light_space_caustic_focus(shadow_uv, fluid_light_depth, fluid_thickness, fluid_coverage);
        refracted_floor_coord = refracted_floor_caustic_coord(
            world_pos,
            shadow_uv,
            fluid_light_depth,
            fluid_thickness,
            water_depth
        );
    }
    refracted_floor_coord = mix(world_floor_coord(world_pos), refracted_floor_coord, has_light_volume);
    float caustic = stable_floor_caustics(refracted_floor_coord, max(water_depth, 0.05), effective_thickness);
    caustic = clamp(caustic * (0.70 + 0.34 * light_focus) + light_focus * 0.055, 0.0, 1.0);

    // Transparent blue shadow cast by the water: a soft multiply tinted blue and deepened
    // by fluid density (a thicker column absorbs more red/green, leaving a bluer shadow).
    float shadow_opacity = clamp(
        surface_shadow_strength * density * (0.70 + 0.30 * max(coverage_density, bounds_density)) * 1.65,
        0.0,
        0.48
    );
    vec3 absorption_tint = exp(-vec3(1.08, 0.52, 0.10) * clamp(effective_thickness * 2.2 + caustic_depth * 0.030, 0.0, 2.4));
    vec3 transmitted_shadow = scene * vec3(0.55, 0.74, 1.06) * absorption_tint + vec3(0.00, 0.10, 0.26) * density;
    vec3 shadowed = mix(scene, transmitted_shadow, shadow_opacity);

    vec3 caustic_color = vec3(0.22, 0.76, 1.12);
    shadowed += caustic_color * caustic * clamp(floor_caustic_strength * 0.13, 0.0, 0.36)
        * caustic_volume * (0.54 + 0.20 * coverage_density);

    FragColor = vec4(clamp(shadowed, 0.0, 1.0), 1.0);
}
"""

fluid_composite_fragment_shader = """
#version 330 core
in vec2 TexCoord;

out vec4 FragColor;

uniform sampler2D scene_texture;
uniform sampler2D fluid_depth_texture;
uniform sampler2D thickness_texture;
uniform sampler2D env_map;
uniform sampler2D scene_depth_texture;
uniform sampler2D shadow_map;
uniform mat4 projection;
uniform mat4 inv_projection;
uniform mat4 inv_view;
uniform mat4 light_space_matrix;
uniform mat3 inv_view_rotation;
uniform vec3 fluid_bounds_lower;
uniform vec3 fluid_bounds_upper;
uniform vec2 texel_size;
uniform vec3 water_color;
uniform vec3 water_deep_color;
uniform float color_gradient_strength;
uniform float opacity;
uniform float reflection_strength;
uniform float refraction_strength;
uniform float env_map_strength;
uniform float env_reflection_lod;
uniform float env_color_preserve;
uniform float env_intensity;
uniform int env_map_available;
uniform vec3 sky_reflection_color;
uniform vec3 ground_reflection_color;
uniform float absorption_strength;
uniform float depth_visualization_strength;
uniform float caustic_strength;
uniform float caustic_scale;
uniform float floor_caustic_strength;
uniform float surface_shadow_strength;
uniform float shadow_radius;
uniform int up_axis;
uniform vec3 sun_direction_view;

const float PI = 3.14159265359;

vec3 reconstruct_view_pos(vec2 uv, float linear_depth)
{
    vec4 clip = vec4(uv * 2.0 - vec2(1.0), 1.0, 1.0);
    vec4 far_view = clip * inv_projection;
    far_view /= max(abs(far_view.w), 1.0e-6);
    vec3 ray = normalize(far_view.xyz);
    return ray * (linear_depth / max(-ray.z, 1.0e-5));
}

vec3 reconstruct_device_depth_view_pos(vec2 uv, float depth)
{
    vec4 clip = vec4(uv * 2.0 - vec2(1.0), depth * 2.0 - 1.0, 1.0);
    vec4 view = clip * inv_projection;
    view /= max(abs(view.w), 1.0e-6);
    return view.xyz;
}

vec3 view_to_world(vec3 view_pos)
{
    vec4 world = vec4(view_pos, 1.0) * inv_view;
    return world.xyz / max(abs(world.w), 1.0e-6);
}

vec2 world_floor_coord(vec3 world_pos)
{
    if (up_axis == 0)
        return world_pos.yz;
    if (up_axis == 1)
        return world_pos.xz;
    return world_pos.xy;
}

float world_up_coord(vec3 world_pos)
{
    if (up_axis == 0)
        return world_pos.x;
    if (up_axis == 1)
        return world_pos.y;
    return world_pos.z;
}

vec3 world_up_vector()
{
    if (up_axis == 0)
        return vec3(1.0, 0.0, 0.0);
    if (up_axis == 1)
        return vec3(0.0, 1.0, 0.0);
    return vec3(0.0, 0.0, 1.0);
}

vec3 fluid_normal(vec2 uv, float depth)
{
    float center_thickness = texture(thickness_texture, uv).r;
    vec2 safe_min = texel_size * 0.5;
    vec2 safe_max = vec2(1.0) - safe_min;
    float normal_radius = clamp(1.5 + center_thickness * 7.0, 1.5, 5.0);
    vec2 dx = vec2(texel_size.x, 0.0) * normal_radius;
    vec2 dy = vec2(0.0, texel_size.y) * normal_radius;
    vec2 uv_l = clamp(uv - dx, safe_min, safe_max);
    vec2 uv_r = clamp(uv + dx, safe_min, safe_max);
    vec2 uv_d = clamp(uv - dy, safe_min, safe_max);
    vec2 uv_u = clamp(uv + dy, safe_min, safe_max);

    float depth_l = texture(fluid_depth_texture, uv_l).r;
    float depth_r = texture(fluid_depth_texture, uv_r).r;
    float depth_d = texture(fluid_depth_texture, uv_d).r;
    float depth_u = texture(fluid_depth_texture, uv_u).r;
    float thick_l = texture(thickness_texture, uv_l).r;
    float thick_r = texture(thickness_texture, uv_r).r;
    float thick_d = texture(thickness_texture, uv_d).r;
    float thick_u = texture(thickness_texture, uv_u).r;
    float min_neighbor_thickness = max(center_thickness * 0.035, 1.0e-5);
    if (depth_l <= 0.0 || thick_l <= min_neighbor_thickness) depth_l = depth;
    if (depth_r <= 0.0 || thick_r <= min_neighbor_thickness) depth_r = depth;
    if (depth_d <= 0.0 || thick_d <= min_neighbor_thickness) depth_d = depth;
    if (depth_u <= 0.0 || thick_u <= min_neighbor_thickness) depth_u = depth;

    vec3 p = reconstruct_view_pos(uv, depth);
    vec3 v = normalize(-p);
    vec3 p_l = reconstruct_view_pos(uv_l, depth_l);
    vec3 p_r = reconstruct_view_pos(uv_r, depth_r);
    vec3 p_d = reconstruct_view_pos(uv_d, depth_d);
    vec3 p_u = reconstruct_view_pos(uv_u, depth_u);
    vec3 tangent_x = p_r - p_l;
    vec3 tangent_y = p_u - p_d;
    vec3 normal_cross = cross(tangent_x, tangent_y);
    if (dot(normal_cross, normal_cross) < 1.0e-10) {
        return v;
    }

    vec3 n = normalize(normal_cross);
    if (dot(n, v) < 0.0)
        n = -n;

    float support = smoothstep(0.0025, 0.035, center_thickness);
    return normalize(mix(v, n, support));
}

vec3 sample_env_map(vec3 dir, float lod)
{
    vec3 dir_up = dir;
    if (up_axis == 0) {
        dir_up = vec3(-dir.y, dir.x, dir.z);
    } else if (up_axis == 2) {
        dir_up = vec3(dir.x, dir.z, -dir.y);
    }
    float u = atan(dir_up.z, dir_up.x) / (2.0 * PI) + 0.5;
    float v = asin(clamp(dir_up.y, -1.0, 1.0)) / PI + 0.5;
    return textureLod(env_map, vec2(fract(u), clamp(v, 0.001, 0.999)), lod).rgb;
}

vec3 water_parallax_env_dir(vec3 reflection_dir, vec3 surface_world)
{
    vec3 dir = normalize(reflection_dir);
    vec3 lower = min(fluid_bounds_lower, fluid_bounds_upper);
    vec3 upper = max(fluid_bounds_lower, fluid_bounds_upper);
    vec3 bounds_size = max(upper - lower, vec3(1.0e-4));
    vec3 center = (lower + upper) * 0.5;
    float largest_extent = max(max(bounds_size.x, bounds_size.y), bounds_size.z);
    vec3 half_extent = max(bounds_size * 1.35, vec3(max(largest_extent * 0.65, 0.75)));

    vec3 safe_dir = dir;
    safe_dir.x = abs(safe_dir.x) < 1.0e-5 ? (safe_dir.x < 0.0 ? -1.0e-5 : 1.0e-5) : safe_dir.x;
    safe_dir.y = abs(safe_dir.y) < 1.0e-5 ? (safe_dir.y < 0.0 ? -1.0e-5 : 1.0e-5) : safe_dir.y;
    safe_dir.z = abs(safe_dir.z) < 1.0e-5 ? (safe_dir.z < 0.0 ? -1.0e-5 : 1.0e-5) : safe_dir.z;

    vec3 t0 = (center - half_extent - surface_world) / safe_dir;
    vec3 t1 = (center + half_extent - surface_world) / safe_dir;
    vec3 t_far = max(t0, t1);
    float t_hit = min(min(t_far.x, t_far.y), t_far.z);
    if (t_hit <= 1.0e-4) {
        return dir;
    }

    vec3 hit = surface_world + dir * t_hit;
    return normalize(hit - center);
}

vec3 sample_water_env_map(vec3 reflection_dir, vec3 surface_world, vec3 n_world, float lod)
{
    vec3 dir = normalize(reflection_dir);
    vec3 boxed_dir = water_parallax_env_dir(dir, surface_world);
    vec3 angular = sample_env_map(dir, lod);
    vec3 boxed = sample_env_map(boxed_dir, lod);
    float top_face = smoothstep(0.18, 0.76, abs(world_up_coord(normalize(n_world))));
    float grazing = smoothstep(0.18, 0.92, 1.0 - abs(dot(normalize(n_world), dir)));
    return mix(angular, boxed, clamp(0.62 + 0.22 * top_face + 0.10 * grazing, 0.0, 0.92));
}

float hash21(vec2 p)
{
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float value_noise(vec2 p)
{
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

float fbm(vec2 p)
{
    float value = 0.0;
    float amp = 0.5;
    for (int i = 0; i < 4; ++i) {
        value += amp * value_noise(p);
        p = p * 2.03 + vec2(17.1, -9.2);
        amp *= 0.5;
    }
    return value;
}

vec3 water_env_tint(vec3 env_color, vec3 water_tint, float preserve_color)
{
    float luma = dot(env_color, vec3(0.2126, 0.7152, 0.0722));
    vec3 filtered = mix(vec3(luma), env_color, preserve_color);
    vec3 aqua_bias = mix(vec3(0.74, 1.03, 1.12), vec3(1.0), preserve_color);
    return filtered * aqua_bias + water_tint * luma * mix(0.16, 0.04, preserve_color);
}

vec3 water_env_limit(vec3 env_color, float luma_limit)
{
    float luma = max(dot(env_color, vec3(0.2126, 0.7152, 0.0722)), 0.0);
    float safe_limit = max(luma_limit, 1.0e-4);
    float compressed_luma = safe_limit * (1.0 - exp(-luma / safe_limit));
    return env_color * (compressed_luma / max(luma, 1.0e-4));
}

vec3 water_sky_ground_reflection(vec3 reflection_dir, vec3 water_tint, float preserve_color)
{
    vec3 r = normalize(reflection_dir);
    float up = world_up_coord(r);
    float sky_blend = smoothstep(0.08, 0.38, up);
    float lift = smoothstep(-0.18, 0.30, up);
    vec3 horizon = mix(ground_reflection_color, sky_reflection_color, sky_blend);
    vec3 limited = water_env_limit(horizon * (0.32 + 0.28 * lift), 0.42);
    return water_env_tint(limited, water_tint, clamp(preserve_color, 0.35, 1.0));
}

vec3 water_analytic_environment(vec3 reflection_dir)
{
    vec3 r = normalize(reflection_dir);
    float up = clamp(world_up_coord(r), -1.0, 1.0);
    float sky_blend = smoothstep(-0.08, 0.52, up);
    float horizon = 1.0 - smoothstep(0.10, 0.86, abs(up));
    vec3 horizon_color = mix(ground_reflection_color, sky_reflection_color, 0.38);
    vec3 env = mix(ground_reflection_color, sky_reflection_color, sky_blend);
    return mix(env, horizon_color, horizon * 0.28);
}

vec3 water_directional_env_reflection(
    vec3 reflection_dir,
    vec3 surface_world,
    vec3 n_world,
    vec3 water_tint,
    float preserve_color,
    float lod,
    float gain,
    float fresnel_term,
    float roughness
)
{
    vec3 r = normalize(reflection_dir);
    vec3 env_dir = water_parallax_env_dir(r, surface_world);
    float reflected_up = clamp(world_up_coord(env_dir), -1.0, 1.0);
    float horizon_surface = 1.0 - abs(reflected_up);
    float sky_energy = smoothstep(-0.04, 0.58, reflected_up);

    vec3 env = env_map_available != 0
        ? sample_water_env_map(r, surface_world, n_world, lod)
        : water_analytic_environment(env_dir);
    env *= gain;

    float luma_limit = mix(0.34, 0.95, max(sky_energy, fresnel_term));
    luma_limit += horizon_surface * 0.16;
    luma_limit *= mix(1.0, 0.70, roughness);
    env = water_env_limit(env, luma_limit);
    return water_env_tint(env, water_tint, preserve_color);
}

float scene_depth_gap(vec2 uv, float water_depth)
{
    float raw_depth = texture(scene_depth_texture, uv).r;
    if (raw_depth >= 0.9999) {
        return 999.0;
    }

    vec3 scene_view = reconstruct_device_depth_view_pos(uv, raw_depth);
    return (-scene_view.z) - water_depth;
}

float refraction_visibility(vec2 uv, float water_depth)
{
    return smoothstep(0.010, 0.075, scene_depth_gap(uv, water_depth));
}

vec3 depth_aware_scene_sample(vec2 uv, float water_depth, vec3 fallback_scene)
{
    vec2 sample_uv = clamp(uv, vec2(0.0), vec2(1.0));
    vec3 sample_scene = texture(scene_texture, sample_uv).rgb;
    return mix(fallback_scene, sample_scene, refraction_visibility(sample_uv, water_depth));
}

vec2 ripple_pattern(vec3 surface_world, vec3 n_world)
{
    vec2 floor_p = world_floor_coord(surface_world);
    vec2 normal_p = world_floor_coord(n_world);
    float height = world_up_coord(surface_world);
    vec2 p = (floor_p + normal_p * 0.060 + vec2(height * 0.10, -height * 0.075)) * vec2(48.0, 70.0);
    float a = sin(p.x + 0.9 * sin(p.y * 0.72));
    float b = sin(p.y * 1.13 + 0.7 * sin(p.x * 0.61));
    float c = sin((p.x + p.y) * 0.57 + 1.2 * sin(p.x * 0.31 - p.y * 0.21));
    return vec2(a + 0.45 * c, b - 0.35 * c);
}

float water_surface_shadow(vec3 surface_world, vec3 n_world)
{
    if (surface_shadow_strength <= 0.0) {
        return 0.0;
    }

    vec4 light_clip = light_space_matrix * vec4(surface_world + n_world * 0.004, 1.0);
    if (light_clip.w <= 0.0) {
        return 0.0;
    }

    vec3 proj = light_clip.xyz / light_clip.w;
    proj = proj * 0.5 + vec3(0.5);
    if (proj.z > 1.0 || any(lessThan(proj.xy, vec2(0.0))) || any(greaterThan(proj.xy, vec2(1.0)))) {
        return 0.0;
    }

    float fade = 1.0;
    float margin = 0.10;
    fade *= smoothstep(0.0, margin, proj.x);
    fade *= smoothstep(0.0, margin, 1.0 - proj.x);
    fade *= smoothstep(0.0, margin, proj.y);
    fade *= smoothstep(0.0, margin, 1.0 - proj.y);

    vec2 texel_size_shadow = 1.0 / textureSize(shadow_map, 0);
    vec2 poisson[8] = vec2[](
        vec2(-0.75, -0.18),
        vec2(-0.34,  0.62),
        vec2( 0.12, -0.78),
        vec2( 0.48,  0.40),
        vec2( 0.86, -0.08),
        vec2(-0.90,  0.50),
        vec2( 0.32,  0.88),
        vec2(-0.18, -0.95)
    );

    float biased_depth = proj.z - 0.00055;
    float shadow = 0.0;
    for (int i = 0; i < 8; ++i) {
        float map_depth = texture(shadow_map, proj.xy + poisson[i] * shadow_radius * texel_size_shadow).r;
        shadow += map_depth < biased_depth ? 1.0 : 0.0;
    }
    return shadow * 0.125 * fade;
}

vec3 tropical_gradient(vec3 surface_world, vec3 n_world, float thickness)
{
    float optical_depth = thickness * max(absorption_strength, 0.0);
    float depth_mix = 1.0 - exp(-optical_depth * (0.85 + 0.25 * depth_visualization_strength));
    vec2 floor_p = world_floor_coord(surface_world);
    vec2 normal_p = world_floor_coord(n_world);
    float height = world_up_coord(surface_world);
    vec2 p = floor_p + normal_p * 0.085 + vec2(height * 0.075, -height * 0.055);
    float wave = 0.5 + 0.5 * sin(p.x * 27.0 + sin(p.y * 39.0));
    float shelf = smoothstep(0.05, 1.3, optical_depth);
    float gradient_mix = clamp(depth_mix * 0.82 + wave * 0.10 + shelf * 0.08, 0.0, 1.0);
    vec3 shallow = mix(water_color, vec3(0.42, 0.76, 1.0), 0.16);
    vec3 mid = mix(shallow, vec3(0.10, 0.48, 0.88), 0.42);
    vec3 graded = mix(mid, water_deep_color, smoothstep(0.18, 0.95, gradient_mix));
    return mix(water_color, graded, color_gradient_strength);
}

void main()
{
    float depth = texture(fluid_depth_texture, TexCoord).r;
    if (depth <= 0.0)
        discard;

    vec3 p = reconstruct_view_pos(TexCoord, depth);
    vec3 n = fluid_normal(TexCoord, depth);
    vec3 v = normalize(-p);
    float surface_thickness = texture(thickness_texture, TexCoord).r;
    vec4 water_clip = projection * vec4(p, 1.0);
    float water_device_depth = clamp(water_clip.z / max(abs(water_clip.w), 1.0e-6) * 0.5 + 0.5, 0.0, 1.0);

    vec3 surface_world = view_to_world(p);
    vec3 n_world_base = normalize(inv_view_rotation * n);

    float surface_optical_depth = surface_thickness * max(absorption_strength, 0.0);
    vec2 ripple = ripple_pattern(surface_world, n_world_base);
    float ripple_weight = smoothstep(0.01, 0.24, surface_thickness)
        * (1.0 - smoothstep(4.0, 8.0, surface_optical_depth));
    n = normalize(n + vec3(ripple * 0.012 * ripple_weight, 0.0));
    float fresnel = pow(clamp(1.0 - dot(n, v), 0.0, 1.0), 5.0);
    float water_fresnel = 0.08 + 0.92 * fresnel;
    vec3 n_world = normalize(inv_view_rotation * n);
    float normal_up = clamp(world_up_coord(n_world), -1.0, 1.0);
    float top_surface = smoothstep(0.16, 0.72, normal_up);
    float side_surface = 1.0 - top_surface;

    float refraction_body = mix(0.28, 1.0, top_surface)
        * mix(1.0, 0.42, smoothstep(1.2, 4.8, surface_optical_depth));
    vec2 refract_offset = n.xy * refraction_strength * (0.22 + clamp(surface_optical_depth, 0.0, 1.35)) * refraction_body;
    refract_offset += ripple * refraction_strength * 0.024 * ripple_weight * top_surface;
    vec2 refract_uv = clamp(TexCoord + refract_offset, vec2(0.0), vec2(1.0));
    float refracted_thickness = texture(thickness_texture, refract_uv).r;
    float thickness = max(refracted_thickness, surface_thickness * 0.35);
    float optical_depth = thickness * max(absorption_strength, 0.0);
    float depth_mix = clamp(1.0 - exp(-optical_depth * 0.85), 0.0, 1.0);
    float alpha = clamp((1.0 - exp(-optical_depth * 1.35)) * opacity, 0.0, opacity);
    vec3 scene = texture(scene_texture, TexCoord).rgb;
    vec3 refracted_r = depth_aware_scene_sample(TexCoord + refract_offset * 1.10, depth, scene);
    vec3 refracted_g = depth_aware_scene_sample(refract_uv, depth, scene);
    vec3 refracted_b = depth_aware_scene_sample(TexCoord + refract_offset * 0.88, depth, scene);
    vec3 scene_refracted = vec3(refracted_r.r, refracted_g.g, refracted_b.b);
    scene_refracted = mix(scene_refracted, scene, side_surface * 0.58);
    vec3 reflection_ray = reflect(-v, n);
    float reflection_slider = clamp(reflection_strength / 0.6, 0.0, 1.0);
    float surface_roughness = clamp(
        0.07 + ripple_weight * 0.26 + depth_mix * 0.14 + (1.0 - water_fresnel) * 0.08 + (1.0 - top_surface) * 0.10,
        0.0,
        1.0
    );
    float water_env_gain = env_intensity / (1.0 + 0.95 * max(env_intensity, 0.0));
    vec3 reflection_world = normalize(inv_view_rotation * reflection_ray);
    vec3 env_dir = water_parallax_env_dir(reflection_world, surface_world);
    float env_reflection_lod_dynamic = clamp(env_reflection_lod + surface_roughness * 2.2 - water_fresnel * 0.34, 0.0, 8.0);
    vec3 refracted_eye = refract(-v, n, 1.0 / 1.333);
    if (length(refracted_eye) < 0.001)
        refracted_eye = reflect(-v, n);
    vec3 refracted_world = normalize(inv_view_rotation * refracted_eye);
    vec3 env_refracted_raw = (env_map_available != 0
            ? sample_env_map(refracted_world, max(env_reflection_lod + 2.0, 2.0))
            : water_analytic_environment(refracted_world))
        * water_env_gain;
    env_refracted_raw = water_env_limit(env_refracted_raw, 0.24) * 0.36;

    vec3 l = normalize(sun_direction_view);
    vec3 h = normalize(l + v);
    float spec = pow(max(dot(n, h), 0.0), 220.0) * (0.34 + env_map_strength * 0.42);
    spec *= mix(0.10, 1.0, top_surface);

    vec3 gradient_color = tropical_gradient(surface_world, n_world, thickness);
    vec3 env_reflected = water_directional_env_reflection(
        reflection_world,
        surface_world,
        n_world,
        gradient_color,
        env_color_preserve,
        env_reflection_lod_dynamic,
        water_env_gain,
        water_fresnel,
        surface_roughness
    );
    vec3 env_refracted = water_env_tint(env_refracted_raw, gradient_color, clamp(env_color_preserve * 0.45, 0.0, 1.0));

    float raw_bottom_depth = texture(scene_depth_texture, TexCoord).r;
    float water_column_depth = 0.0;
    vec3 bottom_view = p;
    vec3 bottom_world = surface_world;
    float scene_gap = 999.0;
    float bottom_visibility = 0.0;
    float foreground_occlusion = 0.0;
    if (raw_bottom_depth < 0.9999) {
        bottom_view = reconstruct_device_depth_view_pos(TexCoord, raw_bottom_depth);
        bottom_world = view_to_world(bottom_view);
        water_column_depth = max(world_up_coord(surface_world) - world_up_coord(bottom_world), 0.0);
        float candidate_gap = (-bottom_view.z) - depth;
        float scene_depth_ahead = water_device_depth - raw_bottom_depth;
        if (candidate_gap > 0.0 || scene_depth_ahead <= 0.00035) {
            scene_gap = max(candidate_gap, 0.0);
            bottom_visibility = 1.0 - smoothstep(0.00015, 0.0015, scene_depth_ahead);
        } else {
            foreground_occlusion = smoothstep(0.00035, 0.0040, scene_depth_ahead);
        }
    }

    float column_optical_depth = max(
        optical_depth,
        water_column_depth * max(absorption_strength, 0.0) * 0.72 * bottom_visibility
    );
    float column_depth_mix = clamp(1.0 - exp(-column_optical_depth * 0.85), 0.0, 1.0);
    vec3 column_absorption = exp(-vec3(1.55, 0.46, 0.10) * column_optical_depth);
    float column_alpha = clamp((1.0 - exp(-column_optical_depth * 1.05)) * opacity, 0.0, opacity);
    alpha = max(alpha, column_alpha);
    alpha *= mix(0.22, 0.58, max(depth_mix, column_depth_mix));
    float env_transmission = clamp(
        env_map_strength * refraction_strength * 2.6 * (0.28 + 0.72 * (1.0 - column_depth_mix)),
        0.0,
        0.22
    );
    vec3 transmitted_scene = mix(scene_refracted, env_refracted, env_transmission);

    float floor_transmission = floor_caustic_strength * 0.35
        * max(smoothstep(0.02, 0.42, water_column_depth), smoothstep(0.020, 0.26, scene_gap) * 0.72)
        * (1.0 - smoothstep(2.8, 5.2, water_column_depth))
        * smoothstep(0.02, 0.65, thickness);
    vec3 floor_translucency = vec3(0.42, 0.82, 1.05) + gradient_color * 0.22;
    float floor_receiver_depth = max(
        smoothstep(0.015, 0.28, water_column_depth),
        smoothstep(0.012, 0.22, scene_gap) * 0.86
    );
    float thickness_receiver = smoothstep(0.006, 0.18, thickness);
    float water_volume_shadow = clamp(
        surface_shadow_strength * thickness_receiver * (0.62 + 0.38 * (1.0 - top_surface)) * 1.85,
        0.0,
        0.82
    );
    float water_floor_shadow = clamp(
        surface_shadow_strength * 1.35
            * bottom_visibility
            * floor_receiver_depth
            * thickness_receiver,
        0.0,
        0.88
    );
    float water_shadow_amount = max(water_floor_shadow, water_volume_shadow);
    vec3 water_shadow_filter = vec3(0.16, 0.32, 0.96);
    transmitted_scene = mix(
        transmitted_scene,
        transmitted_scene * water_shadow_filter + gradient_color * 0.015,
        water_shadow_amount
    );
    transmitted_scene = mix(
        transmitted_scene,
        transmitted_scene * vec3(0.72, 0.90, 1.06) + floor_translucency * 0.18,
        clamp(floor_transmission * 0.16, 0.0, 0.42)
    );

    vec3 transmitted = transmitted_scene * column_absorption * (1.0 - 0.34 * column_depth_mix)
        + gradient_color * (0.30 + 0.54 * column_depth_mix);
    transmitted = mix(transmitted, gradient_color, clamp(0.16 + 0.28 * column_depth_mix, 0.0, 0.46));
    float reflection_volume = smoothstep(0.006, 0.12, thickness);
    float column_reflection = mix(1.0, smoothstep(0.04, 0.75, water_column_depth), bottom_visibility);
    float reflection_visibility = clamp(
        mix(0.22, 1.0, reflection_volume * column_reflection) * (0.72 + 0.28 * water_fresnel),
        0.12,
        1.0
    );
    // View-dependent environment reflection only. Screen-space reflection was removed because
    // its ray-march produced camera-following streaks on the surface; the env map gives a stable,
    // Fresnel-weighted reflection instead.
    vec3 reflection_color = env_reflected;
    float env_shape_reflection = mix(0.18, 0.66, top_surface) * (0.30 + 0.70 * water_fresnel);
    env_shape_reflection += side_surface * (0.08 + 0.22 * water_fresnel);
    float reflection_mix_raw = env_map_strength * (0.10 + 0.46 * water_fresnel + 0.06 * top_surface)
        + reflection_slider * env_shape_reflection;
    float reflection_mix = clamp(reflection_mix_raw * mix(0.58, 1.08, reflection_visibility), 0.0, 0.82);
    vec3 water = mix(transmitted, reflection_color, reflection_mix);
    float reflected_up = clamp(world_up_coord(env_dir), -1.0, 1.0);
    float reflected_sky = smoothstep(-0.02, 0.52, reflected_up);
    float directional_reflection = clamp(
        (reflection_slider * 0.22 + env_map_strength * 0.42) * (0.38 + 0.62 * water_fresnel),
        0.0,
        0.48
    );
    water = mix(water, env_reflected, directional_reflection * reflection_visibility);
    water *= mix(vec3(0.72, 0.86, 1.04), vec3(1.06, 1.09, 1.04), top_surface * (0.55 + 0.45 * reflected_sky));
    water *= mix(vec3(1.0), vec3(0.48, 0.64, 1.02), water_shadow_amount * 0.72);
    water += floor_transmission * floor_translucency * (0.030 + 0.035 * (1.0 - reflection_mix));
    vec3 side_body = mix(
        gradient_color * vec3(0.60, 0.78, 1.05),
        env_reflected,
        clamp(0.16 + reflection_slider * 0.26 + env_map_strength * 0.58, 0.0, 0.58)
    );
    water = mix(water, side_body, clamp(side_surface * 0.86, 0.0, 0.90));
    float view_side = clamp(1.0 - abs(dot(n, v)), 0.0, 1.0);
    float depth_shadow = depth_visualization_strength
        * clamp(max(depth_mix, column_depth_mix) + smoothstep(0.02, 0.45, water_column_depth), 0.0, 1.0);
    water *= mix(vec3(1.0), vec3(0.70, 0.84, 1.0), clamp(depth_shadow * 0.42, 0.0, 0.55));
    water += gradient_color * (0.065 + 0.11 * view_side) * depth_visualization_strength;
    float final_env_reflection = clamp(
        (env_map_strength * 0.38 + reflection_slider * 0.30)
            * (0.32 + 0.68 * water_fresnel)
            * reflection_visibility
            + side_surface * (env_map_strength * 0.20 + reflection_slider * 0.14),
        0.0,
        0.54
    );
    water = mix(water, env_reflected, final_env_reflection);
    float surface_shadow = water_surface_shadow(surface_world, n_world);
    float shadow_amount = clamp(surface_shadow * surface_shadow_strength, 0.0, 0.82);
    water += vec3(spec) * mix(1.0, 0.42, shadow_amount);
    water *= mix(vec3(1.0), vec3(0.70, 0.84, 0.98), shadow_amount);
    alpha = max(alpha, opacity * 0.08 * smoothstep(0.006, 0.080, surface_thickness));
    alpha = max(alpha, opacity * 0.42 * side_surface * smoothstep(0.004, 0.060, surface_thickness));
    alpha = max(
        alpha,
        opacity * 0.12 * reflection_visibility * clamp(reflection_mix + final_env_reflection, 0.0, 1.0)
            * smoothstep(0.006, 0.080, surface_thickness)
    );
    alpha = min(alpha, opacity * mix(0.56, 0.62, clamp(reflection_mix + final_env_reflection, 0.0, 1.0)));
    alpha *= (1.0 - foreground_occlusion);

    gl_FragDepth = mix(water_device_depth, raw_bottom_depth, foreground_occlusion);
    FragColor = vec4(mix(scene, water, alpha), 1.0);
}
"""


def str_buffer(string: str):
    """Convert string to C-style char pointer for OpenGL."""
    return ctypes.c_char_p(string.encode("utf-8"))


def arr_pointer(arr: np.ndarray):
    """Convert numpy array to C-style float pointer for OpenGL."""
    return arr.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))


class ShaderGL:
    """Base class for OpenGL shader wrappers."""

    def __init__(self):
        self.shader_program = None
        self._gl = None

    def _get_uniform_location(self, name: str):
        """Get uniform location for given name."""
        if self.shader_program is None:
            raise RuntimeError("Shader not initialized")
        return self._gl.glGetUniformLocation(self.shader_program.id, str_buffer(name))

    def use(self):
        """Bind this shader for use."""
        if self.shader_program is None:
            raise RuntimeError("Shader not initialized")
        self._gl.glUseProgram(self.shader_program.id)

    def __enter__(self):
        """Context manager entry - bind shader."""
        self.use()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass  # OpenGL doesn't need explicit unbinding


class ShaderShape(ShaderGL):
    """Shader for rendering 3D shapes with lighting and shadows."""

    def __init__(self, gl):
        super().__init__()
        from pyglet.graphics.shader import Shader, ShaderProgram

        self._gl = gl
        self.shader_program = ShaderProgram(
            Shader(shape_vertex_shader, "vertex"), Shader(shape_fragment_shader, "fragment")
        )

        # Get all uniform locations
        with self:
            self.loc_view = self._get_uniform_location("view")
            self.loc_projection = self._get_uniform_location("projection")
            self.loc_view_pos = self._get_uniform_location("view_pos")
            self.loc_light_space_matrix = self._get_uniform_location("light_space_matrix")
            self.loc_shadow_map = self._get_uniform_location("shadow_map")
            self.loc_albedo_map = self._get_uniform_location("albedo_map")
            self.loc_env_map = self._get_uniform_location("env_map")
            self.loc_env_intensity = self._get_uniform_location("env_intensity")
            self.loc_fog_color = self._get_uniform_location("fogColor")
            self.loc_up_axis = self._get_uniform_location("up_axis")
            self.loc_sun_direction = self._get_uniform_location("sun_direction")
            self.loc_light_color = self._get_uniform_location("light_color")
            self.loc_ground_color = self._get_uniform_location("ground_color")
            self.loc_sky_color = self._get_uniform_location("sky_color")
            self.loc_shadow_radius = self._get_uniform_location("shadow_radius")
            self.loc_diffuse_scale = self._get_uniform_location("diffuse_scale")
            self.loc_specular_scale = self._get_uniform_location("specular_scale")
            self.loc_spotlight_enabled = self._get_uniform_location("spotlight_enabled")
            self.loc_shadow_extents = self._get_uniform_location("shadow_extents")
            self.loc_exposure = self._get_uniform_location("exposure")

    def update(
        self,
        view_matrix: np.ndarray,
        projection_matrix: np.ndarray,
        view_pos: tuple[float, float, float],
        fog_color: tuple[float, float, float],
        up_axis: int,
        sun_direction: tuple[float, float, float],
        light_color: tuple[float, float, float] = (2.0, 2.0, 2.0),
        ground_color: tuple[float, float, float] = (0.3, 0.3, 0.35),
        sky_color: tuple[float, float, float] = (0.8, 0.8, 0.85),
        enable_shadows: bool = False,
        shadow_texture: int | None = None,
        light_space_matrix: np.ndarray | None = None,
        env_texture: int | None = None,
        env_intensity: float = 1.0,
        shadow_radius: float = 3.0,
        diffuse_scale: float = 1.0,
        specular_scale: float = 1.0,
        spotlight_enabled: bool = True,
        shadow_extents: float = 10.0,
        exposure: float = 1.6,
    ):
        """Update all shader uniforms."""
        with self:
            # Basic matrices
            self._gl.glUniformMatrix4fv(self.loc_view, 1, self._gl.GL_FALSE, arr_pointer(view_matrix))
            self._gl.glUniformMatrix4fv(self.loc_projection, 1, self._gl.GL_FALSE, arr_pointer(projection_matrix))
            self._gl.glUniform3f(self.loc_view_pos, *view_pos)

            # Lighting
            self._gl.glUniform3f(self.loc_sun_direction, *sun_direction)
            self._gl.glUniform3f(self.loc_light_color, *light_color)
            self._gl.glUniform3f(self.loc_ground_color, *ground_color)
            self._gl.glUniform3f(self.loc_sky_color, *sky_color)
            self._gl.glUniform1f(self.loc_shadow_radius, shadow_radius)
            self._gl.glUniform1f(self.loc_diffuse_scale, diffuse_scale)
            self._gl.glUniform1f(self.loc_specular_scale, specular_scale)
            self._gl.glUniform1i(self.loc_spotlight_enabled, int(spotlight_enabled))
            self._gl.glUniform1f(self.loc_shadow_extents, shadow_extents)
            self._gl.glUniform1f(self.loc_exposure, exposure)

            # Fog and rendering options
            self._gl.glUniform3f(self.loc_fog_color, *fog_color)
            self._gl.glUniform1i(self.loc_up_axis, up_axis)

            # Shadows
            # if enable_shadows and shadow_texture is not None and light_space_matrix is not None:
            self._gl.glActiveTexture(self._gl.GL_TEXTURE0)
            self._gl.glBindTexture(self._gl.GL_TEXTURE_2D, shadow_texture)
            self._gl.glUniform1i(self.loc_shadow_map, 0)
            self._gl.glUniformMatrix4fv(
                self.loc_light_space_matrix, 1, self._gl.GL_FALSE, arr_pointer(light_space_matrix)
            )
            self._gl.glUniform1i(self.loc_albedo_map, 1)
            self._gl.glActiveTexture(self._gl.GL_TEXTURE2)
            if env_texture is not None:
                self._gl.glBindTexture(self._gl.GL_TEXTURE_2D, env_texture)
            else:
                from .opengl import RendererGL  # noqa: PLC0415

                self._gl.glBindTexture(self._gl.GL_TEXTURE_2D, RendererGL.get_fallback_texture())
            self._gl.glUniform1i(self.loc_env_map, 2)
            self._gl.glUniform1f(self.loc_env_intensity, float(env_intensity))


class ShaderSky(ShaderGL):
    """Shader for rendering sky background."""

    def __init__(self, gl):
        super().__init__()
        from pyglet.graphics.shader import Shader, ShaderProgram

        self._gl = gl
        self.shader_program = ShaderProgram(
            Shader(sky_vertex_shader, "vertex"), Shader(sky_fragment_shader, "fragment")
        )

        # Get all uniform locations
        with self:
            self.loc_view = self._get_uniform_location("view")
            self.loc_projection = self._get_uniform_location("projection")
            self.loc_sky_upper = self._get_uniform_location("sky_upper")
            self.loc_sky_lower = self._get_uniform_location("sky_lower")
            self.loc_far_plane = self._get_uniform_location("far_plane")
            self.loc_view_pos = self._get_uniform_location("view_pos")
            self.loc_sun_direction = self._get_uniform_location("sun_direction")
            self.loc_up_axis = self._get_uniform_location("up_axis")

    def update(
        self,
        view_matrix: np.ndarray,
        projection_matrix: np.ndarray,
        camera_pos: tuple[float, float, float],
        camera_far: float,
        sky_upper: tuple[float, float, float],
        sky_lower: tuple[float, float, float],
        sun_direction: tuple[float, float, float],
        up_axis: int = 2,
    ):
        """Update all shader uniforms."""
        with self:
            # Matrices and view position
            self._gl.glUniformMatrix4fv(self.loc_view, 1, self._gl.GL_FALSE, arr_pointer(view_matrix))
            self._gl.glUniformMatrix4fv(self.loc_projection, 1, self._gl.GL_FALSE, arr_pointer(projection_matrix))
            self._gl.glUniform3f(self.loc_view_pos, *camera_pos)
            self._gl.glUniform1f(self.loc_far_plane, camera_far * 0.9)  # moves sphere slightly inside far clip plane

            # Sky colors and settings
            self._gl.glUniform3f(self.loc_sky_upper, *sky_upper)
            self._gl.glUniform3f(self.loc_sky_lower, *sky_lower)
            self._gl.glUniform3f(self.loc_sun_direction, *sun_direction)
            self._gl.glUniform1i(self.loc_up_axis, up_axis)


class ShadowShader(ShaderGL):
    """Shader for rendering shadow maps."""

    def __init__(self, gl):
        super().__init__()
        from pyglet.graphics.shader import Shader, ShaderProgram

        self._gl = gl
        self.shader_program = ShaderProgram(
            Shader(shadow_vertex_shader, "vertex"), Shader(shadow_fragment_shader, "fragment")
        )

        # Get uniform locations
        with self:
            self.loc_light_space_matrix = self._get_uniform_location("light_space_matrix")

    def update(self, light_space_matrix: np.ndarray):
        """Update light space matrix for shadow rendering."""
        with self:
            self._gl.glUniformMatrix4fv(
                self.loc_light_space_matrix, 1, self._gl.GL_FALSE, arr_pointer(light_space_matrix)
            )


class FrameShader(ShaderGL):
    """Shader for rendering the final frame buffer to screen."""

    def __init__(self, gl):
        super().__init__()
        from pyglet.graphics.shader import Shader, ShaderProgram

        self._gl = gl
        self.shader_program = ShaderProgram(
            Shader(frame_vertex_shader, "vertex"), Shader(frame_fragment_shader, "fragment")
        )

        # Get uniform locations
        with self:
            self.loc_texture = self._get_uniform_location("texture_sampler")

    def update(self, texture_unit: int = 0):
        """Update texture uniform."""
        with self:
            self._gl.glUniform1i(self.loc_texture, texture_unit)


class FluidParticleShader(ShaderGL):
    """Shader that rasterizes fluid particles as sphere impostors."""

    def __init__(self, gl):
        super().__init__()
        from pyglet.graphics.shader import Shader, ShaderProgram

        self._gl = gl
        self.shader_program = ShaderProgram(
            Shader(fluid_particle_vertex_shader, "vertex"),
            Shader(fluid_particle_fragment_shader, "fragment"),
            Shader(fluid_particle_geometry_shader, "geometry"),
        )
        with self:
            self.loc_view = self._get_uniform_location("view")
            self.loc_projection = self._get_uniform_location("projection")
            self.loc_inv_projection = self._get_uniform_location("inv_projection")
            self.loc_texel_size = self._get_uniform_location("texel_size")
            self.loc_output_thickness = self._get_uniform_location("output_thickness")
            self.loc_thickness_scale = self._get_uniform_location("thickness_scale")

    def update(
        self,
        view_matrix: np.ndarray,
        projection_matrix: np.ndarray,
        inv_projection_matrix: np.ndarray,
        texel_size: tuple[float, float],
        output_thickness: bool,
        thickness_scale: float,
    ):
        with self:
            self._gl.glUniformMatrix4fv(self.loc_view, 1, self._gl.GL_FALSE, arr_pointer(view_matrix))
            self._gl.glUniformMatrix4fv(self.loc_projection, 1, self._gl.GL_FALSE, arr_pointer(projection_matrix))
            self._gl.glUniformMatrix4fv(
                self.loc_inv_projection, 1, self._gl.GL_FALSE, arr_pointer(inv_projection_matrix)
            )
            self._gl.glUniform2f(self.loc_texel_size, texel_size[0], texel_size[1])
            self._gl.glUniform1i(self.loc_output_thickness, int(output_thickness))
            self._gl.glUniform1f(self.loc_thickness_scale, float(thickness_scale))


class FluidDiffuseShader(ShaderGL):
    """Shader that renders Flex-style diffuse foam/spray particles."""

    def __init__(self, gl):
        super().__init__()
        from pyglet.graphics.shader import Shader, ShaderProgram

        self._gl = gl
        self.shader_program = ShaderProgram(
            Shader(fluid_diffuse_vertex_shader, "vertex"),
            Shader(fluid_diffuse_fragment_shader, "fragment"),
            Shader(fluid_diffuse_geometry_shader, "geometry"),
        )
        with self:
            self.loc_view = self._get_uniform_location("view")
            self.loc_projection = self._get_uniform_location("projection")
            self.loc_radius = self._get_uniform_location("radius")
            self.loc_motion_blur_scale = self._get_uniform_location("motion_blur_scale")
            self.loc_diffuse_expansion = self._get_uniform_location("diffuse_expansion")
            self.loc_diffuse_color = self._get_uniform_location("diffuse_color")
            self.loc_alpha = self._get_uniform_location("alpha")
            self.loc_scene_depth_texture = self._get_uniform_location("scene_depth_texture")
            self.loc_fluid_depth_texture = self._get_uniform_location("fluid_depth_texture")
            self.loc_shadow_map = self._get_uniform_location("shadow_map")
            self.loc_inv_projection = self._get_uniform_location("inv_projection")
            self.loc_light_space_matrix = self._get_uniform_location("light_space_matrix")
            self.loc_sun_direction_view = self._get_uniform_location("sun_direction_view")
            self.loc_texel_size = self._get_uniform_location("texel_size")
            self.loc_depth_mode = self._get_uniform_location("depth_mode")
            self.loc_inscatter = self._get_uniform_location("inscatter")
            self.loc_outscatter = self._get_uniform_location("outscatter")
            self.loc_shadow_strength = self._get_uniform_location("shadow_strength")
            self.loc_shadow_enabled = self._get_uniform_location("shadow_enabled")

    def update(
        self,
        view_matrix: np.ndarray,
        projection_matrix: np.ndarray,
        inv_projection_matrix: np.ndarray,
        radius: float,
        motion_blur_scale: float,
        diffuse_expansion: float,
        diffuse_color: tuple[float, float, float],
        alpha: float,
        scene_depth_unit: int,
        fluid_depth_unit: int,
        shadow_unit: int,
        light_space_matrix: np.ndarray,
        sun_direction_view: tuple[float, float, float],
        texel_size: tuple[float, float],
        depth_mode: int,
        inscatter: float,
        outscatter: float,
        shadow_strength: float,
        shadow_enabled: bool,
    ):
        with self:
            self._gl.glUniformMatrix4fv(self.loc_view, 1, self._gl.GL_FALSE, arr_pointer(view_matrix))
            self._gl.glUniformMatrix4fv(self.loc_projection, 1, self._gl.GL_FALSE, arr_pointer(projection_matrix))
            self._gl.glUniform1f(self.loc_radius, float(radius))
            self._gl.glUniform1f(self.loc_motion_blur_scale, float(motion_blur_scale))
            self._gl.glUniform1f(self.loc_diffuse_expansion, float(diffuse_expansion))
            self._gl.glUniform3f(self.loc_diffuse_color, *diffuse_color)
            self._gl.glUniform1f(self.loc_alpha, float(alpha))
            self._gl.glUniform1i(self.loc_scene_depth_texture, scene_depth_unit)
            self._gl.glUniform1i(self.loc_fluid_depth_texture, fluid_depth_unit)
            self._gl.glUniform1i(self.loc_shadow_map, shadow_unit)
            self._gl.glUniformMatrix4fv(
                self.loc_inv_projection, 1, self._gl.GL_FALSE, arr_pointer(inv_projection_matrix)
            )
            self._gl.glUniformMatrix4fv(
                self.loc_light_space_matrix, 1, self._gl.GL_FALSE, arr_pointer(light_space_matrix)
            )
            self._gl.glUniform3f(self.loc_sun_direction_view, *sun_direction_view)
            self._gl.glUniform2f(self.loc_texel_size, texel_size[0], texel_size[1])
            self._gl.glUniform1i(self.loc_depth_mode, int(depth_mode))
            self._gl.glUniform1f(self.loc_inscatter, float(inscatter))
            self._gl.glUniform1f(self.loc_outscatter, float(outscatter))
            self._gl.glUniform1f(self.loc_shadow_strength, float(shadow_strength))
            self._gl.glUniform1i(self.loc_shadow_enabled, int(shadow_enabled))


class FluidBlurShader(ShaderGL):
    """Separable bilateral blur for the fluid linear-depth buffer."""

    def __init__(self, gl):
        super().__init__()
        from pyglet.graphics.shader import Shader, ShaderProgram

        self._gl = gl
        self.shader_program = ShaderProgram(
            Shader(frame_vertex_shader, "vertex"), Shader(fluid_blur_fragment_shader, "fragment")
        )
        with self:
            self.loc_depth_texture = self._get_uniform_location("depth_texture")
            self.loc_guide_texture = self._get_uniform_location("guide_texture")
            self.loc_texel_size = self._get_uniform_location("texel_size")
            self.loc_direction = self._get_uniform_location("direction")
            self.loc_filter_radius = self._get_uniform_location("filter_radius")
            self.loc_max_depth_delta = self._get_uniform_location("max_depth_delta")
            self.loc_depth_edge_falloff = self._get_uniform_location("depth_edge_falloff")
            self.loc_max_radial_samples = self._get_uniform_location("max_radial_samples")
            self.loc_use_guide_texture = self._get_uniform_location("use_guide_texture")

    def update(
        self,
        texture_unit: int,
        guide_unit: int,
        texel_size: tuple[float, float],
        direction: tuple[float, float],
        filter_radius: float,
        max_depth_delta: float,
        depth_edge_falloff: float,
        max_radial_samples: int,
        use_guide_texture: bool,
    ):
        with self:
            self._gl.glUniform1i(self.loc_depth_texture, texture_unit)
            self._gl.glUniform1i(self.loc_guide_texture, guide_unit)
            self._gl.glUniform2f(self.loc_texel_size, texel_size[0], texel_size[1])
            self._gl.glUniform2f(self.loc_direction, direction[0], direction[1])
            self._gl.glUniform1f(self.loc_filter_radius, float(filter_radius))
            self._gl.glUniform1f(self.loc_max_depth_delta, max_depth_delta)
            self._gl.glUniform1f(self.loc_depth_edge_falloff, float(depth_edge_falloff))
            self._gl.glUniform1i(self.loc_max_radial_samples, int(max_radial_samples))
            self._gl.glUniform1i(self.loc_use_guide_texture, int(use_guide_texture))


class FluidShadowShader(ShaderGL):
    """Apply water-volume transmission and caustics to the opaque scene."""

    def __init__(self, gl):
        super().__init__()
        from pyglet.graphics.shader import Shader, ShaderProgram

        self._gl = gl
        self.shader_program = ShaderProgram(
            Shader(frame_vertex_shader, "vertex"), Shader(fluid_shadow_fragment_shader, "fragment")
        )
        with self:
            self.loc_scene_texture = self._get_uniform_location("scene_texture")
            self.loc_scene_depth_texture = self._get_uniform_location("scene_depth_texture")
            self.loc_fluid_shadow_depth_texture = self._get_uniform_location("fluid_shadow_depth_texture")
            self.loc_fluid_shadow_thickness_texture = self._get_uniform_location("fluid_shadow_thickness_texture")
            self.loc_inv_projection = self._get_uniform_location("inv_projection")
            self.loc_inv_view = self._get_uniform_location("inv_view")
            self.loc_light_projection = self._get_uniform_location("light_projection")
            self.loc_light_view = self._get_uniform_location("light_view")
            self.loc_light_space_matrix = self._get_uniform_location("light_space_matrix")
            self.loc_sun_direction_world = self._get_uniform_location("sun_direction_world")
            self.loc_fluid_bounds_lower = self._get_uniform_location("fluid_bounds_lower")
            self.loc_fluid_bounds_upper = self._get_uniform_location("fluid_bounds_upper")
            self.loc_caustic_scale = self._get_uniform_location("caustic_scale")
            self.loc_floor_caustic_strength = self._get_uniform_location("floor_caustic_strength")
            self.loc_surface_shadow_strength = self._get_uniform_location("surface_shadow_strength")
            self.loc_up_axis = self._get_uniform_location("up_axis")

    def update(
        self,
        scene_unit: int,
        scene_depth_unit: int,
        fluid_shadow_depth_unit: int,
        fluid_shadow_thickness_unit: int,
        inv_projection: np.ndarray,
        inv_view: np.ndarray,
        light_projection: np.ndarray,
        light_view: np.ndarray,
        light_space_matrix: np.ndarray,
        sun_direction_world: tuple[float, float, float],
        fluid_bounds_lower: np.ndarray,
        fluid_bounds_upper: np.ndarray,
        caustic_scale: float,
        floor_caustic_strength: float,
        surface_shadow_strength: float,
        up_axis: int,
    ):
        with self:
            self._gl.glUniform1i(self.loc_scene_texture, scene_unit)
            self._gl.glUniform1i(self.loc_scene_depth_texture, scene_depth_unit)
            self._gl.glUniform1i(self.loc_fluid_shadow_depth_texture, fluid_shadow_depth_unit)
            self._gl.glUniform1i(self.loc_fluid_shadow_thickness_texture, fluid_shadow_thickness_unit)
            self._gl.glUniformMatrix4fv(self.loc_inv_projection, 1, self._gl.GL_FALSE, arr_pointer(inv_projection))
            self._gl.glUniformMatrix4fv(self.loc_inv_view, 1, self._gl.GL_FALSE, arr_pointer(inv_view))
            self._gl.glUniformMatrix4fv(self.loc_light_projection, 1, self._gl.GL_FALSE, arr_pointer(light_projection))
            self._gl.glUniformMatrix4fv(self.loc_light_view, 1, self._gl.GL_FALSE, arr_pointer(light_view))
            self._gl.glUniformMatrix4fv(
                self.loc_light_space_matrix, 1, self._gl.GL_FALSE, arr_pointer(light_space_matrix)
            )
            self._gl.glUniform3f(self.loc_sun_direction_world, *sun_direction_world)
            self._gl.glUniform3f(
                self.loc_fluid_bounds_lower,
                float(fluid_bounds_lower[0]),
                float(fluid_bounds_lower[1]),
                float(fluid_bounds_lower[2]),
            )
            self._gl.glUniform3f(
                self.loc_fluid_bounds_upper,
                float(fluid_bounds_upper[0]),
                float(fluid_bounds_upper[1]),
                float(fluid_bounds_upper[2]),
            )
            self._gl.glUniform1f(self.loc_caustic_scale, float(caustic_scale))
            self._gl.glUniform1f(self.loc_floor_caustic_strength, float(floor_caustic_strength))
            self._gl.glUniform1f(self.loc_surface_shadow_strength, float(surface_shadow_strength))
            self._gl.glUniform1i(self.loc_up_axis, int(up_axis))


class FluidCompositeShader(ShaderGL):
    """Composite smoothed fluid depth/thickness over the scene color."""

    def __init__(self, gl):
        super().__init__()
        from pyglet.graphics.shader import Shader, ShaderProgram

        self._gl = gl
        self.shader_program = ShaderProgram(
            Shader(frame_vertex_shader, "vertex"), Shader(fluid_composite_fragment_shader, "fragment")
        )
        with self:
            self.loc_scene_texture = self._get_uniform_location("scene_texture")
            self.loc_fluid_depth_texture = self._get_uniform_location("fluid_depth_texture")
            self.loc_thickness_texture = self._get_uniform_location("thickness_texture")
            self.loc_env_map = self._get_uniform_location("env_map")
            self.loc_scene_depth_texture = self._get_uniform_location("scene_depth_texture")
            self.loc_shadow_map = self._get_uniform_location("shadow_map")
            self.loc_projection = self._get_uniform_location("projection")
            self.loc_inv_projection = self._get_uniform_location("inv_projection")
            self.loc_inv_view = self._get_uniform_location("inv_view")
            self.loc_light_space_matrix = self._get_uniform_location("light_space_matrix")
            self.loc_inv_view_rotation = self._get_uniform_location("inv_view_rotation")
            self.loc_fluid_bounds_lower = self._get_uniform_location("fluid_bounds_lower")
            self.loc_fluid_bounds_upper = self._get_uniform_location("fluid_bounds_upper")
            self.loc_texel_size = self._get_uniform_location("texel_size")
            self.loc_water_color = self._get_uniform_location("water_color")
            self.loc_water_deep_color = self._get_uniform_location("water_deep_color")
            self.loc_color_gradient_strength = self._get_uniform_location("color_gradient_strength")
            self.loc_opacity = self._get_uniform_location("opacity")
            self.loc_reflection_strength = self._get_uniform_location("reflection_strength")
            self.loc_refraction_strength = self._get_uniform_location("refraction_strength")
            self.loc_env_map_strength = self._get_uniform_location("env_map_strength")
            self.loc_env_reflection_lod = self._get_uniform_location("env_reflection_lod")
            self.loc_env_color_preserve = self._get_uniform_location("env_color_preserve")
            self.loc_env_intensity = self._get_uniform_location("env_intensity")
            self.loc_env_map_available = self._get_uniform_location("env_map_available")
            self.loc_sky_reflection_color = self._get_uniform_location("sky_reflection_color")
            self.loc_ground_reflection_color = self._get_uniform_location("ground_reflection_color")
            self.loc_absorption_strength = self._get_uniform_location("absorption_strength")
            self.loc_depth_visualization_strength = self._get_uniform_location("depth_visualization_strength")
            self.loc_caustic_strength = self._get_uniform_location("caustic_strength")
            self.loc_caustic_scale = self._get_uniform_location("caustic_scale")
            self.loc_floor_caustic_strength = self._get_uniform_location("floor_caustic_strength")
            self.loc_surface_shadow_strength = self._get_uniform_location("surface_shadow_strength")
            self.loc_shadow_radius = self._get_uniform_location("shadow_radius")
            self.loc_up_axis = self._get_uniform_location("up_axis")
            self.loc_sun_direction_view = self._get_uniform_location("sun_direction_view")

    def update(
        self,
        scene_unit: int,
        depth_unit: int,
        thickness_unit: int,
        env_unit: int,
        scene_depth_unit: int,
        shadow_unit: int,
        projection_matrix: np.ndarray,
        inv_projection: np.ndarray,
        inv_view: np.ndarray,
        light_space_matrix: np.ndarray,
        inv_view_rotation: np.ndarray,
        fluid_bounds_lower: np.ndarray,
        fluid_bounds_upper: np.ndarray,
        texel_size: tuple[float, float],
        water_color: tuple[float, float, float],
        water_deep_color: tuple[float, float, float],
        color_gradient_strength: float,
        opacity: float,
        reflection_strength: float,
        refraction_strength: float,
        env_map_strength: float,
        env_reflection_lod: float,
        env_color_preserve: float,
        env_intensity: float,
        env_map_available: bool,
        sky_reflection_color: tuple[float, float, float],
        ground_reflection_color: tuple[float, float, float],
        absorption_strength: float,
        depth_visualization_strength: float,
        caustic_strength: float,
        caustic_scale: float,
        floor_caustic_strength: float,
        surface_shadow_strength: float,
        shadow_radius: float,
        foam_strength: float,
        foam_scale: float,
        up_axis: int,
        sun_direction_view: tuple[float, float, float],
    ):
        with self:
            self._gl.glUniform1i(self.loc_scene_texture, scene_unit)
            self._gl.glUniform1i(self.loc_fluid_depth_texture, depth_unit)
            self._gl.glUniform1i(self.loc_thickness_texture, thickness_unit)
            self._gl.glUniform1i(self.loc_env_map, env_unit)
            self._gl.glUniform1i(self.loc_scene_depth_texture, scene_depth_unit)
            self._gl.glUniform1i(self.loc_shadow_map, shadow_unit)
            self._gl.glUniformMatrix4fv(self.loc_projection, 1, self._gl.GL_FALSE, arr_pointer(projection_matrix))
            self._gl.glUniformMatrix4fv(self.loc_inv_projection, 1, self._gl.GL_FALSE, arr_pointer(inv_projection))
            self._gl.glUniformMatrix4fv(self.loc_inv_view, 1, self._gl.GL_FALSE, arr_pointer(inv_view))
            self._gl.glUniformMatrix4fv(
                self.loc_light_space_matrix, 1, self._gl.GL_FALSE, arr_pointer(light_space_matrix)
            )
            self._gl.glUniformMatrix3fv(
                self.loc_inv_view_rotation, 1, self._gl.GL_FALSE, arr_pointer(inv_view_rotation)
            )
            self._gl.glUniform3f(
                self.loc_fluid_bounds_lower,
                float(fluid_bounds_lower[0]),
                float(fluid_bounds_lower[1]),
                float(fluid_bounds_lower[2]),
            )
            self._gl.glUniform3f(
                self.loc_fluid_bounds_upper,
                float(fluid_bounds_upper[0]),
                float(fluid_bounds_upper[1]),
                float(fluid_bounds_upper[2]),
            )
            self._gl.glUniform2f(self.loc_texel_size, texel_size[0], texel_size[1])
            self._gl.glUniform3f(self.loc_water_color, *water_color)
            self._gl.glUniform3f(self.loc_water_deep_color, *water_deep_color)
            self._gl.glUniform1f(self.loc_color_gradient_strength, float(color_gradient_strength))
            self._gl.glUniform1f(self.loc_opacity, float(opacity))
            self._gl.glUniform1f(self.loc_reflection_strength, float(reflection_strength))
            self._gl.glUniform1f(self.loc_refraction_strength, float(refraction_strength))
            self._gl.glUniform1f(self.loc_env_map_strength, float(env_map_strength))
            self._gl.glUniform1f(self.loc_env_reflection_lod, float(env_reflection_lod))
            self._gl.glUniform1f(self.loc_env_color_preserve, float(env_color_preserve))
            self._gl.glUniform1f(self.loc_env_intensity, float(env_intensity))
            self._gl.glUniform1i(self.loc_env_map_available, int(env_map_available))
            self._gl.glUniform3f(self.loc_sky_reflection_color, *sky_reflection_color)
            self._gl.glUniform3f(self.loc_ground_reflection_color, *ground_reflection_color)
            self._gl.glUniform1f(self.loc_absorption_strength, float(absorption_strength))
            self._gl.glUniform1f(self.loc_depth_visualization_strength, float(depth_visualization_strength))
            self._gl.glUniform1f(self.loc_caustic_strength, float(caustic_strength))
            self._gl.glUniform1f(self.loc_caustic_scale, float(caustic_scale))
            self._gl.glUniform1f(self.loc_floor_caustic_strength, float(floor_caustic_strength))
            self._gl.glUniform1f(self.loc_surface_shadow_strength, float(surface_shadow_strength))
            self._gl.glUniform1f(self.loc_shadow_radius, float(shadow_radius))
            self._gl.glUniform1i(self.loc_up_axis, int(up_axis))
            self._gl.glUniform3f(self.loc_sun_direction_view, *sun_direction_view)


wireframe_vertex_shader = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

uniform mat4 view;
uniform mat4 projection;
uniform mat4 world;

out vec3 vertexColor;

void main()
{
    vec4 worldPos = world * vec4(aPos, 1.0);
    vertexColor = aColor;
    gl_Position = projection * view * worldPos;
}
"""

wireframe_geometry_shader = """
#version 330 core
layout (lines) in;
layout (triangle_strip, max_vertices = 6) out;

in vec3 vertexColor[2];

out vec3 lineColor;

uniform float inv_asp_ratio;
uniform float line_width;

void main()
{
    vec4 s = gl_in[0].gl_Position;
    vec4 e = gl_in[1].gl_Position;

    if (s.w <= 0.0 || e.w <= 0.0) return;

    vec2 s_ndc = s.xy / s.w;
    vec2 e_ndc = e.xy / e.w;
    float s_depth = s.z / s.w;
    float e_depth = e.z / e.w;

    // Compute perpendicular in screen (aspect-corrected) space so line
    // width is uniform on non-square viewports.
    float safe_asp = max(inv_asp_ratio, 1e-6);
    vec2 dir_ndc = e_ndc - s_ndc;
    vec2 dir_scr = vec2(dir_ndc.x / safe_asp, dir_ndc.y);
    vec2 right_scr = normalize(vec2(dir_scr.y, -dir_scr.x));
    vec2 right = vec2(right_scr.x * safe_asp, right_scr.y);

    vec3 color = 0.5 * (vertexColor[0] + vertexColor[1]);
    vec2 xy = 0.5 * line_width * right;

    gl_Position = vec4(s_ndc - xy, s_depth, 1); lineColor = color;
    EmitVertex();
    gl_Position = vec4(e_ndc + xy, e_depth, 1); lineColor = color;
    EmitVertex();
    gl_Position = vec4(s_ndc + xy, s_depth, 1); lineColor = color;
    EmitVertex();
    EndPrimitive();

    gl_Position = vec4(s_ndc - xy, s_depth, 1); lineColor = color;
    EmitVertex();
    gl_Position = vec4(e_ndc - xy, e_depth, 1); lineColor = color;
    EmitVertex();
    gl_Position = vec4(e_ndc + xy, e_depth, 1); lineColor = color;
    EmitVertex();
    EndPrimitive();
}
"""

wireframe_fragment_shader = """
#version 330 core
in vec3 lineColor;
out vec4 FragColor;

uniform float alpha;

void main()
{
    FragColor = vec4(lineColor, alpha);
}
"""


class ShaderLine(ShaderGL):
    """Geometry-shader-based line renderer that expands GL_LINES into screen-space quads."""

    def __init__(self, gl):
        super().__init__()
        from pyglet.graphics.shader import Shader, ShaderProgram

        self._gl = gl
        self.shader_program = ShaderProgram(
            Shader(wireframe_vertex_shader, "vertex"),
            Shader(wireframe_geometry_shader, "geometry"),
            Shader(wireframe_fragment_shader, "fragment"),
        )

        with self:
            self.loc_view = self._get_uniform_location("view")
            self.loc_projection = self._get_uniform_location("projection")
            self.loc_world = self._get_uniform_location("world")
            self.loc_inv_asp_ratio = self._get_uniform_location("inv_asp_ratio")
            self.loc_line_width = self._get_uniform_location("line_width")
            self.loc_alpha = self._get_uniform_location("alpha")

    def update_frame(
        self,
        view_matrix: np.ndarray,
        projection_matrix: np.ndarray,
        inv_asp_ratio: float,
        line_width: float = 0.003,
        alpha: float = 0.7,
    ):
        """Set per-frame uniforms (call once before rendering all wireframe shapes)."""
        self._gl.glUniformMatrix4fv(self.loc_view, 1, self._gl.GL_FALSE, arr_pointer(view_matrix))
        self._gl.glUniformMatrix4fv(self.loc_projection, 1, self._gl.GL_FALSE, arr_pointer(projection_matrix))
        self._gl.glUniform1f(self.loc_inv_asp_ratio, float(inv_asp_ratio))
        self._gl.glUniform1f(self.loc_line_width, float(line_width))
        self._gl.glUniform1f(self.loc_alpha, float(alpha))

    def set_world(self, world: np.ndarray):
        """Set the per-shape world matrix uniform."""
        self._gl.glUniformMatrix4fv(self.loc_world, 1, self._gl.GL_FALSE, arr_pointer(world))


arrow_geometry_shader = """
#version 330 core
layout (lines) in;
layout (triangle_strip, max_vertices = 9) out;

in vec3 vertexColor[2];
out vec3 lineColor;

uniform float inv_asp_ratio;
uniform float line_width;
uniform float arrow_size;

void main()
{
    vec4 s = gl_in[0].gl_Position;
    vec4 e = gl_in[1].gl_Position;
    if (s.w <= 0.0 || e.w <= 0.0) return;

    vec2 s_ndc = s.xy / s.w;
    vec2 e_ndc = e.xy / e.w;
    float s_depth = s.z / s.w;
    float e_depth = e.z / e.w;

    // Work in screen space (aspect-corrected) so arrows look correct on
    // non-square viewports.  screen_x = ndc_x / inv_asp_ratio.
    float safe_asp = max(inv_asp_ratio, 1e-6);
    vec2 dir_ndc = e_ndc - s_ndc;
    vec2 dir_scr = vec2(dir_ndc.x / safe_asp, dir_ndc.y);
    float len = length(dir_scr);

    vec3 color = 0.5 * (vertexColor[0] + vertexColor[1]);

    // Degenerate case: line points into/out of screen
    if (len < 1e-6) {
        float r = arrow_size * 0.4;
        vec2 up = vec2(0.0, r);
        vec2 rt = vec2(r * safe_asp, 0.0);
        gl_Position = vec4(e_ndc + up, e_depth, 1); lineColor = color; EmitVertex();
        gl_Position = vec4(e_ndc - rt, e_depth, 1); lineColor = color; EmitVertex();
        gl_Position = vec4(e_ndc + rt, e_depth, 1); lineColor = color; EmitVertex();
        EndPrimitive();
        return;
    }

    // fwd/right in screen space, then convert offsets back to NDC (scale x by safe_asp)
    vec2 fwd_scr = dir_scr / len;
    vec2 right_scr = vec2(fwd_scr.y, -fwd_scr.x);
    vec2 fwd   = vec2(fwd_scr.x * safe_asp, fwd_scr.y);
    vec2 right = vec2(right_scr.x * safe_asp, right_scr.y);

    // Shorten the line body so it ends at the arrowhead base
    vec2 xy = 0.5 * line_width * right;
    vec2 e_body = e_ndc - fwd * arrow_size;

    gl_Position = vec4(s_ndc  - xy, s_depth, 1); lineColor = color; EmitVertex();
    gl_Position = vec4(e_body + xy, e_depth, 1); lineColor = color; EmitVertex();
    gl_Position = vec4(s_ndc  + xy, s_depth, 1); lineColor = color; EmitVertex();
    EndPrimitive();

    gl_Position = vec4(s_ndc  - xy, s_depth, 1); lineColor = color; EmitVertex();
    gl_Position = vec4(e_body - xy, e_depth, 1); lineColor = color; EmitVertex();
    gl_Position = vec4(e_body + xy, e_depth, 1); lineColor = color; EmitVertex();
    EndPrimitive();

    // Triangle 3: arrowhead with tip exactly at the endpoint
    vec2 tip    = e_ndc;
    vec2 base_l = e_body - right * arrow_size * 0.5;
    vec2 base_r = e_body + right * arrow_size * 0.5;

    gl_Position = vec4(tip,    e_depth, 1); lineColor = color; EmitVertex();
    gl_Position = vec4(base_l, e_depth, 1); lineColor = color; EmitVertex();
    gl_Position = vec4(base_r, e_depth, 1); lineColor = color; EmitVertex();
    EndPrimitive();
}
"""


class ShaderArrow(ShaderGL):
    """Geometry-shader-based arrow renderer: wide line + arrowhead triangle per segment."""

    def __init__(self, gl):
        super().__init__()
        from pyglet.graphics.shader import Shader, ShaderProgram

        self._gl = gl
        self.shader_program = ShaderProgram(
            Shader(wireframe_vertex_shader, "vertex"),
            Shader(arrow_geometry_shader, "geometry"),
            Shader(wireframe_fragment_shader, "fragment"),
        )

        with self:
            self.loc_view = self._get_uniform_location("view")
            self.loc_projection = self._get_uniform_location("projection")
            self.loc_world = self._get_uniform_location("world")
            self.loc_inv_asp_ratio = self._get_uniform_location("inv_asp_ratio")
            self.loc_line_width = self._get_uniform_location("line_width")
            self.loc_arrow_size = self._get_uniform_location("arrow_size")
            self.loc_alpha = self._get_uniform_location("alpha")

    def update_frame(
        self,
        view_matrix: np.ndarray,
        projection_matrix: np.ndarray,
        inv_asp_ratio: float,
        line_width: float = 0.003,
        arrow_size: float = 0.01,
        alpha: float = 1.0,
    ):
        """Set per-frame uniforms (call once before rendering all arrow batches)."""
        self._gl.glUniformMatrix4fv(self.loc_view, 1, self._gl.GL_FALSE, arr_pointer(view_matrix))
        self._gl.glUniformMatrix4fv(self.loc_projection, 1, self._gl.GL_FALSE, arr_pointer(projection_matrix))
        self._gl.glUniform1f(self.loc_inv_asp_ratio, float(inv_asp_ratio))
        self._gl.glUniform1f(self.loc_line_width, float(line_width))
        self._gl.glUniform1f(self.loc_arrow_size, float(arrow_size))
        self._gl.glUniform1f(self.loc_alpha, float(alpha))

    def set_world(self, world: np.ndarray):
        """Set the per-shape world matrix uniform."""
        self._gl.glUniformMatrix4fv(self.loc_world, 1, self._gl.GL_FALSE, arr_pointer(world))


edge_fragment_shader = """
#version 330 core
out vec4 FragColor;
uniform vec4 edge_color;
void main()
{
    FragColor = edge_color;
}
"""


class ShaderEdge(ShaderGL):
    """Flat-color shader for the edge/wireframe overlay pass."""

    def __init__(self, gl):
        super().__init__()
        from pyglet.graphics.shader import Shader, ShaderProgram

        self._gl = gl
        self.shader_program = ShaderProgram(
            Shader(shape_vertex_shader, "vertex"), Shader(edge_fragment_shader, "fragment")
        )

        with self:
            self.loc_view = self._get_uniform_location("view")
            self.loc_projection = self._get_uniform_location("projection")
            self.loc_edge_color = self._get_uniform_location("edge_color")
            self.loc_light_space_matrix = self._get_uniform_location("light_space_matrix")

    def update(
        self,
        view_matrix: np.ndarray,
        projection_matrix: np.ndarray,
        edge_color: tuple[float, float, float, float] = (0.05, 0.05, 0.05, 1.0),
        light_space_matrix: np.ndarray | None = None,
    ):
        with self:
            self._gl.glUniformMatrix4fv(self.loc_view, 1, self._gl.GL_FALSE, arr_pointer(view_matrix))
            self._gl.glUniformMatrix4fv(self.loc_projection, 1, self._gl.GL_FALSE, arr_pointer(projection_matrix))
            self._gl.glUniform4f(self.loc_edge_color, *edge_color)
            lsm = light_space_matrix if light_space_matrix is not None else np.eye(4, dtype=np.float32)
            self._gl.glUniformMatrix4fv(self.loc_light_space_matrix, 1, self._gl.GL_FALSE, arr_pointer(lsm))
