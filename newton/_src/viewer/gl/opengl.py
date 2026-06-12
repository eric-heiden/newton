# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import ctypes
import io
import os
import sys

import numpy as np
import warp as wp

from newton import Mesh

from ...utils.mesh import compute_vertex_normals
from ...utils.texture import normalize_texture
from .shaders import (
    FluidBlurShader,
    FluidCompositeShader,
    FluidDiffuseShader,
    FluidParticleShader,
    FluidShadowShader,
    FrameShader,
    ShaderArrow,
    ShaderEdge,
    ShaderLine,
    ShaderShape,
    ShaderSky,
    ShadowShader,
)

ENABLE_CUDA_INTEROP = False
ENABLE_GL_CHECKS = False

wp.set_module_options({"enable_backward": False})


def check_gl_error():
    if not ENABLE_GL_CHECKS:
        return

    from pyglet import gl

    error = gl.glGetError()
    if error != gl.GL_NO_ERROR:
        error_strings = {
            gl.GL_INVALID_ENUM: "GL_INVALID_ENUM",
            gl.GL_INVALID_VALUE: "GL_INVALID_VALUE",
            gl.GL_INVALID_OPERATION: "GL_INVALID_OPERATION",
            gl.GL_INVALID_FRAMEBUFFER_OPERATION: "GL_INVALID_FRAMEBUFFER_OPERATION",
            gl.GL_OUT_OF_MEMORY: "GL_OUT_OF_MEMORY",
        }
        error_name = error_strings.get(error, f"Unknown error code: {error}")

        import traceback  # noqa: PLC0415

        stack = traceback.format_stack()
        print(f"OpenGL error: {error_name} ({error:#x})")
        print(f"Called from: {''.join(stack[-2:-1])}")


def _upload_texture_from_file(gl, texture_image: np.ndarray) -> int:
    image = normalize_texture(
        texture_image,
        flip_vertical=True,
        require_channels=True,
        scale_unit_range=True,
    )
    if image is None:
        return 0
    channels = image.shape[2]
    if image.size == 0:
        return 0
    max_size = gl.GLint()
    gl.glGetIntegerv(gl.GL_MAX_TEXTURE_SIZE, max_size)
    if image.shape[0] > max_size.value or image.shape[1] > max_size.value:
        return 0
    texture_id = gl.GLuint()
    gl.glGenTextures(1, texture_id)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)

    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

    format_enum = gl.GL_RGBA if channels == 4 else gl.GL_RGB
    row_stride = image.shape[1] * channels
    prev_alignment = None
    if row_stride % 4 != 0:
        prev_alignment = gl.GLint()
        gl.glGetIntegerv(gl.GL_UNPACK_ALIGNMENT, prev_alignment)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    gl.glTexImage2D(
        gl.GL_TEXTURE_2D,
        0,
        format_enum,
        image.shape[1],
        image.shape[0],
        0,
        format_enum,
        gl.GL_UNSIGNED_BYTE,
        image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
    )
    if prev_alignment is not None:
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, prev_alignment.value)
    gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    return texture_id


@wp.struct
class RenderVertex:
    pos: wp.vec3
    normal: wp.vec3
    uv: wp.vec2


@wp.struct
class LineVertex:
    pos: wp.vec3
    color: wp.vec3


@wp.kernel
def fill_vertex_data(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    uvs: wp.array[wp.vec2],
    vertices: wp.array[RenderVertex],
):
    tid = wp.tid()

    vertices[tid].pos = points[tid]

    if normals:
        vertices[tid].normal = normals[tid]

    if uvs:
        vertices[tid].uv = uvs[tid]


@wp.kernel
def fill_line_vertex_data(
    starts: wp.array[wp.vec3],
    ends: wp.array[wp.vec3],
    colors: wp.array[wp.vec3],
    vertices: wp.array[LineVertex],
):
    tid = wp.tid()

    # Each line has 2 vertices (begin and end)
    vertex_idx = tid * 2

    # First vertex (line begin)
    vertices[vertex_idx].pos = starts[tid]
    vertices[vertex_idx].color = colors[tid]

    # Second vertex (line end)
    vertices[vertex_idx + 1].pos = ends[tid]
    vertices[vertex_idx + 1].color = colors[tid]


class MeshGL:
    """Encapsulates mesh data and OpenGL buffers for a shape."""

    def __init__(self, num_points, num_indices, device, hidden=False, backface_culling=True):
        """Initialize mesh data with vertices and indices."""
        gl = RendererGL.gl

        self.num_points = num_points
        self.num_indices = num_indices

        # Store references to input buffers and rendering data
        self.device = device
        self.hidden = hidden
        self.backface_culling = backface_culling

        self.vertices = wp.zeros(num_points, dtype=RenderVertex, device=self.device)
        self.indices = None
        self.normals = None  # scratch buffer used during normal recomputation
        self.texture_id = None

        # Set up vertex attributes in the packed format the shaders expect
        self.vertex_byte_size = 12 + 12 + 8
        self.index_byte_size = 4

        self.vbo_size = self.vertex_byte_size * num_points
        self.ebo_size = self.index_byte_size * num_indices

        # Create OpenGL buffers
        self.vao = gl.GLuint()
        gl.glGenVertexArrays(1, self.vao)
        gl.glBindVertexArray(self.vao)

        self.vbo = gl.GLuint()
        gl.glGenBuffers(1, self.vbo)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.vbo_size, None, gl.GL_STATIC_DRAW)

        self.ebo = gl.GLuint()
        gl.glGenBuffers(1, self.ebo)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo_size, None, gl.GL_STATIC_DRAW)

        # positions (location 0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, self.vertex_byte_size, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)

        # normals (location 1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, self.vertex_byte_size, ctypes.c_void_p(3 * 4))
        gl.glEnableVertexAttribArray(1)

        # uv coordinates (location 2)
        gl.glVertexAttribPointer(2, 2, gl.GL_FLOAT, gl.GL_FALSE, self.vertex_byte_size, ctypes.c_void_p(6 * 4))
        gl.glEnableVertexAttribArray(2)

        # set constant instance transform
        gl.glDisableVertexAttribArray(3)
        gl.glDisableVertexAttribArray(4)
        gl.glDisableVertexAttribArray(5)
        gl.glDisableVertexAttribArray(6)
        gl.glDisableVertexAttribArray(7)
        gl.glDisableVertexAttribArray(8)
        gl.glDisableVertexAttribArray(9)

        #   column 0  (1,0,0,0)
        gl.glVertexAttrib4f(3, 1.0, 0.0, 0.0, 0.0)
        #   column 1  (0,1,0,0)
        gl.glVertexAttrib4f(4, 0.0, 1.0, 0.0, 0.0)
        #   column 2  (0,0,1,0)
        gl.glVertexAttrib4f(5, 0.0, 0.0, 1.0, 0.0)
        #   column 3  (0,0,0,1)
        gl.glVertexAttrib4f(6, 0.0, 0.0, 0.0, 1.0)

        gl.glBindVertexArray(0)

        # Per-mesh albedo and material (applied in render()).
        self.color = (0.7, 0.5, 0.3)
        self.material = (0.5, 0.0, 0.0, 0.0)

        # Create CUDA-GL interop buffer for efficient updates
        if ENABLE_CUDA_INTEROP and self.device.is_cuda:
            self.vertex_cuda_buffer = wp.RegisteredGLBuffer(int(self.vbo.value), self.device)
        else:
            self.vertex_cuda_buffer = None
        self._points = None

    def destroy(self):
        """Clean up OpenGL resources."""
        gl = RendererGL.gl
        try:
            if hasattr(self, "vao"):
                gl.glDeleteVertexArrays(1, self.vao)
            if hasattr(self, "vbo"):
                gl.glDeleteBuffers(1, self.vbo)
            if hasattr(self, "ebo"):
                gl.glDeleteBuffers(1, self.ebo)
            if hasattr(self, "texture_id") and self.texture_id is not None:
                gl.glDeleteTextures(1, self.texture_id)
        except Exception:
            # Ignore any errors if the GL context has already been torn down
            pass

    def update(self, points, indices, normals, uvs, texture=None):
        """Update vertex positions in the VBO.

        Args:
            points: New point positions (warp array or numpy array)
            scale: Scaling factor for positions
        """
        gl = RendererGL.gl

        if len(points) != len(self.vertices):
            raise RuntimeError("Number of points does not match")

        self._points = points

        # only update indices the first time (no topology changes)
        if self.indices is None:
            self.indices = wp.clone(indices).view(dtype=wp.uint32)
            self.num_indices = int(len(self.indices))

            host_indices = self.indices.numpy()
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            gl.glBufferData(
                gl.GL_ELEMENT_ARRAY_BUFFER, host_indices.nbytes, host_indices.ctypes.data, gl.GL_STATIC_DRAW
            )

        # If normals are missing, compute them before packing vertex data.
        if points is not None and normals is None:
            self.recompute_normals()
            normals = self.normals

        # update gfx vertices
        wp.launch(
            fill_vertex_data,
            dim=len(self.vertices),
            inputs=[points, normals, uvs],
            outputs=[self.vertices],
            device=self.device,
        )

        # upload vertices to GL
        if ENABLE_CUDA_INTEROP and self.vertices.device.is_cuda:
            # upload points via CUDA if possible
            vbo_vertices = self.vertex_cuda_buffer.map(dtype=RenderVertex, shape=self.vertices.shape)
            wp.copy(vbo_vertices, self.vertices)
            self.vertex_cuda_buffer.unmap()

        else:
            host_vertices = self.vertices.numpy()
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, host_vertices.nbytes, host_vertices.ctypes.data, gl.GL_STATIC_DRAW)

        self.update_texture(texture)

    def recompute_normals(self):
        if self._points is None or self.indices is None:
            return
        self.normals = compute_vertex_normals(
            self._points,
            self.indices,
            normals=self.normals,
            device=self.device,
        )

    def update_texture(self, texture=None):
        gl = RendererGL.gl
        texture_image = None
        if texture is not None:
            from ...utils.texture import load_texture  # noqa: PLC0415

            texture_image = load_texture(texture)

        if texture_image is None:
            if self.texture_id is not None:
                try:
                    gl.glDeleteTextures(1, self.texture_id)
                except Exception:
                    pass
                self.texture_id = None
            return

        if self.texture_id is not None:
            try:
                gl.glDeleteTextures(1, self.texture_id)
            except Exception:
                pass
            self.texture_id = None

        texture_id = _upload_texture_from_file(gl, texture_image)
        if not texture_id:
            return
        self.texture_id = texture_id

    def render(self):
        if not self.hidden:
            gl = RendererGL.gl

            if self.backface_culling:
                gl.glEnable(gl.GL_CULL_FACE)
            else:
                gl.glDisable(gl.GL_CULL_FACE)

            gl.glActiveTexture(gl.GL_TEXTURE1)
            if self.texture_id is not None:
                gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
            else:
                gl.glBindTexture(gl.GL_TEXTURE_2D, RendererGL.get_fallback_texture())

            # Set per-mesh albedo and material (global state, not per-VAO).
            gl.glVertexAttrib3f(7, *self.color)
            gl.glVertexAttrib4f(8, *self.material)

            gl.glBindVertexArray(self.vao)
            gl.glDrawElements(gl.GL_TRIANGLES, self.num_indices, gl.GL_UNSIGNED_INT, None)
            gl.glBindVertexArray(0)


class LinesGL:
    """Encapsulates line data and OpenGL buffers for line rendering."""

    def __init__(self, max_lines, device, hidden=False):
        """Initialize line data with the specified maximum number of lines.

        Args:
            max_lines: Maximum number of lines that can be rendered
            device: Warp device to use
            hidden: Whether the lines are initially hidden
        """
        gl = RendererGL.gl

        self.max_lines = max_lines
        self.max_vertices = max_lines * 2  # Each line has 2 vertices
        self.num_lines = 0  # Current number of active lines to render

        # Store references to input buffers and rendering data
        self.device = device
        self.hidden = hidden

        self.vertices = wp.zeros(self.max_vertices, dtype=LineVertex, device=self.device)

        # Set up vertex attributes for lines (position + color)
        self.vertex_byte_size = 12 + 12  # 3 floats for pos + 3 floats for color
        self.vbo_size = self.vertex_byte_size * self.max_vertices

        # Create OpenGL buffers
        self.vao = gl.GLuint()
        gl.glGenVertexArrays(1, self.vao)
        gl.glBindVertexArray(self.vao)

        self.vbo = gl.GLuint()
        gl.glGenBuffers(1, self.vbo)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.vbo_size, None, gl.GL_DYNAMIC_DRAW)

        # positions (location 0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, self.vertex_byte_size, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)

        # colors (location 1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, self.vertex_byte_size, ctypes.c_void_p(3 * 4))
        gl.glEnableVertexAttribArray(1)

        gl.glBindVertexArray(0)

        # Create CUDA-GL interop buffer for efficient updates
        if ENABLE_CUDA_INTEROP and self.device.is_cuda:
            self.vertex_cuda_buffer = wp.RegisteredGLBuffer(int(self.vbo.value), self.device)
        else:
            self.vertex_cuda_buffer = None

    def destroy(self):
        """Clean up OpenGL resources."""
        gl = RendererGL.gl
        try:
            if hasattr(self, "vao"):
                gl.glDeleteVertexArrays(1, self.vao)
            if hasattr(self, "vbo"):
                gl.glDeleteBuffers(1, self.vbo)
        except Exception:
            # Ignore any errors if the GL context has already been torn down
            pass

    def update(self, starts, ends, colors):
        """Update line data in the VBO.

        Args:
            starts: Array of line start positions (warp array of vec3) or None
            ends: Array of line end positions (warp array of vec3) or None
            colors: Array of line colors (warp array of vec3) or None
        """
        gl = RendererGL.gl

        # Handle None values by setting line count to zero
        if starts is None or ends is None or colors is None:
            self.num_lines = 0
            return

        # Update current line count
        self.num_lines = len(starts)

        if self.num_lines > self.max_lines:
            raise RuntimeError(f"Number of lines ({self.num_lines}) exceeds maximum ({self.max_lines})")
        if len(ends) != self.num_lines:
            raise RuntimeError("Number of line ends does not match line begins")
        if len(colors) != self.num_lines:
            raise RuntimeError("Number of line colors does not match line begins")

        # Only update vertex data if we have lines to render
        if self.num_lines > 0:
            # Update line vertex data using the kernel
            wp.launch(
                fill_line_vertex_data,
                dim=self.num_lines,
                inputs=[starts, ends, colors],
                outputs=[self.vertices],
                device=self.device,
            )

        # Upload vertices to GL
        if ENABLE_CUDA_INTEROP and self.vertices.device.is_cuda:
            # Upload points via CUDA if possible
            vbo_vertices = self.vertex_cuda_buffer.map(dtype=LineVertex, shape=self.vertices.shape)
            wp.copy(vbo_vertices, self.vertices)
            self.vertex_cuda_buffer.unmap()
        else:
            host_vertices = self.vertices.numpy()
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, host_vertices.nbytes, host_vertices.ctypes.data, gl.GL_DYNAMIC_DRAW)

    def render(self):
        if not self.hidden and self.num_lines > 0:
            gl = RendererGL.gl

            gl.glDisable(gl.GL_CULL_FACE)  # Lines don't need culling

            gl.glBindVertexArray(self.vao)
            # Only render vertices for the current number of lines
            current_vertices = self.num_lines * 2
            gl.glDrawArrays(gl.GL_LINES, 0, current_vertices)
            gl.glBindVertexArray(0)


class WireframeShapeGL:
    """Per-shape wireframe edge data rendered via GL_LINES with a geometry shader.

    Stores interleaved (position, color) vertex data in model space.
    The World matrix is set per-shape by the caller before drawing.

    Multiple instances can share the same VAO/VBO when created via
    :meth:`create_shared`.  Only the *owner* (``_owns_gl == True``)
    deletes the GL resources on :meth:`destroy`.
    """

    def __init__(self, vertex_data: np.ndarray):
        """Create a wireframe shape that owns its GL resources."""
        gl = RendererGL.gl
        self.num_vertices = len(vertex_data)
        self.hidden = False
        self.world_matrix = np.eye(4, dtype=np.float32)
        self._owns_gl = True

        vertex_byte_size = 6 * 4

        self.vao = gl.GLuint()
        gl.glGenVertexArrays(1, self.vao)
        gl.glBindVertexArray(self.vao)

        self.vbo = gl.GLuint()
        gl.glGenBuffers(1, self.vbo)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)

        data = vertex_data.astype(np.float32)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data.ctypes.data, gl.GL_STATIC_DRAW)

        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, vertex_byte_size, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, vertex_byte_size, ctypes.c_void_p(3 * 4))
        gl.glEnableVertexAttribArray(1)

        gl.glBindVertexArray(0)

    @classmethod
    def create_shared(cls, owner: "WireframeShapeGL") -> "WireframeShapeGL":
        """Create an instance that shares *owner*'s VAO/VBO."""
        obj = cls.__new__(cls)
        obj.vao = owner.vao
        obj.vbo = owner.vbo
        obj.num_vertices = owner.num_vertices
        obj.hidden = False
        obj.world_matrix = np.eye(4, dtype=np.float32)
        obj._owns_gl = False
        return obj

    def destroy(self):
        """Free GL resources if this instance owns them."""
        if not getattr(self, "_owns_gl", False):
            return
        gl = RendererGL.gl
        try:
            if hasattr(self, "vao"):
                gl.glDeleteVertexArrays(1, self.vao)
            if hasattr(self, "vbo"):
                gl.glDeleteBuffers(1, self.vbo)
        except Exception:
            pass

    def render(self):
        if self.hidden or self.num_vertices == 0:
            return
        gl = RendererGL.gl
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_LINES, 0, self.num_vertices)
        gl.glBindVertexArray(0)


@wp.kernel
def update_vbo_transforms(
    instance_transforms: wp.array[wp.transform],
    instance_scalings: wp.array[wp.vec3],
    vbo_transforms: wp.array[wp.mat44],
):
    """Update VBO with simple instance transformation matrices."""
    tid = wp.tid()

    # Get transform and scaling
    transform = instance_transforms[tid]

    if instance_scalings:
        s = instance_scalings[tid]
    else:
        s = wp.vec3(1.0, 1.0, 1.0)

    # Extract position and rotation
    p = wp.transform_get_translation(transform)
    q = wp.transform_get_rotation(transform)

    # Build rotation matrix
    R = wp.quat_to_matrix(q)

    # Apply scaling
    vbo_transforms[tid] = wp.mat44(
        R[0, 0] * s[0],
        R[1, 0] * s[0],
        R[2, 0] * s[0],
        0.0,
        R[0, 1] * s[1],
        R[1, 1] * s[1],
        R[2, 1] * s[1],
        0.0,
        R[0, 2] * s[2],
        R[1, 2] * s[2],
        R[2, 2] * s[2],
        0.0,
        p[0],
        p[1],
        p[2],
        1.0,
    )


@wp.kernel
def update_vbo_transforms_from_points(
    points: wp.array[wp.vec3],
    widths: wp.array[wp.float32],
    vbo_transforms: wp.array[wp.mat44],
):
    """Update VBO with simple instance transformation matrices."""
    tid = wp.tid()

    # Get transform and scaling
    p = points[tid]

    if widths:
        s = widths[tid]
    else:
        s = 1.0

    # Build rotation matrix
    R = wp.identity(n=3, dtype=wp.float32)

    # Apply scaling
    vbo_transforms[tid] = wp.mat44(
        R[0, 0] * s,
        R[1, 0] * s,
        R[2, 0] * s,
        0.0,
        R[0, 1] * s,
        R[1, 1] * s,
        R[2, 1] * s,
        0.0,
        R[0, 2] * s,
        R[1, 2] * s,
        R[2, 2] * s,
        0.0,
        p[0],
        p[1],
        p[2],
        1.0,
    )


class MeshInstancerGL:
    """
    Handles instanced rendering for a mesh.
    Note the vertices must be in the 8-dimensional format:
        [3D point, 3D normal, UV texture coordinates]
    """

    def __init__(self, num_instances, mesh):
        self.mesh = mesh
        self.device = mesh.device
        self.hidden = False
        self.instance_transform_buffer = None
        self.instance_color_buffer = None
        self.instance_material_buffer = None

        self.instance_transform_cuda_buffer = None

        self.allocate(num_instances)
        self.active_instances = num_instances

    def __del__(self):
        gl = RendererGL.gl

        if self.instance_transform_cuda_buffer is not None:
            try:
                gl.glDeleteBuffers(1, self.instance_transform_cuda_buffer)
            except Exception:
                # Ignore any errors (e.g., context already destroyed)
                pass

        if hasattr(self, "vao") and self.vao is not None:
            try:
                gl.glDeleteVertexArrays(1, self.vao)
                gl.glDeleteBuffers(1, self.instance_transform_buffer)
                gl.glDeleteBuffers(1, self.instance_color_buffer)
                gl.glDeleteBuffers(1, self.instance_material_buffer)
            except Exception:
                # Ignore any errors during interpreter shutdown
                pass

    def allocate(self, num_instances):
        gl = RendererGL.gl

        self.world_xforms = wp.zeros(num_instances, dtype=wp.mat44, device=self.device)

        self.vao = gl.GLuint()
        self.instance_transform_buffer = gl.GLuint()
        self.instance_color_buffer = gl.GLuint()
        self.instance_material_buffer = gl.GLuint()
        self.num_instances = num_instances

        gl.glGenVertexArrays(1, self.vao)
        gl.glBindVertexArray(self.vao)

        # -------------------------
        # index buffer

        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.mesh.ebo)

        # ------------------------
        # mesh buffers

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.mesh.vbo)

        # positions
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, self.mesh.vertex_byte_size, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)
        # normals
        gl.glVertexAttribPointer(
            1,
            3,
            gl.GL_FLOAT,
            gl.GL_FALSE,
            self.mesh.vertex_byte_size,
            ctypes.c_void_p(3 * 4),
        )
        gl.glEnableVertexAttribArray(1)
        # uv coordinates
        gl.glVertexAttribPointer(
            2,
            2,
            gl.GL_FLOAT,
            gl.GL_FALSE,
            self.mesh.vertex_byte_size,
            ctypes.c_void_p(6 * 4),
        )
        gl.glEnableVertexAttribArray(2)

        self.transform_byte_size = 16 * 4  # sizeof(mat44)
        self.color_byte_size = 3 * 4  # sizeof(vec3)
        self.material_byte_size = 4 * 4  # sizeof(vec4)

        self.instance_transform_buffer_size = self.transform_byte_size * self.num_instances
        self.instance_color_buffer_size = self.color_byte_size * self.num_instances
        self.instance_material_buffer_size = self.material_byte_size * self.num_instances

        # ------------------------
        # transform buffer

        gl.glGenBuffers(1, self.instance_transform_buffer)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.instance_transform_buffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.instance_transform_buffer_size, None, gl.GL_DYNAMIC_DRAW)

        # Send transforms as vec4 columns because vertex attributes cannot carry a full mat4 directly.
        for i in range(4):
            gl.glVertexAttribPointer(
                3 + i, 4, gl.GL_FLOAT, gl.GL_FALSE, self.transform_byte_size, ctypes.c_void_p(i * 16)
            )
            gl.glEnableVertexAttribArray(3 + i)
            gl.glVertexAttribDivisor(3 + i, 1)

        # ------------------------
        # colors

        gl.glGenBuffers(1, self.instance_color_buffer)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.instance_color_buffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.instance_color_buffer_size, None, gl.GL_STATIC_DRAW)

        gl.glVertexAttribPointer(7, 3, gl.GL_FLOAT, gl.GL_FALSE, self.color_byte_size, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(7)
        gl.glVertexAttribDivisor(7, 1)

        # ------------------------
        # materials buffer
        host_materials = np.zeros(self.num_instances * 4, dtype=np.float32)

        gl.glGenBuffers(1, self.instance_material_buffer)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.instance_material_buffer)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER, self.instance_material_buffer_size, host_materials.ctypes.data, gl.GL_STATIC_DRAW
        )

        gl.glVertexAttribPointer(8, 4, gl.GL_FLOAT, gl.GL_FALSE, self.material_byte_size, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(8)
        gl.glVertexAttribDivisor(8, 1)

        gl.glBindVertexArray(0)

        # Create CUDA buffer for instance transforms
        if ENABLE_CUDA_INTEROP and self.device.is_cuda:
            self._instance_transform_cuda_buffer = wp.RegisteredGLBuffer(
                int(self.instance_transform_buffer.value), self.device, flags=wp.RegisteredGLBuffer.WRITE_DISCARD
            )
        else:
            self._instance_transform_cuda_buffer = None

    def update_from_transforms(
        self,
        transforms: wp.array = None,
        scalings: wp.array = None,
        colors: wp.array = None,
        materials: wp.array = None,
    ):
        if transforms is None:
            active_count = 0
        else:
            active_count = len(transforms)

            if active_count > self.num_instances:
                raise ValueError(
                    f"Active instance count ({active_count}) exceeds allocated capacity ({self.num_instances})."
                )
            if scalings is not None and len(scalings) != active_count:
                raise ValueError("Number of scalings must match number of transforms")

        if active_count > 0:
            wp.launch(
                update_vbo_transforms,
                dim=active_count,
                inputs=[
                    transforms,
                    scalings,
                ],
                outputs=[
                    self.world_xforms,
                ],
                device=self.device,
                record_tape=False,
            )

        self.active_instances = active_count
        # Upload the full buffer; only the first `active_instances` rows are rendered
        self._update_vbo(self.world_xforms, colors, materials)

    # helper to update instance transforms from points
    def update_from_points(self, points, widths, colors):
        if points is None:
            active = 0
        else:
            active = len(points)

        if active > self.num_instances:
            raise ValueError("Active point count exceeds allocated capacity. Reallocate before updating.")

        self.active_instances = active

        if self.active_instances > 0 and (points is not None or widths is not None):
            wp.launch(
                update_vbo_transforms_from_points,
                dim=self.active_instances,
                inputs=[
                    points,
                    widths,
                ],
                outputs=[
                    self.world_xforms,
                ],
                device=self.device,
                record_tape=False,
            )

        self._update_vbo(self.world_xforms, colors, None)

    # upload to vbo
    def _update_vbo(self, xforms, colors, materials):
        gl = RendererGL.gl

        if ENABLE_CUDA_INTEROP and self.device.is_cuda:
            vbo_transforms = self._instance_transform_cuda_buffer.map(dtype=wp.mat44, shape=(self.num_instances,))
            wp.copy(vbo_transforms, xforms)
            self._instance_transform_cuda_buffer.unmap()
        else:
            host_transforms = xforms.numpy()
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.instance_transform_buffer)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, host_transforms.nbytes, host_transforms.ctypes.data, gl.GL_DYNAMIC_DRAW)

        # update other properties through CPU for now
        if colors is not None:
            host_colors = colors.numpy()
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.instance_color_buffer)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, host_colors.nbytes, host_colors.ctypes.data, gl.GL_STATIC_DRAW)

        if materials is not None:
            host_materials = materials.numpy()
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.instance_material_buffer)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, host_materials.nbytes, host_materials.ctypes.data, gl.GL_STATIC_DRAW)

    def update_from_pinned(self, host_transforms_np, count, colors=None, materials=None):
        """Upload pre-computed mat44 transforms from pinned host memory to GL.

        Args:
            host_transforms_np: Numpy array slice of mat44 transforms.
            count: Number of active instances.
            colors: Optional wp.array of per-instance colors.
            materials: Optional wp.array of per-instance materials.
        """
        gl = RendererGL.gl
        if count > self.num_instances:
            raise ValueError(f"Active instance count ({count}) exceeds allocated capacity ({self.num_instances}).")
        self.active_instances = count
        if count > 0:
            nbytes = count * self.transform_byte_size
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.instance_transform_buffer)
            gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, nbytes, host_transforms_np.ctypes.data)
        if colors is not None:
            host_colors = colors.numpy()
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.instance_color_buffer)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, host_colors.nbytes, host_colors.ctypes.data, gl.GL_STATIC_DRAW)
        if materials is not None:
            host_materials = materials.numpy()
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.instance_material_buffer)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, host_materials.nbytes, host_materials.ctypes.data, gl.GL_STATIC_DRAW)

    def render(self):
        gl = RendererGL.gl

        if self.hidden:
            return

        if self.mesh.backface_culling:
            gl.glEnable(gl.GL_CULL_FACE)
        else:
            gl.glDisable(gl.GL_CULL_FACE)

        gl.glActiveTexture(gl.GL_TEXTURE1)
        if self.mesh.texture_id is not None:
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.mesh.texture_id)
        else:
            gl.glBindTexture(gl.GL_TEXTURE_2D, RendererGL.get_fallback_texture())

        gl.glBindVertexArray(self.vao)
        gl.glDrawElementsInstanced(
            gl.GL_TRIANGLES, self.mesh.num_indices, gl.GL_UNSIGNED_INT, None, self.active_instances
        )
        gl.glBindVertexArray(0)


@wp.kernel
def _pack_fluid_particle_data(
    points: wp.array[wp.vec3],
    radii: wp.array[float],
    use_radii: int,
    uniform_radius: float,
    radius_scale: float,
    anisotropy: wp.array[wp.vec4],
    anisotropy_secondary: wp.array[wp.vec4],
    anisotropy_tertiary: wp.array[wp.vec4],
    dest: wp.array[float],
    bounds: wp.array[float],
):
    tid = wp.tid()
    p = points[tid]
    a = anisotropy[tid]
    a2 = anisotropy_secondary[tid]
    a3 = anisotropy_tertiary[tid]
    r = uniform_radius
    if use_radii != 0:
        r = radii[tid]
    r *= radius_scale
    if a[3] <= 0.0:
        # Inactive particles collapse to zero-radius splats the shaders skip.
        r = 0.0

    base = tid * 16
    dest[base + 0] = p[0]
    dest[base + 1] = p[1]
    dest[base + 2] = p[2]
    dest[base + 3] = r
    dest[base + 4] = a[0]
    dest[base + 5] = a[1]
    dest[base + 6] = a[2]
    dest[base + 7] = a[3]
    dest[base + 8] = a2[0]
    dest[base + 9] = a2[1]
    dest[base + 10] = a2[2]
    dest[base + 11] = a2[3]
    dest[base + 12] = a3[0]
    dest[base + 13] = a3[1]
    dest[base + 14] = a3[2]
    dest[base + 15] = a3[3]

    if r > 0.0:
        wp.atomic_min(bounds, 0, p[0] - r)
        wp.atomic_min(bounds, 1, p[1] - r)
        wp.atomic_min(bounds, 2, p[2] - r)
        wp.atomic_max(bounds, 3, p[0] + r)
        wp.atomic_max(bounds, 4, p[1] + r)
        wp.atomic_max(bounds, 5, p[2] + r)


class FluidGL:
    """GPU buffer for screen-space fluid particle samples."""

    # Material/tuning fields preserved across capacity resizes.
    _MATERIAL_FIELDS = (
        "hidden",
        "color",
        "deep_color",
        "color_gradient_strength",
        "opacity",
        "radius_scale",
        "thickness_scale",
        "smoothing_iterations",
        "smoothing_radius",
        "smoothing_depth_edge_falloff",
        "smoothing_max_samples",
        "reflection_strength",
        "refraction_strength",
        "env_map_strength",
        "env_reflection_lod",
        "env_color_preserve",
        "absorption_strength",
        "depth_visualization_strength",
        "caustic_strength",
        "caustic_scale",
        "floor_caustic_strength",
        "surface_shadow_strength",
        "foam_strength",
        "foam_scale",
        "anisotropy_strength",
        "anisotropy_min",
        "anisotropy_max",
        "anisotropy_max_particles",
        "render_smoothing",
    )

    def __init__(self, capacity: int):
        gl = RendererGL.gl
        self.capacity = max(int(capacity), 1)
        self.active_particles = 0
        self.hidden = False
        self.color = (0.10, 0.50, 0.80)
        self.deep_color = (0.01, 0.09, 0.34)
        self.color_gradient_strength = 0.20
        self.opacity = 1.00
        self.radius_scale = 1.34
        self.thickness_scale = 0.55
        self.smoothing_iterations = 7
        self.smoothing_radius = 3.83
        self.smoothing_depth_edge_falloff = 5.5
        self.smoothing_max_samples = 4
        self.reflection_strength = 0.528
        self.refraction_strength = 0.038
        self.env_map_strength = 1.02
        self.env_reflection_lod = 1.8
        self.env_color_preserve = 0.57
        self.absorption_strength = 1.2
        self.depth_visualization_strength = 2.13
        self.caustic_strength = 3.03
        self.caustic_scale = 37.1
        self.floor_caustic_strength = 1.15
        self.surface_shadow_strength = 0.35
        self.foam_strength = 0.99
        self.foam_scale = 5.0
        self.bounds_valid = False
        self.bounds_lower = np.zeros(3, dtype=np.float32)
        self.bounds_upper = np.zeros(3, dtype=np.float32)
        self.anisotropy_strength = 0.82
        self.anisotropy_min = 0.1
        self.anisotropy_max = 2.0
        self.anisotropy_max_particles = 25000
        self.render_smoothing = 0.45
        self._particle_stride = 16 * 4
        self._packed_gpu = None
        self._bounds_gpu = None
        self._dummy_radii_gpu = None
        self.vertex_cuda_buffer = None
        self._bounds_reset = np.array([1.0e9, 1.0e9, 1.0e9, -1.0e9, -1.0e9, -1.0e9], dtype=np.float32)

        self.vao = gl.GLuint()
        self.vbo = gl.GLuint()
        gl.glGenVertexArrays(1, self.vao)
        gl.glGenBuffers(1, self.vbo)

        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.capacity * self._particle_stride, None, gl.GL_DYNAMIC_DRAW)
        gl.glVertexAttribPointer(0, 4, gl.GL_FLOAT, gl.GL_FALSE, self._particle_stride, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(1, 4, gl.GL_FLOAT, gl.GL_FALSE, self._particle_stride, ctypes.c_void_p(4 * 4))
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(2, 4, gl.GL_FLOAT, gl.GL_FALSE, self._particle_stride, ctypes.c_void_p(8 * 4))
        gl.glEnableVertexAttribArray(2)
        gl.glVertexAttribPointer(3, 4, gl.GL_FLOAT, gl.GL_FALSE, self._particle_stride, ctypes.c_void_p(12 * 4))
        gl.glEnableVertexAttribArray(3)
        gl.glBindVertexArray(0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

    def destroy(self):
        gl = RendererGL.gl
        self.vertex_cuda_buffer = None
        self._packed_gpu = None
        self._bounds_gpu = None
        self._dummy_radii_gpu = None
        if getattr(self, "vao", None) is not None:
            gl.glDeleteVertexArrays(1, self.vao)
            gl.glDeleteBuffers(1, self.vbo)
            self.vao = None
            self.vbo = None

    def _resize(self, capacity: int):
        preserved = {name: getattr(self, name) for name in self._MATERIAL_FIELDS}
        self.destroy()
        self.__init__(capacity)
        for name, value in preserved.items():
            setattr(self, name, value)

    def update(
        self,
        points,
        radii,
        color: tuple[float, float, float],
        deep_color: tuple[float, float, float],
        color_gradient_strength: float,
        opacity: float,
        radius_scale: float,
        thickness_scale: float,
        smoothing_iterations: int,
        smoothing_radius: float,
        smoothing_depth_edge_falloff: float,
        smoothing_max_samples: int,
        reflection_strength: float,
        refraction_strength: float,
        env_map_strength: float,
        env_reflection_lod: float,
        env_color_preserve: float,
        absorption_strength: float,
        depth_visualization_strength: float,
        caustic_strength: float,
        caustic_scale: float,
        floor_caustic_strength: float,
        surface_shadow_strength: float,
        foam_strength: float,
        foam_scale: float,
        hidden: bool,
        render_points=None,
        anisotropy=None,
        anisotropy_secondary=None,
        anisotropy_tertiary=None,
    ):
        if points is None:
            self.active_particles = 0
            self.hidden = True
            self.bounds_valid = False
            return

        gl = RendererGL.gl
        self.hidden = hidden
        self.color = tuple(float(c) for c in color)
        self.deep_color = tuple(float(c) for c in deep_color)
        self.color_gradient_strength = float(np.clip(color_gradient_strength, 0.0, 1.0))
        self.opacity = float(np.clip(opacity, 0.0, 1.0))
        self.radius_scale = float(max(radius_scale, 0.0))
        self.thickness_scale = float(max(thickness_scale, 0.0))
        self.smoothing_iterations = max(int(smoothing_iterations), 0)
        self.smoothing_radius = float(max(smoothing_radius, 0.0))
        self.smoothing_depth_edge_falloff = float(max(smoothing_depth_edge_falloff, 0.0))
        self.smoothing_max_samples = max(min(int(smoothing_max_samples), 8), 0)
        self.reflection_strength = float(max(reflection_strength, 0.0))
        self.refraction_strength = float(max(refraction_strength, 0.0))
        self.env_map_strength = float(max(env_map_strength, 0.0))
        self.env_reflection_lod = float(np.clip(env_reflection_lod, 0.0, 8.0))
        self.env_color_preserve = float(np.clip(env_color_preserve, 0.0, 1.0))
        self.absorption_strength = float(max(absorption_strength, 0.0))
        self.depth_visualization_strength = float(max(depth_visualization_strength, 0.0))
        self.caustic_strength = float(max(caustic_strength, 0.0))
        self.caustic_scale = float(max(caustic_scale, 1.0))
        self.floor_caustic_strength = float(max(floor_caustic_strength, 0.0))
        self.surface_shadow_strength = float(max(surface_shadow_strength, 0.0))
        self.foam_strength = float(max(foam_strength, 0.0))
        self.foam_scale = float(max(foam_scale, 1.0))

        # Fast path: when the solver provides device-side render buffers, pack
        # the interleaved vertex data and bounds on the GPU. This avoids four
        # separate device-to-host copies plus numpy filtering per frame.
        scalar_radius = radii is None or isinstance(radii, (int, float, np.integer, np.floating))
        if (
            render_points is not None
            and anisotropy is not None
            and anisotropy_secondary is not None
            and anisotropy_tertiary is not None
            and isinstance(render_points, wp.array)
            and render_points.device.is_cuda
            and (scalar_radius or (isinstance(radii, wp.array) and radii.device == render_points.device))
        ):
            self._update_from_device(render_points, radii, anisotropy, anisotropy_secondary, anisotropy_tertiary)
            return

        host_points = points.numpy().astype(np.float32, copy=False)
        source_count = host_points.shape[0]
        if radii is None:
            radius_values = np.full(source_count, 0.1, dtype=np.float32)
        elif isinstance(radii, (int, float, np.integer, np.floating)):
            radius_values = np.full(source_count, float(radii), dtype=np.float32)
        else:
            radius_values = radii.numpy().astype(np.float32, copy=False)
        radius_values = np.asarray(radius_values * self.radius_scale, dtype=np.float32)

        host_render_points = None
        if render_points is not None:
            host_render_points = render_points.numpy().astype(np.float32, copy=False)
            if host_render_points.shape != host_points.shape:
                raise ValueError("Fluid render_points must have the same shape as points.")

        active = None
        host_anisotropy = None
        host_anisotropy_secondary = None
        host_anisotropy_tertiary = None
        if anisotropy is not None:
            host_anisotropy = anisotropy.numpy().astype(np.float32, copy=False)
            if host_anisotropy.shape != (source_count, 4):
                raise ValueError("Fluid anisotropy must have shape [particle_count, 4].")
            active = host_anisotropy[:, 3] > 0.0
            if not np.all(active):
                host_points = host_points[active]
                radius_values = radius_values[active]
                host_anisotropy = host_anisotropy[active]
                if host_render_points is not None:
                    host_render_points = host_render_points[active]

        if anisotropy_secondary is not None:
            host_anisotropy_secondary = anisotropy_secondary.numpy().astype(np.float32, copy=False)
            if host_anisotropy_secondary.shape != (source_count, 4):
                raise ValueError("Fluid anisotropy_secondary must have shape [particle_count, 4].")
            if active is not None and not np.all(active):
                host_anisotropy_secondary = host_anisotropy_secondary[active]

        if anisotropy_tertiary is not None:
            host_anisotropy_tertiary = anisotropy_tertiary.numpy().astype(np.float32, copy=False)
            if host_anisotropy_tertiary.shape != (source_count, 4):
                raise ValueError("Fluid anisotropy_tertiary must have shape [particle_count, 4].")
            if active is not None and not np.all(active):
                host_anisotropy_tertiary = host_anisotropy_tertiary[active]

        count = host_points.shape[0]
        if count > self.capacity:
            self._resize(max(count, self.capacity * 2))

        self.active_particles = count
        if count > 0:
            bounds_pad = float(np.max(radius_values)) if radius_values.size else 0.0
            self.bounds_valid = True
            self.bounds_lower = (np.min(host_points, axis=0) - bounds_pad).astype(np.float32, copy=False)
            self.bounds_upper = (np.max(host_points, axis=0) + bounds_pad).astype(np.float32, copy=False)
        else:
            self.bounds_valid = False
            return

        if (
            host_render_points is None
            or host_anisotropy is None
            or host_anisotropy_secondary is None
            or host_anisotropy_tertiary is None
        ):
            (
                fallback_render_points,
                fallback_anisotropy,
                fallback_anisotropy_secondary,
                fallback_anisotropy_tertiary,
            ) = self._compute_render_anisotropy(host_points, radius_values)
            if host_render_points is None:
                host_render_points = fallback_render_points
            if host_anisotropy is None:
                host_anisotropy = fallback_anisotropy
            if host_anisotropy_secondary is None:
                host_anisotropy_secondary = fallback_anisotropy_secondary
            if host_anisotropy_tertiary is None:
                host_anisotropy_tertiary = fallback_anisotropy_tertiary

        particle_data = np.empty((count, 16), dtype=np.float32)
        particle_data[:, :3] = host_render_points
        particle_data[:, 3] = radius_values
        particle_data[:, 4:8] = host_anisotropy
        particle_data[:, 8:12] = host_anisotropy_secondary
        particle_data[:, 12:16] = host_anisotropy_tertiary

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, particle_data.nbytes, particle_data.ctypes.data)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

    def _update_from_device(self, render_points, radii, anisotropy, anisotropy_secondary, anisotropy_tertiary):
        gl = RendererGL.gl
        count = int(len(render_points))
        if count == 0:
            self.active_particles = 0
            self.bounds_valid = False
            return
        if count > self.capacity:
            self._resize(max(count, self.capacity * 2))

        device = render_points.device
        if self._packed_gpu is None or len(self._packed_gpu) < self.capacity * 16 or self._packed_gpu.device != device:
            self._packed_gpu = wp.empty(self.capacity * 16, dtype=float, device=device)
            self._bounds_gpu = wp.empty(6, dtype=float, device=device)
            self._dummy_radii_gpu = wp.zeros(1, dtype=float, device=device)
        self._bounds_gpu.assign(self._bounds_reset)

        if isinstance(radii, wp.array):
            radii_array = radii
            use_radii = 1
            uniform_radius = 0.0
        else:
            radii_array = self._dummy_radii_gpu
            use_radii = 0
            uniform_radius = 0.1 if radii is None else float(radii)

        use_interop = ENABLE_CUDA_INTEROP
        dest = None
        if use_interop:
            if self.vertex_cuda_buffer is None:
                self.vertex_cuda_buffer = wp.RegisteredGLBuffer(
                    int(self.vbo.value), device, flags=wp.RegisteredGLBuffer.WRITE_DISCARD
                )
            dest = self.vertex_cuda_buffer.map(dtype=wp.float32, shape=(self.capacity * 16,))
        else:
            dest = self._packed_gpu

        wp.launch(
            _pack_fluid_particle_data,
            dim=count,
            inputs=[
                render_points,
                radii_array,
                use_radii,
                uniform_radius,
                self.radius_scale,
                anisotropy,
                anisotropy_secondary,
                anisotropy_tertiary,
                dest,
                self._bounds_gpu,
            ],
            device=device,
        )

        if use_interop:
            self.vertex_cuda_buffer.unmap()
        else:
            packed_host = self._packed_gpu[: count * 16].numpy()
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
            gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, packed_host.nbytes, packed_host.ctypes.data)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

        bounds_host = self._bounds_gpu.numpy()
        self.active_particles = count
        if bounds_host[0] <= bounds_host[3]:
            self.bounds_valid = True
            self.bounds_lower = bounds_host[:3].copy()
            self.bounds_upper = bounds_host[3:].copy()
        else:
            self.bounds_valid = False

    def _compute_render_anisotropy(
        self, points: np.ndarray, radii: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        count = int(points.shape[0])
        render_points = np.array(points, dtype=np.float32, copy=True)
        anisotropy = np.zeros((count, 4), dtype=np.float32)
        anisotropy_secondary = np.zeros((count, 4), dtype=np.float32)
        anisotropy_tertiary = np.zeros((count, 4), dtype=np.float32)
        if count == 0:
            return render_points, anisotropy, anisotropy_secondary, anisotropy_tertiary

        anisotropy[:, 0] = 1.0
        anisotropy[:, 3] = 1.0
        anisotropy_secondary[:, 1] = 1.0
        anisotropy_secondary[:, 3] = 1.0
        anisotropy_tertiary[:, 2] = 1.0
        anisotropy_tertiary[:, 3] = 1.0
        positive_radii = radii[radii > 0.0]
        if positive_radii.size == 0 or count < 4 or count > self.anisotropy_max_particles:
            return render_points, anisotropy, anisotropy_secondary, anisotropy_tertiary

        base_radius = float(np.median(positive_radii))
        support = max(base_radius * 2.6, 1.0e-5)
        support_sq = support * support
        cell_coords = np.floor(points / support).astype(np.int32, copy=False)
        buckets: dict[tuple[int, int, int], list[int]] = {}
        for particle_index, cell in enumerate(cell_coords):
            buckets.setdefault((int(cell[0]), int(cell[1]), int(cell[2])), []).append(particle_index)

        offsets = [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)]
        identity_axis = np.array((1.0, 0.0, 0.0), dtype=np.float32)
        secondary_fallback = np.array((0.0, 1.0, 0.0), dtype=np.float32)
        tertiary_fallback = np.array((0.0, 0.0, 1.0), dtype=np.float32)
        for particle_index, point in enumerate(points):
            cell = cell_coords[particle_index]
            candidates: list[int] = []
            for dx, dy, dz in offsets:
                candidates.extend(buckets.get((int(cell[0] + dx), int(cell[1] + dy), int(cell[2] + dz)), ()))
            if len(candidates) < 4:
                continue

            neighbor_points = points[np.asarray(candidates, dtype=np.int32)]
            delta = neighbor_points - point
            dist_sq = np.einsum("ij,ij->i", delta, delta)
            mask = dist_sq < support_sq
            if int(np.count_nonzero(mask)) < 4:
                continue

            neighbor_points = neighbor_points[mask]
            dist_sq = dist_sq[mask]
            weights = np.square(np.maximum(1.0 - dist_sq / support_sq, 0.0)).astype(np.float32)
            weight_sum = float(np.sum(weights))
            if weight_sum <= 1.0e-8:
                continue

            center = np.sum(neighbor_points * weights[:, None], axis=0) / weight_sum
            render_points[particle_index] = point * (1.0 - self.render_smoothing) + center * self.render_smoothing

            if int(np.count_nonzero(dist_sq > 1.0e-10)) < 3:
                continue

            centered = neighbor_points - center
            covariance = (centered * weights[:, None]).T @ centered / weight_sum
            covariance += np.eye(3, dtype=np.float32) * (base_radius * base_radius * 0.025)
            try:
                values, vectors = np.linalg.eigh(covariance)
            except np.linalg.LinAlgError:
                continue

            order = np.argsort(values)[::-1]
            values = values[order]
            major_axis = vectors[:, order[0]].astype(np.float32, copy=False)
            if not np.all(np.isfinite(major_axis)) or float(np.dot(major_axis, major_axis)) < 1.0e-8:
                major_axis = identity_axis
            secondary_axis = vectors[:, order[1]].astype(np.float32, copy=False)
            if not np.all(np.isfinite(secondary_axis)) or float(np.dot(secondary_axis, secondary_axis)) < 1.0e-8:
                secondary_axis = secondary_fallback
            secondary_axis = secondary_axis - major_axis * float(np.dot(secondary_axis, major_axis))
            secondary_norm = float(np.linalg.norm(secondary_axis))
            if secondary_norm <= 1.0e-8:
                secondary_axis = secondary_fallback
            else:
                secondary_axis = secondary_axis / secondary_norm
            tertiary_axis = vectors[:, order[2]].astype(np.float32, copy=False)
            if not np.all(np.isfinite(tertiary_axis)) or float(np.dot(tertiary_axis, tertiary_axis)) < 1.0e-8:
                tertiary_axis = np.cross(major_axis, secondary_axis).astype(np.float32, copy=False)
            tertiary_axis = tertiary_axis - major_axis * float(np.dot(tertiary_axis, major_axis))
            tertiary_axis = tertiary_axis - secondary_axis * float(np.dot(tertiary_axis, secondary_axis))
            tertiary_norm = float(np.linalg.norm(tertiary_axis))
            if tertiary_norm <= 1.0e-8:
                tertiary_axis = np.cross(major_axis, secondary_axis).astype(np.float32, copy=False)
                tertiary_norm = float(np.linalg.norm(tertiary_axis))
            if tertiary_norm <= 1.0e-8:
                tertiary_axis = tertiary_fallback
            else:
                tertiary_axis = tertiary_axis / tertiary_norm

            spread_major = float(np.sqrt(max(values[0], 0.0)))
            spread_minor = float(np.sqrt(max(values[-1], 0.0)))
            eccentricity = max((spread_major - spread_minor) / max(base_radius, 1.0e-6), 0.0)
            min_axis_scale = max(float(self.anisotropy_min), 0.01)
            max_axis_scale = max(float(self.anisotropy_max), min_axis_scale)
            major_min_scale = max(min_axis_scale, 1.0)
            major_max_scale = max(max_axis_scale, major_min_scale)
            stretch = 1.0 + self.anisotropy_strength * min(eccentricity, major_max_scale - 1.0)
            stretch = float(np.clip(stretch, major_min_scale, major_max_scale))
            stretch_strength = np.clip((stretch - 1.0) / max(major_max_scale - 1.0, 1.0e-6), 0.0, 1.0)
            minor_min_scale = min(min_axis_scale, 1.0)
            minor_span = 1.0 - minor_min_scale
            anisotropy[particle_index, :3] = major_axis
            anisotropy[particle_index, 3] = stretch
            anisotropy_secondary[particle_index, :3] = secondary_axis
            anisotropy_secondary[particle_index, 3] = float(
                np.clip(1.0 - 0.70 * stretch_strength * minor_span, min_axis_scale, max_axis_scale)
            )
            anisotropy_tertiary[particle_index, :3] = tertiary_axis
            anisotropy_tertiary[particle_index, 3] = float(
                np.clip(1.0 - stretch_strength * minor_span, min_axis_scale, max_axis_scale)
            )

        return render_points, anisotropy, anisotropy_secondary, anisotropy_tertiary

    def render(self):
        if self.hidden or self.active_particles == 0:
            return
        gl = RendererGL.gl
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_POINTS, 0, self.active_particles)
        gl.glBindVertexArray(0)


class FluidDiffuseGL:
    """GPU buffer for secondary diffuse foam/spray particles."""

    def __init__(self, capacity: int):
        gl = RendererGL.gl
        self.capacity = max(int(capacity), 1)
        self.active_particles = 0
        self.hidden = False
        self.radius = 0.025
        self.color = (1.0, 1.0, 1.0)
        self.alpha = 0.75
        self.motion_blur_scale = 1.0
        self.expansion = 0.65
        self.inscatter = 0.38
        self.outscatter = 0.18
        self.shadow_strength = 0.42
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
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.capacity * 4 * 4, None, gl.GL_DYNAMIC_DRAW)
        gl.glVertexAttribPointer(0, 4, gl.GL_FLOAT, gl.GL_FALSE, 4 * 4, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.velocity_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.capacity * 4 * 4, None, gl.GL_DYNAMIC_DRAW)
        gl.glVertexAttribPointer(1, 4, gl.GL_FLOAT, gl.GL_FALSE, 4 * 4, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)
        gl.glBindVertexArray(0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

    def destroy(self):
        gl = RendererGL.gl
        if getattr(self, "vao", None) is not None:
            gl.glDeleteVertexArrays(1, self.vao)
            gl.glDeleteBuffers(1, self.position_vbo)
            gl.glDeleteBuffers(1, self.velocity_vbo)
            self.vao = None
            self.position_vbo = None
            self.velocity_vbo = None

    def _resize(self, capacity: int):
        self.destroy()
        self.__init__(capacity)

    def update(
        self,
        positions,
        velocities,
        radius: float,
        color: tuple[float, float, float],
        alpha: float,
        motion_blur_scale: float,
        expansion: float,
        inscatter: float,
        outscatter: float,
        shadow_strength: float,
        hidden: bool,
    ):
        if positions is None:
            self.active_particles = 0
            self.hidden = True
            return

        self.hidden = hidden
        self.radius = float(max(radius, 0.0))
        self.color = tuple(float(c) for c in color)
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.motion_blur_scale = float(max(motion_blur_scale, 0.0))
        self.expansion = float(max(expansion, 0.0))
        self.inscatter = float(max(inscatter, 0.0))
        self.outscatter = float(max(outscatter, 0.0))
        self.shadow_strength = float(max(shadow_strength, 0.0))

        host_positions = positions.numpy().astype(np.float32, copy=False)
        if velocities is None:
            host_velocities = np.zeros_like(host_positions)
        else:
            host_velocities = velocities.numpy().astype(np.float32, copy=False)

        live = host_positions[:, 3] > 0.0
        host_positions = np.ascontiguousarray(host_positions[live])
        host_velocities = np.ascontiguousarray(host_velocities[live])
        count = int(host_positions.shape[0])
        if count > self.capacity:
            self._resize(max(count, self.capacity * 2))

        self.active_particles = count
        self._host_positions = host_positions
        self._host_velocities = host_velocities
        self._upload_host_arrays()

    def _upload_host_arrays(self):
        if self.active_particles == 0:
            return
        gl = RendererGL.gl
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.position_vbo)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, self._host_positions.nbytes, self._host_positions.ctypes.data)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.velocity_vbo)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, self._host_velocities.nbytes, self._host_velocities.ctypes.data)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

    def sort_for_view(self, view_matrix):
        if self.active_particles <= 1:
            return

        view = np.asarray(view_matrix, dtype=np.float32).reshape(4, 4).transpose()
        homogeneous = np.ones((self.active_particles, 4), dtype=np.float32)
        homogeneous[:, :3] = self._host_positions[:, :3]
        view_positions = (view @ homogeneous.T).T
        order = np.argsort(view_positions[:, 2], kind="mergesort")
        self._host_positions = np.ascontiguousarray(self._host_positions[order])
        self._host_velocities = np.ascontiguousarray(self._host_velocities[order])
        self._upload_host_arrays()

    def render(self):
        if self.hidden or self.active_particles == 0:
            return
        gl = RendererGL.gl
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_POINTS, 0, self.active_particles)
        gl.glBindVertexArray(0)


class RendererGL:
    gl = None  # Class-level variable to hold the imported module
    _fallback_texture = None  # 1x1 white texture bound when no albedo is set (suppresses macOS GL warning)

    @classmethod
    def initialize_gl(cls):
        if cls.gl is None:  # Only import if not already imported
            from pyglet import gl

            cls.gl = gl

    @classmethod
    def get_fallback_texture(cls):
        """Return a 1x1 white RGBA texture, creating it on first use."""
        if cls._fallback_texture is None:
            gl = cls.gl
            tex = gl.GLuint()
            gl.glGenTextures(1, tex)
            gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
            pixel = (gl.GLubyte * 4)(255, 255, 255, 255)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, 1, 1, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, pixel)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            cls._fallback_texture = tex
        return cls._fallback_texture

    def __init__(self, title="Newton", screen_width=1920, screen_height=1080, vsync=True, headless=None, device=None):
        self.draw_sky = True
        self.draw_fps = True
        self.draw_shadows = True
        self.draw_wireframe = False
        self.wireframe_line_width = 1.5  # pixels
        self.line_width = 1.5  # pixels, for all log_lines batches
        self.arrow_scale = 1.0  # screen-space multiplier on arrow line width and arrowhead size
        self.arrow_length_scale = 1.0  # multiplier on contact-arrow world-space length
        self.joint_scale = 1.0  # multiplier on joint-axis line length
        self.com_scale = 1.0  # multiplier on COM sphere radius
        self.draw_edges = False
        self._edge_color = (0.05, 0.05, 0.05, 1.0)

        self.background_color = (68.0 / 255.0, 161.0 / 255.0, 255.0 / 255.0)

        self.sky_upper = self.background_color
        self.sky_lower = (40.0 / 255.0, 44.0 / 255.0, 55.0 / 255.0)

        # Lighting settings
        self._shadow_radius = 3.0
        self._diffuse_scale = 1.0
        self._specular_scale = 1.0
        self.spotlight_enabled = True
        self._shadow_extents = 10.0
        self._exposure = 1.6

        # Hemispherical ambient light colors, interpolated by dot(N, up).
        # Decoupled from the sky background so the visible sky can be a
        # saturated blue while the ambient fill stays neutral — a stand-in
        # for a proper irradiance map that we don't precompute yet.
        self.ambient_sky = (0.8, 0.8, 0.85)
        self.ambient_ground = (0.3, 0.3, 0.35)

        # On Wayland, PyOpenGL defaults to EGL which cannot see the GLX context
        # that pyglet creates via XWayland. Force GLX so both libraries agree.
        # Must be set before PyOpenGL is first imported (platform is selected
        # once at import time).
        if "PYOPENGL_PLATFORM" not in os.environ:
            # WAYLAND_DISPLAY is the primary indicator; XDG_SESSION_TYPE is
            # checked as a fallback for sessions where the socket is not yet set.
            is_wayland = bool(os.environ.get("WAYLAND_DISPLAY")) or os.environ.get("XDG_SESSION_TYPE") == "wayland"
            if is_wayland:
                os.environ["PYOPENGL_PLATFORM"] = "glx"

        try:
            import pyglet

            # disable error checking for performance
            pyglet.options["debug_gl"] = False

            # try imports
            from pyglet.graphics.shader import Shader, ShaderProgram  # noqa: F401
            from pyglet.math import Vec3 as PyVec3  # noqa: F401

            RendererGL.initialize_gl()
            gl = RendererGL.gl
        except ImportError as e:
            raise Exception("OpenGLRenderer requires pyglet (version >= 2.0) to be installed.") from e

        self._title = title

        try:
            # try to enable MSAA
            config = pyglet.gl.Config(sample_buffers=1, samples=8, double_buffer=True)
            self.window = pyglet.window.Window(
                width=screen_width,
                height=screen_height,
                caption=title,
                resizable=True,
                vsync=vsync,
                visible=not headless,
                config=config,
            )
            gl.glEnable(gl.GL_MULTISAMPLE)
            # remember sample count for later (e.g., resolving FBO)
            self.msaa_samples = 4
        except pyglet.window.NoSuchConfigException:
            print("Warning: Could not get MSAA config, falling back to non-AA.")
            self.window = pyglet.window.Window(
                width=screen_width,
                height=screen_height,
                caption=title,
                resizable=True,
                vsync=vsync,
                visible=not headless,
            )
            self.msaa_samples = 0

        self._set_icon()

        # Pyglet on Windows 8+ (where _always_dwm=True) disables the GL
        # swap interval to avoid double-syncing with DWM, but then also
        # skips calling DwmFlush() in flip() due to a condition bug.
        # We call DwmFlush() ourselves in present() to work around this.
        self._dwm_flush = None
        if sys.platform == "win32" and getattr(self.window, "_always_dwm", False):
            try:
                self._dwm_flush = ctypes.windll.dwmapi.DwmFlush
            except (AttributeError, OSError):
                pass

        if headless is None:
            self.headless = pyglet.options.get("headless", False)
        else:
            self.headless = headless
        self.app = pyglet.app

        # making window current opengl rendering context
        self._make_current()

        self._screen_width, self._screen_height = self.window.get_framebuffer_size()

        self._camera_speed = 0.04
        self._last_x, self._last_y = self._screen_width // 2, self._screen_height // 2
        self._key_callbacks = []
        self._key_release_callbacks = []

        self._env_texture = None
        self._env_intensity = 1.0
        self._env_path = None
        self._env_texture_obj = None

        default_env = os.path.join(os.path.dirname(__file__), "newton_envmap.jpg")
        if os.path.exists(default_env):
            self._env_path = default_env
        self._mouse_drag_callbacks = []
        self._mouse_press_callbacks = []
        self._mouse_release_callbacks = []
        self._mouse_motion_callbacks = []
        self._mouse_scroll_callbacks = []
        self._resize_callbacks = []

        # Initialize device and shape lookup
        self._device = device if device is not None else wp.get_device()
        self._shape_lookup = {}

        self._shadow_fbo = None
        self._shadow_texture = None
        self._shadow_shader = None
        self._shadow_width = 4096
        self._shadow_height = 4096
        self._light_space_matrix = np.eye(4, dtype=np.float32)

        self._frame_texture = None
        self._frame_depth_texture = None
        self._frame_fbo = None
        self._frame_pbo = None
        self._fluid_fbo = None
        self._fluid_blur_fbo = None
        self._fluid_depth_texture = None
        self._fluid_depth_smooth_texture = None
        self._fluid_thickness_texture = None
        self._fluid_thickness_smooth_texture = None
        self._fluid_scene_texture = None
        self._fluid_scene_depth_texture = None
        self._fluid_shadow_depth_texture = None
        self._fluid_shadow_thickness_texture = None
        self._fluid_shadow_depth_attachment = None
        self._fluid_shadow_size = 2048

        self._sun_direction = None  # set on first render based on camera up_axis
        self._light_view_matrix = np.eye(4, dtype=np.float32)
        self._light_projection_matrix = np.eye(4, dtype=np.float32)

        self._light_color = (1.0, 1.0, 1.0)

        check_gl_error()

        if not headless:
            # set up our own event handling so we can synchronously render frames
            # by calling update() in a loop
            from pyglet.window import Window

            Window._enable_event_queue = False

            self.window.dispatch_pending_events()

            platform_event_loop = self.app.platform_event_loop
            platform_event_loop.start()

            # start event loop
            # self.app.event_loop.dispatch_event("on_enter")

        # create frame buffer for rendering to a texture
        self._setup_shadow_buffer()
        self._setup_frame_buffer()
        self._setup_fluid_buffers()
        self._setup_sky_mesh()
        self._setup_frame_mesh()

        self._shadow_shader = ShadowShader(gl)
        self._shape_shader = ShaderShape(gl)
        self._edge_shader = ShaderEdge(gl)
        self._frame_shader = FrameShader(gl)
        self._sky_shader = ShaderSky(gl)
        self._wireframe_shader = ShaderLine(gl)
        self._arrow_shader = ShaderArrow(gl)
        self._fluid_particle_shader = FluidParticleShader(gl)
        self._fluid_diffuse_shader = FluidDiffuseShader(gl)
        self._fluid_blur_shader = FluidBlurShader(gl)
        self._fluid_shadow_shader = FluidShadowShader(gl)
        self._fluid_composite_shader = FluidCompositeShader(gl)

        if not headless:
            self._setup_window_callbacks()

    @property
    def shadow_radius(self) -> float:
        return self._shadow_radius

    @shadow_radius.setter
    def shadow_radius(self, value: float):
        self._shadow_radius = max(float(value), 0.0)

    @property
    def fluid_shadow_size(self) -> int:
        return self._fluid_shadow_size

    @fluid_shadow_size.setter
    def fluid_shadow_size(self, value: int):
        size = max(min(int(value), 4096), 256)
        if size == self._fluid_shadow_size:
            return

        self._fluid_shadow_size = size
        if getattr(self, "_fluid_shadow_depth_texture", None) is not None:
            self._setup_fluid_buffers()

    @property
    def diffuse_scale(self) -> float:
        return self._diffuse_scale

    @diffuse_scale.setter
    def diffuse_scale(self, value: float):
        self._diffuse_scale = max(float(value), 0.0)

    @property
    def specular_scale(self) -> float:
        return self._specular_scale

    @specular_scale.setter
    def specular_scale(self, value: float):
        self._specular_scale = max(float(value), 0.0)

    @property
    def shadow_extents(self) -> float:
        return self._shadow_extents

    @shadow_extents.setter
    def shadow_extents(self, value: float):
        self._shadow_extents = max(float(value), 1e-4)

    @property
    def exposure(self) -> float:
        return self._exposure

    @exposure.setter
    def exposure(self, value: float):
        self._exposure = max(float(value), 0.0)

    def update(self):
        self._make_current()

        if not self.headless:
            import pyglet

            pyglet.clock.tick()

            self.app.platform_event_loop.step(0.001)  # 1ms app polling latency
            try:
                self.window.dispatch_events()
            except (ctypes.ArgumentError, TypeError):
                # Handle known issue with pyglet xlib backend on some Linux configurations
                # where window handle can have wrong type in XCheckWindowEvent
                # This is a non-fatal error that can be safely ignored
                pass

    def render(self, camera, objects, lines=None, wireframe_shapes=None, arrows=None, fluids=None, fluid_diffuse=None):
        gl = RendererGL.gl
        self._make_current()

        gl.glClearColor(*self.sky_upper, 1)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthMask(True)
        gl.glDepthRange(0.0, 1.0)

        self.camera = camera

        # Lazy-init sun direction based on camera up axis
        if self._sun_direction is None:
            _sun_dirs = {
                0: np.array((0.8, 0.2, -0.3)),  # X-up
                1: np.array((0.2, 0.8, -0.3)),  # Y-up
                2: np.array((0.2, -0.3, 0.8)),  # Z-up
            }
            d = _sun_dirs.get(camera.up_axis, _sun_dirs[2])
            self._sun_direction = d / np.linalg.norm(d)

        # Store matrices for other methods
        self._view_matrix = self.camera.get_view_matrix()
        self._projection_matrix = self.camera.get_projection_matrix()
        if self.draw_shadows or fluids:
            self._update_light_matrices()

        # Lazy-load environment map after a valid GL context is active
        if self._env_path is not None and self._env_texture is None:
            try:
                self.set_environment_map(self._env_path)
            except Exception:
                pass
            self._env_path = None

        # 1. render depth of scene to texture (from light's perspective)
        gl.glViewport(0, 0, self._shadow_width, self._shadow_height)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._shadow_fbo)
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT)

        if self.draw_shadows:
            # Note: lines are skipped during shadow pass since they don't cast shadows
            self._render_shadow_map(objects)

        # reset viewport
        gl.glViewport(0, 0, self._screen_width, self._screen_height)

        # select target framebuffer (MSAA or regular) for scene rendering
        target_fbo = self._frame_msaa_fbo if getattr(self, "msaa_samples", 0) > 0 else self._frame_fbo

        # ---------------------------------------
        # Set texture as render target for MSAA resolve

        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, target_fbo)
        gl.glDrawBuffer(gl.GL_COLOR_ATTACHMENT0)

        gl.glClearColor(*self.sky_upper, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glBindVertexArray(0)

        self._render_scene(objects)

        msaa_resolved = False
        if getattr(self, "msaa_samples", 0) > 0 and self._frame_msaa_fbo is not None:
            self._resolve_msaa_frame()
            msaa_resolved = True
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._frame_fbo)
            gl.glDrawBuffer(gl.GL_COLOR_ATTACHMENT0)

        rendered_fluid_surface = False
        if fluids:
            rendered_fluid_surface = self._render_fluids(fluids, fluid_diffuse)

        if fluid_diffuse and not rendered_fluid_surface:
            self._render_fluid_diffuse(fluid_diffuse)

        # Render lines after main scene and fluid composition.
        if lines:
            self._render_lines(lines)

        if arrows:
            self._render_arrows(arrows)

        if wireframe_shapes:
            self._render_wireframe_shapes(wireframe_shapes)

        # ------------------------------------------------------------------
        # If MSAA is enabled, resolve the multi-sample buffer into texture FBO
        # ------------------------------------------------------------------
        if not msaa_resolved and getattr(self, "msaa_samples", 0) > 0 and self._frame_msaa_fbo is not None:
            self._resolve_msaa_frame()

        # ------------------------------------------------------------------
        # Draw resolved texture to the screen
        # ------------------------------------------------------------------
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glViewport(0, 0, self._screen_width, self._screen_height)

        # render frame buffer texture to screen
        if self._frame_fbo is not None:
            with self._frame_shader:
                gl.glActiveTexture(gl.GL_TEXTURE0)
                gl.glBindTexture(gl.GL_TEXTURE_2D, self._frame_texture)
                self._frame_shader.update(0)

                gl.glBindVertexArray(self._frame_vao)
                gl.glDrawElements(gl.GL_TRIANGLES, len(self._frame_indices), gl.GL_UNSIGNED_INT, None)
                gl.glBindVertexArray(0)
                gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        if self.draw_fps:
            gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            gl.glEnable(gl.GL_BLEND)

        err = gl.glGetError()
        assert err == gl.GL_NO_ERROR, hex(err)

    def _resolve_msaa_frame(self):
        gl = RendererGL.gl
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self._frame_msaa_fbo)
        gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)

        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, self._frame_fbo)
        gl.glDrawBuffer(gl.GL_COLOR_ATTACHMENT0)

        gl.glBlitFramebuffer(
            0,
            0,
            self._screen_width,
            self._screen_height,
            0,
            0,
            self._screen_width,
            self._screen_height,
            gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT,
            gl.GL_NEAREST,
        )

    def present(self):
        if not self.headless:
            if self._dwm_flush is not None and self.window._interval:
                self._dwm_flush()
            self.window.flip()

    def resize(self, width, height):
        self._screen_width, self._screen_height = self.window.get_framebuffer_size()
        self._setup_frame_buffer()
        self._setup_fluid_buffers()

    def set_title(self, title):
        self.window.set_caption(title)

    def set_vsync(self, enabled: bool):
        """Enable or disable vertical synchronization (vsync).

        Args:
            enabled: If True, enable vsync; if False, disable vsync.
        """
        self.window.set_vsync(enabled)

    def get_vsync(self) -> bool:
        """Get the current vsync state.

        Returns:
            True if vsync is enabled, False otherwise.
        """
        return self.window.vsync

    def has_exit(self):
        return self.app.event_loop.has_exit

    def close(self):
        self._make_current()

        if not self.headless:
            self.app.event_loop.dispatch_event("on_exit")
            self.app.platform_event_loop.stop()

        RendererGL._fallback_texture = None
        self.window.close()

    def _setup_window_callbacks(self):
        """Set up the basic window event handlers."""
        import pyglet

        self.window.push_handlers(on_draw=self._on_draw)
        self.window.push_handlers(on_resize=self._on_window_resize)
        self.window.push_handlers(on_key_press=self._on_key_press)
        self.window.push_handlers(on_key_release=self._on_key_release)
        self.window.push_handlers(on_close=self._on_close)

        self._key_handler = pyglet.window.key.KeyStateHandler()
        self.window.push_handlers(self._key_handler)

        self.window.push_handlers(on_mouse_press=self._on_mouse_press)
        self.window.push_handlers(on_mouse_release=self._on_mouse_release)

        self.window.on_mouse_scroll = self._on_scroll
        self.window.on_mouse_drag = self._on_mouse_drag
        self.window.on_mouse_motion = self._on_mouse_motion

    def register_key_press(self, callback):
        """Register a callback for key press events.

        Args:
            callback: Function that takes (symbol, modifiers) parameters
        """
        self._key_callbacks.append(callback)

    def register_key_release(self, callback):
        """Register a callback for key release events.

        Args:
            callback: Function that takes (symbol, modifiers) parameters
        """
        self._key_release_callbacks.append(callback)

    def register_mouse_press(self, callback):
        """Register a callback for mouse press events.

        Args:
            callback: Function that takes (x, y, button, modifiers) parameters
        """
        self._mouse_press_callbacks.append(callback)

    def register_mouse_release(self, callback):
        """Register a callback for mouse release events.

        Args:
            callback: Function that takes (x, y, button, modifiers) parameters
        """
        self._mouse_release_callbacks.append(callback)

    def register_mouse_drag(self, callback):
        """Register a callback for mouse drag events.

        Args:
            callback: Function that takes (x, y, dx, dy, buttons, modifiers) parameters
        """
        self._mouse_drag_callbacks.append(callback)

    def register_mouse_motion(self, callback):
        """Register a callback for mouse motion events.

        Args:
            callback: Function that takes (x, y, dx, dy) parameters
        """
        self._mouse_motion_callbacks.append(callback)

    def register_mouse_scroll(self, callback):
        """Register a callback for mouse scroll events.

        Args:
            callback: Function that takes (x, y, scroll_x, scroll_y) parameters
        """
        self._mouse_scroll_callbacks.append(callback)

    def register_resize(self, callback):
        """Register a callback for window resize events.

        Args:
            callback: Function that takes (width, height) parameters
        """
        self._resize_callbacks.append(callback)

    def register_update(self, callback):
        """Register a per-frame update callback receiving dt (seconds)."""
        self._update_callbacks.append(callback)

    def _on_key_press(self, symbol, modifiers):
        # update key state
        for callback in self._key_callbacks:
            callback(symbol, modifiers)

    def _on_key_release(self, symbol, modifiers):
        # update key state
        for callback in self._key_release_callbacks:
            callback(symbol, modifiers)

    def _on_mouse_press(self, x, y, button, modifiers):
        """Handle mouse button press events."""
        for callback in self._mouse_press_callbacks:
            callback(x, y, button, modifiers)

    def _on_mouse_release(self, x, y, button, modifiers):
        """Handle mouse button release events."""
        for callback in self._mouse_release_callbacks:
            callback(x, y, button, modifiers)

    def _on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        # Then call registered callbacks
        for callback in self._mouse_drag_callbacks:
            callback(x, y, dx, dy, buttons, modifiers)

    def _on_mouse_motion(self, x, y, dx, dy):
        """Handle mouse motion events."""
        for callback in self._mouse_motion_callbacks:
            callback(x, y, dx, dy)

    def _on_scroll(self, x, y, scroll_x, scroll_y):
        for callback in self._mouse_scroll_callbacks:
            callback(x, y, scroll_x, scroll_y)

    def _on_window_resize(self, width, height):
        self.resize(width, height)

        for callback in self._resize_callbacks:
            callback(width, height)

    def _on_close(self):
        self.close()

    def _on_draw(self):
        pass

    # public query for key state
    def is_key_down(self, symbol: int) -> bool:
        if self.headless:
            return False

        return bool(self._key_handler[symbol])

    def _setup_sky_mesh(self):
        gl = RendererGL.gl

        # create VAO, VBO, and EBO
        self._sky_vao = gl.GLuint()
        gl.glGenVertexArrays(1, self._sky_vao)
        gl.glBindVertexArray(self._sky_vao)

        sky_mesh = Mesh.create_sphere(
            1.0,
            num_latitudes=32,
            num_longitudes=32,
            reverse_winding=True,
            compute_inertia=False,
        )
        vertices = np.hstack([sky_mesh.vertices, sky_mesh.normals, sky_mesh.uvs]).astype(np.float32, copy=False)
        indices = sky_mesh.indices.astype(np.uint32, copy=False)
        self._sky_tri_count = len(indices)

        self._sky_vbo = gl.GLuint()
        gl.glGenBuffers(1, self._sky_vbo)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._sky_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices.ctypes.data, gl.GL_STATIC_DRAW)

        self._sky_ebo = gl.GLuint()
        gl.glGenBuffers(1, self._sky_ebo)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._sky_ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices.ctypes.data, gl.GL_STATIC_DRAW)

        # set up vertex attributes
        vertex_stride = vertices.shape[1] * vertices.itemsize
        # positions
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, vertex_stride, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)
        # normals
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, vertex_stride, ctypes.c_void_p(3 * vertices.itemsize))
        gl.glEnableVertexAttribArray(1)
        # uv coordinates
        gl.glVertexAttribPointer(2, 2, gl.GL_FLOAT, gl.GL_FALSE, vertex_stride, ctypes.c_void_p(6 * vertices.itemsize))
        gl.glEnableVertexAttribArray(2)

        gl.glBindVertexArray(0)

        # unbind the VBO and VAO
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)

        check_gl_error()

    def _setup_frame_buffer(self):
        gl = RendererGL.gl

        # Ensure MSAA member variables exist even on first call
        if not hasattr(self, "_frame_msaa_color_rb"):
            self._frame_msaa_color_rb = None
        if not hasattr(self, "_frame_msaa_depth_rb"):
            self._frame_msaa_depth_rb = None
        if not hasattr(self, "_frame_msaa_fbo"):
            self._frame_msaa_fbo = None

        self._make_current()

        if self._frame_texture is None:
            self._frame_texture = gl.GLuint()
            gl.glGenTextures(1, self._frame_texture)
        if self._frame_depth_texture is None:
            self._frame_depth_texture = gl.GLuint()
            gl.glGenTextures(1, self._frame_depth_texture)

        # set up RGB texture
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._frame_texture)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGB,
            self._screen_width,
            self._screen_height,
            0,
            gl.GL_RGB,
            gl.GL_UNSIGNED_BYTE,
            None,
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

        # set up depth texture
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._frame_depth_texture)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_DEPTH_COMPONENT32,
            self._screen_width,
            self._screen_height,
            0,
            gl.GL_DEPTH_COMPONENT,
            gl.GL_FLOAT,
            None,
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        # create a framebuffer object (FBO)
        if self._frame_fbo is None:
            self._frame_fbo = gl.GLuint()
            gl.glGenFramebuffers(1, self._frame_fbo)
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._frame_fbo)

            # attach the texture to the FBO as its color attachment
            gl.glFramebufferTexture2D(
                gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self._frame_texture, 0
            )
            # attach the depth texture to the FBO as its depth attachment
            gl.glFramebufferTexture2D(
                gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_TEXTURE_2D, self._frame_depth_texture, 0
            )

            if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
                print("Framebuffer is not complete!", flush=True)
                gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
                sys.exit(1)

        # unbind the FBO (switch back to the default framebuffer)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        if self._frame_pbo is None:
            self._frame_pbo = gl.GLuint()
            gl.glGenBuffers(1, self._frame_pbo)  # generate 1 buffer reference
        # binding to this buffer
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, self._frame_pbo)

        # allocate memory for PBO
        rgb_bytes_per_pixel = 3
        depth_bytes_per_pixel = 4
        pixels = np.zeros(
            (self._screen_height, self._screen_width, rgb_bytes_per_pixel + depth_bytes_per_pixel), dtype=np.uint8
        )
        gl.glBufferData(gl.GL_PIXEL_PACK_BUFFER, pixels.nbytes, pixels.ctypes.data, gl.GL_DYNAMIC_DRAW)
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)

        # ---------------------------------------------------------------------
        # Additional: create MSAA framebuffer if multi-sampling is enabled
        # ---------------------------------------------------------------------
        if getattr(self, "msaa_samples", 0) > 0:
            # color renderbuffer
            if self._frame_msaa_color_rb is None:
                self._frame_msaa_color_rb = gl.GLuint()
                gl.glGenRenderbuffers(1, self._frame_msaa_color_rb)
            gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self._frame_msaa_color_rb)
            gl.glRenderbufferStorageMultisample(
                gl.GL_RENDERBUFFER, self.msaa_samples, gl.GL_RGB8, self._screen_width, self._screen_height
            )

            # depth renderbuffer
            if self._frame_msaa_depth_rb is None:
                self._frame_msaa_depth_rb = gl.GLuint()
                gl.glGenRenderbuffers(1, self._frame_msaa_depth_rb)
            gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self._frame_msaa_depth_rb)
            gl.glRenderbufferStorageMultisample(
                gl.GL_RENDERBUFFER, self.msaa_samples, gl.GL_DEPTH_COMPONENT32, self._screen_width, self._screen_height
            )

            # FBO
            if self._frame_msaa_fbo is None:
                self._frame_msaa_fbo = gl.GLuint()
                gl.glGenFramebuffers(1, self._frame_msaa_fbo)
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._frame_msaa_fbo)
            gl.glFramebufferRenderbuffer(
                gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_RENDERBUFFER, self._frame_msaa_color_rb
            )
            gl.glFramebufferRenderbuffer(
                gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, self._frame_msaa_depth_rb
            )

            if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
                print("Warning: MSAA framebuffer incomplete, disabling MSAA.")
                self.msaa_samples = 0
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        check_gl_error()

    def _setup_fluid_buffers(self):
        gl = RendererGL.gl
        self._make_current()

        def ensure_texture(texture):
            if texture is not None:
                return texture
            texture = gl.GLuint()
            gl.glGenTextures(1, texture)
            return texture

        self._fluid_depth_texture = ensure_texture(self._fluid_depth_texture)
        self._fluid_depth_smooth_texture = ensure_texture(self._fluid_depth_smooth_texture)
        self._fluid_thickness_texture = ensure_texture(self._fluid_thickness_texture)
        self._fluid_thickness_smooth_texture = ensure_texture(self._fluid_thickness_smooth_texture)
        self._fluid_scene_texture = ensure_texture(self._fluid_scene_texture)
        self._fluid_scene_depth_texture = ensure_texture(self._fluid_scene_depth_texture)
        self._fluid_shadow_depth_texture = ensure_texture(self._fluid_shadow_depth_texture)
        self._fluid_shadow_thickness_texture = ensure_texture(self._fluid_shadow_thickness_texture)
        self._fluid_shadow_depth_attachment = ensure_texture(self._fluid_shadow_depth_attachment)

        for texture in (
            self._fluid_depth_texture,
            self._fluid_depth_smooth_texture,
            self._fluid_thickness_texture,
            self._fluid_thickness_smooth_texture,
        ):
            gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D,
                0,
                gl.GL_R32F,
                self._screen_width,
                self._screen_height,
                0,
                gl.GL_RED,
                gl.GL_FLOAT,
                None,
            )
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

        gl.glBindTexture(gl.GL_TEXTURE_2D, self._fluid_scene_texture)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGB,
            self._screen_width,
            self._screen_height,
            0,
            gl.GL_RGB,
            gl.GL_UNSIGNED_BYTE,
            None,
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        gl.glBindTexture(gl.GL_TEXTURE_2D, self._fluid_scene_depth_texture)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_DEPTH_COMPONENT32,
            self._screen_width,
            self._screen_height,
            0,
            gl.GL_DEPTH_COMPONENT,
            gl.GL_FLOAT,
            None,
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        for texture in (self._fluid_shadow_depth_texture, self._fluid_shadow_thickness_texture):
            gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D,
                0,
                gl.GL_R32F,
                self._fluid_shadow_size,
                self._fluid_shadow_size,
                0,
                gl.GL_RED,
                gl.GL_FLOAT,
                None,
            )
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_BORDER)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_BORDER)
            border_color = [0.0, 0.0, 0.0, 0.0]
            gl.glTexParameterfv(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_BORDER_COLOR, (gl.GLfloat * 4)(*border_color))

        gl.glBindTexture(gl.GL_TEXTURE_2D, self._fluid_shadow_depth_attachment)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_DEPTH_COMPONENT32,
            self._fluid_shadow_size,
            self._fluid_shadow_size,
            0,
            gl.GL_DEPTH_COMPONENT,
            gl.GL_FLOAT,
            None,
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_BORDER)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_BORDER)
        border_color = [1.0, 1.0, 1.0, 1.0]
        gl.glTexParameterfv(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_BORDER_COLOR, (gl.GLfloat * 4)(*border_color))
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        if self._fluid_fbo is None:
            self._fluid_fbo = gl.GLuint()
            gl.glGenFramebuffers(1, self._fluid_fbo)
        if self._fluid_blur_fbo is None:
            self._fluid_blur_fbo = gl.GLuint()
            gl.glGenFramebuffers(1, self._fluid_blur_fbo)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._fluid_fbo)
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self._fluid_depth_texture, 0
        )
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_TEXTURE_2D, self._frame_depth_texture, 0
        )
        gl.glDrawBuffer(gl.GL_COLOR_ATTACHMENT0)
        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            print("Warning: fluid framebuffer incomplete; fluid rendering will be skipped.")

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._fluid_blur_fbo)
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self._fluid_depth_smooth_texture, 0
        )
        gl.glDrawBuffer(gl.GL_COLOR_ATTACHMENT0)
        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            print("Warning: fluid blur framebuffer incomplete; smoothing will be skipped.")

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        check_gl_error()

    def _setup_frame_mesh(self):
        gl = RendererGL.gl

        # fmt: off
        # set up VBO for the quad that is rendered to the user window with the texture
        self._frame_vertices = np.array([
            # Positions  TexCoords
            -1.0, -1.0,  0.0, 0.0,
             1.0, -1.0,  1.0, 0.0,
             1.0,  1.0,  1.0, 1.0,
            -1.0,  1.0,  0.0, 1.0
        ], dtype=np.float32)
        # fmt: on

        self._frame_indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

        self._frame_vao = gl.GLuint()
        gl.glGenVertexArrays(1, self._frame_vao)
        gl.glBindVertexArray(self._frame_vao)

        self._frame_vbo = gl.GLuint()
        gl.glGenBuffers(1, self._frame_vbo)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._frame_vbo)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER, self._frame_vertices.nbytes, self._frame_vertices.ctypes.data, gl.GL_STATIC_DRAW
        )

        self._frame_ebo = gl.GLuint()
        gl.glGenBuffers(1, self._frame_ebo)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._frame_ebo)
        gl.glBufferData(
            gl.GL_ELEMENT_ARRAY_BUFFER, self._frame_indices.nbytes, self._frame_indices.ctypes.data, gl.GL_STATIC_DRAW
        )

        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 4 * self._frame_vertices.itemsize, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(
            1,
            2,
            gl.GL_FLOAT,
            gl.GL_FALSE,
            4 * self._frame_vertices.itemsize,
            ctypes.c_void_p(2 * self._frame_vertices.itemsize),
        )
        gl.glEnableVertexAttribArray(1)

        check_gl_error()

    def _setup_shadow_buffer(self):
        gl = RendererGL.gl

        self._make_current()

        # create depth texture FBO
        self._shadow_fbo = gl.GLuint()
        gl.glGenFramebuffers(1, self._shadow_fbo)

        self._shadow_texture = gl.GLuint()
        gl.glGenTextures(1, self._shadow_texture)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._shadow_texture)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_DEPTH_COMPONENT,
            self._shadow_width,
            self._shadow_height,
            0,
            gl.GL_DEPTH_COMPONENT,
            gl.GL_FLOAT,
            None,
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_BORDER)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_BORDER)
        border_color = [1.0, 1.0, 1.0, 1.0]
        gl.glTexParameterfv(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_BORDER_COLOR, (gl.GLfloat * 4)(*border_color))

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._shadow_fbo)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_TEXTURE_2D, self._shadow_texture, 0)
        gl.glDrawBuffer(gl.GL_NONE)
        gl.glReadBuffer(gl.GL_NONE)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        check_gl_error()

    def _render_shadow_map(self, objects):
        gl = RendererGL.gl

        self._make_current()

        self._shadow_shader.update(self._light_space_matrix)

        # render from light's point of view (skip objects that don't cast shadows)
        shadow_objects = {k: v for k, v in objects.items() if getattr(v, "cast_shadow", True)}
        with self._shadow_shader:
            self._draw_objects(shadow_objects)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        check_gl_error()

    def _update_light_matrices(self):
        from pyglet.math import Mat4, Vec3

        extents = self.shadow_extents
        light_near = 1.0
        light_far = 1000.0
        camera_pos = np.array(self.camera.pos, dtype=np.float32)
        light_pos = camera_pos + self._sun_direction * extents
        light_proj = Mat4.orthogonal_projection(-extents, extents, -extents, extents, light_near, light_far)
        light_view = Mat4.look_at(Vec3(*light_pos), Vec3(*camera_pos), Vec3(*self._scene_up_vector()))

        self._light_projection_matrix = np.array(light_proj, dtype=np.float32)
        self._light_view_matrix = np.array(light_view, dtype=np.float32)
        self._light_space_matrix = np.array(light_proj @ light_view, dtype=np.float32)

    def _scene_up_vector(self) -> tuple[float, float, float]:
        if self.camera.up_axis == 0:
            return (1.0, 0.0, 0.0)
        if self.camera.up_axis == 1:
            return (0.0, 1.0, 0.0)
        return (0.0, 0.0, 1.0)

    def _update_fluid_light_matrices(self, active_fluids) -> None:
        valid_bounds = [fluid for fluid in active_fluids if getattr(fluid, "bounds_valid", False)]
        if not valid_bounds:
            return

        from pyglet.math import Mat4, Vec3

        lower = np.min(np.stack([fluid.bounds_lower for fluid in valid_bounds]), axis=0).astype(np.float32)
        upper = np.max(np.stack([fluid.bounds_upper for fluid in valid_bounds]), axis=0).astype(np.float32)
        up_axis = int(self.camera.up_axis)
        sun_direction = np.asarray(self._sun_direction, dtype=np.float32)
        sun_up = float(sun_direction[up_axis])
        receiver_up = min(float(lower[up_axis]), 0.0)
        corners = np.array(
            [(x, y, z) for x in (lower[0], upper[0]) for y in (lower[1], upper[1]) for z in (lower[2], upper[2])],
            dtype=np.float32,
        )
        projected_corners = []
        if sun_up > 1.0e-4:
            for corner in corners:
                travel = max((float(corner[up_axis]) - receiver_up) / sun_up, 0.0)
                projected_corners.append(corner - sun_direction * travel)
        if projected_corners:
            shadow_points = np.concatenate((corners, np.asarray(projected_corners, dtype=np.float32)), axis=0)
        else:
            shadow_points = corners
        shadow_lower = np.min(shadow_points, axis=0)
        shadow_upper = np.max(shadow_points, axis=0)
        center = (shadow_lower + shadow_upper) * 0.5
        half_diagonal = float(np.linalg.norm((shadow_upper - shadow_lower) * 0.5))
        ortho_extent = max(half_diagonal * 1.20, self.shadow_extents * 0.25, 1.0)
        light_distance = max(ortho_extent * 3.0, 8.0)
        light_near = 0.05
        light_far = light_distance + max(ortho_extent * 4.0, 16.0)
        light_pos = center + sun_direction * light_distance

        light_proj = Mat4.orthogonal_projection(
            -ortho_extent,
            ortho_extent,
            -ortho_extent,
            ortho_extent,
            light_near,
            light_far,
        )
        light_view = Mat4.look_at(Vec3(*light_pos), Vec3(*center), Vec3(*self._scene_up_vector()))

        self._light_projection_matrix = np.array(light_proj, dtype=np.float32)
        self._light_view_matrix = np.array(light_view, dtype=np.float32)
        self._light_space_matrix = np.array(light_proj @ light_view, dtype=np.float32)

    def _draw_frame_quad(self):
        gl = RendererGL.gl
        gl.glBindVertexArray(self._frame_vao)
        gl.glDrawElements(gl.GL_TRIANGLES, len(self._frame_indices), gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)

    def _blur_fluid_scalar(
        self,
        source_texture,
        target_texture,
        direction: tuple[float, float],
        filter_radius: float,
        depth_edge_falloff: float,
        max_radial_samples: int,
        guide_texture=None,
        max_depth_delta: float = 0.25,
    ) -> None:
        gl = RendererGL.gl
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._fluid_blur_fbo)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, target_texture, 0)
        gl.glDrawBuffer(gl.GL_COLOR_ATTACHMENT0)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_BLEND)
        gl.glViewport(0, 0, self._screen_width, self._screen_height)

        with self._fluid_blur_shader:
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, source_texture)
            gl.glActiveTexture(gl.GL_TEXTURE1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, guide_texture if guide_texture is not None else source_texture)
            self._fluid_blur_shader.update(
                texture_unit=0,
                guide_unit=1,
                texel_size=(1.0 / max(self._screen_width, 1), 1.0 / max(self._screen_height, 1)),
                direction=direction,
                filter_radius=filter_radius,
                max_depth_delta=max_depth_delta,
                depth_edge_falloff=depth_edge_falloff,
                max_radial_samples=max_radial_samples,
                use_guide_texture=guide_texture is not None,
            )
            self._draw_frame_quad()

    def _blur_fluid_depth(
        self,
        source_texture,
        target_texture,
        direction: tuple[float, float],
        filter_radius: float,
        depth_edge_falloff: float,
        max_radial_samples: int,
    ) -> None:
        self._blur_fluid_scalar(
            source_texture,
            target_texture,
            direction,
            filter_radius,
            depth_edge_falloff,
            max_radial_samples,
            guide_texture=None,
            max_depth_delta=0.55,
        )

    def _copy_frame_to_fluid_scene(self) -> None:
        gl = RendererGL.gl
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self._frame_fbo)
        gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, self._fluid_blur_fbo)
        gl.glFramebufferTexture2D(
            gl.GL_DRAW_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self._fluid_scene_texture, 0
        )
        gl.glDrawBuffer(gl.GL_COLOR_ATTACHMENT0)
        gl.glBlitFramebuffer(
            0,
            0,
            self._screen_width,
            self._screen_height,
            0,
            0,
            self._screen_width,
            self._screen_height,
            gl.GL_COLOR_BUFFER_BIT,
            gl.GL_NEAREST,
        )

    @staticmethod
    def _fluid_smoothing_budget(requested_iterations: int) -> tuple[int, int, float]:
        requested_iterations = max(int(requested_iterations), 0)
        if requested_iterations == 0:
            return 0, 0, 1.0

        depth_iterations = min(requested_iterations, 6)
        thickness_iterations = max(1, min(depth_iterations // 2, 3))
        radius_scale = float(np.sqrt(requested_iterations / max(depth_iterations, 1)))
        return depth_iterations, thickness_iterations, radius_scale

    def _render_fluid_shadow_maps(self, active_fluids, material_fluid) -> None:
        gl = RendererGL.gl
        light_projection = np.asarray(self._light_projection_matrix, dtype=np.float32).reshape(4, 4).transpose()
        light_inv_projection = np.linalg.inv(light_projection).transpose()
        shadow_texel_size = (
            1.0 / max(self._fluid_shadow_size, 1),
            1.0 / max(self._fluid_shadow_size, 1),
        )
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._fluid_fbo)
        gl.glViewport(0, 0, self._fluid_shadow_size, self._fluid_shadow_size)
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_TEXTURE_2D, self._fluid_shadow_depth_attachment, 0
        )

        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self._fluid_shadow_depth_texture, 0
        )
        gl.glDrawBuffer(gl.GL_COLOR_ATTACHMENT0)
        gl.glClearBufferfv(gl.GL_COLOR, 0, (gl.GLfloat * 4)(0.0, 0.0, 0.0, 0.0))
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glDepthMask(True)
        gl.glDisable(gl.GL_BLEND)
        if hasattr(gl, "GL_PROGRAM_POINT_SIZE"):
            gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)

        with self._fluid_particle_shader:
            self._fluid_particle_shader.update(
                self._light_view_matrix,
                self._light_projection_matrix,
                light_inv_projection,
                shadow_texel_size,
                output_thickness=False,
                thickness_scale=1.0,
            )
            for fluid in active_fluids:
                fluid.render()

        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self._fluid_shadow_thickness_texture, 0
        )
        gl.glDrawBuffer(gl.GL_COLOR_ATTACHMENT0)
        gl.glClearBufferfv(gl.GL_COLOR, 0, (gl.GLfloat * 4)(0.0, 0.0, 0.0, 0.0))
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDepthMask(False)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE)
        with self._fluid_particle_shader:
            self._fluid_particle_shader.update(
                self._light_view_matrix,
                self._light_projection_matrix,
                light_inv_projection,
                shadow_texel_size,
                output_thickness=True,
                thickness_scale=material_fluid.thickness_scale,
            )
            for fluid in active_fluids:
                fluid.render()
        gl.glDisable(gl.GL_BLEND)
        gl.glDepthMask(True)
        gl.glViewport(0, 0, self._screen_width, self._screen_height)

    def _apply_fluid_shadow_to_scene(
        self,
        inv_projection: np.ndarray,
        inv_view: np.ndarray,
        material_fluid,
        fluid_bounds_lower: np.ndarray,
        fluid_bounds_upper: np.ndarray,
        scene_depth_texture=None,
        copy_to_frame: bool = True,
    ) -> None:
        gl = RendererGL.gl
        depth_texture = scene_depth_texture if scene_depth_texture is not None else self._frame_depth_texture

        self._copy_frame_to_fluid_scene()
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._fluid_blur_fbo)
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self._fluid_scene_texture, 0
        )
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_TEXTURE_2D, 0, 0)
        gl.glDrawBuffer(gl.GL_COLOR_ATTACHMENT0)
        gl.glViewport(0, 0, self._screen_width, self._screen_height)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_BLEND)

        with self._fluid_shadow_shader:
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self._frame_texture)
            gl.glActiveTexture(gl.GL_TEXTURE1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, depth_texture)
            gl.glActiveTexture(gl.GL_TEXTURE2)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self._fluid_shadow_depth_texture)
            gl.glActiveTexture(gl.GL_TEXTURE3)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self._fluid_shadow_thickness_texture)
            self._fluid_shadow_shader.update(
                scene_unit=0,
                scene_depth_unit=1,
                fluid_shadow_depth_unit=2,
                fluid_shadow_thickness_unit=3,
                inv_projection=inv_projection,
                inv_view=inv_view,
                light_projection=self._light_projection_matrix,
                light_view=self._light_view_matrix,
                light_space_matrix=self._light_space_matrix,
                sun_direction_world=(
                    float(self._sun_direction[0]),
                    float(self._sun_direction[1]),
                    float(self._sun_direction[2]),
                ),
                fluid_bounds_lower=fluid_bounds_lower,
                fluid_bounds_upper=fluid_bounds_upper,
                water_color=material_fluid.color,
                absorption_strength=material_fluid.absorption_strength,
                caustic_scale=material_fluid.caustic_scale,
                floor_caustic_strength=material_fluid.floor_caustic_strength,
                surface_shadow_strength=material_fluid.surface_shadow_strength,
                up_axis=self.camera.up_axis,
            )
            self._draw_frame_quad()

        if not copy_to_frame:
            return

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._frame_fbo)
        gl.glDrawBuffer(gl.GL_COLOR_ATTACHMENT0)
        with self._frame_shader:
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self._fluid_scene_texture)
            self._frame_shader.update(0)
            self._draw_frame_quad()

    def _render_fluids(self, fluids, fluid_diffuse=None):
        gl = RendererGL.gl
        active_fluids = [fluid for fluid in fluids.values() if not fluid.hidden and fluid.active_particles > 0]
        if not active_fluids:
            return False

        material_fluid = active_fluids[0]
        valid_bounds = [fluid for fluid in active_fluids if fluid.bounds_valid]
        if valid_bounds:
            fluid_bounds_lower = np.min([fluid.bounds_lower for fluid in valid_bounds], axis=0).astype(np.float32)
            fluid_bounds_upper = np.max([fluid.bounds_upper for fluid in valid_bounds], axis=0).astype(np.float32)
        else:
            fluid_bounds_lower = np.zeros(3, dtype=np.float32)
            fluid_bounds_upper = np.zeros(3, dtype=np.float32)

        view = np.asarray(self._view_matrix, dtype=np.float32).reshape(4, 4).transpose()
        projection = np.asarray(self._projection_matrix, dtype=np.float32).reshape(4, 4).transpose()
        inv_projection = np.linalg.inv(projection).transpose()
        inv_view = np.linalg.inv(view).transpose()
        inv_view_rotation = np.linalg.inv(view[:3, :3]).transpose()
        screen_texel_size = (
            1.0 / max(self._screen_width, 1),
            1.0 / max(self._screen_height, 1),
        )

        # Nearest fluid surface depth.
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._fluid_fbo)
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self._fluid_depth_texture, 0
        )
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_TEXTURE_2D, self._fluid_scene_depth_texture, 0
        )
        gl.glDrawBuffer(gl.GL_COLOR_ATTACHMENT0)
        gl.glViewport(0, 0, self._screen_width, self._screen_height)
        gl.glClearBufferfv(gl.GL_COLOR, 0, (gl.GLfloat * 4)(0.0, 0.0, 0.0, 0.0))
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glDepthMask(True)
        gl.glDisable(gl.GL_BLEND)
        if hasattr(gl, "GL_PROGRAM_POINT_SIZE"):
            gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)

        with self._fluid_particle_shader:
            self._fluid_particle_shader.update(
                self._view_matrix,
                self._projection_matrix,
                inv_projection,
                screen_texel_size,
                output_thickness=False,
                thickness_scale=1.0,
            )
            for fluid in active_fluids:
                fluid.render()

        # Optical thickness accumulated through all visible fluid particles.
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self._fluid_thickness_texture, 0
        )
        gl.glClearBufferfv(gl.GL_COLOR, 0, (gl.GLfloat * 4)(0.0, 0.0, 0.0, 0.0))
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDepthMask(False)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE)
        with self._fluid_particle_shader:
            self._fluid_particle_shader.update(
                self._view_matrix,
                self._projection_matrix,
                inv_projection,
                screen_texel_size,
                output_thickness=True,
                thickness_scale=material_fluid.thickness_scale,
            )
            for fluid in active_fluids:
                fluid.render()
        gl.glDisable(gl.GL_BLEND)
        gl.glDepthMask(True)

        depth_iterations, thickness_iterations, smoothing_radius_scale = self._fluid_smoothing_budget(
            material_fluid.smoothing_iterations
        )
        depth_filter_radius = material_fluid.smoothing_radius * smoothing_radius_scale

        depth_texture = self._fluid_depth_texture
        scratch_texture = self._fluid_depth_smooth_texture
        for _ in range(depth_iterations):
            self._blur_fluid_depth(
                depth_texture,
                scratch_texture,
                (1.0, 0.0),
                depth_filter_radius,
                material_fluid.smoothing_depth_edge_falloff,
                material_fluid.smoothing_max_samples,
            )
            depth_texture, scratch_texture = scratch_texture, depth_texture
            self._blur_fluid_depth(
                depth_texture,
                scratch_texture,
                (0.0, 1.0),
                depth_filter_radius,
                material_fluid.smoothing_depth_edge_falloff,
                material_fluid.smoothing_max_samples,
            )
            depth_texture, scratch_texture = scratch_texture, depth_texture

        thickness_texture = self._fluid_thickness_texture
        thickness_scratch_texture = self._fluid_thickness_smooth_texture
        thickness_radius = max(material_fluid.smoothing_radius * smoothing_radius_scale * 0.72, 0.2)
        for _ in range(thickness_iterations):
            self._blur_fluid_scalar(
                thickness_texture,
                thickness_scratch_texture,
                (1.0, 0.0),
                thickness_radius,
                material_fluid.smoothing_depth_edge_falloff,
                material_fluid.smoothing_max_samples,
                guide_texture=depth_texture,
                max_depth_delta=0.35,
            )
            thickness_texture, thickness_scratch_texture = thickness_scratch_texture, thickness_texture
            self._blur_fluid_scalar(
                thickness_texture,
                thickness_scratch_texture,
                (0.0, 1.0),
                thickness_radius,
                material_fluid.smoothing_depth_edge_falloff,
                material_fluid.smoothing_max_samples,
                guide_texture=depth_texture,
                max_depth_delta=0.35,
            )
            thickness_texture, thickness_scratch_texture = thickness_scratch_texture, thickness_texture

        fluid_shadows_enabled = self.draw_shadows and (
            material_fluid.surface_shadow_strength > 0.0 or material_fluid.floor_caustic_strength > 0.0
        )
        scene_light_projection_matrix = np.array(self._light_projection_matrix, copy=True)
        scene_light_view_matrix = np.array(self._light_view_matrix, copy=True)
        scene_light_space_matrix = np.array(self._light_space_matrix, copy=True)
        if fluid_shadows_enabled:
            self._update_fluid_light_matrices(active_fluids)
            self._render_fluid_shadow_maps(active_fluids, material_fluid)

        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self._frame_fbo)
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, self._fluid_blur_fbo)
        gl.glFramebufferTexture2D(
            gl.GL_DRAW_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_TEXTURE_2D, self._fluid_scene_depth_texture, 0
        )
        gl.glBlitFramebuffer(
            0,
            0,
            self._screen_width,
            self._screen_height,
            0,
            0,
            self._screen_width,
            self._screen_height,
            gl.GL_DEPTH_BUFFER_BIT,
            gl.GL_NEAREST,
        )

        if fluid_shadows_enabled:
            self._apply_fluid_shadow_to_scene(
                inv_projection,
                inv_view,
                material_fluid,
                fluid_bounds_lower,
                fluid_bounds_upper,
                scene_depth_texture=self._fluid_scene_depth_texture,
                copy_to_frame=True,
            )
            self._light_projection_matrix = scene_light_projection_matrix
            self._light_view_matrix = scene_light_view_matrix
            self._light_space_matrix = scene_light_space_matrix
        else:
            self._copy_frame_to_fluid_scene()

        if fluid_diffuse:
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._fluid_blur_fbo)
            gl.glFramebufferTexture2D(
                gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self._fluid_scene_texture, 0
            )
            self._render_fluid_diffuse(
                fluid_diffuse,
                fluid_depth_texture=depth_texture,
                depth_mode=1,
                target_fbo=self._fluid_blur_fbo,
            )

        gl.glBindTexture(gl.GL_TEXTURE_2D, self._fluid_scene_texture)
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        # Composite over the frame color.
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._frame_fbo)
        gl.glDrawBuffer(gl.GL_COLOR_ATTACHMENT0)
        gl.glViewport(0, 0, self._screen_width, self._screen_height)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_ALWAYS)
        gl.glDepthMask(True)
        gl.glDisable(gl.GL_BLEND)

        sun_view = view[:3, :3] @ np.asarray(self._sun_direction, dtype=np.float32)
        norm = np.linalg.norm(sun_view)
        if norm > 0.0:
            sun_view = sun_view / norm

        with self._fluid_composite_shader:
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self._fluid_scene_texture)
            gl.glActiveTexture(gl.GL_TEXTURE1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, depth_texture)
            gl.glActiveTexture(gl.GL_TEXTURE2)
            gl.glBindTexture(gl.GL_TEXTURE_2D, thickness_texture)
            gl.glActiveTexture(gl.GL_TEXTURE3)
            if self._env_texture is not None:
                gl.glBindTexture(gl.GL_TEXTURE_2D, self._env_texture)
            else:
                gl.glBindTexture(gl.GL_TEXTURE_2D, RendererGL.get_fallback_texture())
            gl.glActiveTexture(gl.GL_TEXTURE4)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self._fluid_scene_depth_texture)
            gl.glActiveTexture(gl.GL_TEXTURE5)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self._shadow_texture)
            self._fluid_composite_shader.update(
                scene_unit=0,
                depth_unit=1,
                thickness_unit=2,
                env_unit=3,
                scene_depth_unit=4,
                shadow_unit=5,
                projection_matrix=self._projection_matrix,
                inv_projection=inv_projection,
                inv_view=inv_view,
                light_space_matrix=self._light_space_matrix,
                inv_view_rotation=inv_view_rotation,
                fluid_bounds_lower=fluid_bounds_lower,
                fluid_bounds_upper=fluid_bounds_upper,
                texel_size=(1.0 / max(self._screen_width, 1), 1.0 / max(self._screen_height, 1)),
                water_color=material_fluid.color,
                water_deep_color=material_fluid.deep_color,
                color_gradient_strength=material_fluid.color_gradient_strength,
                opacity=material_fluid.opacity,
                reflection_strength=material_fluid.reflection_strength,
                refraction_strength=material_fluid.refraction_strength,
                env_map_strength=material_fluid.env_map_strength,
                env_reflection_lod=material_fluid.env_reflection_lod,
                env_color_preserve=material_fluid.env_color_preserve,
                absorption_strength=material_fluid.absorption_strength,
                depth_visualization_strength=material_fluid.depth_visualization_strength,
                env_intensity=self._env_intensity,
                env_map_available=self._env_texture is not None,
                sky_reflection_color=self.sky_upper,
                ground_reflection_color=self.ambient_ground,
                caustic_strength=material_fluid.caustic_strength,
                caustic_scale=material_fluid.caustic_scale,
                floor_caustic_strength=0.0,
                surface_shadow_strength=0.0,
                shadow_radius=self.shadow_radius,
                foam_strength=material_fluid.foam_strength,
                foam_scale=material_fluid.foam_scale,
                up_axis=self.camera.up_axis,
                sun_direction_view=(float(sun_view[0]), float(sun_view[1]), float(sun_view[2])),
            )
            self._draw_frame_quad()

        gl.glDepthFunc(gl.GL_LESS)

        if fluid_diffuse:
            self._render_fluid_diffuse(fluid_diffuse, fluid_depth_texture=depth_texture, depth_mode=2)

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LESS)
        check_gl_error()
        return True

    def _render_fluid_diffuse(self, diffuse_batches, fluid_depth_texture=None, depth_mode: int = 0, target_fbo=None):
        active_batches = [
            batch for batch in diffuse_batches.values() if not batch.hidden and batch.active_particles > 0
        ]
        if not active_batches:
            return

        gl = RendererGL.gl
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._frame_fbo if target_fbo is None else target_fbo)
        gl.glDrawBuffer(gl.GL_COLOR_ATTACHMENT0)
        gl.glViewport(0, 0, self._screen_width, self._screen_height)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDepthMask(False)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE_MINUS_SRC_ALPHA)

        with self._fluid_diffuse_shader:
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self._frame_depth_texture)
            gl.glActiveTexture(gl.GL_TEXTURE1)
            if fluid_depth_texture is not None:
                gl.glBindTexture(gl.GL_TEXTURE_2D, fluid_depth_texture)
            else:
                gl.glBindTexture(gl.GL_TEXTURE_2D, RendererGL.get_fallback_texture())
            gl.glActiveTexture(gl.GL_TEXTURE2)
            gl.glBindTexture(
                gl.GL_TEXTURE_2D,
                self._shadow_texture if self._shadow_texture is not None else RendererGL.get_fallback_texture(),
            )
            sun = np.asarray(
                self._sun_direction if self._sun_direction is not None else (0.2, -0.3, 0.8),
                dtype=np.float32,
            )
            view_matrix = np.asarray(self._view_matrix, dtype=np.float32).reshape(4, 4).transpose()
            projection_matrix = np.asarray(self._projection_matrix, dtype=np.float32).reshape(4, 4).transpose()
            inv_projection_matrix = np.linalg.inv(projection_matrix).transpose()
            sun_view = view_matrix[:3, :3] @ sun
            up_world = np.zeros(3, dtype=np.float32)
            up_world[int(self.camera.up_axis)] = 1.0
            up_view = view_matrix[:3, :3] @ up_world
            for batch in active_batches:
                batch.sort_for_view(self._view_matrix)
                self._fluid_diffuse_shader.update(
                    view_matrix=self._view_matrix,
                    projection_matrix=self._projection_matrix,
                    inv_projection_matrix=inv_projection_matrix,
                    radius=batch.radius,
                    motion_blur_scale=batch.motion_blur_scale,
                    diffuse_expansion=batch.expansion,
                    up_axis_view=(float(up_view[0]), float(up_view[1]), float(up_view[2])),
                    diffuse_color=batch.color,
                    alpha=batch.alpha,
                    scene_depth_unit=0,
                    fluid_depth_unit=1,
                    shadow_unit=2,
                    light_space_matrix=self._light_space_matrix,
                    sun_direction_view=(float(sun_view[0]), float(sun_view[1]), float(sun_view[2])),
                    texel_size=(1.0 / max(self._screen_width, 1), 1.0 / max(self._screen_height, 1)),
                    depth_mode=depth_mode,
                    inscatter=batch.inscatter,
                    outscatter=batch.outscatter,
                    shadow_strength=batch.shadow_strength,
                    shadow_enabled=self.draw_shadows and self._shadow_texture is not None,
                )
                batch.render()

        gl.glDisable(gl.GL_BLEND)
        gl.glDepthMask(True)
        gl.glDepthFunc(gl.GL_LESS)
        check_gl_error()

    def _render_scene(self, objects):
        gl = RendererGL.gl

        if self.draw_sky:
            self._draw_sky()

        if self.draw_wireframe:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)

        self._shape_shader.update(
            view_matrix=self._view_matrix,
            projection_matrix=self._projection_matrix,
            view_pos=self.camera.pos,
            fog_color=self.sky_lower,
            up_axis=self.camera.up_axis,
            sun_direction=self._sun_direction,
            enable_shadows=self.draw_shadows,
            shadow_texture=self._shadow_texture,
            light_space_matrix=self._light_space_matrix,
            light_color=self._light_color,
            sky_color=self.ambient_sky,
            ground_color=self.ambient_ground,
            env_texture=self._env_texture,
            env_intensity=self._env_intensity,
            shadow_radius=self.shadow_radius,
            diffuse_scale=self.diffuse_scale,
            specular_scale=self.specular_scale,
            spotlight_enabled=self.spotlight_enabled,
            shadow_extents=self.shadow_extents,
            exposure=self.exposure,
        )

        with self._shape_shader:
            self._draw_objects(objects)

        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

        # Edge overlay: redraw the same geometry as lines with polygon offset
        # to avoid z-fighting (per @mmacklin review on #2300).
        if self.draw_edges:
            # Skip objects that opted out of the edge overlay (e.g. ground
            # planes) via the per-object draw_edge flag. Mirrors the cast_shadow
            # filter in _render_shadow_map and keeps the decision off the checker
            # material bit (see #2808 review).
            edge_objects = {k: v for k, v in objects.items() if getattr(v, "draw_edge", True)}
            self._edge_shader.update(
                view_matrix=self._view_matrix,
                projection_matrix=self._projection_matrix,
                edge_color=self._edge_color,
                light_space_matrix=self._light_space_matrix,
            )
            gl.glEnable(gl.GL_POLYGON_OFFSET_LINE)
            gl.glPolygonOffset(-1.0, -1.0)
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
            with self._edge_shader:
                self._draw_objects(edge_objects)
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
            gl.glDisable(gl.GL_POLYGON_OFFSET_LINE)

        check_gl_error()

    def _render_lines(self, lines):
        """Render all line objects using the geometry-shader wide-line pipeline."""
        gl = RendererGL.gl
        inv_asp = float(self._screen_height) / float(max(self._screen_width, 1))
        clip_width = max(0.0, self.line_width) * 2.0 / max(self._screen_height, 1)

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        identity = np.eye(4, dtype=np.float32)
        with self._wireframe_shader:
            self._wireframe_shader.update_frame(
                self._view_matrix,
                self._projection_matrix,
                inv_asp,
                line_width=clip_width,
                alpha=1.0,
            )
            self._wireframe_shader.set_world(identity)
            for line_obj in lines.values():
                if hasattr(line_obj, "render"):
                    line_obj.render()

        gl.glDisable(gl.GL_BLEND)
        check_gl_error()

    def _render_arrows(self, arrows):
        """Render arrow batches (wide line + arrowhead triangle per segment)."""
        gl = RendererGL.gl
        inv_asp = float(self._screen_height) / float(max(self._screen_width, 1))
        scale = max(0.0, self.arrow_scale)
        clip_width = (2.0 * scale) * 2.0 / max(self._screen_height, 1)
        clip_arrow = (8.0 * scale) * 2.0 / max(self._screen_height, 1)

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        identity = np.eye(4, dtype=np.float32)
        with self._arrow_shader:
            self._arrow_shader.update_frame(
                self._view_matrix,
                self._projection_matrix,
                inv_asp,
                line_width=clip_width,
                arrow_size=clip_arrow,
                alpha=1.0,
            )
            self._arrow_shader.set_world(identity)
            for arrow_obj in arrows.values():
                if hasattr(arrow_obj, "render"):
                    arrow_obj.render()

        gl.glDisable(gl.GL_BLEND)
        check_gl_error()

    def _render_wireframe_shapes(self, wireframe_shapes):
        """Render wireframe shapes using the geometry-shader line expansion."""
        gl = RendererGL.gl
        inv_asp = float(self._screen_height) / float(max(self._screen_width, 1))
        clip_width = self.wireframe_line_width * 2.0 / max(self._screen_height, 1)

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        with self._wireframe_shader:
            self._wireframe_shader.update_frame(
                self._view_matrix, self._projection_matrix, inv_asp, line_width=clip_width
            )
            for shape in wireframe_shapes.values():
                if not shape.hidden and shape.num_vertices > 0:
                    self._wireframe_shader.set_world(shape.world_matrix)
                    shape.render()

        gl.glDisable(gl.GL_BLEND)
        check_gl_error()

    def _draw_objects(self, objects):
        for o in objects.values():
            if hasattr(o, "render"):
                o.render()

        check_gl_error()

    def _draw_sky(self):
        gl = RendererGL.gl

        self._make_current()

        self._sky_shader.update(
            view_matrix=self._view_matrix,
            projection_matrix=self._projection_matrix,
            camera_pos=self.camera.pos,
            camera_far=self.camera.far,
            sky_upper=self.sky_upper,
            sky_lower=self.sky_lower,
            sun_direction=self._sun_direction,
            up_axis=self.camera.up_axis,
        )

        gl.glBindVertexArray(self._sky_vao)
        gl.glDrawElements(gl.GL_TRIANGLES, self._sky_tri_count, gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)

        check_gl_error()

    def set_environment_map(self, path: str, intensity: float = 1.0) -> None:
        gl = RendererGL.gl
        from ...utils.texture import load_texture_from_file  # noqa: PLC0415

        image = load_texture_from_file(path)
        if image is None:
            return
        if self._env_texture is not None:
            try:
                gl.glDeleteTextures(1, self._env_texture)
            except Exception:
                pass
            self._env_texture = None
        self._env_texture = _upload_texture_from_file(gl, image)
        self._env_texture_obj = None
        self._env_intensity = float(intensity)

    def _make_current(self):
        try:
            self.window.switch_to()
        except AttributeError:
            # The window could be in the process of being closed, in which case
            # its corresponding context might have been destroyed and set to `None`.
            pass

    def _set_icon(self):
        import pyglet

        def load_icon(filename):
            filename = os.path.join(os.path.dirname(__file__), filename)

            if not os.path.exists(filename):
                raise FileNotFoundError(
                    f"Error: Icon file '{filename}' not found. Please run the 'generate_icons.py' script first."
                )

            with open(filename, "rb") as f:
                icon_bytes = f.read()

            icon_stream = io.BytesIO(icon_bytes)
            icon = pyglet.image.load(filename=filename, file=icon_stream)

            return icon

        icons = [load_icon("icon_16.png"), load_icon("icon_32.png"), load_icon("icon_64.png")]

        # 5. Create the window and set the icon
        self.window.set_icon(*icons)
