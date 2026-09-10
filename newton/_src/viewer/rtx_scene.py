# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render-product consumption, independent of Newton's USD scene generation."""

from __future__ import annotations

from typing import Any

import numpy as np


class _RtxScene:
    def __init__(self, stage: Any, renderer: Any, product: str, color_output: str):
        if not product.startswith("/") or not color_output.startswith("/"):
            raise ValueError("Render product and color output must be absolute prim paths")
        self.stage = stage
        self.renderer = renderer
        self.product = product
        self.color_output = color_output
        self.products = None
        self.ordinal = -1
        self.closed = False
        self.window = None

    def render(self, ordinal: int, delta_time: float):
        if self.closed:
            raise RuntimeError("The viewer is closed")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0 or ordinal < self.ordinal:
            raise ValueError("Render ordinal must be nonnegative and must not decrease")
        if not np.isfinite(delta_time) or delta_time < 0:
            raise ValueError("delta_time must be finite and nonnegative")
        # step() completes before the next state publication. No renderer-owned
        # scene API, implicit stage commit, or shader extraction is involved.
        self.products = self.renderer.step(render_products={self.product}, delta_time=delta_time, ordinal=ordinal)
        self.ordinal = ordinal
        return self.products

    def pixels(self) -> np.ndarray:
        if self.closed or self.products is None:
            raise RuntimeError("A completed render is required")
        import ovrtx

        for frame in self.products[self.product].frames:
            if self.color_output in frame.render_vars:
                with frame.render_vars[self.color_output].map(device=ovrtx.Device.CPU) as mapping:
                    return np.array(np.from_dlpack(mapping), copy=True)
        raise RuntimeError(f"Render product has no color output {self.color_output!r}")

    def display(self, *, vsync: bool) -> None:
        # A small fixed-camera preview. The prototype deliberately keeps camera
        # and scene editing in the caller. Headless rendering avoids this copy.
        import pyglet

        pixels = self.pixels()
        height, width = pixels.shape[:2]
        if pixels.dtype != np.uint8 or pixels.shape != (height, width, 4):
            raise ValueError("Window preview requires an RGBA uint8 color output")
        if self.window is None:
            self.window = pyglet.window.Window(width=width, height=height, caption="Newton RTX", vsync=vsync)
        self.window.switch_to()
        self.window.dispatch_events()
        if self.window.has_exit:
            return
        self.window.clear()
        pyglet.image.ImageData(width, height, "RGBA", pixels.tobytes(), pitch=-width * 4).blit(0, 0)
        self.window.flip()

    def close(self) -> None:
        self.products = None
        if self.window is not None:
            self.window.close()
            self.window = None
        self.closed = True
        # Renderer and stage belong to the caller; do not detach or destroy.
