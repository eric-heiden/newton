# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free image helpers for MCP observations: PNG codec, tiling, labels."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np

# 5x7 glyphs, one string of five characters per row ("#" = ink).
_GLYPHS = {
    " ": ["     "] * 7,
    "0": [" ### ", "#   #", "#  ##", "# # #", "##  #", "#   #", " ### "],
    "1": ["  #  ", " ##  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### "],
    "2": [" ### ", "#   #", "    #", "   # ", "  #  ", " #   ", "#####"],
    "3": ["#####", "   # ", "  #  ", "   # ", "    #", "#   #", " ### "],
    "4": ["   # ", "  ## ", " # # ", "#  # ", "#####", "   # ", "   # "],
    "5": ["#####", "#    ", "#### ", "    #", "    #", "#   #", " ### "],
    "6": ["  ## ", " #   ", "#    ", "#### ", "#   #", "#   #", " ### "],
    "7": ["#####", "    #", "   # ", "  #  ", " #   ", " #   ", " #   "],
    "8": [" ### ", "#   #", "#   #", " ### ", "#   #", "#   #", " ### "],
    "9": [" ### ", "#   #", "#   #", " ####", "    #", "   # ", " ##  "],
    "A": [" ### ", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"],
    "B": ["#### ", "#   #", "#   #", "#### ", "#   #", "#   #", "#### "],
    "C": [" ### ", "#   #", "#    ", "#    ", "#    ", "#   #", " ### "],
    "D": ["#### ", "#   #", "#   #", "#   #", "#   #", "#   #", "#### "],
    "E": ["#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#####"],
    "F": ["#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#    "],
    "G": [" ### ", "#   #", "#    ", "# ###", "#   #", "#   #", " ####"],
    "H": ["#   #", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"],
    "I": [" ### ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### "],
    "J": ["  ###", "   # ", "   # ", "   # ", "   # ", "#  # ", " ##  "],
    "K": ["#   #", "#  # ", "# #  ", "##   ", "# #  ", "#  # ", "#   #"],
    "L": ["#    ", "#    ", "#    ", "#    ", "#    ", "#    ", "#####"],
    "M": ["#   #", "## ##", "# # #", "# # #", "#   #", "#   #", "#   #"],
    "N": ["#   #", "#   #", "##  #", "# # #", "#  ##", "#   #", "#   #"],
    "O": [" ### ", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "],
    "P": ["#### ", "#   #", "#   #", "#### ", "#    ", "#    ", "#    "],
    "Q": [" ### ", "#   #", "#   #", "#   #", "# # #", "#  # ", " ## #"],
    "R": ["#### ", "#   #", "#   #", "#### ", "# #  ", "#  # ", "#   #"],
    "S": [" ####", "#    ", "#    ", " ### ", "    #", "    #", "#### "],
    "T": ["#####", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  "],
    "U": ["#   #", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "],
    "V": ["#   #", "#   #", "#   #", "#   #", "#   #", " # # ", "  #  "],
    "W": ["#   #", "#   #", "#   #", "# # #", "# # #", "# # #", " # # "],
    "X": ["#   #", "#   #", " # # ", "  #  ", " # # ", "#   #", "#   #"],
    "Y": ["#   #", "#   #", " # # ", "  #  ", "  #  ", "  #  ", "  #  "],
    "Z": ["#####", "    #", "   # ", "  #  ", " #   ", "#    ", "#####"],
    ".": ["     ", "     ", "     ", "     ", "     ", " ##  ", " ##  "],
    ",": ["     ", "     ", "     ", "     ", " ##  ", "  #  ", " #   "],
    ":": ["     ", " ##  ", " ##  ", "     ", " ##  ", " ##  ", "     "],
    "-": ["     ", "     ", "     ", "#####", "     ", "     ", "     "],
    "+": ["     ", "  #  ", "  #  ", "#####", "  #  ", "  #  ", "     "],
    "=": ["     ", "     ", "#####", "     ", "#####", "     ", "     "],
    "_": ["     ", "     ", "     ", "     ", "     ", "     ", "#####"],
    "/": ["    #", "    #", "   # ", "  #  ", " #   ", "#    ", "#    "],
    "(": ["   # ", "  #  ", " #   ", " #   ", " #   ", "  #  ", "   # "],
    ")": [" #   ", "  #  ", "   # ", "   # ", "   # ", "  #  ", " #   "],
    "[": [" ### ", " #   ", " #   ", " #   ", " #   ", " #   ", " ### "],
    "]": [" ### ", "   # ", "   # ", "   # ", "   # ", "   # ", " ### "],
    "%": ["##   ", "##  #", "   # ", "  #  ", " #   ", "#  ##", "   ##"],
    "#": [" # # ", " # # ", "#####", " # # ", "#####", " # # ", " # # "],
    "|": ["  #  "] * 7,
    "?": [" ### ", "#   #", "    #", "   # ", "  #  ", "     ", "  #  "],
}
_FONT = {char: np.array([[cell == "#" for cell in row] for row in rows], dtype=bool) for char, rows in _GLYPHS.items()}


def encode_png(rgb: np.ndarray) -> bytes:
    """Encode a top-left HxWx3 uint8 image without an optional imaging dependency."""
    height, width, _ = rgb.shape

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    scanlines = np.zeros((height, width * 3 + 1), dtype=np.uint8)
    scanlines[:, 1:] = np.ascontiguousarray(rgb, dtype=np.uint8).reshape(height, width * 3)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(scanlines.tobytes(), 6))
        + chunk(b"IEND", b"")
    )


def decode_png(data: bytes) -> np.ndarray:
    """Decode a non-interlaced 8-bit gray/RGB/RGBA (optionally alpha-gray) PNG to HxWx3 uint8."""
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Not a PNG file")
    position, idat, header = 8, [], None
    while position < len(data):
        (length,) = struct.unpack(">I", data[position : position + 4])
        kind = data[position + 4 : position + 8]
        body = data[position + 8 : position + 8 + length]
        position += 12 + length
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            idat.append(body)
        elif kind == b"IEND":
            break
    if header is None:
        raise ValueError("PNG has no header")
    width, height, depth, color, _, _, interlace = header
    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(color)
    if depth != 8 or channels is None or interlace:
        raise ValueError("Only non-interlaced 8-bit gray/RGB/RGBA PNG images are supported")
    if width * height > 16_777_216:
        raise ValueError("PNG exceeds 16 megapixels")
    raw = np.frombuffer(zlib.decompress(b"".join(idat)), dtype=np.uint8)
    stride = width * channels
    raw = raw.reshape(height, stride + 1)
    out = np.zeros((height, stride), dtype=np.int32)
    previous = np.zeros(stride, dtype=np.int32)
    for y in range(height):
        kind, line = raw[y, 0], raw[y, 1:].astype(np.int32)
        if kind == 0:
            row = line
        elif kind == 2:
            row = (line + previous) & 255
        else:
            row = np.zeros(stride, dtype=np.int32)
            for x in range(stride):
                left = row[x - channels] if x >= channels else 0
                if kind == 1:
                    row[x] = (line[x] + left) & 255
                elif kind == 3:
                    row[x] = (line[x] + ((left + previous[x]) >> 1)) & 255
                else:
                    up_left = previous[x - channels] if x >= channels else 0
                    p = left + previous[x] - up_left
                    pa, pb, pc = abs(p - left), abs(p - previous[x]), abs(p - up_left)
                    predictor = left if pa <= pb and pa <= pc else previous[x] if pb <= pc else up_left
                    row[x] = (line[x] + predictor) & 255
        out[y] = row
        previous = row
    image = out.astype(np.uint8).reshape(height, width, channels)
    if channels <= 2:
        return np.repeat(image[..., :1], 3, axis=-1)
    return np.ascontiguousarray(image[..., :3])


def load_image(source: Any) -> np.ndarray:
    """Load a PNG (or any Pillow-readable format when Pillow is installed) as RGB uint8."""
    data = Path(source).read_bytes()
    try:
        import io  # noqa: PLC0415

        from PIL import Image
    except ImportError:
        # The built-in decoder is exact but slow for per-pixel predictive filters.
        return decode_png(data)
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def to_rgb(image: Any) -> np.ndarray:
    """Convert arrays, Pillow images, matplotlib figures, PNG bytes, or paths to RGB uint8."""
    if isinstance(image, bytes | bytearray):
        try:
            import io  # noqa: PLC0415

            from PIL import Image
        except ImportError:
            return decode_png(bytes(image))
        return np.asarray(Image.open(io.BytesIO(bytes(image))).convert("RGB"))
    if isinstance(image, str | Path):
        return load_image(image)
    if hasattr(image, "savefig") and hasattr(image, "canvas"):
        import io  # noqa: PLC0415

        buffer = io.BytesIO()
        image.savefig(buffer, format="png", dpi=getattr(image, "dpi", 100))
        return to_rgb(buffer.getvalue())
    if hasattr(image, "convert") and hasattr(image, "size") and not isinstance(image, np.ndarray):
        return np.asarray(image.convert("RGB"))
    array = np.asarray(image.numpy() if hasattr(image, "numpy") else image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3 or array.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected an HxW, HxWx3, or HxWx4 image; got shape {array.shape}")
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    array = array[..., :3]
    if array.dtype.kind == "f":
        finite = np.nan_to_num(array.astype(np.float64), nan=0.0, posinf=1.0, neginf=0.0)
        scale = 255.0 if finite.max(initial=0.0) <= 1.0 else 1.0
        array = np.clip(finite * scale, 0, 255)
    return np.ascontiguousarray(array, dtype=np.uint8)


def draw_label(rgb: np.ndarray, text: str, x: int = 3, y: int = 3, scale: int | None = None) -> None:
    """Draw white-on-black ASCII text in place; unsupported characters render as '?'."""
    height, width = rgb.shape[:2]
    scale = scale or (2 if min(width, height) >= 320 else 1)
    text = text.upper()[: max(1, (width - x) // (6 * scale))]
    box_w, box_h = len(text) * 6 * scale + 2 * scale, 9 * scale
    rgb[y : y + box_h, x : x + box_w] = 0
    for i, char in enumerate(text):
        glyph = _FONT.get(char, _FONT["?"])
        ink = np.kron(glyph, np.ones((scale, scale), dtype=bool))
        top, left = y + scale, x + scale + i * 6 * scale
        region = rgb[top : top + ink.shape[0], left : left + ink.shape[1]]
        region[ink[: region.shape[0], : region.shape[1]]] = 255


def tile(images: list[list[np.ndarray | None]], labels: list[list[str]] | None = None, gap: int = 4) -> np.ndarray:
    """Tile RGB images row-major on a white background, padding ragged cells."""
    heights = [max((im.shape[0] for im in row if im is not None), default=0) for row in images]
    columns = max(len(row) for row in images)
    widths = [
        max((row[c].shape[1] for row in images if c < len(row) and row[c] is not None), default=0)
        for c in range(columns)
    ]
    sheet = np.full((sum(heights) + gap * (len(images) - 1), sum(widths) + gap * (columns - 1), 3), 255, dtype=np.uint8)
    top = 0
    for r, row in enumerate(images):
        left = 0
        for c in range(columns):
            if c < len(row) and row[c] is not None:
                image = row[c]
                sheet[top : top + image.shape[0], left : left + image.shape[1]] = image
                if labels is not None and labels[r][c]:
                    view = sheet[top : top + image.shape[0], left : left + image.shape[1]]
                    draw_label(view, labels[r][c])
            left += widths[c] + gap
        top += heights[r] + gap
    return sheet


def _gray(image: np.ndarray) -> np.ndarray:
    return image[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114])


def _box_mean(values: np.ndarray, radius: int) -> np.ndarray:
    """Mean over a (2 radius + 1)^2 window, clamped at the borders, via summed-area tables."""
    padded = np.pad(values, radius + 1, mode="edge")
    table = padded.cumsum(axis=0).cumsum(axis=1)
    size = 2 * radius + 1
    window = table[size:, size:] - table[:-size, size:] - table[size:, :-size] + table[:-size, :-size]
    return window[: values.shape[0], : values.shape[1]] / (size * size)


def _edges(gray: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude."""
    padded = np.pad(gray, 1, mode="edge")
    gx = (padded[:-2, 2:] + 2 * padded[1:-1, 2:] + padded[2:, 2:]) - (
        padded[:-2, :-2] + 2 * padded[1:-1, :-2] + padded[2:, :-2]
    )
    gy = (padded[2:, :-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:]) - (
        padded[:-2, :-2] + 2 * padded[:-2, 1:-1] + padded[:-2, 2:]
    )
    return np.hypot(gx, gy)


def image_metrics(simulated: np.ndarray, reference: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Similarity of two equally sized RGB images, optionally restricted to ``mask``.

    Returns PSNR [dB] of RGB, SSIM of luminance (7 x 7 windows), and edge NCC, the normalized
    cross-correlation of Sobel gradient magnitudes, which tracks geometric alignment under
    lighting and material differences. Higher is better for all three.
    """
    if simulated.shape[:2] != reference.shape[:2]:
        raise ValueError(f"Image sizes differ: simulated {simulated.shape}, reference {reference.shape}")
    weights = np.ones(simulated.shape[:2]) if mask is None else np.asarray(mask, dtype=np.float64)
    if weights.shape != simulated.shape[:2] or weights.sum() <= 0:
        raise ValueError("mask must match the image size and select at least one pixel")
    a = simulated[..., :3].astype(np.float64)
    b = reference[..., :3].astype(np.float64)
    mse = float((((a - b) ** 2).mean(axis=-1) * weights).sum() / weights.sum())
    psnr = float("inf") if mse == 0 else 10.0 * np.log10(255.0**2 / mse)
    ga, gb = _gray(a), _gray(b)
    mu_a, mu_b = _box_mean(ga, 3), _box_mean(gb, 3)
    var_a = _box_mean(ga * ga, 3) - mu_a**2
    var_b = _box_mean(gb * gb, 3) - mu_b**2
    cov = _box_mean(ga * gb, 3) - mu_a * mu_b
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ssim_map = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))
    ssim = float((ssim_map * weights).sum() / weights.sum())
    ea, eb = _edges(ga), _edges(gb)
    ea = ea - (ea * weights).sum() / weights.sum()
    eb = eb - (eb * weights).sum() / weights.sum()
    denominator = np.sqrt((ea * ea * weights).sum() * (eb * eb * weights).sum())
    edge_ncc = float((ea * eb * weights).sum() / denominator) if denominator > 0 else 0.0
    return {"psnr_db": round(psnr, 3), "ssim": round(ssim, 4), "edge_ncc": round(edge_ncc, 4)}


def comparison_panel(simulated: np.ndarray, reference: np.ndarray, kind: str = "mismatch", threshold: int = 24):
    """A third panel for simulated-versus-reference views.

    ``mismatch`` tints pixels differing by more than ``threshold`` magenta (for synthetic references),
    ``edges`` draws simulated edges in magenta and reference edges in green over the dimmed reference
    (overlap appears white; for real photos), and ``blend`` averages both images.
    """
    if kind == "mismatch":
        return compare(simulated, reference, threshold)[0]
    if kind == "blend":
        return ((simulated[..., :3].astype(np.uint16) + reference[..., :3].astype(np.uint16)) // 2).astype(np.uint8)
    if kind == "edges":
        panel = np.repeat((0.35 * _gray(reference))[..., None], 3, axis=-1)
        sim_edges, ref_edges = _edges(_gray(simulated)), _edges(_gray(reference))
        strong_sim = sim_edges > max(np.percentile(sim_edges, 90), 1.0)
        strong_ref = ref_edges > max(np.percentile(ref_edges, 90), 1.0)
        panel[strong_ref] = (60, 220, 60)
        panel[strong_sim] = (230, 40, 200)
        panel[strong_sim & strong_ref] = (255, 255, 255)
        return panel.astype(np.uint8)
    raise ValueError("comparison must be 'mismatch', 'edges', or 'blend'")


def compare(simulated: np.ndarray, reference: np.ndarray, threshold: int = 24) -> tuple[np.ndarray, dict]:
    """Return a mismatch panel and pixel statistics for equally sized images.

    The panel shows the reference in grayscale with pixels that differ by more
    than ``threshold`` (max channel difference) tinted magenta.
    """
    if simulated.shape != reference.shape:
        raise ValueError(f"Image sizes differ: simulated {simulated.shape}, reference {reference.shape}")
    difference = np.abs(simulated.astype(np.int16) - reference.astype(np.int16)).max(axis=-1)
    mismatch = difference > threshold
    gray = (0.6 * reference.mean(axis=-1)).astype(np.uint8)
    panel = np.repeat(gray[..., None], 3, axis=-1)
    panel[mismatch] = (255, 0, 200)
    stats = {
        "mean_abs_difference": round(float(np.abs(simulated.astype(np.int16) - reference.astype(np.int16)).mean()), 3),
        "mismatch_fraction": round(float(mismatch.mean()), 5),
        "mismatch_threshold": threshold,
        **image_metrics(simulated, reference),
    }
    return panel, stats
