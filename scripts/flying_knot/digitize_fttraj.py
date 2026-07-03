# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Digitize the flying-knot end-effector trajectory from the paper figure.

The figure (``fttraj.png``, from https://flying-knots.github.io/) plots the
hand/end-effector position of the overhand flying-knot throw:
x (red), y (blue), z (green) in meters over t in [0, 0.7] s. Solid lines are
the executed/learned command, dashed lines the human demonstration; the two
are within a few cm of each other. We extract a per-column color centroid
(blending solid+dashed where both are present) and fit the paper's 8-knot
Bezier parametrization (Appendix D) per axis.

Outputs (in this directory):
  - fttraj_digitized.npz: t [n], pos [n,3], bezier control points [8,3]
  - fttraj_digitized_overlay.png: extraction + fit overlaid on the figure data
"""

from pathlib import Path

import matplotlib  # noqa: TID253

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: TID253
import numpy as np
from PIL import Image  # noqa: TID253
from scipy.special import comb  # noqa: TID253

HERE = Path(__file__).parent

# Axis calibration (data coordinates of the tick range).
T_MIN, T_MAX = 0.0, 0.7
P_MIN, P_MAX = -0.2, 0.8


def find_axes_box(img: np.ndarray) -> tuple[int, int, int, int]:
    """Locate the plot spines (matplotlib axes rectangle) as pixel bounds."""
    dark = (img[..., :3] < 90).all(axis=-1)
    col_counts = dark.sum(axis=0)
    row_counts = dark.sum(axis=1)
    h, w = dark.shape
    # Spines are the long dark straight lines.
    cols = np.where(col_counts > 0.5 * h)[0]
    rows = np.where(row_counts > 0.5 * w)[0]
    if len(cols) < 2 or len(rows) < 2:
        raise RuntimeError("could not locate axes spines")
    return cols[0], cols[-1], rows[0], rows[-1]  # left, right, top, bottom


def tick_positions(img: np.ndarray, left: int, right: int, top: int, bottom: int):
    """Find tick marks just outside the axes box (below bottom spine, left of left spine)."""
    dark = (img[..., :3] < 90).all(axis=-1)
    # Bottom ticks: dark pixels a few rows below the bottom spine.
    band = dark[bottom + 2 : bottom + 5, :].any(axis=0)
    xticks = _cluster_centers(np.where(band)[0])
    # Left ticks: dark pixels a few cols left of the left spine.
    band = dark[:, left - 5 : left - 2].any(axis=1)
    yticks = _cluster_centers(np.where(band)[0])
    return xticks, yticks


def _cluster_centers(idx: np.ndarray) -> list[float]:
    if len(idx) == 0:
        return []
    groups = np.split(idx, np.where(np.diff(idx) > 3)[0] + 1)
    return [float(g.mean()) for g in groups]


def extract_curves(img: np.ndarray, left, right, top, bottom, px_to_t, px_to_p):
    """Per-column colored-pixel centroid for each of the 3 axes (r, b, g)."""
    rgb = img[..., :3].astype(np.int32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    masks = {
        "x": (r > 120) & (r - g > 50) & (r - b > 50),  # red
        "y": (b > 100) & (b - r > 40) & (b - g > 30),  # blue
        "z": (g > 100) & (g - r > 30) & (g - b > 30),  # green
    }
    curves = {}
    for name, mask in masks.items():
        m = mask.copy()
        m[: top + 2] = False
        m[bottom - 1 :] = False
        m[:, : left + 2] = False
        m[:, right - 1 :] = False
        ts, ps = [], []
        prev = None
        for col in range(left + 2, right - 1):
            rows = np.where(m[:, col])[0]
            if len(rows) == 0:
                continue
            clusters = np.split(rows, np.where(np.diff(rows) > 4)[0] + 1)
            centers = np.array([c.mean() for c in clusters])
            weights = np.array([len(c) for c in clusters], dtype=float)
            if prev is None:
                center = float(np.average(centers, weights=weights))
            else:
                # Prefer clusters continuous with the previous column; blend all
                # clusters within 12 px of the tracked value (solid + dashed).
                near = np.abs(centers - prev) < 12
                if near.any():
                    center = float(np.average(centers[near], weights=weights[near]))
                else:
                    center = float(centers[np.argmin(np.abs(centers - prev))])
            prev = center
            ts.append(px_to_t(col))
            ps.append(px_to_p(center))
        curves[name] = (np.array(ts), np.array(ps))
    return curves


def bernstein_matrix(n_ctrl: int, s: np.ndarray) -> np.ndarray:
    n = n_ctrl - 1
    return np.stack([comb(n, i) * s**i * (1 - s) ** (n - i) for i in range(n_ctrl)], axis=1)


def fit_bezier(t: np.ndarray, p: np.ndarray, n_ctrl: int = 8, t_end: float = 0.7) -> np.ndarray:
    s = np.clip(t / t_end, 0.0, 1.0)
    A = bernstein_matrix(n_ctrl, s)
    ctrl, *_ = np.linalg.lstsq(A, p, rcond=None)
    return ctrl


def main():
    img = np.asarray(Image.open(HERE / "fttraj.png").convert("RGB"))
    left, right, top, bottom = find_axes_box(img)
    xticks, yticks = tick_positions(img, left, right, top, bottom)
    print(f"axes box: left={left} right={right} top={top} bottom={bottom}")
    print(f"x ticks px: {[f'{v:.0f}' for v in xticks]}")
    print(f"y ticks px: {[f'{v:.0f}' for v in yticks]}")

    # Calibrate: first/last tick correspond to T_MIN..T_MAX and P_MAX..P_MIN (y down).
    x0, x1 = xticks[0], xticks[-1]
    y0, y1 = yticks[0], yticks[-1]  # top tick = P_MAX, bottom tick = P_MIN

    def px_to_t(col):
        return T_MIN + (col - x0) / (x1 - x0) * (T_MAX - T_MIN)

    def px_to_p(row):
        return P_MAX + (row - y0) / (y1 - y0) * (P_MIN - P_MAX)

    curves = extract_curves(img, left, right, top, bottom, px_to_t, px_to_p)

    # Resample all three onto a common time grid.
    t_grid = np.linspace(0.0, 0.7, 141)
    pos = np.zeros((len(t_grid), 3))
    for i, name in enumerate(["x", "y", "z"]):
        ts, ps = curves[name]
        pos[:, i] = np.interp(t_grid, ts, ps)
        print(f"{name}: {len(ts)} columns, range [{ps.min():.3f}, {ps.max():.3f}] m")

    ctrl = np.zeros((8, 3))
    for i in range(3):
        ctrl[:, i] = fit_bezier(t_grid, pos[:, i])
    fit = bernstein_matrix(8, t_grid / 0.7) @ ctrl
    err = np.abs(fit - pos).max(axis=0)
    print(f"bezier max fit error per axis [m]: {err}")

    np.savez(
        HERE / "fttraj_digitized.npz",
        t=t_grid,
        pos=pos,
        bezier_ctrl=ctrl,
        t_end=0.7,
    )

    # Overlay figure.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {"x": "tab:red", "y": "tab:blue", "z": "tab:green"}
    for i, name in enumerate(["x", "y", "z"]):
        ts, ps = curves[name]
        ax.plot(ts, ps, ".", ms=2, color=colors[name], alpha=0.35, label=f"{name} extracted")
        ax.plot(t_grid, fit[:, i], "-", color=colors[name], lw=2, label=f"{name} Bezier fit")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position [m]")
    ax.set_title("Digitized flying-knot EE trajectory + 8-knot Bezier fit")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "fttraj_digitized_overlay.png", dpi=150)
    print("wrote fttraj_digitized.npz and fttraj_digitized_overlay.png")


if __name__ == "__main__":
    main()
