# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the MuJoCo + SolverMACFluid coupled examples."""

from __future__ import annotations

import json
import os
from collections.abc import Callable

import numpy as np
import warp as wp
from newton.solvers.experimental.coupled import SolverCoupledProxy

import newton
from newton.solvers import SolverMACFluid, SolverMuJoCo


def add_macfluid_args(parser) -> None:
    """Add the CLI arguments shared by the MAC-fluid coupled examples."""
    parser.add_argument("--fluid-res", type=int, default=48, help="Fluid grid cells along the longest tank axis.")
    parser.add_argument("--pressure-iterations", type=int, default=120, help="Fixed CG iterations per pressure solve.")
    parser.add_argument("--viscosity", type=float, default=1.0e-4, help="Kinematic viscosity [m^2/s].")
    parser.add_argument("--proxy-iterations", type=int, default=1, help="Proxy coupling passes per step.")
    parser.add_argument(
        "--proxy-relaxation",
        type=float,
        default=1.0,
        help="Proxy feedback relaxation factor (< 1 damps lagged coupling feedback).",
    )
    parser.add_argument(
        "--proxy-relaxation-mode",
        type=str,
        choices=["fixed", "aitken"],
        default="fixed",
        help="Proxy feedback relaxation mode.",
    )
    parser.add_argument(
        "--proxy-mode",
        type=str,
        choices=["staggered", "lagged"],
        default="staggered",
        help="Proxy coupling mode. Staggered avoids the free-body feedback rewind, "
        "which is inconsistent for joint-constrained bodies and can destabilize lagged coupling.",
    )
    parser.add_argument(
        "--mass-scale",
        type=float,
        default=1.0,
        help="Proxy effective-mass scale; values > 1 stabilize bodies whose hydrodynamic added mass exceeds their inertia.",
    )
    parser.add_argument("--rigid-substeps", type=int, default=4, help="MuJoCo substeps per coupled step.")
    parser.add_argument("--metrics-output", type=str, default=None, help="Write per-frame metrics JSON to this path.")
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Rigid-only comparison run: step MuJoCo without the fluid.",
    )


def make_coupled_fluid_solver(
    model: newton.Model,
    fluid_config: SolverMACFluid.Config,
    *,
    rigid_bodies: list[int],
    joints: list[int],
    args,
    mujoco_kwargs: dict | None = None,
) -> tuple[SolverCoupledProxy, SolverMACFluid]:
    """Build the standard MuJoCo (source) + MAC fluid (destination) proxy coupling."""
    kwargs = {"use_mujoco_contacts": False, "njmax": 50}
    kwargs.update(mujoco_kwargs or {})
    coupled = SolverCoupledProxy(
        model=model,
        entries=[
            SolverCoupledProxy.Entry(
                name="mjc",
                solver=lambda v: SolverMuJoCo(model=v, **kwargs),
                bodies=rigid_bodies,
                joints=joints,
                substeps=args.rigid_substeps,
            ),
            SolverCoupledProxy.Entry(
                name="fluid",
                solver=lambda v: SolverMACFluid(v, fluid_config),
                in_place=True,
            ),
        ],
        coupling=SolverCoupledProxy.Config(
            proxies=[
                SolverCoupledProxy.Proxy(
                    source="mjc",
                    destination="fluid",
                    bodies=rigid_bodies,
                    mode=getattr(args, "proxy_mode", "staggered"),
                    mass_scale=getattr(args, "mass_scale", 1.0),
                    proxy_relaxation=getattr(args, "proxy_relaxation", 1.0),
                    proxy_relaxation_mode=getattr(args, "proxy_relaxation_mode", "fixed"),
                    # the fluid treats proxies as immersed boundaries; no
                    # rigid contact detection is needed for the fluid entry
                    collision_pipeline=lambda _model: None,
                )
            ],
            iterations=args.proxy_iterations,
        ),
    )
    return coupled, coupled.solver("fluid")


def fluid_body_wrenches(coupled: SolverCoupledProxy, fluid: SolverMACFluid, dt: float, body_count: int) -> np.ndarray:
    """Hydrodynamic wrench per parent-model body [N, N·m] from the last fluid step."""
    wrench = np.zeros((body_count, 6), dtype=np.float64)
    local_to_global = coupled.entry_body_local_to_global("fluid").numpy()
    impulses = fluid.body_impulse.numpy()
    n = min(len(local_to_global), len(impulses))
    for local in range(n):
        g = int(local_to_global[local])
        if 0 <= g < body_count:
            wrench[g] = impulses[local] / dt
    return wrench


def capture_frame_graph(model: newton.Model, simulate: Callable[[], None], warmup: Callable[[], None] | None = None):
    """Capture one coupled frame in a CUDA graph (no-op on CPU).

    Args:
        model: Model providing the target device.
        simulate: Callable advancing the simulation by one frame.
        warmup: Optional callable run once before capture so that first-step
            allocations (e.g. MuJoCo internals) happen outside the graph.
    """
    if not model.device.is_cuda:
        return None
    with wp.ScopedDevice(model.device):
        if warmup is not None:
            warmup()
        with wp.ScopedCapture() as capture:
            simulate()
    return capture.graph


def colormap(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Map scalars to a blue-white-red diverging colormap, shape (n, 3)."""
    t = np.clip((values - vmin) / max(vmax - vmin, 1.0e-12), 0.0, 1.0)
    r = np.clip(2.0 * t, 0.0, 1.0)
    b = np.clip(2.0 * (1.0 - t), 0.0, 1.0)
    g = 1.0 - 2.0 * np.abs(t - 0.5)
    return np.stack([r, g * 0.9, b], axis=-1).astype(np.float32)


class FluidSliceVisualizer:
    """Renders one grid slice of the fluid as colored points in the viewer.

    Points are colored by pressure or velocity magnitude; solid cells are
    drawn dark. This host-side visualization runs outside the simulation
    graph.
    """

    def __init__(self, fluid: SolverMACFluid, axis: int = 1, index: int | None = None):
        self.fluid = fluid
        self.axis = axis
        nx, ny, nz = fluid.nx, fluid.ny, fluid.nz
        self.index = index if index is not None else (nx, ny, nz)[axis] // 2

        dims = [nx, ny, nz]
        dims[axis] = 1
        ii, jj, kk = np.meshgrid(np.arange(dims[0]), np.arange(dims[1]), np.arange(dims[2]), indexing="ij")
        idx = [ii, jj, kk]
        idx[axis][:] = self.index
        origin = np.array([fluid.origin[0], fluid.origin[1], fluid.origin[2]])
        pts = (np.stack([idx[0], idx[1], idx[2]], axis=-1).reshape(-1, 3) + 0.5) * fluid.dx + origin
        self._points = wp.array(pts.astype(np.float32), dtype=wp.vec3, device=fluid.model.device)
        self._slice_index = tuple(
            idx[a].reshape(-1) if a != axis else np.full(pts.shape[0], self.index) for a in range(3)
        )
        self._colors = wp.zeros(pts.shape[0], dtype=wp.vec3, device=fluid.model.device)

    def log(self, viewer, name: str = "/fluid_slice", field: str = "speed", scale: float | None = None):
        f = self.fluid
        i, j, k = self._slice_index
        labels = f.cell_label.numpy()[i, j, k]
        if field == "pressure":
            values = f.pressure.numpy()[i, j, k]
            vmax = scale if scale is not None else max(np.abs(values).max(), 1.0e-6)
            colors = colormap(values, -vmax, vmax)
        else:
            u = f.velocity_u.numpy()
            v = f.velocity_v.numpy()
            w = f.velocity_w.numpy()
            speed = np.sqrt(
                (0.5 * (u[i, j, k] + u[i + 1, j, k])) ** 2
                + (0.5 * (v[i, j, k] + v[i, j + 1, k])) ** 2
                + (0.5 * (w[i, j, k] + w[i, j, k + 1])) ** 2
            )
            vmax = scale if scale is not None else max(speed.max(), 1.0e-6)
            colors = colormap(speed, 0.0, vmax)
        colors[labels != -2] = (0.15, 0.15, 0.15)
        self._colors.assign(colors)
        viewer.log_points(name, points=self._points, radii=0.45 * f.dx, colors=self._colors)


class MetricsRecorder:
    """Accumulates per-frame metrics and writes them to a JSON file."""

    def __init__(self, path: str | None):
        self.path = path
        self.frames: list[dict] = []
        self.meta: dict = {}

    def record(self, **kwargs) -> None:
        frame = {}
        for key, value in kwargs.items():
            if isinstance(value, np.ndarray):
                frame[key] = value.tolist()
            elif isinstance(value, (np.floating, np.integer)):
                frame[key] = value.item()
            else:
                frame[key] = value
        self.frames.append(frame)

    def save(self, path: str | None = None) -> None:
        path = path or self.path
        if not path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"meta": self.meta, "frames": self.frames}, f)
        print(f"Saved metrics to {path}")


def log_tank_outline(viewer, fluid: SolverMACFluid, name: str = "/tank"):
    """Draw the wireframe outline of the fluid domain box."""
    o = np.array([fluid.origin[0], fluid.origin[1], fluid.origin[2]])
    e = o + np.array([fluid.nx, fluid.ny, fluid.nz]) * fluid.dx
    c = [
        (o[0], o[1], o[2]),
        (e[0], o[1], o[2]),
        (e[0], e[1], o[2]),
        (o[0], e[1], o[2]),
        (o[0], o[1], e[2]),
        (e[0], o[1], e[2]),
        (e[0], e[1], e[2]),
        (o[0], e[1], e[2]),
    ]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    starts = np.array([c[a] for a, _ in edges], dtype=np.float32)
    ends = np.array([c[b] for _, b in edges], dtype=np.float32)
    device = fluid.model.device
    colors = wp.full(len(edges), value=wp.vec3(0.6, 0.75, 0.85), dtype=wp.vec3, device=device)
    viewer.log_lines(
        name,
        wp.array(starts, dtype=wp.vec3, device=device),
        wp.array(ends, dtype=wp.vec3, device=device),
        colors,
    )
