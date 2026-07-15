# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MAC-grid incompressible fluid solver."""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.mac_fluid import kernels as K
from newton._src.solvers.mac_fluid.grid import sample_u, sample_v, sample_w
from newton.solvers import SolverBase, SolverMACFluid
from newton.solvers.experimental.coupled import CouplingInterface, SolverCoupled, SolverCoupledProxy
from newton.tests.unittest_utils import add_function_test, get_test_devices

GRAVITY = 9.81


@wp.kernel(enable_backward=False)
def _sample_velocity_kernel(
    u: wp.array3d[float],
    v: wp.array3d[float],
    w: wp.array3d[float],
    origin: wp.vec3,
    dx: float,
    points: wp.array[wp.vec3],
    out: wp.array[wp.vec3],
):
    i = wp.tid()
    p = points[i]
    out[i] = wp.vec3(
        sample_u(u, origin, dx, p),
        sample_v(v, origin, dx, p),
        sample_w(w, origin, dx, p),
    )


def _make_tank_model(device, gravity=-GRAVITY, bodies=()):
    """Closed 1 m tank with optional rigid bodies: list of (pos, radius, mass)."""
    builder = newton.ModelBuilder(gravity=gravity)
    body_ids = []
    for pos, radius, mass in bodies:
        b = builder.add_body(xform=wp.transform(wp.vec3(*pos), wp.quat_identity()), mass=mass)
        builder.add_shape_sphere(b, radius=radius)
        body_ids.append(b)
    model = builder.finalize(device=device)
    return model, body_ids


def _make_solver(model, res=16, iters=60, viscosity=0.0, **kwargs):
    cfg = SolverMACFluid.Config(
        resolution=(res, res, res),
        cell_size=1.0 / res,
        origin=(0.0, 0.0, 0.0),
        pressure_iterations=iters,
        kinematic_viscosity=viscosity,
        **kwargs,
    )
    return SolverMACFluid(model, cfg)


def test_interpolation_linear_field(test: unittest.TestCase, device):
    """Trilinear MAC interpolation must reproduce a linear velocity field exactly."""
    model, _ = _make_tank_model(device, gravity=0.0)
    solver = _make_solver(model, res=8)
    g = solver.grid
    dx = solver.dx

    # velocity = A x + b (linear in position), evaluated at the staggered locations
    A = np.array([[0.3, -0.2, 0.1], [0.5, 0.4, -0.3], [-0.1, 0.2, 0.6]])
    b = np.array([0.05, -0.02, 0.03])

    def fill(component, axis):
        nxs = component.shape
        vals = np.zeros(nxs, dtype=np.float32)
        for i in range(nxs[0]):
            for j in range(nxs[1]):
                for k in range(nxs[2]):
                    p = np.array([i + 0.5, j + 0.5, k + 0.5]) * dx
                    p[axis] = (i, j, k)[axis] * dx
                    vals[i, j, k] = A[axis] @ p + b[axis]
        component.assign(vals)

    fill(g.u, 0)
    fill(g.v, 1)
    fill(g.w, 2)

    rng = np.random.default_rng(42)
    # sample strictly inside the staggered support of all components
    pts = rng.uniform(1.5 * dx, 1.0 - 1.5 * dx, size=(50, 3)).astype(np.float32)
    points = wp.array(pts, dtype=wp.vec3, device=device)
    out = wp.zeros(len(pts), dtype=wp.vec3, device=device)
    wp.launch(
        _sample_velocity_kernel,
        dim=len(pts),
        inputs=[g.u, g.v, g.w, solver.origin, dx, points, out],
        device=device,
    )
    expected = pts @ A.T + b
    np.testing.assert_allclose(out.numpy(), expected, atol=1.0e-5)


def test_projection_reduces_divergence(test: unittest.TestCase, device):
    """Pressure projection must reduce the divergence of a random field by orders of magnitude."""
    model, _ = _make_tank_model(device, gravity=0.0)
    solver = _make_solver(model, res=16, iters=120)
    g = solver.grid

    rng = np.random.default_rng(7)
    g.u.assign(rng.uniform(-1.0, 1.0, size=g.u.shape).astype(np.float32))
    g.v.assign(rng.uniform(-1.0, 1.0, size=g.v.shape).astype(np.float32))
    g.w.assign(rng.uniform(-1.0, 1.0, size=g.w.shape).astype(np.float32))

    s0, s1 = model.state(), model.state()
    solver.step(s0, s1, None, None, 1.0e-3)
    d = solver.read_diagnostics()

    test.assertGreater(d["div_l2_pre"], 1.0)
    test.assertLess(d["div_l2_post"], 1.0e-3 * d["div_l2_pre"])
    test.assertLess(d["div_linf_post"], 1.0e-2 * d["div_linf_pre"])
    # residual reduced by >= 5 orders of magnitude relative to ||b||_2
    b_norm = d["div_l2_pre"] * math.sqrt(d["fluid_cell_count"])
    test.assertLess(d["pressure_residual"], 1.0e-5 * b_norm)


def test_closed_domain_null_space(test: unittest.TestCase, device):
    """Hydrostatic equilibrium in a closed all-Neumann tank.

    The singular Poisson system must remain solvable (compatibility fix), the
    fluid must stay quiescent, and the pressure must approach the hydrostatic
    profile.
    """
    model, _ = _make_tank_model(device)
    solver = _make_solver(model, res=16, iters=150)

    s0, s1 = model.state(), model.state()
    dt = 1.0 / 60.0
    for _ in range(5):
        solver.step(s0, s1, None, None, dt)

    d = solver.read_diagnostics()
    # velocities stay ~0: post-projection velocity magnitude via divergence + direct check
    u = solver.velocity_u.numpy()
    v = solver.velocity_v.numpy()
    w = solver.velocity_w.numpy()
    max_vel = max(np.abs(u).max(), np.abs(v).max(), np.abs(w).max())
    test.assertLess(max_vel, 1.0e-4, f"fluid should stay quiescent, max |u| = {max_vel}")
    test.assertLess(d["div_linf_post"], 1.0e-4)

    # hydrostatic pressure gradient: dp/dz = -rho * g between vertical neighbors
    p = solver.pressure.numpy()
    dp = p[:, :, 1:] - p[:, :, :-1]
    expected = -solver.density * GRAVITY * solver.dx
    np.testing.assert_allclose(dp, expected, rtol=2.0e-2, atol=abs(expected) * 2.0e-2)


def test_viscous_diffusion_operator(test: unittest.TestCase, device):
    """The explicit diffusion operator must match the reference 7-point stencil.

    The velocity update of the viscosity stage is compared face-by-face
    against a NumPy reference (out-of-array neighbors are zero-velocity
    no-slip walls; domain-boundary faces are constrained and not updated).
    """
    model, _ = _make_tank_model(device, gravity=0.0)
    solver = _make_solver(model, res=12, viscosity=1.0e-3)
    g = solver.grid
    dt = 1.0 / 120.0
    coeff = 1.0e-3 * dt / (solver.dx * solver.dx)

    rng = np.random.default_rng(3)
    u0 = rng.uniform(-1.0, 1.0, size=g.u.shape).astype(np.float32)
    g.u.assign(u0)

    # rasterize once so cell labels are valid, then apply the operator
    s0, s1 = model.state(), model.state()
    solver.step(s0, s1, None, None, dt)  # populates labels
    g.u.assign(u0)
    wp.launch(
        K.diffuse_u_kernel,
        dim=g.u.shape,
        inputs=[
            solver.origin,
            solver.dx,
            coeff,
            solver.density * solver.dx**3,
            g.cell_label,
            g.cell_sdf,
            g.u,
            g.u_tmp,
            solver.body_impulse,
            solver._dummy_body_q,
            solver._dummy_body_com,
            solver._diag_vec,
        ],
        device=device,
    )

    # NumPy reference: 7-point diffusion with zero ghost values
    padded = np.zeros((u0.shape[0] + 2, u0.shape[1] + 2, u0.shape[2] + 2), dtype=np.float64)
    padded[1:-1, 1:-1, 1:-1] = u0
    lap = (
        padded[:-2, 1:-1, 1:-1]
        + padded[2:, 1:-1, 1:-1]
        + padded[1:-1, :-2, 1:-1]
        + padded[1:-1, 2:, 1:-1]
        + padded[1:-1, 1:-1, :-2]
        + padded[1:-1, 1:-1, 2:]
        - 6.0 * u0
    )
    expected = u0 + coeff * lap
    expected[0, :, :] = u0[0, :, :]  # constrained domain-boundary faces
    expected[-1, :, :] = u0[-1, :, :]

    np.testing.assert_allclose(g.u_tmp.numpy(), expected, atol=1.0e-6)

    # decay rate of a sinusoidal shear profile matches the discrete eigenvalue
    nz = u0.shape[2]
    profile = np.sin(2.0 * np.pi * (np.arange(nz) + 0.5) / nz)
    u_sin = np.broadcast_to(profile, g.u.shape).astype(np.float32).copy()
    g.u.assign(u_sin)
    wp.launch(
        K.diffuse_u_kernel,
        dim=g.u.shape,
        inputs=[
            solver.origin,
            solver.dx,
            coeff,
            solver.density * solver.dx**3,
            g.cell_label,
            g.cell_sdf,
            g.u,
            g.u_tmp,
            solver.body_impulse,
            solver._dummy_body_q,
            solver._dummy_body_com,
            solver._diag_vec,
        ],
        device=device,
    )
    factor = 1.0 - 2.0 * coeff * (1.0 - np.cos(2.0 * np.pi / nz))
    interior = g.u_tmp.numpy()[2:-2, 2:-2, 2:-2]
    np.testing.assert_allclose(interior, u_sin[2:-2, 2:-2, 2:-2] * factor, atol=1.0e-6)


def test_viscosity_dissipates_energy(test: unittest.TestCase, device):
    """Viscosity must dissipate kinetic energy faster than the inviscid pipeline."""

    def run(viscosity):
        model, _ = _make_tank_model(device, gravity=0.0)
        solver = _make_solver(model, res=16, iters=80, viscosity=viscosity)
        g = solver.grid
        rng = np.random.default_rng(11)
        g.u.assign(rng.uniform(-1.0, 1.0, size=g.u.shape).astype(np.float32))
        g.v.assign(rng.uniform(-1.0, 1.0, size=g.v.shape).astype(np.float32))
        g.w.assign(rng.uniform(-1.0, 1.0, size=g.w.shape).astype(np.float32))
        s0, s1 = model.state(), model.state()
        for _ in range(10):
            solver.step(s0, s1, None, None, 1.0 / 120.0)
        u, v, w = solver.velocity_u.numpy(), solver.velocity_v.numpy(), solver.velocity_w.numpy()
        return (u**2).sum() + (v**2).sum() + (w**2).sum()

    ke_inviscid = run(0.0)
    ke_viscous = run(2.0e-3)
    test.assertGreater(ke_inviscid, 0.0)
    test.assertLess(ke_viscous, 0.9 * ke_inviscid)


def test_stationary_body_buoyancy(test: unittest.TestCase, device):
    """A stationary submerged sphere must feel the discrete buoyancy force.

    In discrete hydrostatic equilibrium, the pressure surface force on the
    voxelized body equals rho_fluid * g * V_voxel exactly (discrete divergence
    theorem), where V_voxel is the volume of the solid cells.
    """
    radius = 0.2
    analytic = 1000.0 * GRAVITY * 4.0 / 3.0 * math.pi * radius**3

    def run(res):
        model, _bodies = _make_tank_model(device, bodies=[((0.5, 0.5, 0.5), radius, 10.0)])
        solver = _make_solver(model, res=res, iters=300)
        s0, s1 = model.state(), model.state()
        for _ in range(3):
            solver.step(s0, s1, None, None, 1.0 / 60.0)
        d = solver.read_diagnostics()
        labels = solver.cell_label.numpy()
        v_voxel = int((labels == 0).sum()) * solver.dx**3
        return d, v_voxel, solver

    d, v_voxel, solver = run(24)
    expected_fz = solver.density * GRAVITY * v_voxel
    wrench = d["body_wrench"][0]

    # lateral force and torque ~ 0 by symmetry
    test.assertLess(abs(wrench[0]), 0.02 * expected_fz)
    test.assertLess(abs(wrench[1]), 0.02 * expected_fz)
    test.assertLess(np.abs(wrench[3:]).max(), 0.05 * expected_fz)

    # buoyancy magnitude within the O(dx) voxelization error
    np.testing.assert_allclose(wrench[2], expected_fz, rtol=0.25)
    np.testing.assert_allclose(wrench[2], analytic, rtol=0.3)

    # the buoyancy error must shrink with resolution (first-order boundary)
    d16, v16, _ = run(16)
    d32, v32, s32 = run(32)
    err16 = abs(d16["body_wrench"][0][2] / (s32.density * GRAVITY * v16) - 1.0)
    err32 = abs(d32["body_wrench"][0][2] / (s32.density * GRAVITY * v32) - 1.0)
    test.assertLess(err32, err16)

    # fluid stays quiescent around the stationary body
    test.assertLess(d["noslip_max"], 1.0e-3)


def test_translating_body_drag(test: unittest.TestCase, device):
    """A translating sphere must feel an opposing hydrodynamic force."""
    model, _ = _make_tank_model(device, gravity=0.0, bodies=[((0.5, 0.5, 0.5), 0.15, 10.0)])
    solver = _make_solver(model, res=24, iters=150, viscosity=1.0e-4)

    s0, s1 = model.state(), model.state()
    speed = 0.5
    qd = np.zeros((1, 6), dtype=np.float32)
    qd[0, 0] = speed  # linear x
    s0.body_qd.assign(qd)

    dt = 1.0 / 120.0
    for _ in range(3):
        solver.step(s0, s1, None, None, dt)
    d = solver.read_diagnostics()

    wrench = d["body_wrench"][0]
    test.assertLess(wrench[0], 0.0, f"drag must oppose +x motion, got fx = {wrench[0]}")
    test.assertGreater(abs(wrench[0]), 10.0 * abs(wrench[1]))
    test.assertGreater(abs(wrench[0]), 10.0 * abs(wrench[2]))
    # Tangential slip is bounded (binary voxelized boundary: the normal
    # component is enforced exactly, the tangential one only through
    # viscosity, and potential flow speeds up to ~1.5x around a sphere).
    test.assertLess(d["noslip_max"], 2.0 * speed)
    test.assertLess(d["noslip_mean"], 0.6 * speed)


def test_rotating_body_torque(test: unittest.TestCase, device):
    """A spinning sphere in viscous fluid must feel an opposing torque."""
    model, _ = _make_tank_model(device, gravity=0.0, bodies=[((0.5, 0.5, 0.5), 0.2, 10.0)])
    solver = _make_solver(model, res=24, iters=100, viscosity=1.0e-3)

    s0, s1 = model.state(), model.state()
    omega = 4.0
    qd = np.zeros((1, 6), dtype=np.float32)
    qd[0, 5] = omega  # angular z
    s0.body_qd.assign(qd)

    dt = 1.0 / 240.0
    for _ in range(5):
        solver.step(s0, s1, None, None, dt)
    d = solver.read_diagnostics()

    wrench = d["body_wrench"][0]
    test.assertLess(wrench[5], 0.0, f"viscous torque must oppose +z spin, got tz = {wrench[5]}")
    test.assertGreater(abs(wrench[5]), 5.0 * abs(wrench[3]))
    test.assertGreater(abs(wrench[5]), 5.0 * abs(wrench[4]))


def test_momentum_balance(test: unittest.TestCase, device):
    """Boundary impulses on the fluid must balance the fluid momentum change exactly.

    This verifies the equal-and-opposite bookkeeping between fluid and
    boundaries in the discrete system (up to float32 reduction error).
    """
    model, _ = _make_tank_model(device, bodies=[((0.5, 0.5, 0.6), 0.15, 10.0)])
    solver = _make_solver(model, res=16, iters=120, viscosity=1.0e-4)

    s0, s1 = model.state(), model.state()
    qd = np.zeros((1, 6), dtype=np.float32)
    qd[0, 2] = -0.4
    s0.body_qd.assign(qd)

    dt = 1.0 / 120.0
    solver.step(s0, s1, None, None, dt)
    d = solver.read_diagnostics()

    scale = solver.density * GRAVITY * dt  # reference impulse scale [N s / m^3] * V
    err = np.abs(np.array(d["momentum_balance_error"])).max()
    imp = np.abs(np.array(d["boundary_impulse_pressure"])).max()
    test.assertLess(err, 1.0e-3 * max(imp, scale), f"momentum balance error {err} vs impulse scale {imp}")


def test_harvest_proxy_wrench_matches_body_impulse(test: unittest.TestCase, device):
    """The coupling harvest hook must return body_impulse / dt at proxy indices."""
    model, _ = _make_tank_model(device, bodies=[((0.5, 0.5, 0.5), 0.2, 10.0)])
    solver = _make_solver(model, res=16, iters=80)

    # mark the body as a proxy the way the coupler would
    flags = model.body_flags.numpy()
    flags[0] |= int(newton.BodyFlags.PROXY)
    model.body_flags.assign(flags)

    s0, s1 = model.state(), model.state()
    dt = 1.0 / 60.0
    solver.step(s0, s1, None, None, dt)

    mapping = wp.array([0], dtype=int, device=device)
    out = wp.zeros(1, dtype=wp.spatial_vector, device=device)
    solver.coupling_harvest_proxy_wrenches(
        mapping, out, body_qd_before=s0.body_qd, state=s0, state_out=s1, contacts=None, dt=dt
    )
    expected = solver.body_impulse.numpy()[0] / dt
    np.testing.assert_allclose(out.numpy()[0], expected, rtol=1.0e-6)
    test.assertGreater(expected[2], 0.0)  # buoyancy points up


def test_coupled_iteration_restore(test: unittest.TestCase, device):
    """An iteration restart must restore the beginning-of-step fluid state bit-exactly."""
    model, _ = _make_tank_model(device, bodies=[((0.5, 0.5, 0.5), 0.15, 10.0)])
    solver = _make_solver(model, res=16, iters=60, viscosity=1.0e-4)

    s0, s1 = model.state(), model.state()
    qd = np.zeros((1, 6), dtype=np.float32)
    qd[0, 0] = 0.3
    s0.body_qd.assign(qd)
    dt = 1.0 / 60.0

    # a genuinely transient state: random flow keeps evolving under advection
    rng = np.random.default_rng(5)
    solver.grid.u.assign(rng.uniform(-1.0, 1.0, size=solver.grid.u.shape).astype(np.float32))
    solver.grid.v.assign(rng.uniform(-1.0, 1.0, size=solver.grid.v.shape).astype(np.float32))
    solver.grid.w.assign(rng.uniform(-1.0, 1.0, size=solver.grid.w.shape).astype(np.float32))
    solver.step(s0, s1, None, None, dt)

    u_before = solver.velocity_u.numpy().copy()

    solver.step(s0, s1, None, None, dt)
    u_first = solver.velocity_u.numpy().copy()
    imp_first = solver.body_impulse.numpy().copy()

    # coupler repeats the same physical interval
    solver.coupling_notify_input_state_update(s0, 0, iteration_restart=True, dt=dt)
    solver.step(s0, s1, None, None, dt)
    u_second = solver.velocity_u.numpy().copy()
    imp_second = solver.body_impulse.numpy().copy()

    test.assertFalse(np.array_equal(u_before, u_first), "step must change the state")
    # deterministic reductions make the repeated interval bit-exact
    np.testing.assert_array_equal(u_first, u_second)
    # wrench accumulation uses atomics; allow reduction-order rounding only
    np.testing.assert_allclose(imp_first, imp_second, rtol=1.0e-5, atol=1.0e-7)

    # without a restart the fluid advances (no accidental restore)
    solver.step(s0, s1, None, None, dt)
    test.assertFalse(np.array_equal(solver.velocity_u.numpy().copy(), u_second))


class _KinematicBodySolver(SolverBase, CouplingInterface):
    """Test source solver: integrates body_q from a fixed body_qd, records body_f."""

    def __init__(self, model):
        super().__init__(model)
        self.received_body_f = []

    def step(self, state_in, state_out, control, contacts, dt):
        del control, contacts
        self.received_body_f.append(state_in.body_f.numpy().copy())
        body_q = state_in.body_q.numpy()
        body_qd = state_in.body_qd.numpy()
        for b in range(body_q.shape[0]):
            body_q[b, :3] = body_q[b, :3] + dt * body_qd[b, :3]
        state_out.body_q.assign(body_q)
        state_out.body_qd.assign(body_qd)


def test_proxy_coupled_end_to_end(test: unittest.TestCase, device):
    """Full SolverCoupledProxy round trip: pose transfer in, wrench feedback out.

    Verifies that the rigid source receives hydrodynamic feedback forces and
    that multiple coupling iterations do not advance the fluid state further
    than a single iteration (fluid-state restoration).
    """
    results = {}
    for iterations in (1, 3):
        builder = newton.ModelBuilder(gravity=-GRAVITY)
        b = builder.add_body(xform=wp.transform(wp.vec3(0.5, 0.5, 0.5), wp.quat_identity()), mass=10.0)
        shape = builder.add_shape_sphere(b, radius=0.2)
        model = builder.finalize(device=device)

        sources = []

        def make_source(view, sources=sources):
            s = _KinematicBodySolver(view)
            sources.append(s)
            return s

        fluid_cfg = SolverMACFluid.Config(resolution=(16, 16, 16), cell_size=1.0 / 16.0, pressure_iterations=80)

        coupled = SolverCoupledProxy(
            model=model,
            entries=[
                SolverCoupled.Entry(name="rigid", solver=make_source, bodies=[b], shapes=[shape]),
                SolverCoupled.Entry(
                    name="fluid",
                    solver=lambda v, cfg=fluid_cfg: SolverMACFluid(v, cfg),
                    in_place=True,
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="rigid",
                        destination="fluid",
                        bodies=[b],
                        collision_pipeline=lambda _model: None,
                    )
                ],
                iterations=iterations,
            ),
        )

        state_0 = model.state()
        state_1 = model.state()
        dt = 1.0 / 60.0
        for _ in range(2):
            state_0.clear_forces()
            coupled.step(state_0, state_1, None, None, dt)
            state_0, state_1 = state_1, state_0

        fluid_solver = coupled.solver("fluid")
        results[iterations] = {
            "u": fluid_solver.velocity_u.numpy().copy(),
            "body_f": sources[0].received_body_f,
            "impulse": fluid_solver.body_impulse.numpy().copy(),
        }

    # hydrodynamic feedback reaches the rigid source after the first pass
    later_forces = np.array(results[1]["body_f"][1:])
    test.assertGreater(np.abs(later_forces).max(), 1.0, "rigid solver must receive fluid feedback")
    # buoyancy dominates: upward force on the submerged sphere
    test.assertGreater(results[1]["impulse"][0][2], 0.0)

    # more coupling iterations must not advance the fluid extra physical steps:
    # the fluid state remains comparable between 1 and 3 iterations (it would
    # roughly triple its gravity-driven transient if iterations accumulated)
    u1 = np.abs(results[1]["u"]).max()
    u3 = np.abs(results[3]["u"]).max()
    test.assertLess(abs(u3 - u1), 0.5 * max(u1, 1.0e-6))


def test_cpu_cuda_consistency(test: unittest.TestCase, device):
    """The same scenario must produce consistent results on CPU and the test device."""
    if device.is_cpu:
        return

    def run(dev):
        model, _ = _make_tank_model(dev, bodies=[((0.5, 0.5, 0.6), 0.15, 10.0)])
        solver = _make_solver(model, res=16, iters=100, viscosity=1.0e-4)
        s0, s1 = model.state(), model.state()
        qd = np.zeros((1, 6), dtype=np.float32)
        qd[0, 2] = -0.3
        s0.body_qd.assign(qd)
        for _ in range(3):
            solver.step(s0, s1, None, None, 1.0 / 120.0)
        return solver.velocity_u.numpy(), solver.body_impulse.numpy()

    u_dev, imp_dev = run(device)
    u_cpu, imp_cpu = run("cpu")

    np.testing.assert_allclose(u_dev, u_cpu, atol=5.0e-5)
    np.testing.assert_allclose(imp_dev, imp_cpu, rtol=1.0e-3, atol=1.0e-4)


def test_cuda_graph_capture(test: unittest.TestCase, device):
    """Solver steps must be capturable and replayable in a CUDA graph."""
    if device.is_cpu:
        return

    model, _ = _make_tank_model(device, bodies=[((0.5, 0.5, 0.5), 0.15, 10.0)])
    solver = _make_solver(model, res=16, iters=60, viscosity=1.0e-4)
    s0, s1 = model.state(), model.state()
    dt = 1.0 / 60.0

    # warm up module loads outside of capture
    solver.step(s0, s1, None, None, dt)
    solver.reset(s0)

    with wp.ScopedDevice(device):
        with wp.ScopedCapture() as capture:
            solver.step(s0, s1, None, None, dt)
        for _ in range(5):
            wp.capture_launch(capture.graph)

    d = solver.read_diagnostics()
    test.assertTrue(np.isfinite(solver.velocity_u.numpy()).all())
    test.assertTrue(np.isfinite(d["body_wrench"]).all())
    test.assertLess(d["div_l2_post"], 1.0e-2)
    test.assertGreater(d["body_wrench"][0][2], 0.0)  # buoyancy


devices = get_test_devices()


class TestSolverMACFluid(unittest.TestCase):
    pass


add_function_test(
    TestSolverMACFluid, "test_interpolation_linear_field", test_interpolation_linear_field, devices=devices
)
add_function_test(
    TestSolverMACFluid, "test_projection_reduces_divergence", test_projection_reduces_divergence, devices=devices
)
add_function_test(TestSolverMACFluid, "test_closed_domain_null_space", test_closed_domain_null_space, devices=devices)
add_function_test(
    TestSolverMACFluid, "test_viscous_diffusion_operator", test_viscous_diffusion_operator, devices=devices
)
add_function_test(
    TestSolverMACFluid, "test_viscosity_dissipates_energy", test_viscosity_dissipates_energy, devices=devices
)
add_function_test(TestSolverMACFluid, "test_stationary_body_buoyancy", test_stationary_body_buoyancy, devices=devices)
add_function_test(TestSolverMACFluid, "test_translating_body_drag", test_translating_body_drag, devices=devices)
add_function_test(TestSolverMACFluid, "test_rotating_body_torque", test_rotating_body_torque, devices=devices)
add_function_test(TestSolverMACFluid, "test_momentum_balance", test_momentum_balance, devices=devices)
add_function_test(
    TestSolverMACFluid,
    "test_harvest_proxy_wrench_matches_body_impulse",
    test_harvest_proxy_wrench_matches_body_impulse,
    devices=devices,
)
add_function_test(TestSolverMACFluid, "test_coupled_iteration_restore", test_coupled_iteration_restore, devices=devices)
add_function_test(TestSolverMACFluid, "test_proxy_coupled_end_to_end", test_proxy_coupled_end_to_end, devices=devices)
add_function_test(TestSolverMACFluid, "test_cpu_cuda_consistency", test_cpu_cuda_consistency, devices=devices)
add_function_test(TestSolverMACFluid, "test_cuda_graph_capture", test_cuda_graph_capture, devices=devices)


if __name__ == "__main__":
    unittest.main()
