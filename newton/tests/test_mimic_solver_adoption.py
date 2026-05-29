# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherstone, SolverXPBD


def _make_mimic_chain_model() -> newton.Model:
    builder = newton.ModelBuilder()
    leader_body = builder.add_link(mass=1.0, label="leader_body")
    follower_body = builder.add_link(mass=1.0, label="follower_body")
    leader_joint = builder.add_joint_revolute(
        parent=-1,
        child=leader_body,
        axis=(0.0, 0.0, 1.0),
        label="leader_joint",
    )
    follower_joint = builder.add_joint_mimic(
        parent=leader_body,
        child=follower_body,
        leader_joint=leader_joint,
        coef0=0.2,
        coef1=2.0,
        axis=(0.0, 0.0, 1.0),
        mimic_type=newton.JointType.REVOLUTE,
        label="follower_joint",
    )
    builder.add_articulation([leader_joint, follower_joint])
    return builder.finalize()


def test_featherstone_fk_refresh_evaluates_mimic_followers():
    model = _make_mimic_chain_model()
    state_in = model.state()
    expected = model.state()
    state_out = model.state()

    q = state_in.joint_q.numpy()
    qd = state_in.joint_qd.numpy()
    q[0] = 0.3
    qd[0] = 0.4
    state_in.joint_q.assign(q)
    state_in.joint_qd.assign(qd)
    expected.joint_q.assign(q)
    expected.joint_qd.assign(qd)

    newton.eval_fk(model, expected.joint_q, expected.joint_qd, expected)
    solver = SolverFeatherstone(model)
    solver.step(state_in, state_out, model.control(), model.contacts(), 0.0)
    wp.synchronize()

    follower = model.body_label.index("follower_body")
    np.testing.assert_allclose(
        state_out.body_q.numpy()[follower],
        expected.body_q.numpy()[follower],
        atol=1.0e-5,
    )


def test_xpbd_joint_force_path_accepts_mimic_followers(capsys):
    model = _make_mimic_chain_model()
    state_in = model.state()
    state_out = model.state()

    q = state_in.joint_q.numpy()
    q[0] = 0.3
    state_in.joint_q.assign(q)
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)

    solver = SolverXPBD(model)
    solver.step(state_in, state_out, model.control(), model.contacts(), 1.0 / 60.0)
    wp.synchronize()

    assert "joint type not handled in apply_joint_forces" not in capsys.readouterr().out


def test_xpbd_joint_force_kernel_explicitly_skips_mimic_followers():
    kernel_source = Path("newton/_src/solvers/xpbd/kernels.py").read_text(encoding="utf-8")
    assert "JointType.MIMIC" in kernel_source
