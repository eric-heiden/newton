# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


def test_mimic_report_video_capture_uses_viewergl_get_frame():
    script = Path("reports/mimic_joints_prototype/benchmark_mimic_assets.py").read_text(encoding="utf-8")

    assert "ViewerGL" in script
    assert ".get_frame(" in script
    assert "matplotlib" not in script
    assert ".scatter(" not in script
    assert ".plot(" not in script


def test_mimic_report_benchmark_includes_solver_step_matrix():
    script = Path("reports/mimic_joints_prototype/benchmark_mimic_assets.py").read_text(encoding="utf-8")

    assert "solver_step_timing" in script
    for solver_name in (
        "SolverSemiImplicit",
        "SolverXPBD",
        "SolverFeatherstone",
        "SolverVBD",
        "SolverMuJoCo",
    ):
        assert solver_name in script
