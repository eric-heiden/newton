# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Live MCP evaluate(): 32 cases (8 frictions x 4 pushes) of a hosted puck
# script with SolverMuJoCo, with the session's kept 32-world model (warm) and
# with a new model built for the call.
###########################################################################

import tempfile
import textwrap
from pathlib import Path

import warp as wp
from asv_runner.benchmarks.mark import skip_benchmark_if

wp.config.enable_backward = False
wp.config.log_level = wp.LOG_WARNING

SCRIPT = textwrap.dedent(
    """
    import warp as wp

    import newton


    class Example:
        def __init__(self, viewer, args):
            builder = newton.ModelBuilder()
            newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
            builder.add_ground_plane(label="floor")
            puck = builder.add_body(xform=wp.transform((0.0, 0.0, 0.05), wp.quat_identity()), label="puck")
            builder.add_shape_box(puck, hx=0.05, hy=0.05, hz=0.05, label="puck_geom")
            self.model = builder.finalize()
            self.solver = newton.solvers.SolverMuJoCo(self.model)
            self.state_0, self.state_1 = self.model.state(), self.model.state()
            self.control = self.model.control()
            self.frame_dt = 0.01
            self.sim_dt = 0.002

        def step(self):
            for _ in range(5):
                self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
                self.state_0, self.state_1 = self.state_1, self.state_0
    """
)

CELL = """
import numpy as np

def setup(world, mu, push):
    world.set_model("shape_material_mu", mu, labels=["puck_geom", "floor"])
    world.set_state("joint_qd", [push, 0.0, 0.0, 0.0, 0.0, 0.0], labels="puck*")

def score(records, cases):
    x = records["puck"][:, :, 0, 0]
    return {"slide": x[-1] - x[0]}

CASES = dict(frames=100, setup=setup, score=score, record={"puck": ("body_q", "puck")}, every=10)
CANDIDATES, SCENARIOS = list(np.linspace(0.1, 1.0, 8)), [0.5, 1.0, 2.0, 3.0]
"""


class McpLiveEvaluate:
    repeat = 3
    number = 1

    def setup(self):
        if wp.get_cuda_device_count() == 0:
            return
        from newton.mcp import ExampleHost  # noqa: PLC0415

        self.directory = tempfile.TemporaryDirectory()
        script = Path(self.directory.name) / "puck.py"
        script.write_text(SCRIPT)
        self.session = ExampleHost(script).session(artifact_directory=self.directory.name)
        self.session.dispatch("execute", {"code": CELL})
        self.time_evaluate_warm()

    def teardown(self):
        if wp.get_cuda_device_count() == 0:
            return
        self.session.close()
        self.directory.cleanup()

    def _evaluate(self):
        self.session.dispatch("execute", {"code": "evaluate(CANDIDATES, SCENARIOS, **CASES)"})

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_evaluate_warm(self):
        self._evaluate()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_evaluate_new_model(self):
        self.session._batch().close()
        self._evaluate()


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {"McpLiveEvaluate": McpLiveEvaluate}

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-b", "--bench", default=None, action="append", choices=benchmark_list.keys())
    args = parser.parse_known_args()[0]
    for key in args.bench or benchmark_list.keys():
        run_benchmark(benchmark_list[key])
