# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# BatchRollout.evaluate: 32 cases (8 candidates x 4 scenarios) of a puck on
# a floor with SolverMuJoCo, as one 32-world batch and one case at a time on
# a single-world rollout (the equivalent sequential loop).
###########################################################################

import warp as wp
from asv_runner.benchmarks.mark import skip_benchmark_if

wp.config.enable_backward = False
wp.config.log_level = wp.LOG_WARNING

import numpy as np

import newton
from newton.utils import BatchRollout

CANDIDATES = list(np.linspace(0.1, 1.0, 8))
SCENARIOS = [0.5, 1.0, 2.0, 3.0]
FRAMES = 100


def _build():
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.add_ground_plane(label="floor")
    puck = builder.add_body(xform=wp.transform((0.0, 0.0, 0.05), wp.quat_identity()), label="puck")
    builder.add_shape_box(puck, hx=0.05, hy=0.05, hz=0.05, label="puck_geom")
    return builder


def _setup(world, mu, push):
    world.set_model("shape_material_mu", mu, labels=["puck_geom", "floor"])
    world.set_state("joint_qd", [push, 0.0, 0.0, 0.0, 0.0, 0.0], labels="puck*")


def _score(records, cases):
    x = records["puck"][:, :, 0, 0]
    return {"slide": x[-1] - x[0]}


class BatchRolloutEvaluate:
    repeat = 3
    number = 1

    def setup(self):
        if wp.get_cuda_device_count() == 0:
            return
        self.batched = BatchRollout(_build, 32, solver=newton.solvers.SolverMuJoCo, dt=0.002, substeps=5)
        self.single = BatchRollout(_build, 1, solver=newton.solvers.SolverMuJoCo, dt=0.002, substeps=5)
        self.time_evaluate_batched()
        self.time_evaluate_sequential()

    def _evaluate(self, rollout):
        rollout.evaluate(
            CANDIDATES,
            SCENARIOS,
            frames=FRAMES,
            setup=_setup,
            record={"puck": ("body_q", "puck")},
            every=10,
            score=_score,
        )

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_evaluate_batched(self):
        self._evaluate(self.batched)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_evaluate_sequential(self):
        self._evaluate(self.single)


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {"BatchRolloutEvaluate": BatchRolloutEvaluate}

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-b", "--bench", default=None, action="append", choices=benchmark_list.keys())
    args = parser.parse_known_args()[0]
    for key in args.bench or benchmark_list.keys():
        run_benchmark(benchmark_list[key])
