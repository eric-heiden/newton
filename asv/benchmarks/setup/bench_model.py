# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import itertools
import time
import numpy as np

import numpy as np
import warp as wp

wp.config.enable_backward = False
wp.config.quiet = True

from newton.examples.example_mujoco import Example
from newton.solvers import SolverMuJoCo

nconmax = {
    "humanoid": None,
    "g1": 150,
    "h1": 150,
    "cartpole": None,
    "ant": None,
    "quadruped": None,
}

class KpiInitializeModel:
    params = (["humanoid", "g1", "h1", "cartpole", "ant", "quadruped"], [4096, 8192])
    param_names = ["robot", "num_envs"]

    rounds = 1
    number = 1
    repeat = 3
    min_run_count = 1
    timeout = 3600

    def setup_cache(self):
        wp.init()

        if wp.get_cuda_device_count() == 0:
            raise NotImplementedError("CUDA is not available")

        timings = {}
        timings["modelBuilder"] = {}
        timings["solver"] = {}

        for robot, num_envs in itertools.product(self.params[0], self.params[1]):
            builder = Example.create_model_builder(robot, num_envs, randomize=True, seed=123)
            # finalize model
            _model = builder.finalize()

            # Load the model to cache the kernels
            solver = SolverMuJoCo(_model, ncon_per_env=nconmax[robot])
            del solver

            wp.synchronize_device()

            timings_modelBuilder = []
            timings_solver = []

            for _ in range(self.repeat):

                modelBuilder_beg = time.perf_counter()
                builder = Example.create_model_builder(robot, num_envs, randomize=True, seed=123)

                # finalize model
                model = builder.finalize()
                wp.synchronize_device()
                modelBuilder_end = time.perf_counter()
                timings_modelBuilder.append(modelBuilder_end - modelBuilder_beg)

            
                solver_beg = time.perf_counter()
                solver = SolverMuJoCo(model, ncon_per_env=nconmax[robot])
                wp.synchronize_device()
                solver_end = time.perf_counter()
                timings_solver.append(solver_end - solver_beg)

                del solver
                del model

            timings["modelBuilder"][(robot, num_envs)] = np.median(timings_modelBuilder) * 1000
            timings["solver"][(robot, num_envs)] = np.median(timings_solver) * 1000

        return timings

    def track_time_initialize_solverMuJoCo(self, timings, robot, num_envs):
        return timings["solver"][(robot, num_envs)]
    track_time_initialize_solverMuJoCo.unit = "ms"
    
    def track_time_initialize_model(self, timings, robot, num_envs):
        return timings["modelBuilder"][(robot, num_envs)]
    track_time_initialize_model.unit = "ms"

class FastInitializeModel:
    params = (["humanoid", "g1", "h1", "cartpole", "ant", "quadruped"], [128, 256])
    param_names = ["robot", "num_envs"]

    rounds = 1
    number = 1
    repeat = 3
    min_run_count = 1

    def setup_cache(self):
        wp.init()

        if wp.get_cuda_device_count() == 0:
            raise NotImplementedError("CUDA is not available")

        timings = {}
        timings["modelBuilder"] = {}
        timings["solver"] = {}

        for robot, num_envs in itertools.product(self.params[0], self.params[1]):
            builder = Example.create_model_builder(robot, num_envs, randomize=True, seed=123)
            # finalize model
            _model = builder.finalize()

            # Load the model to cache the kernels
            solver = SolverMuJoCo(_model, ncon_per_env=nconmax[robot])
            del solver

            wp.synchronize_device()

            timings_modelBuilder = []
            timings_solver = []

            for _ in range(self.repeat):

                modelBuilder_beg = time.perf_counter()
                builder = Example.create_model_builder(robot, num_envs, randomize=True, seed=123)

                # finalize model
                model = builder.finalize()
                wp.synchronize_device()
                modelBuilder_end = time.perf_counter()
                timings_modelBuilder.append(modelBuilder_end - modelBuilder_beg)

            
                solver_beg = time.perf_counter()
                solver = SolverMuJoCo(model, ncon_per_env=nconmax[robot])
                wp.synchronize_device()
                solver_end = time.perf_counter()
                timings_solver.append(solver_end - solver_beg)

                del solver
                del model

            timings["modelBuilder"][(robot, num_envs)] = np.median(timings_modelBuilder) * 1000
            timings["solver"][(robot, num_envs)] = np.median(timings_solver) * 1000

        return timings

    def track_time_initialize_solverMuJoCo(self, timings, robot, num_envs):
        unit = "ms"
        return timings["solver"][(robot, num_envs)]
    track_time_initialize_solverMuJoCo.unit = "ms"
    
    def track_time_initialize_model(self, timings, robot, num_envs):
        unit = "ms"
        return timings["modelBuilder"][(robot, num_envs)]
    track_time_initialize_model.unit = "ms"

    def peakmem_initialize_model_cpu(self, _, robot, num_envs):
        gc.collect()

        with wp.ScopedDevice("cpu"):
            builder = Example.create_model_builder(robot, num_envs, randomize=True, seed=123)

            # finalize model
            model = builder.finalize()

        del model


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "KpiInitializeModel": KpiInitializeModel,
        "FastInitializeModel": FastInitializeModel,
    }

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-b", "--bench", default=None, action="append", choices=benchmark_list.keys(), help="Run a single benchmark."
    )
    args = parser.parse_known_args()[0]

    if args.bench is None:
        benchmarks = benchmark_list.keys()
    else:
        benchmarks = args.bench

    for key in benchmarks:
        benchmark = benchmark_list[key]
        run_benchmark(benchmark)
