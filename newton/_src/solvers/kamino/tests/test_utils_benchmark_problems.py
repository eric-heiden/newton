# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

from newton._src.solvers.kamino._src.utils.benchmark.configs import make_benchmark_configs
from newton._src.solvers.kamino._src.utils.benchmark.problems import (
    BenchmarkProblemNameToConfigFn,
    make_benchmark_problems,
)
from newton._src.solvers.kamino._src.utils.benchmark.runner import (
    BenchmarkSim,
    NewtonExampleBenchmarkSim,
    make_benchmark_simulator,
)
from newton._src.solvers.kamino._src.utils.sim import Simulator


class TestBenchmarkProblems(unittest.TestCase):
    def test_registry_contains_humanoid_problem(self):
        self.assertIn("humanoid", BenchmarkProblemNameToConfigFn)
        self.assertIn("allegro_hand", BenchmarkProblemNameToConfigFn)
        self.assertIn("heterogeneous", BenchmarkProblemNameToConfigFn)
        self.assertIn("cloth_hanging", BenchmarkProblemNameToConfigFn)
        self.assertIn("fourbar", BenchmarkProblemNameToConfigFn)

    def test_fourbar_problem_builds_builder(self):
        problem_set = make_benchmark_problems(names=["fourbar"], num_worlds=1)
        problem, control, camera = problem_set["fourbar"]

        self.assertIsNotNone(control)
        self.assertIsNotNone(camera)

        builder = problem.factory()
        self.assertGreater(builder.num_bodies, 0)
        self.assertGreater(builder.num_joints, 0)

    def test_cloth_problem_builds_newton_runtime_and_steps(self):
        sim_config = Simulator.Config()
        problem_set = make_benchmark_problems(names=["cloth_hanging"], num_worlds=1)
        problem, control, camera = problem_set["cloth_hanging"]

        self.assertIsNone(control)
        self.assertIsNone(camera)

        simulator = make_benchmark_simulator(
            problem=problem,
            configs=sim_config,
            device="cpu",
            max_steps=2,
            use_cuda_graph=False,
            viewer=False,
        )

        self.assertIsInstance(simulator, NewtonExampleBenchmarkSim)
        self.assertEqual(simulator.device.alias, "cpu")
        self.assertIsNone(simulator.viewer)
        self.assertGreater(simulator.model.particle_count, 0)

        simulator.step_once()
        self.assertEqual(simulator.step_count, 1)

    def test_fourbar_problem_builds_kamino_runtime(self):
        sim_config = Simulator.Config()
        problem_set = make_benchmark_problems(names=["fourbar"], num_worlds=1)
        problem, _control, _camera = problem_set["fourbar"]

        simulator = make_benchmark_simulator(
            problem=problem,
            configs=sim_config,
            device="cpu",
            max_steps=2,
            use_cuda_graph=False,
            viewer=False,
        )

        self.assertIsInstance(simulator, BenchmarkSim)

    def test_heterogeneous_problem_builds_kamino_runtime(self):
        sim_config = Simulator.Config()
        problem_set = make_benchmark_problems(names=["heterogeneous"], num_worlds=1)
        problem, control, camera = problem_set["heterogeneous"]

        self.assertIsNone(control)
        self.assertIsNotNone(camera)

        simulator = make_benchmark_simulator(
            problem=problem,
            configs=sim_config,
            device="cpu",
            max_steps=1,
            use_cuda_graph=False,
            viewer=False,
        )

        self.assertIsInstance(simulator, BenchmarkSim)
        self.assertGreater(simulator.builder.num_bodies, 0)

    def test_allegro_hand_problem_builds_newton_example_config(self):
        problem_set = make_benchmark_problems(names=["allegro_hand"], num_worlds=2)
        problem, control, camera = problem_set["allegro_hand"]

        self.assertEqual(problem.runtime, "newton_example")
        self.assertIsNone(control)
        self.assertIsNotNone(camera)

        example_config = problem.factory()
        self.assertIsNotNone(example_config.args_factory)
        args = example_config.args_factory()
        self.assertEqual(args.world_count, 2)

    def test_benchmark_configs_validate_with_simulator_config(self):
        for config in make_benchmark_configs(include_default=False).values():
            simulator_config = Simulator.Config(dt=0.001, solver=config)
            self.assertIn(simulator_config.solver.dynamics.linear_solver_type, {"LLTB", "CR"})
