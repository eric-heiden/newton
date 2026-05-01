# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import argparse
import gc
import time

import warp as wp

import newton
import newton.examples

from ....examples import print_progress_bar
from ...core.builder import ModelBuilderKamino
from ...utils import logger as msg
from ...utils.control.rand import RandomJointController
from ...utils.device import get_device_malloc_info
from ...utils.sim import SimulationLogger, Simulator, ViewerKamino
from .metrics import BenchmarkMetrics
from .problems import BenchmarkProblemDefinition, CameraConfig, ControlConfig, NewtonExampleConfig, ProblemDimensions

###
# Types
###


class BenchmarkSim:
    def __init__(
        self,
        builder: ModelBuilderKamino,
        configs: Simulator.Config,
        control: ControlConfig | None = None,
        camera: CameraConfig | None = None,
        device: wp.DeviceLike = None,
        use_cuda_graph: bool = False,
        max_steps: int = 1000,
        seed: int = 0,
        viewer: bool = False,
        logging: bool = False,
        physics_metrics: bool = False,
    ):
        # Cache the device and other internal flags
        self.builder: ModelBuilderKamino = builder
        self.device = wp.get_device(device)
        self.use_cuda_graph: bool = use_cuda_graph
        self.max_steps: int = max_steps

        # Override the default compute_solution_metrics toggle in the
        # simulator configs based on the benchmark configuration
        configs.solver.compute_solution_metrics = physics_metrics

        # Create a simulator
        msg.info("Building the simulator...")
        with wp.ScopedDevice(self.device):
            self.sim = Simulator(builder=builder, config=configs, device=self.device)

        if control is None or not control.disable_controller:
            # Create a random-action controller for the model
            self.ctlr = RandomJointController(
                model=self.sim.model,
                seed=seed,
                decimation=control.decimation if control else None,
                scale=control.scale if control else None,
            )

            # Define a callback function to wrap the execution of the controller
            def control_callback(simulator: Simulator):
                self.ctlr.compute(
                    time=simulator.solver.data.time,
                    control=simulator.control,
                )

            # Set the control callbacks into the simulator
            self.sim.set_control_callback(control_callback)

        # Initialize the data logger
        self.logger: SimulationLogger | None = None
        if logging:
            msg.info("Creating the sim data logger...")
            self.logger = SimulationLogger(self.max_steps, self.sim, builder)

        # Initialize the 3D viewer
        self.viewer: ViewerKamino | None = None
        if viewer:
            msg.info("Creating the 3D viewer...")
            self.viewer = ViewerKamino(
                builder=self.builder,
                simulator=self.sim,
            )
            if hasattr(self.viewer, "set_camera") and camera is not None:
                self.viewer.set_camera(wp.vec3(*camera.position), camera.pitch, camera.yaw)

        # Declare and initialize the optional computation graphs
        # NOTE: These are used for most efficient GPU runtime
        self.reset_graph = None
        self.step_graph = None

        # Capture CUDA graph if requested and available
        self._capture()

        # Warm-start the simulator before rendering
        # NOTE: This compiles and loads the warp kernels prior to execution
        msg.info("Warming up simulator...")
        self.step_once()
        self.reset()

    ###
    # Operations
    ###

    def reset(self):
        with wp.ScopedDevice(self.device):
            if self.reset_graph:
                wp.capture_launch(self.reset_graph)
            else:
                self.sim.reset()
            if not self.use_cuda_graph and self.logger:
                self.logger.reset()
                self.logger.log()

    def step(self):
        with wp.ScopedDevice(self.device):
            if self.step_graph:
                wp.capture_launch(self.step_graph)
            else:
                self.sim.step()
            if not self.use_cuda_graph and self.logger:
                self.logger.log()

    def step_once(self):
        self.step()

    def render(self):
        if self.viewer:
            self.viewer.render_frame()

    def test(self):
        pass

    def plot(self, path: str | None = None, show: bool = False):
        if self.logger:
            self.logger.plot_solver_info(path=path, show=show)
            self.logger.plot_joint_tracking(path=path, show=show)
            self.logger.plot_solution_metrics(path=path, show=show)

    ###
    # Internals
    ###

    def _capture(self):
        with wp.ScopedDevice(self.device):
            if self.use_cuda_graph:
                msg.info("Running with CUDA graphs...")
                with wp.ScopedCapture(self.device) as reset_capture:
                    self.sim.reset()
                self.reset_graph = reset_capture.graph
                with wp.ScopedCapture(self.device) as step_capture:
                    self.sim.step()
                self.step_graph = step_capture.graph
            else:
                msg.info("Running with kernels...")


class NewtonExampleBenchmarkSim:
    def __init__(
        self,
        example_config: NewtonExampleConfig,
        device: wp.DeviceLike = None,
        max_steps: int = 1000,
        viewer: bool = False,
    ):
        self.device = wp.get_device(device)
        self.max_steps = max_steps
        self._steps = 0
        self._viewer_enabled = viewer

        self._example_viewer = newton.viewer.ViewerNull(num_frames=max_steps)
        self.viewer = self._example_viewer if viewer else None
        with wp.ScopedDevice(self.device):
            args = example_config.args_factory() if example_config.args_factory is not None else None
            self.example = example_config.example_type(
                viewer=self._example_viewer,
                args=args,
                **example_config.kwargs,
            )

    @property
    def model(self) -> newton.Model:
        return self.example.model

    @property
    def step_count(self) -> int:
        return self._steps

    def step(self):
        self.example.step()
        self._steps += 1

    def step_once(self):
        self.step()

    def render(self):
        if self._viewer_enabled:
            self.example.render()
            self.viewer.end_frame()


###
# Functions
###


def run_single_benchmark_with_viewer(
    args: argparse.Namespace,
    simulator: BenchmarkSim | NewtonExampleBenchmarkSim,
) -> tuple[float, float]:
    start_time = time.time()
    if isinstance(simulator, BenchmarkSim):
        newton.examples.run(simulator, args)
    else:
        for _ in range(simulator.max_steps):
            simulator.step_once()
            simulator.render()
            wp.synchronize()
    stop_time = time.time()
    return start_time, stop_time


def run_single_benchmark_with_progress(
    simulator: BenchmarkSim | NewtonExampleBenchmarkSim,
) -> tuple[float, float]:
    start_time = time.time()
    for step_idx in range(simulator.max_steps):
        simulator.step_once()
        wp.synchronize()
        print_progress_bar(step_idx + 1, simulator.max_steps, start_time, prefix="Progress", suffix="")
    stop_time = time.time()
    return start_time, stop_time


def run_single_benchmark_silent(
    simulator: BenchmarkSim | NewtonExampleBenchmarkSim,
) -> tuple[float, float]:
    start_time = time.time()
    for _s in range(simulator.max_steps):
        simulator.step_once()
        wp.synchronize()
    stop_time = time.time()
    return start_time, stop_time


def run_single_benchmark_with_step_metrics(
    problem_idx: int,
    config_idx: int,
    simulator: BenchmarkSim | NewtonExampleBenchmarkSim,
    metrics: BenchmarkMetrics,
) -> tuple[float, float]:
    start_time = time.time()
    step_start_time = float(start_time)
    for step_idx in range(simulator.max_steps):
        simulator.step_once()
        wp.synchronize()
        step_stop_time = time.time()
        solver = simulator.sim.solver if isinstance(simulator, BenchmarkSim) else None
        metrics.record_step(problem_idx, config_idx, step_idx, step_stop_time - step_start_time, solver)
        step_start_time = float(step_stop_time)
    stop_time = time.time()
    return start_time, stop_time


def make_benchmark_simulator(
    problem: BenchmarkProblemDefinition | ModelBuilderKamino,
    configs: Simulator.Config,
    control: ControlConfig | None = None,
    camera: CameraConfig | None = None,
    device: wp.DeviceLike = None,
    use_cuda_graph: bool = True,
    max_steps: int = 1000,
    seed: int = 0,
    viewer: bool = False,
    physics_metrics: bool = False,
) -> BenchmarkSim | NewtonExampleBenchmarkSim:
    if isinstance(problem, BenchmarkProblemDefinition):
        payload = problem.factory()
        if problem.runtime == "kamino":
            if not isinstance(payload, ModelBuilderKamino):
                raise TypeError("Kamino benchmark problems must build a ModelBuilderKamino instance.")
            return BenchmarkSim(
                builder=payload,
                configs=configs,
                control=control,
                camera=camera,
                device=device,
                use_cuda_graph=use_cuda_graph,
                max_steps=max_steps,
                seed=seed,
                viewer=viewer,
                physics_metrics=physics_metrics,
            )

        if problem.runtime == "newton_example":
            if not isinstance(payload, NewtonExampleConfig):
                raise TypeError("Newton benchmark problems must build a NewtonExampleConfig instance.")
            return NewtonExampleBenchmarkSim(
                example_config=payload,
                device=device,
                max_steps=max_steps,
                viewer=viewer,
            )

        raise ValueError(f"Unsupported benchmark runtime '{problem.runtime}'.")

    return BenchmarkSim(
        builder=problem,
        configs=configs,
        control=control,
        camera=camera,
        device=device,
        use_cuda_graph=use_cuda_graph,
        max_steps=max_steps,
        seed=seed,
        viewer=viewer,
        physics_metrics=physics_metrics,
    )


def get_problem_dimensions(simulator: BenchmarkSim | NewtonExampleBenchmarkSim) -> ProblemDimensions:
    if isinstance(simulator, BenchmarkSim):
        model = simulator.sim.model
        return ProblemDimensions(
            num_body_dofs=model.size.max_of_num_body_dofs,
            num_joint_dofs=model.size.max_of_num_joint_dofs,
            min_delassus_dim=model.size.max_of_num_kinematic_joint_cts + model.size.max_of_num_dynamic_joint_cts,
            max_delassus_dim=model.size.max_of_max_total_cts,
        )

    model = simulator.model
    return ProblemDimensions(
        num_body_dofs=int(getattr(model, "body_count", 0) * 6),
        num_joint_dofs=int(getattr(model, "joint_dof_count", 0)),
        min_delassus_dim=-1,
        max_delassus_dim=-1,
    )


def run_single_benchmark(
    problem_idx: int,
    config_idx: int,
    metrics: BenchmarkMetrics,
    args: argparse.Namespace,
    problem: BenchmarkProblemDefinition | ModelBuilderKamino,
    configs: Simulator.Config,
    control: ControlConfig | None = None,
    camera: CameraConfig | None = None,
    device: wp.DeviceLike = None,
    use_cuda_graph: bool = True,
    print_device_info: bool = False,
    progress: bool = False,
):
    simulator = make_benchmark_simulator(
        problem=problem,
        configs=configs,
        control=control,
        camera=camera,
        device=device,
        use_cuda_graph=use_cuda_graph,
        max_steps=args.num_steps,
        seed=args.seed,
        viewer=args.viewer,
        physics_metrics=metrics.physics_metrics is not None,
    )

    msg.info("Starting benchmark run...")
    if simulator.viewer:
        msg.info("Running in Viewer mode...")
        start_time, stop_time = run_single_benchmark_with_viewer(args, simulator)
    else:
        msg.info(f"Running for {simulator.max_steps} steps...")
        if metrics.step_time is not None:
            start_time, stop_time = run_single_benchmark_with_step_metrics(problem_idx, config_idx, simulator, metrics)
        elif progress:
            start_time, stop_time = run_single_benchmark_with_progress(simulator)
        else:
            start_time, stop_time = run_single_benchmark_silent(simulator)
    msg.info("Finished benchmark run.")

    # Record final metrics for the benchmark run
    metrics.record_total(
        problem_idx=problem_idx,
        config_idx=config_idx,
        total_time=stop_time - start_time,
        total_steps=simulator.step_count
        if isinstance(simulator, NewtonExampleBenchmarkSim)
        else int(simulator.sim.solver.data.time.steps.numpy()[0]),
        memory_used=float(wp.get_mempool_used_mem_current(device) if device.is_cuda else 0.0),
    )

    # Record problem dimensions
    problem_name = metrics._problem_names[problem_idx]
    if problem_name not in metrics._problem_dims:
        metrics._problem_dims[problem_name] = get_problem_dimensions(simulator)

    # Optionally also print the total device memory allocated during the benchmark run
    if print_device_info and simulator.device.is_cuda:
        mem_info = get_device_malloc_info(simulator.device)
        msg.info("[Device malloc info]: %s", mem_info)

    # Deallocate simulator to ensure accurate memory consumption measure for the next run
    del simulator
    gc.collect()
    wp.synchronize()
