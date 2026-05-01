# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp
from asv_runner.benchmarks.mark import SkipNotImplemented, skip_benchmark_if

wp.config.enable_backward = False
wp.config.quiet = True

import importlib

import newton.examples
from newton.viewer import ViewerNull

ISAACGYM_ENVS_REPO_URL = "https://github.com/isaac-sim/IsaacGymEnvs.git"
ISAACGYM_NUT_BOLT_FOLDER = "assets/factory/mesh/factory_nut_bolt"
DENSE_PYRAMID_NUM_PYRAMIDS = 6
DENSE_PYRAMID_SIZE = 28

try:
    from newton.examples import download_external_git_folder as _download_external_git_folder
except ImportError:
    from newton._src.utils.download_assets import download_git_folder as _download_external_git_folder


def _import_example_class(module_names: list[str]):
    """Import and return the ``Example`` class from candidate modules.

    Args:
        module_names: Ordered module names to try importing.

    Returns:
        The first successfully imported module's ``Example`` class.

    Raises:
        SkipNotImplemented: If none of the module names can be imported.
    """
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        return module.Example

    raise SkipNotImplemented


def _build_pyramid_example(
    num_frames: int,
    *,
    num_pyramids: int | None = None,
    pyramid_size: int | None = None,
    test_mode: bool | None = None,
):
    """Build the pyramid contact example with optional argument overrides."""
    example_cls = _import_example_class(
        [
            "newton.examples.contacts.example_pyramid",
        ]
    )

    if hasattr(newton.examples, "default_args") and hasattr(example_cls, "create_parser"):
        args = newton.examples.default_args(example_cls.create_parser())
        if num_pyramids is not None:
            args.num_pyramids = num_pyramids
        if pyramid_size is not None:
            args.pyramid_size = pyramid_size
        if test_mode is not None:
            args.test = test_mode
        return example_cls(ViewerNull(num_frames=num_frames), args)

    kwargs = {
        "viewer": ViewerNull(num_frames=num_frames),
        "solver": "xpbd",
        "test_mode": False if test_mode is None else test_mode,
    }
    if num_pyramids is not None:
        kwargs["num_pyramids"] = num_pyramids
    if pyramid_size is not None:
        kwargs["pyramid_size"] = pyramid_size
    return example_cls(**kwargs)


class FastExampleContactSdfDefaults:
    """Benchmark the SDF nut-bolt example default configuration."""

    repeat = 2
    number = 1

    def setup_cache(self):
        _download_external_git_folder(ISAACGYM_ENVS_REPO_URL, ISAACGYM_NUT_BOLT_FOLDER)

    def setup(self):
        example_cls = _import_example_class(
            [
                "newton.examples.contacts.example_nut_bolt_sdf",
            ]
        )
        self.num_frames = 20
        if hasattr(newton.examples, "default_args") and hasattr(example_cls, "create_parser"):
            args = newton.examples.default_args(example_cls.create_parser())
            self.example = example_cls(ViewerNull(num_frames=self.num_frames), args)
        else:
            self.example = example_cls(
                viewer=ViewerNull(num_frames=self.num_frames),
                world_count=100,
                num_per_world=1,
                scene="nut_bolt",
                solver="mujoco",
                test_mode=False,
            )

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self):
        for _ in range(self.num_frames):
            self.example.step()
        wp.synchronize_device()


class FastExampleContactHydroWorkingDefaults:
    """Benchmark the hydroelastic nut-bolt example default configuration."""

    repeat = 2
    number = 1

    def setup_cache(self):
        _download_external_git_folder(ISAACGYM_ENVS_REPO_URL, ISAACGYM_NUT_BOLT_FOLDER)

    def setup(self):
        example_cls = _import_example_class(
            [
                "newton.examples.contacts.example_nut_bolt_hydro",
            ]
        )
        self.num_frames = 20
        if hasattr(newton.examples, "default_args") and hasattr(example_cls, "create_parser"):
            args = newton.examples.default_args(example_cls.create_parser())
            self.example = example_cls(ViewerNull(num_frames=self.num_frames), args)
        else:
            self.example = example_cls(
                viewer=ViewerNull(num_frames=self.num_frames),
                world_count=20,
                num_per_world=1,
                scene="nut_bolt",
                solver="mujoco",
                test_mode=False,
            )

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self):
        for _ in range(self.num_frames):
            self.example.step()
        wp.synchronize_device()


class FastExampleContactPyramidDefaults:
    """Benchmark the box pyramid example with default configuration."""

    repeat = 2
    number = 1

    def setup(self):
        self.num_frames = 20
        self.example = _build_pyramid_example(self.num_frames)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self):
        for _ in range(self.num_frames):
            self.example.step()
        wp.synchronize_device()


class FastExampleContactPyramidDense:
    """Benchmark a denser pyramid stack for contact-heavy solver experiments."""

    repeat = 2
    number = 1

    def setup(self):
        self.num_frames = 10
        self.example = _build_pyramid_example(
            self.num_frames,
            num_pyramids=DENSE_PYRAMID_NUM_PYRAMIDS,
            pyramid_size=DENSE_PYRAMID_SIZE,
            test_mode=True,
        )

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self):
        for _ in range(self.num_frames):
            self.example.step()
        wp.synchronize_device()

    def track_contact_count(self):
        return int(self.example.contacts.rigid_contact_count.numpy()[0])


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "FastExampleContactSdfDefaults": FastExampleContactSdfDefaults,
        "FastExampleContactHydroWorkingDefaults": FastExampleContactHydroWorkingDefaults,
        "FastExampleContactPyramidDefaults": FastExampleContactPyramidDefaults,
        "FastExampleContactPyramidDense": FastExampleContactPyramidDense,
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
