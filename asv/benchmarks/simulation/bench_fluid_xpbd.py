# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure captured fluid stepping across independently simulated worlds."""

import importlib

import numpy as np
import warp as wp
from asv_runner.benchmarks.mark import SkipNotImplemented

wp.config.enable_backward = False
wp.config.log_level = wp.LOG_WARNING


class _FluidWorlds:
    number = 1
    repeat = 3
    rounds = 1
    param_names = ["world_count"]
    frames = 120
    particle_count = 1024

    def setup(self, world_count):
        if not wp.get_device().is_cuda:
            raise SkipNotImplemented

        from newton.viewer import ViewerNull  # noqa: PLC0415

        example_type = importlib.import_module(f"newton.examples.fluid.example_{self.scene}").Example
        args = example_type.create_parser().parse_args(
            ["--world-count", str(world_count), "--particle-count", str(self.particle_count)]
        )
        self.example = example_type(ViewerNull(), args)
        for _ in range(3):
            self.example.step()
        wp.synchronize_device(self.example.model.device)

    def time_simulate(self, world_count):
        for _ in range(self.frames):
            self.example.step()
        wp.synchronize_device(self.example.model.device)

    def teardown(self, world_count):
        for name in ("particle_q", "particle_qd", "body_q", "body_qd"):
            array = getattr(self.example.state_0, name)
            if array is not None and not np.isfinite(array.numpy()).all():
                raise RuntimeError(f"Nonfinite {name} in {self.scene}")
        self.example.viewer.close()


class FluidXPBDWorlds(_FluidWorlds):
    """Track fluid-neighbor and collision scaling with overlapping worlds."""

    scene = "fluid_dam_break"
    params = [1, 8, 32]
    particle_count = 2048


class FluidCupTransferWorlds(_FluidWorlds):
    """Include batched IK, fluid motion, observations, and rewards."""

    scene = "fluid_cup_transfer"
    params = [1, 4, 16]
