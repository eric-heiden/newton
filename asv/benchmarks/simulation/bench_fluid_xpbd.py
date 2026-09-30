# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure captured fluid simulation separately from surface rendering."""

import importlib

import warp as wp


class FastExampleFluidXPBD:
    params = (["dam_break", "archimedes_screw"], [10_000, 100_000])
    param_names = ["scene", "particle_count"]
    repeat = 3
    number = 1
    timeout = 180

    def setup(self, scene, particle_count):
        if not wp.is_cuda_available():
            raise NotImplementedError("Fluid performance benchmarks require CUDA")

        import newton.examples  # noqa: PLC0415
        import newton.viewer  # noqa: PLC0415

        wp.config.enable_backward = False
        wp.config.log_level = wp.LOG_WARNING
        module = importlib.import_module(f"newton.examples.fluid.example_fluid_xpbd_{scene}")
        args = module.Example.create_parser().parse_args(["--particle-count", str(particle_count)])
        with wp.ScopedDevice("cuda:0"):
            self.example = module.Example(newton.viewer.ViewerNull(), args)
            # Compile and capture outside the timed interval.
            for _ in range(3):
                self.example.step()
            wp.synchronize_device(self.example.model.device)

    def time_simulate(self, scene, particle_count):
        for _ in range(120):
            self.example.step()
        wp.synchronize_device(self.example.model.device)

    def teardown(self, scene, particle_count):
        self.example.viewer.close()
