# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compatibility entry point for :mod:`newton.examples.fluid.example_fluid_water_surface`."""

import newton.examples
from newton.examples.fluid.example_fluid_water_surface import Example

if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
