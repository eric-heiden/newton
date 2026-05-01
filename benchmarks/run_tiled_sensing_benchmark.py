# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Entrypoint for generating tiled sensing benchmark artifacts."""

from newton._src.tools.tiled_sensing_benchmark import main

if __name__ == "__main__":
    raise SystemExit(main())
