# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Entrypoint for generating solver benchmark dashboard artifacts."""

from newton._src.tools.solver_benchmark_matrix import main


if __name__ == "__main__":
    raise SystemExit(main())
