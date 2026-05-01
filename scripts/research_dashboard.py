# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Entrypoint for the local research dashboard server."""

from newton._src.tools.research_dashboard import main

if __name__ == "__main__":
    raise SystemExit(main())
