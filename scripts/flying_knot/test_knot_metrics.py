# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the knot metrics used by example_cable_flying_knot.

Run: uv run python scripts/flying_knot/test_knot_metrics.py
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "newton" / "examples" / "cable"))

from example_cable_flying_knot import count_crossings, polyline_writhe


def trefoil(n=400, closed=True):
    t = np.linspace(0, 2 * np.pi, n, endpoint=not closed)
    x = np.sin(t) + 2 * np.sin(2 * t)
    y = np.cos(t) - 2 * np.cos(2 * t)
    z = -np.sin(3 * t)
    return np.stack([x, y, z], axis=1)


def open_overhand(n=400):
    """Open overhand knot: trefoil with an arc removed, ends extended outward."""
    t = np.linspace(0.25, 2 * np.pi - 0.25, n)
    x = np.sin(t) + 2 * np.sin(2 * t)
    y = np.cos(t) - 2 * np.cos(2 * t)
    z = -np.sin(3 * t)
    pts = np.stack([x, y, z], axis=1)
    # Extend both ends away from the knot to mimic a hanging rope's free ends.
    d0 = pts[0] - pts[1]
    d1 = pts[-1] - pts[-2]
    pre = pts[0] + np.outer(np.linspace(3.0, 0.1, 30), d0 / np.linalg.norm(d0))
    post = pts[-1] + np.outer(np.linspace(0.1, 3.0, 30), d1 / np.linalg.norm(d1))
    return np.concatenate([pre, pts, post], axis=0)


class TestWrithe(unittest.TestCase):
    def test_straight_line_zero(self):
        pts = np.stack([np.zeros(50), np.zeros(50), np.linspace(0, 1, 50)], axis=1)
        self.assertAlmostEqual(polyline_writhe(pts), 0.0, places=6)

    def test_planar_circle_zero(self):
        t = np.linspace(0, 2 * np.pi, 100)
        pts = np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], axis=1)
        self.assertLess(abs(polyline_writhe(pts)), 1e-6)

    def test_closed_trefoil(self):
        wr = polyline_writhe(trefoil())
        self.assertGreater(abs(wr), 3.0)
        self.assertLess(abs(wr), 3.8)

    def test_open_overhand(self):
        wr = polyline_writhe(open_overhand())
        self.assertGreater(abs(wr), 2.2)

    def test_hanging_slack_curve(self):
        # Gentle non-planar S-curve, no knot: writhe stays small.
        t = np.linspace(0, 1, 100)
        pts = np.stack([0.1 * np.sin(4 * t), 0.1 * np.cos(5 * t), -t], axis=1)
        self.assertLess(abs(polyline_writhe(pts)), 0.5)


class TestCrossings(unittest.TestCase):
    def test_straight_line(self):
        pts = np.stack([np.linspace(0, 1, 30), np.zeros(30), np.zeros(30)], axis=1)
        self.assertEqual(count_crossings(pts, axis=2), 0)

    def test_trefoil_projection(self):
        # A generic projection of a trefoil has at least 3 crossings.
        n = count_crossings(trefoil(), axis=2)
        self.assertGreaterEqual(n, 3)

    def test_open_overhand_projection(self):
        n = count_crossings(open_overhand(), axis=2)
        self.assertGreaterEqual(n, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
