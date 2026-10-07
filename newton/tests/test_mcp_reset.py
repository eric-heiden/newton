# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reset and restore of hosted examples: rewind what steps write, keep and report cell edits."""

import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

from newton.mcp import ExampleHost

_SCRIPT = textwrap.dedent(
    """
    import numpy as np
    import warp as wp

    import newton


    @wp.kernel
    def drive(schedule: wp.array[float], cursor: wp.array[wp.int32], body_qd: wp.array[wp.spatial_vector]):
        i = cursor[0]
        body_qd[0] = wp.spatial_vector(wp.vec3(schedule[wp.min(i, schedule.shape[0] - 1)], 0.0, 0.0), wp.vec3(0.0))
        cursor[0] = i + 1


    class Example:
        def __init__(self, viewer, args):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            body = builder.add_body()
            builder.add_shape_sphere(body, radius=0.1)
            self.model = builder.finalize()
            self.solver = newton.solvers.SolverSemiImplicit(self.model)
            self.state_0, self.state_1 = self.model.state(), self.model.state()
            self.control = self.model.control()
            self.schedule = wp.full(16, 1.0, dtype=float)
            self.cursor = wp.zeros(1, dtype=wp.int32)
            self.q0 = self.model.body_q.numpy().copy()
            self.frame_dt = 0.1

        def reset(self):
            self.state_0.body_q.assign(self.q0)
            self.cursor.zero_()

        def step(self):
            wp.launch(drive, 1, inputs=[self.schedule, self.cursor, self.state_0.body_qd])
            self.solver.step(self.state_0, self.state_1, self.control, None, self.frame_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
    """
)


class TestMcpReset(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        script = Path(directory.name) / "drive.py"
        script.write_text(_SCRIPT)
        self.session = ExampleHost(script).session(artifact_directory=directory.name)
        self.addCleanup(self.session.close)

    def execute(self, code: str) -> dict:
        return self.session.dispatch("execute", {"code": code})

    def test_reset_keeps_inputs_that_steps_do_not_write(self):
        """Keep a cell's schedule edit across reset, rewind the step cursor, and report both."""
        first = self.execute("rollout(3, start=True, record={'x': 'state.body_q.numpy()[0, 0]'})['x'].tolist()")
        np.testing.assert_allclose(first["result"], [0.0, 0.1, 0.2, 0.3], atol=1e-5)
        result = self.execute(
            "example.schedule.fill_(2.0)\n"
            "r = session.dispatch('reset')\n"
            "x = rollout(3, record={'x': 'state.body_q.numpy()[0, 0]'})['x'].tolist()\n"
            "{'reset': r, 'x': x, 'cursor': int(example.cursor.numpy()[0])}"
        )
        value = result["result"]
        self.assertEqual(value["reset"]["kept"], ["example.schedule"])
        self.assertIn("example.cursor", value["reset"]["rewound"])
        np.testing.assert_allclose(value["x"], [0.0, 0.2, 0.4, 0.6], atol=1e-5)
        self.assertEqual(value["cursor"], 3)
        self.assertIn("reset: kept example.schedule", result["note"])

    def test_reset_reports_initial_values_it_does_not_apply(self):
        """Name example.q0 when it changed, since session reset does not call example.reset()."""
        self.execute("rollout(2)")
        result = self.execute("example.q0[0, 0] = 5.0\nsession.dispatch('reset')['not_applied']")
        self.assertEqual(result["result"], ["example.q0"])
        self.assertIn("example.q0 changed, which reset does not apply (example.reset() reads it)", result["note"])
        self.assertAlmostEqual(float(self.session.state.body_q.numpy()[0, 0]), 0.0)
        # The same fact is not repeated while it stays the same.
        again = self.execute("session.dispatch('reset')\nNone")
        self.assertNotIn("note", again)

    def test_checkpoint_restore_rewinds_step_writes_only(self):
        """Restore rewinds the cursor a step advanced but keeps a later input edit."""
        result = self.execute(
            "rollout(2)\n"
            "session.dispatch('checkpoint', {'name': 'a'})\n"
            "rollout(2)\n"
            "example.schedule.fill_(3.0)\n"
            "r = session.dispatch('restore', {'name': 'a'})\n"
            "[r['kept'], r['rewound'], int(example.cursor.numpy()[0]), float(example.schedule.numpy()[0])]"
        )
        kept, rewound, cursor, schedule = result["result"]
        self.assertEqual(kept, ["example.schedule"])
        self.assertEqual(rewound, ["example.cursor"])
        self.assertEqual(cursor, 2)
        self.assertEqual(schedule, 3.0)
        self.assertIn("restore('a'): kept example.schedule", result["note"])

    def test_restore_does_not_report_included_objects_as_kept(self):
        """Arrays of objects a checkpoint included are restored with them, so restore does not list them as kept."""
        result = self.execute(
            "rollout(2)\n"
            "checkpoint('a', include=['example'])\n"
            "rollout(2)\n"
            "example.schedule.fill_(3.0)\n"
            "r = session.dispatch('restore', {'name': 'a'})\n"
            "[r.get('kept'), r.get('objects_restored'), float(example.schedule.numpy()[0])]"
        )
        kept, restored, schedule = result["result"]
        self.assertIsNone(kept)
        self.assertIn("example.schedule", restored)
        self.assertEqual(schedule, 1.0)
        self.assertNotIn("kept example.schedule", result.get("note", ""))

    def test_cell_steps_teach_writes(self):
        """Steps a cell runs through example.step() count as steps for what reset rewinds."""
        result = self.execute(
            "for _ in range(4):\n"
            "    example.step()\n"
            "control.joint_f.fill_(1.0)\n"
            "r = session.dispatch('reset')\n"
            "[r.get('kept'), r.get('rewound'), int(example.cursor.numpy()[0])]"
        )
        kept, rewound, cursor = result["result"]
        self.assertEqual(kept, ["control.joint_f"])
        self.assertEqual(rewound, ["example.cursor"])
        self.assertEqual(cursor, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
