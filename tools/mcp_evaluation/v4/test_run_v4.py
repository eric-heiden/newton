# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests of the loop harness: workspace snapshots and their verification, prompts, and cache seeds.

CPU only; no agents run. ``python -m unittest tools.mcp_evaluation.v4.test_run_v4``
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tokenize
import unittest
from pathlib import Path
from unittest import mock

from tools.mcp_evaluation.v4 import run_v4
from tools.mcp_evaluation.v4 import snapshots as snap
from tools.mcp_evaluation.v4 import trial_isolation as ti

HERE = Path(__file__).resolve().parent


class TemporaryDirectory(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)


class TestSnapshots(TemporaryDirectory):
    def workspace(self) -> Path:
        workspace = self.root / "work"
        (workspace / "data").mkdir(parents=True)
        (workspace / "__pycache__").mkdir()
        (workspace / "script.py").write_text("PASS = False\n")
        (workspace / "data" / "big.bin").write_bytes(b"x" * 4096)
        (workspace / "__pycache__" / "script.pyc").write_bytes(b"compiled")
        (workspace / "link.py").symlink_to("script.py")
        (workspace / "script.py").chmod(0o640)
        return workspace

    def test_snapshots_store_each_content_once(self):
        workspace = self.workspace()
        store = snap.WorkspaceSnapshots(workspace, self.root / "snapshots")
        first = store.take(0.0)
        self.assertEqual(sorted(first["files"]), ["data/big.bin", "link.py", "script.py"])
        self.assertEqual(first["files"]["link.py"], {"link": "script.py"})
        self.assertEqual(first["name"], "t00000")
        (workspace / "script.py").write_text("PASS = True\n")
        second = store.take(301.4)
        self.assertEqual(second["name"], "t00301")
        self.assertNotEqual(first["files"]["script.py"]["sha256"], second["files"]["script.py"]["sha256"])
        self.assertEqual(first["files"]["data/big.bin"], second["files"]["data/big.bin"])
        # Two versions of the script and one data file.
        self.assertEqual(len(list((self.root / "snapshots" / "objects").iterdir())), 3)
        final = store.take(400.0, final=True)
        self.assertTrue(final["final"])
        self.assertEqual([m["name"] for m in snap.load(self.root / "snapshots")], ["t00000", "t00301", "t00400"])

    def test_unchanged_files_are_not_read_again(self):
        workspace = self.workspace()
        store = snap.WorkspaceSnapshots(workspace, self.root / "snapshots")
        store.prime()
        with mock.patch.object(store, "_store", side_effect=AssertionError("re-read")):
            manifest = store.take(10.0)
        self.assertIn("script.py", manifest["files"])

    def test_materialize_restores_files_modes_and_links(self):
        workspace = self.workspace()
        store = snap.WorkspaceSnapshots(workspace, self.root / "snapshots")
        manifest = store.take(0.0)
        target = self.root / "restored"
        snap.materialize(self.root / "snapshots", manifest, target)
        self.assertEqual((target / "script.py").read_text(), "PASS = False\n")
        self.assertEqual((target / "script.py").stat().st_mode & 0o777, 0o640)
        self.assertEqual((target / "data" / "big.bin").read_bytes(), b"x" * 4096)
        self.assertEqual(os.readlink(target / "link.py"), "script.py")
        self.assertFalse((target / "__pycache__").exists())

    def test_large_files_are_skipped_and_listed(self):
        workspace = self.workspace()
        with mock.patch.object(snap, "MAX_FILE_BYTES", 1000):
            manifest = snap.WorkspaceSnapshots(workspace, self.root / "snapshots").take(0.0)
        self.assertNotIn("data/big.bin", manifest["files"])
        self.assertEqual(manifest["skipped"][0]["path"], "data/big.bin")

    def test_periodic_snapshots_are_timed_from_the_budget_start(self):
        workspace = self.workspace()
        store = snap.WorkspaceSnapshots(workspace, self.root / "snapshots", period=0.2)
        start = time.time()
        store.start(start)
        time.sleep(0.5)
        store.stop()
        seconds = [m["seconds"] for m in snap.load(self.root / "snapshots")]
        self.assertGreaterEqual(len(seconds), 3)
        self.assertLess(seconds[0], 0.1)
        self.assertAlmostEqual(seconds[1], 0.2, delta=0.1)
        self.assertEqual(store.errors, [])

    def test_file_changing_during_every_copy_is_copied_again_later(self):
        workspace = self.workspace()
        store = snap.WorkspaceSnapshots(workspace, self.root / "snapshots")
        script = workspace / "script.py"
        store_file = store._store
        versions = iter(range(1, 10))

        def rewritten_while_copied(path):
            name = store_file(path)
            if path == script:
                time.sleep(0.01)
                script.write_text(f"GAIN = {next(versions)}\n")
            return name

        with mock.patch.object(store, "_store", side_effect=rewritten_while_copied):
            first = store.take(0.0)
        second = store.take(300.0)
        stored = self.root / "snapshots" / "objects" / second["files"]["script.py"]["sha256"]
        self.assertEqual(stored.read_text(), script.read_text())
        self.assertEqual(second["skipped"], [])
        self.assertNotIn("unstable", second["files"]["script.py"])
        self.assertEqual(first["skipped"], [{"path": "script.py", "reason": "changed while copied"}])
        self.assertTrue(first["files"]["script.py"]["unstable"])

    def test_ignore_patterns(self):
        patterns = ["photos/", "episode.npz", "*.png"]
        self.assertTrue(snap.ignored("photos/a/b.json", patterns))
        self.assertTrue(snap.ignored("episode.npz", patterns))
        self.assertFalse(snap.ignored("fits/episode.npz", patterns))
        self.assertTrue(snap.ignored("renders/frame.png", patterns))
        self.assertFalse(snap.ignored("scene_replay.py", patterns))

    def test_submission_digest_ignores_only_listed_files(self):
        manifest = {"files": {"s.py": {"sha256": "a"}, "out.png": {"sha256": "b"}}}
        changed_image = {"files": {"s.py": {"sha256": "a"}, "out.png": {"sha256": "c"}}}
        changed_script = {"files": {"s.py": {"sha256": "d"}, "out.png": {"sha256": "b"}}}
        digest = snap.submission_digest(manifest, ["*.png"])
        self.assertEqual(digest, snap.submission_digest(changed_image, ["*.png"]))
        self.assertNotEqual(digest, snap.submission_digest(changed_script, ["*.png"]))
        self.assertNotEqual(digest, snap.submission_digest(changed_image))


class TestSearch(unittest.TestCase):
    @staticmethod
    def snapshots(count: int) -> list[dict]:
        return [{"name": f"t{300 * i:05d}", "seconds": 300.0 * i} for i in range(count)]

    def check(self, passing):
        calls = []

        def check(index):
            calls.append(index)
            return {"snapshot": f"t{300 * index:05d}", "success": passing(index)}

        return check, calls

    def test_binary_search_finds_the_first_pass_in_logarithmic_checks(self):
        for first in range(12):
            snapshots = self.snapshots(12)
            check, calls = self.check(lambda i, first=first: i >= first)
            result = snap.search(snapshots, [f"d{i}" for i in range(12)], check)
            self.assertEqual(result["first_pass_seconds"], 300.0 * first)
            self.assertEqual(result["last_fail_seconds"], 300.0 * (first - 1) if first else None)
            self.assertLessEqual(len(calls), math.ceil(math.log2(12)) + 1)
            row = result["snapshots"][first]
            self.assertTrue(row["success"])
            self.assertEqual(row["source"], "verifier")
            for row in result["snapshots"]:
                if row["source"] == "inferred":
                    self.assertEqual(row["inferred_success"], row["version"] >= first)

    def test_no_pass_needs_one_check(self):
        check, calls = self.check(lambda i: False)
        result = snap.search(self.snapshots(8), [f"d{i}" for i in range(8)], check)
        self.assertIsNone(result["first_pass_seconds"])
        self.assertEqual(calls, [7])

    def test_unchanged_submissions_are_verified_once(self):
        digests = ["a", "a", "b", "b", "b", "c"]
        check, calls = self.check(lambda i: i >= 2)
        result = snap.search(self.snapshots(6), digests, check, exhaustive=True)
        self.assertEqual(calls, [0, 2, 5])
        self.assertEqual(result["versions"], 3)
        self.assertEqual(result["first_pass_seconds"], 600.0)
        self.assertEqual(result["snapshots"][3]["source"], "same as t00600")

    def test_exhaustive_search_finds_a_pass_that_was_broken_later(self):
        check, calls = self.check(lambda i: i == 2)
        snapshots, digests = self.snapshots(5), [f"d{i}" for i in range(5)]
        self.assertIsNone(snap.search(snapshots, digests, self.check(lambda i: i == 2)[0])["first_pass_seconds"])
        result = snap.search(snapshots, digests, check, exhaustive=True)
        self.assertEqual(result["first_pass_seconds"], 600.0)
        self.assertEqual(calls, [0, 1, 2, 3, 4])

    def test_reverted_submission_that_passed_earlier(self):
        outcome = {"A": True, "B": False, "C": False, "X": False}
        for digests, first, last_fail in (
            (["A", "B", "A"], 0.0, None),
            (["X", "A", "B", "C", "A"], 300.0, 0.0),
            (["X", "A", "B"], None, None),
        ):
            with self.subTest(digests=digests):
                check, _ = self.check(lambda i, digests=digests: outcome[digests[i]])
                result = snap.search(self.snapshots(len(digests)), digests, check)
                self.assertEqual(result["first_pass_seconds"], first)
                self.assertEqual(result["last_fail_seconds"], last_fail)
                self.assertEqual(result["monotonicity_violated"], first is not None and digests[-1] == "A")
                for row in result["snapshots"]:
                    if row["source"] != "inferred":
                        self.assertEqual(row["success"], outcome[row["digest"]])
        # A pass known from an earlier verification counts even when the last version fails.
        known = {"A": {"snapshot": "t00300", "success": True}}
        check, _ = self.check(lambda i: ["X", "A", "B"][i] == "A")
        result = snap.search(self.snapshots(3), ["X", "A", "B"], check, known=known)
        self.assertEqual(result["first_pass_seconds"], 300.0)
        self.assertTrue(result["monotonicity_violated"])

    def test_known_results_are_reused(self):
        known = {f"d{i}": {"snapshot": f"t{300 * i:05d}", "success": i >= 1} for i in range(4)}
        check, calls = self.check(lambda i: True)
        result = snap.search(self.snapshots(4), [f"d{i}" for i in range(4)], check, known=known)
        self.assertEqual(calls, [])
        self.assertEqual(result["first_pass_seconds"], 300.0)


def _fake_task(directory: Path) -> dict:
    (directory / "starter.py").write_text("PASS = False\n")
    return {
        "files": {"s.py": directory / "starter.py"},
        "script": "s.py",
        "host_args": [],
        "verifier": str(HERE / "fake_verify.py"),
        "goal": "Make the fake verifier pass.",
        "warmup": [],
        "private": [],
        "snapshot_ignore": [".git/", "*.log"],
    }


class TestTrialSnapshots(TemporaryDirectory):
    """A restart trial whose agent is a shell script, with snapshots, then the post-trial verification."""

    def setUp(self):
        super().setUp()
        task = _fake_task(self.root)
        caches = self.root / "seed" / "caches"
        (caches / "warp").mkdir(parents=True)
        (caches / "warp" / "kernel.bin").write_text("cached")
        agent = (
            "cat > /dev/null; sleep 0.5; echo '# edit' >> s.py; echo noise > out.log; sleep 0.7; "
            "echo 'PASS = True' >> s.py; sleep 0.6; echo more > out.log"
        )
        for replacement in (
            mock.patch.object(run_v4, "_task", return_value=task),
            mock.patch.object(run_v4, "TRIALS", self.root / "trials"),
            mock.patch.object(run_v4, "SANDBOX", False),
            mock.patch.object(run_v4, "SNAPSHOT_SECONDS", 0.3),
            mock.patch.object(run_v4, "_agent_command", return_value=["bash", "-c", agent]),
            mock.patch.object(run_v4, "parse_events", return_value={}),
            mock.patch.object(ti, "seed_caches", return_value=caches),
            mock.patch.object(ti, "provenance", return_value={"diff": "", "commit": "test"}),
            mock.patch.object(ti, "live_trials", return_value=set()),
            mock.patch.object(run_v4, "newton_source", return_value=None),
        ):
            replacement.start()
            self.addCleanup(replacement.stop)

    def test_snapshots_and_first_pass(self):
        run_dir = self.root / "loop" / "fake-opus-restart-p0"
        prepared = run_v4.prepare(run_dir, "fake", "restart", "opus", 60, "test")
        summary = run_v4.run_trial(prepared)
        self.assertTrue(summary["verification"]["success"])
        self.assertEqual(summary["snapshots"]["errors"], [])
        manifests = snap.load(run_dir / "snapshots")
        self.assertGreaterEqual(len(manifests), 5)
        self.assertEqual(manifests[0]["name"], "t00000")
        self.assertIn("TASK.md", manifests[0]["files"])
        self.assertTrue(manifests[-1]["final"])
        self.assertEqual(summary["snapshots"]["final"], manifests[-1]["name"])
        # The trial's caches are kept for snapshot verification; its sandbox is gone.
        self.assertEqual((run_dir / "snapshots" / "caches" / "warp" / "kernel.bin").read_text(), "cached")
        self.assertFalse((self.root / "trials" / summary["trial_id"]).exists())

        result = run_v4.verify_snapshots(run_dir)
        report = json.loads((run_dir / "snapshot_verification.json").read_text())
        self.assertTrue(report["complete"])
        self.assertEqual(report["mode"], "binary")
        self.assertEqual(report["assumption"], snap.MONOTONE)
        first = next(m for m in manifests if m["name"] == result["first_pass_snapshot"])
        restored = self.root / "restored"
        snap.materialize(run_dir / "snapshots", first, restored)
        self.assertIn("PASS = True", (restored / "s.py").read_text())
        index = manifests.index(first)
        self.assertEqual(result["last_fail_seconds"], manifests[index - 1]["seconds"])
        snap.materialize(run_dir / "snapshots", manifests[index - 1], self.root / "before")
        self.assertNotIn("PASS = True", (self.root / "before" / "s.py").read_text())
        # out.log changes after the pass do not make new versions (snapshot_ignore).
        self.assertEqual(result["versions"], 3)
        sources = {record["source"] for record in report["verifications"]}
        self.assertEqual(sources, {"trial", "verifier"})
        self.assertFalse((self.root / "trials" / summary["trial_id"]).exists())
        verified = next(r["snapshot"] for r in report["verifications"] if r["source"] == "verifier")
        self.assertTrue((run_dir / "snapshots" / "verify" / verified / "verification.json").exists())

        # A second run reuses every result; --all verifies the remaining versions only.
        before = len(report["verifications"])
        run_v4.verify_snapshots(run_dir)
        self.assertEqual(len(json.loads((run_dir / "snapshot_verification.json").read_text())["verifications"]), before)
        everything = run_v4.verify_snapshots(run_dir, exhaustive=True)
        self.assertEqual(len(everything["verifications"]), 3)
        self.assertEqual(everything["first_pass_seconds"], result["first_pass_seconds"])

    def test_no_periodic_snapshot_after_the_agent_ends(self):
        stop = snap.WorkspaceSnapshots.stop

        def slow_stop(store):
            time.sleep(0.8)  # longer than SNAPSHOT_SECONDS: ticks in this window would follow the final snapshot
            stop(store)

        run_dir = self.root / "loop" / "fake-opus-restart-p0"
        with mock.patch.object(snap.WorkspaceSnapshots, "stop", slow_stop):
            run_v4.run_trial(run_v4.prepare(run_dir, "fake", "restart", "opus", 60, "test"))
        manifests = snap.load(run_dir / "snapshots")
        self.assertTrue(manifests[-1]["final"])
        self.assertEqual(sum(m["final"] for m in manifests), 1)

    def test_final_snapshot_is_the_last_version(self):
        run_dir = self.root / "loop" / "fake-opus-restart-p0"
        summary = run_v4.run_trial(run_v4.prepare(run_dir, "fake", "restart", "opus", 60, "test"))
        final = next(m for m in snap.load(run_dir / "snapshots") if m["final"])
        # A periodic snapshot labelled after the final one, of a workspace that no longer passes.
        late = dict(final, name="t99999", seconds=final["seconds"] + 1.0, final=False)
        late["files"] = {name: entry for name, entry in final["files"].items() if name != "s.py"}
        (run_dir / "snapshots" / "t99999.json").write_text(json.dumps(late))
        result = run_v4.verify_snapshots(run_dir)
        self.assertNotIn("t99999", [row["name"] for row in result["snapshots"]])
        self.assertEqual(result["snapshots"][-1]["name"], final["name"])
        self.assertEqual(result["snapshots"][-1]["source"], "trial")
        self.assertEqual(result["snapshots"][-1]["success"], summary["verification"]["success"])

    def test_final_snapshot_takes_the_reverified_verdict(self):
        run_dir = self.root / "loop" / "fake-opus-restart-p0"
        summary = run_v4.run_trial(run_v4.prepare(run_dir, "fake", "restart", "opus", 60, "test"))
        self.assertTrue(summary["verification"]["success"])
        first = run_v4.verify_snapshots(run_dir)
        self.assertEqual(first["final_verdict_source"], "trial")
        self.assertIsNotNone(first["first_pass_seconds"])
        # A later re-verification with a stricter verifier fails the final workspace: it replaces the trial's
        # verdict, and under the monotonicity assumption no version passed.
        (run_dir / "reverify").mkdir()
        rejected = {"success": False, "failed_checks": ["stricter"], "commit": "abc"}
        (run_dir / "reverify" / "result.json").write_text(json.dumps(rejected))
        result = run_v4.verify_snapshots(run_dir)
        self.assertEqual(result["final_verdict_source"], "reverify")
        final = result["snapshots"][-1]
        self.assertEqual((final["source"], final["success"]), ("reverify", False))
        records = [r for r in result["verifications"] if r["digest"] == final["digest"]]
        self.assertEqual(len(records), 1)
        self.assertEqual((records[0]["failed_checks"], records[0]["verifier_commit"]), (["stricter"], "abc"))
        self.assertIsNone(result["first_pass_seconds"])
        self.assertTrue(result["monotonicity_violated"] is False)

    def test_agents_run_without_auto_memory(self):
        run_dir = self.root / "loop" / "fake-opus-restart-p0"
        agent = 'cat > /dev/null; echo "$CLAUDE_CODE_DISABLE_AUTO_MEMORY" > memory.txt'
        with mock.patch.object(run_v4, "_agent_command", return_value=["bash", "-c", agent]):
            run_v4.run_trial(run_v4.prepare(run_dir, "fake", "restart", "opus", 60, "test"))
        self.assertEqual((run_dir / "workspace" / "memory.txt").read_text().strip(), "1")

    def test_refuses_while_trials_run(self):
        run_dir = self.root / "loop" / "fake-opus-restart-p0"
        run_v4.run_trial(run_v4.prepare(run_dir, "fake", "restart", "opus", 60, "test"))
        with mock.patch.object(ti, "live_trials", return_value={"abc123"}):
            with self.assertRaisesRegex(RuntimeError, "between iterations"):
                run_v4.verify_snapshots(run_dir)


def _codex_session(path: Path, totals: list[tuple[int, int, int]], images: int = 0) -> None:
    """A Codex session log with one cumulative token count per (input, cached input, output) and image views."""
    events = [{"type": "session_meta", "payload": {"id": "s"}}]
    for _ in range(images):
        item = {"type": "ImageView", "path": "file:///w/a.png"}
        events.append({"type": "event_msg", "payload": {"type": "item_completed", "item": item}})
    for tokens, cached, output in totals:
        usage = {"input_tokens": tokens, "cached_input_tokens": cached, "cache_write_input_tokens": 0}
        usage |= {"output_tokens": output, "reasoning_output_tokens": 0, "total_tokens": tokens + output}
        info = {"total_token_usage": usage, "last_token_usage": usage}
        events.append({"type": "event_msg", "payload": {"type": "token_count", "info": info}})
    events.append({"type": "event_msg", "payload": {"type": "turn_aborted", "reason": "interrupted"}})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(event) + "\n" for event in events))


class TestUsage(TemporaryDirectory):
    """Usage of a run the harness interrupted is recovered from Codex's session log or marked unknown, never 0."""

    def setUp(self):
        super().setUp()
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()
        self.zero = {"input_tokens": 0, "cached_input_tokens": 0, "cache_write_tokens": 0, "output_tokens": 0}
        self.zero["uncached_input_plus_output"] = 0

    def transcript(self, *events: dict) -> None:
        (self.run_dir / "agent.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))

    def test_interrupted_codex_run_takes_the_session_token_count(self):
        self.transcript({"type": "thread.started"}, {"type": "turn.started"})
        home = self.root / "codex-home"
        _codex_session(home / "sessions/2026/10/07/rollout-a.jsonl", [(100, 0, 5), (230, 110, 9)], images=2)
        record = run_v4._usage_record(self.run_dir, "codex", dict(self.zero), str(home))
        self.assertEqual(record["usage_source"], "codex_session")
        self.assertFalse(record["usage_complete"])
        self.assertEqual(record["usage"]["input_tokens"], 230)
        self.assertEqual(record["usage"]["cached_input_tokens"], 110)
        self.assertEqual(record["usage"]["output_tokens"], 9)
        self.assertEqual(record["usage"]["uncached_input_plus_output"], 129)
        self.assertEqual(record["codex_image_views"], 2)
        self.assertTrue((self.run_dir / "codex-sessions" / "rollout-a.jsonl").exists())

    def test_completed_codex_run_keeps_its_final_usage(self):
        self.transcript({"type": "turn.completed", "usage": {"input_tokens": 7}})
        home = self.root / "codex-home"
        _codex_session(home / "sessions/2026/10/07/rollout-a.jsonl", [(7, 0, 1)], images=1)
        usage = dict(self.zero, input_tokens=7)
        record = run_v4._usage_record(self.run_dir, "codex", usage, str(home))
        self.assertEqual((record["usage_source"], record["usage_complete"]), ("final", True))
        self.assertEqual(record["usage"], usage)
        self.assertEqual(record["codex_image_views"], 1)

    def test_unknown_usage_is_none_not_zero(self):
        self.transcript({"type": "assistant", "message": {"content": []}})
        record = run_v4._usage_record(self.run_dir, "claude", dict(self.zero), None)
        self.assertEqual((record["usage_source"], record["usage_complete"]), (None, False))
        self.assertEqual(set(record["usage"]), set(self.zero))
        self.assertTrue(all(value is None for value in record["usage"].values()))
        self.transcript({"type": "turn.started"})
        record = run_v4._usage_record(self.run_dir, "codex", dict(self.zero), str(self.root / "empty-home"))
        self.assertIsNone(record["usage"]["output_tokens"])
        self.assertIsNone(record["codex_image_views"])

    def test_claude_result_is_complete(self):
        self.transcript({"type": "result", "usage": {}})
        record = run_v4._usage_record(self.run_dir, "claude", dict(self.zero), None)
        self.assertEqual((record["usage_source"], record["usage_complete"]), ("final", True))
        self.assertNotIn("codex_image_views", record)

    def test_codex_keeps_its_session_log_only_with_its_own_home(self):
        command = run_v4._agent_command({"cli": "codex", "model": "m", "effort": "low"}, self.root, None)
        self.assertIn("--ephemeral", command)
        kept = run_v4.keep_codex_session(command, {"CODEX_HOME": str(self.root / "codex-home")})
        self.assertEqual(kept, [arg for arg in command if arg != "--ephemeral"])
        self.assertEqual(run_v4.keep_codex_session(command, {}), command)
        claude = ["claude", "-p", "--ephemeral"]
        self.assertEqual(run_v4.keep_codex_session(claude, {"CODEX_HOME": "/x"}), claude)


class TestNewtonSource(TemporaryDirectory):
    """Finished trials are verified again on the newton package of the commit they ran on."""

    def setUp(self):
        super().setUp()
        self.repo = self.root / "repo"
        (self.repo / "newton").mkdir(parents=True)
        (self.repo / "tools").mkdir()

        def git(*args):
            return subprocess.run(
                ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
                cwd=self.repo,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()

        self.git = git
        git("init", "-q")
        (self.repo / "newton" / "__init__.py").write_text("VERSION = 1\n")
        (self.repo / "tools" / "verify.py").write_text("RULE = 1\n")
        git("add", "-A")
        git("commit", "-q", "-m", "one")
        self.first = git("rev-parse", "HEAD")
        patcher = mock.patch.object(run_v4, "VERIFY_SOURCES", self.root / "sources")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_same_newton_uses_the_tree(self):
        (self.repo / "tools" / "verify.py").write_text("RULE = 2\n")  # verifier changes do not matter
        self.git("commit", "-q", "-am", "verifier fix")
        self.assertIsNone(run_v4.newton_source(self.first, self.repo))
        self.assertIsNone(run_v4.newton_source(None, self.repo))

    def test_changed_newton_is_exported_once(self):
        (self.repo / "newton" / "__init__.py").write_text("VERSION = 2\n")
        self.git("commit", "-q", "-am", "newton change")
        source = run_v4.newton_source(self.first, self.repo)
        self.assertEqual(source, self.root / "sources" / self.git("rev-parse", f"{self.first}:newton"))
        self.assertEqual((source / "newton" / "__init__.py").read_text(), "VERSION = 1\n")
        self.assertFalse((source / "tools").exists())
        (source / "newton" / "marker").write_text("")
        self.assertTrue((run_v4.newton_source(self.first, self.repo) / "newton" / "marker").exists())
        # A later commit with the same package (a verifier fix) shares the copy.
        self.git("checkout", "-q", self.first, "--", "newton")
        (self.repo / "tools" / "verify.py").write_text("RULE = 2\n")
        self.git("commit", "-q", "-am", "revert newton, fix verifier")
        same = self.git("rev-parse", "HEAD")
        (self.repo / "newton" / "__init__.py").write_text("VERSION = 3\n")
        self.git("commit", "-q", "-am", "newton change")
        self.assertEqual(run_v4.newton_source(same, self.repo), source)
        # Uncommitted and untracked changes of the tree count as well.
        self.assertIsNone(run_v4.newton_source(self.git("rev-parse", "HEAD"), self.repo))
        (self.repo / "newton" / "extra.py").write_text("")
        self.assertIsNotNone(run_v4.newton_source(self.git("rev-parse", "HEAD"), self.repo))

    def test_unknown_commit(self):
        with self.assertRaisesRegex(RuntimeError, "not in"):
            run_v4.newton_source("0" * 40, self.repo)

    def test_rerun_env_puts_the_trial_newton_first(self):
        with mock.patch.object(run_v4, "newton_source", return_value=Path("/sources/c")):
            env = run_v4.rerun_env({"trial_id": "t", "commit": "c"}, self.root / "sandbox", "snapshot")
        self.assertEqual(env["PYTHONPATH"], f"/sources/c{os.pathsep}{run_v4.ROOT}")
        self.assertEqual(env["NEWTON_TRIAL_ID"], "t-snapshot")
        with mock.patch.object(run_v4, "newton_source", return_value=None):
            env = run_v4.rerun_env({"trial_id": "t", "commit": "c"}, self.root / "sandbox", "snapshot")
        self.assertEqual(env["PYTHONPATH"], str(run_v4.ROOT))


class TestAgentVisibleText(unittest.TestCase):
    """Agents of both conditions can read newton/ and docs/ (the sandbox masks only tools/mcp_evaluation)."""

    ROOT = HERE.parents[2]
    SUFFIXES = (".py", ".rst", ".md", ".txt", ".toml", ".json", ".yaml", ".yml", ".cfg", ".ini")

    @classmethod
    def text(cls, path: Path) -> str:
        """Prose of a file: comments and strings of Python files (not identifiers such as h10), else everything."""
        source = path.read_text(errors="replace")
        if path.suffix != ".py":
            return source
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
        except (tokenize.TokenError, SyntaxError):
            return source
        return "\n".join(token.string for token in tokens if token.type in (tokenize.COMMENT, tokenize.STRING))

    def test_newton_and_docs_do_not_name_the_study(self):
        tasks = {d.name for d in HERE.iterdir() if d.is_dir() and "_" in d.name and not d.name.startswith("_")}
        tasks |= {"grasp_drift", "g1_track", "g1_hard"}
        patterns = (
            re.compile(r"\b(" + "|".join(sorted(map(re.escape, tasks))) + r")\b"),
            re.compile(r"\b[ih]1[0-9]\b"),
            re.compile(r"\bMCP trials?\b|\bpaired (study|trials?)\b|\bmcp_evaluation\b", re.IGNORECASE),
        )
        hits = []
        for top in ("newton", "docs"):
            for directory, names, files in os.walk(self.ROOT / top):
                names[:] = [name for name in names if name not in ("__pycache__", "_build")]
                for name in files:
                    path = Path(directory) / name
                    if path.suffix in self.SUFFIXES:
                        text = self.text(path)
                        hits += [
                            f"{path.relative_to(self.ROOT)}: {m.group(0)}" for p in patterns for m in p.finditer(text)
                        ]
        self.assertEqual(hits, [])


FACTS = {"gpu": "one MIG 1g.24gb slice (24 GB of GPU memory) of a GPU", "cpu_cores": 4, "cpu_count_reported": 192}


class TestPrompt(unittest.TestCase):
    def test_both_conditions_state_the_same_environment_facts(self):
        with mock.patch.object(run_v4, "_host_guide", return_value="GUIDE"):
            prompts = {
                condition: run_v4.prompt_for("g1_mpc", condition, Path("/w"), 5400, None, dict(FACTS, gnu_time=False))
                for condition in ("mcp", "restart")
            }
        environment = run_v4._environment(dict(FACTS, gnu_time=False))
        self.assertIn("one MIG 1g.24gb slice (24 GB of GPU memory) of a GPU and 4 CPU cores", environment)
        self.assertIn("os.cpu_count() reports 192", environment)
        self.assertIn("shared by every process of this task", environment)
        self.assertIn("GNU time (/usr/bin/time) is not installed", environment)
        for prompt in prompts.values():
            self.assertIn(environment, prompt)
        self.assertNotIn("not installed", run_v4._environment(dict(FACTS, gnu_time=True)))

    def test_both_conditions_name_the_same_newton_utilities(self):
        import importlib  # noqa: PLC0415

        prompts = [
            run_v4.prompt_for("abc_scratch", condition, Path("/w"), 5400, "GUIDE", FACTS)
            for condition in ("mcp", "restart")
        ]
        block = run_v4._newton_tools()
        for prompt in prompts:
            self.assertEqual(prompt.count(block), 1)
        self.assertEqual(block.count("\n"), len(run_v4.NEWTON_UTILITIES) + 1)
        # Every API a line names exists, so the prompt cannot advertise a removed or renamed symbol.
        for names, line in run_v4.NEWTON_UTILITIES:
            for name in names:
                parts = name.split(".")
                for split in range(len(parts), 0, -1):
                    try:
                        target = importlib.import_module(".".join(parts[:split]))
                    except ImportError:
                        continue
                    for attribute in parts[split:]:
                        target = getattr(target, attribute)
                    break
                else:
                    self.fail(f"{name} does not resolve")
                self.assertIn(parts[-1], line)

    def test_both_conditions_state_the_verifier_rules_and_shell_facts(self):
        facts = {
            "abc_scratch": (
                "fits of the tray's silhouette in photos/01_top.jpg and photos/11_top.jpg",
                "no fruit's lowest collision point may rise more than 1 cm above its lowest point at the start",
                "scans scene_replay.py and the workspace modules it imports",
            ),
            "g1_mpc": ("scans every Python file in the workspace",),
        }
        for task, lines in facts.items():
            for cli in ("claude", "codex"):
                prompts = [
                    run_v4.prompt_for(task, condition, Path("/w"), 5400, "GUIDE", FACTS, cli)
                    for condition in ("mcp", "restart")
                ]
                for prompt in prompts:
                    for line in lines:
                        self.assertIn(line, prompt)
                    self.assertIn(run_v4.SCAN_NAMES, prompt)
                    self.assertNotRegex(prompt, r"\{[a-z_]+\}")  # every placeholder filled
                    self.assertEqual(prompt.count(run_v4.SHELL_FACTS["claude"]), int(cli == "claude"))
        self.assertIn("120 s", run_v4.SHELL_FACTS["claude"])
        self.assertIn("`sleep N` with N of 25 or more is refused", run_v4.SHELL_FACTS["claude"])

    def test_scan_names_cover_the_verifiers_rules(self):
        from tools.mcp_evaluation.v4.abc_scratch import verify as scratch  # noqa: PLC0415
        from tools.mcp_evaluation.v4.g1_mpc import verify as mpc  # noqa: PLC0415

        for verifier in (scratch, mpc):
            for name in verifier.SUSPICIOUS_MODULES | verifier.SUSPICIOUS_BUILTINS | {"__import__"}:
                self.assertIn(name, run_v4.SCAN_NAMES)

    def test_guide_describes_the_hosted_workers(self):
        workspace = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, workspace)
        (workspace / "s.py").write_text("")
        task = {"script": "s.py", "host_args": ["--x", "1"]}
        with mock.patch.object(run_v4, "WORKERS", 2), mock.patch.object(run_v4, "MAX_WORKERS", 4):
            command = run_v4._host_command(workspace, task, workspace / "c.json")
            guide = run_v4._host_guide(workspace, task)
        self.assertEqual(command[command.index("--workers") + 1], "2")
        self.assertEqual(command[command.index("--max-workers") + 1], "4")
        self.assertEqual(command[-3:], ["--", "--x", "1"])
        self.assertIn("`workers`: 2 sibling copies", guide)
        self.assertIn("(0 to 4)", guide)
        self.assertIn("jobs.start", guide)
        with mock.patch.object(run_v4, "WORKERS", 0), mock.patch.object(run_v4, "MAX_WORKERS", 0):
            self.assertNotIn("`workers`", run_v4._host_guide(workspace, task))


class TestEnvironment(TemporaryDirectory):
    def test_python_wrappers_run_the_project_environment(self):
        directory = ti.python_wrappers(self.root / "bin", run_v4.PYTHON)
        for name in ("python", "python3"):
            output = subprocess.run(
                [str(directory / name), "-c", "import sys, numpy, PIL; print(sys.prefix)"],
                capture_output=True,
                text=True,
                check=True,
                env={"PATH": os.defpath, "CUDA_VISIBLE_DEVICES": ""},
            ).stdout.strip()
            self.assertEqual(Path(output).resolve(), (run_v4.ROOT / ".venv").resolve())

    def test_gpu_description(self):
        listing = (
            "GPU 0: NVIDIA RTX PRO 6000 Blackwell Server Edition (UUID: GPU-1)\n"
            "  MIG 1g.24gb     Device  0: (UUID: MIG-2)\n"
        )
        result = subprocess.CompletedProcess([], 0, stdout=listing)
        with mock.patch.object(ti.subprocess, "run", return_value=result):
            self.assertEqual(
                ti._gpu(),
                "one MIG 1g.24gb slice (24 GB of GPU memory) of an NVIDIA RTX PRO 6000 Blackwell Server Edition",
            )
        plain = [
            subprocess.CompletedProcess([], 0, stdout="GPU 0: NVIDIA L40 (UUID: GPU-1)\n"),
            subprocess.CompletedProcess([], 0, stdout="46068\n"),
        ]
        with mock.patch.object(ti.subprocess, "run", side_effect=plain):
            self.assertEqual(ti._gpu(), "one NVIDIA L40 (45 GB of GPU memory)")
        with mock.patch.dict(os.environ, {"NEWTON_TRIAL_GPU": "one test GPU", "NEWTON_TRIAL_CPUS": "8"}):
            facts = ti.hardware()
        self.assertEqual((facts["gpu"], facts["cpu_cores"]), ("one test GPU", 8))

    def test_live_trials_finds_tagged_processes(self):
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            env={**os.environ, "NEWTON_TRIAL_ID": "h16-test-live"},
        )
        try:
            self.assertIn("h16-test-live", ti.live_trials())
        finally:
            process.kill()
            process.wait()
        self.assertNotIn("h16-test-live", ti.live_trials())

    def test_seed_coverage(self):
        seed = self.root / "caches"
        (seed / "warp" / "1.0" / "wp_kernel_a_1234567").mkdir(parents=True)
        log = self.root / "agent.jsonl"
        log.write_text(
            "Module kernel_a 1234567 load on device 'cuda:0' took 5.00 ms  (compiled)\n"
            "Module kernel_b 89abcde load on device 'cuda:0' took 9.00 ms  (compiled)\n"
            "Module kernel_c 1111111 load on device 'cuda:0' took 0.10 ms  (cached)\n"
        )
        self.assertEqual(
            ti.seed_coverage(seed, [log]),
            {"compiled": 2, "in_seed": ["kernel_a_1234567"], "missing": ["kernel_b_89abcde"]},
        )


class TestWarmShapes(unittest.TestCase):
    def test_shapes_are_added_after_every_mjcf_import(self):
        import newton  # noqa: PLC0415

        mjcf = '<mujoco><worldbody><geom type="plane" size="1 1 0.1"/></worldbody></mujoco>'
        original = newton.ModelBuilder.add_mjcf
        self.addCleanup(setattr, newton.ModelBuilder, "add_mjcf", original)
        shapes = {"bodies": [run_v4._KINDS] * 3, "static": ["box", "mesh"], "origin": run_v4._TABLE}
        ti._patch_add_mjcf(shapes)
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        self.assertEqual(builder.body_count, 3)
        self.assertEqual(builder.joint_dof_count, 18)
        added = {newton.GeoType(t).name for t in builder.shape_type[1:]}
        self.assertEqual(added, {"SPHERE", "ELLIPSOID", "CAPSULE", "CYLINDER", "BOX", "CONVEX_MESH", "MESH"})
        self.assertEqual(builder.shape_count, 1 + 3 * len(run_v4._KINDS) + 2)

    def test_variants_step_with_the_contacts_their_solver_uses(self):
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, directory)
        script = directory / "fake_example.py"
        script.write_text(
            "class Example:\n"
            "    def __init__(self, viewer, args):\n"
            "        self.model, self.solver, self.collision_pipeline, self.steps = 'model', None, 'pipeline', []\n"
            "    def step(self):\n"
            "        self.steps.append((self.solver, self.collision_pipeline))\n"
        )

        class Solver:
            def __init__(self, model, **options):
                self.options = options

        import newton  # noqa: PLC0415
        from newton.mcp import ExampleHost  # noqa: PLC0415

        hosts = []
        original_build = ExampleHost.build

        def build(host, *args, **kwargs):
            hosts.append(host)
            return original_build(host, *args, **kwargs)

        variants = [{"use_mujoco_contacts": True}, {"cone": "elliptic"}, {"use_mujoco_contacts": False}]
        with mock.patch.object(newton.solvers, "SolverMuJoCo", Solver), mock.patch.object(ExampleHost, "build", build):
            ti.warm_solvers(str(script), variants)
        steps = [(solver.options, pipeline) for solver, pipeline in hosts[0].example.steps]
        # SolverMuJoCo finds its own contacts unless use_mujoco_contacts=False (an omitted option means True).
        self.assertEqual(steps, [(variants[0], None), (variants[1], None), (variants[2], "pipeline")])

    def test_task_seeds_name_every_solver_setting_explicitly(self):
        warmups = run_v4._task("abc_scratch")["warmup"]
        shape_runs = [command for command in warmups if len(command) == 6]
        self.assertEqual(len(shape_runs), len(run_v4.STATION_SHAPE_SEEDS))
        for command in [c for c in warmups if "warm-solvers" in c]:
            for variant in json.loads(command[4]):
                self.assertIn("use_mujoco_contacts", variant)


if __name__ == "__main__":
    unittest.main()
