# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import datetime
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from newton._src.tools.research_dashboard import (
    _PaperclipRelayConfig,
    ResearchDashboardRequestHandler,
    _fetch_issue_comments_after,
    _render_index_html,
    assess_research_artifact_freshness,
    build_research_dashboard_payload,
    default_research_artifact_path,
    default_research_max_artifact_age_hours,
    make_argument_parser,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class ResearchDashboardPayloadTest(unittest.TestCase):
    def test_build_research_dashboard_payload_normalizes_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "research.json"
            _write_json(
                artifact_path,
                {
                    "entries": [
                        {
                            "id": "paper-1",
                            "section": "Papers",
                            "kind": "paper",
                            "title": "Contact-aware policy transfer",
                            "summary": "Tracks recent contact handling results.",
                            "date": "2026-03-30T08:30:00Z",
                            "source": {"label": "Paper", "url": "https://example.com/paper"},
                            "implementation_notes": "Translate the evaluation setup into Newton examples.",
                            "next_steps": ["Draft a simulator parity checklist."],
                            "comments": [{"author": "Research Scientist", "body": "High relevance."}],
                            "tags": ["control", "contacts"],
                        }
                    ],
                    "chat": {"mode": "stub", "agent_name": "Research Scientist"},
                },
            )

            with mock.patch.dict(os.environ, {}, clear=True):
                payload = build_research_dashboard_payload(artifact_path)

            self.assertEqual(payload["status"], "ready")
            self.assertEqual(payload["stats"]["entry_count"], 1)
            self.assertEqual(payload["stats"]["section_count"], 1)
            self.assertEqual(payload["stats"]["timeline_count"], 1)
            self.assertEqual(payload["stats"]["comment_count"], 1)
            self.assertEqual(payload["sections"][0]["title"], "Papers")
            self.assertEqual(payload["timeline"][0]["entry_id"], "paper-1")
            self.assertEqual(payload["entries"][0]["next_steps"][0], "Draft a simulator parity checklist.")
            self.assertEqual(payload["entries"][0]["comments"][0]["author"], "Research Scientist")

    def test_build_research_dashboard_payload_reports_missing_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "missing.json"

            with mock.patch.dict(os.environ, {}, clear=True):
                payload = build_research_dashboard_payload(artifact_path)

            self.assertEqual(payload["status"], "missing")
            self.assertFalse(payload["artifact_available"])
            self.assertEqual(payload["stats"]["entry_count"], 0)

    def test_assess_research_artifact_freshness_uses_semantic_timestamps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "research.json"
            _write_json(
                artifact_path,
                {
                    "generated_at": "2026-03-29T10:00:00+00:00",
                    "entries": [
                        {
                            "id": "paper-1",
                            "title": "Old evidence",
                            "date": "2026-03-29T12:00:00+00:00",
                        }
                    ],
                },
            )

            freshness = assess_research_artifact_freshness(
                artifact_path,
                24.0,
                now=datetime.datetime(2026, 4, 1, 12, 0, tzinfo=datetime.timezone.utc),
            )

            self.assertTrue(freshness["is_stale"])
            self.assertEqual(freshness["source"], "artifact")
            self.assertEqual(freshness["artifact_timestamp"], "2026-03-29T12:00:00+00:00")
            self.assertGreater(freshness["age_hours"], 24.0)


class ResearchDashboardHandlerTest(unittest.TestCase):
    def test_request_handler_serves_payload_and_live_chat_reply(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "research.json"
            _write_json(
                artifact_path,
                {
                    "entries": [
                        {
                            "id": "project-1",
                            "section": "Projects",
                            "kind": "project",
                            "title": "Robotics benchmark repo",
                            "summary": "A structured project entry.",
                            "date": "2026-03-29T18:10:00Z",
                            "next_steps": ["Mirror the task taxonomy."],
                        }
                    ],
                    "chat": {"mode": "stub", "agent_name": "Research Scientist"},
                },
            )

            handler = partial(ResearchDashboardRequestHandler, artifact_path=artifact_path)
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                relay_config = _PaperclipRelayConfig(
                    api_url="http://paperclip.test",
                    api_key="token",
                    run_id="run-1",
                    issue_id="issue-1",
                    agent_name="Research Scientist",
                    timeout_seconds=5,
                    poll_interval_seconds=0.01,
                )
                with mock.patch(
                    "newton._src.tools.research_dashboard._load_paperclip_relay_config",
                    return_value=relay_config,
                ), mock.patch(
                    "newton._src.tools.research_dashboard._post_chat_prompt",
                    return_value={"id": "comment-1"},
                ), mock.patch(
                    "newton._src.tools.research_dashboard._wait_for_chat_reply",
                    return_value={"status": "ok", "reply": "Implement the benchmark taxonomy next.", "comment_id": "comment-2"},
                ):
                    base_url = f"http://127.0.0.1:{server.server_port}"
                    with urllib.request.urlopen(f"{base_url}/api/research") as response:
                        payload = json.loads(response.read().decode("utf-8"))
                    self.assertEqual(payload["stats"]["entry_count"], 1)
                    self.assertEqual(payload["chat"]["mode"], "relay")

                    request = urllib.request.Request(
                        f"{base_url}/api/chat",
                        data=json.dumps({"message": "What should we implement next?"}).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(request) as response:
                        chat_payload = json.loads(response.read().decode("utf-8"))
                    self.assertEqual(chat_payload["status"], "ok")
                    self.assertEqual(chat_payload["agent_name"], "Research Scientist")
                    self.assertIn("benchmark taxonomy", chat_payload["reply"])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_request_handler_returns_gateway_timeout_when_relay_does_not_answer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "research.json"
            _write_json(
                artifact_path,
                {
                    "entries": [
                        {
                            "id": "project-1",
                            "section": "Projects",
                            "kind": "project",
                            "title": "Robotics benchmark repo",
                            "summary": "A structured project entry.",
                            "date": "2026-03-29T18:10:00Z",
                        }
                    ],
                },
            )

            handler = partial(ResearchDashboardRequestHandler, artifact_path=artifact_path)
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                relay_config = _PaperclipRelayConfig(
                    api_url="http://paperclip.test",
                    api_key="token",
                    run_id="run-1",
                    issue_id="issue-1",
                    agent_name="Research Scientist",
                    timeout_seconds=3,
                    poll_interval_seconds=0.01,
                )
                with mock.patch(
                    "newton._src.tools.research_dashboard._load_paperclip_relay_config",
                    return_value=relay_config,
                ), mock.patch(
                    "newton._src.tools.research_dashboard._post_chat_prompt",
                    return_value={"id": "comment-1"},
                ), mock.patch(
                    "newton._src.tools.research_dashboard._wait_for_chat_reply",
                    return_value={"status": "timeout", "reply": "", "comment_id": ""},
                ):
                    base_url = f"http://127.0.0.1:{server.server_port}"
                    request = urllib.request.Request(
                        f"{base_url}/api/chat",
                        data=json.dumps({"message": "What should we implement next?"}).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with self.assertRaises(urllib.error.HTTPError) as ctx:
                        urllib.request.urlopen(request)
                    self.assertEqual(ctx.exception.code, 504)
                    payload = json.loads(ctx.exception.read().decode("utf-8"))
                    self.assertEqual(payload["status"], "error")
                    self.assertIn("did not answer within 3 seconds", payload["error"])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_request_handler_creates_entry_and_lists_it_via_api(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "research.json"
            _write_json(
                artifact_path,
                {
                    "entries": [
                        {
                            "id": "project-1",
                            "section": "Projects",
                            "kind": "project",
                            "title": "Robotics benchmark repo",
                            "summary": "A structured project entry.",
                            "date": "2026-03-29T18:10:00Z",
                            "tags": ["benchmarks"],
                        }
                    ],
                    "chat": {"mode": "stub", "agent_name": "Research Scientist"},
                },
            )

            handler = partial(ResearchDashboardRequestHandler, artifact_path=artifact_path)
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base_url = f"http://127.0.0.1:{server.server_port}"
                request = urllib.request.Request(
                    f"{base_url}/api/research/entries",
                    data=json.dumps(
                        {
                            "title": "Differentiable contact survey",
                            "section": "Papers",
                            "kind": "paper",
                            "summary": "Collects contact-gradient references.",
                            "implementation_notes": "Use it to scope a Newton query spike.",
                            "next_steps": ["Create a prototype issue."],
                            "tags": ["differentiable", "contacts"],
                            "author": "Research Scientist",
                            "note": "Relevant to the current dashboard work.",
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request) as response:
                    self.assertEqual(response.status, 201)
                    created_entry = json.loads(response.read().decode("utf-8"))

                self.assertEqual(created_entry["id"], "differentiable-contact-survey")
                self.assertEqual(created_entry["comments"][0]["author"], "Research Scientist")

                with urllib.request.urlopen(
                    f"{base_url}/api/research/entries?q=differentiable&section=Papers&tag=contacts"
                ) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                self.assertEqual(payload["count"], 1)
                self.assertEqual(payload["entries"][0]["id"], "differentiable-contact-survey")

                with urllib.request.urlopen(f"{base_url}/api/research/entries/differentiable-contact-survey") as response:
                    fetched_entry = json.loads(response.read().decode("utf-8"))
                self.assertEqual(fetched_entry["title"], "Differentiable contact survey")

                dashboard_payload = build_research_dashboard_payload(artifact_path)
                self.assertEqual(dashboard_payload["stats"]["entry_count"], 2)
                self.assertEqual(dashboard_payload["timeline"][0]["entry_id"], "differentiable-contact-survey")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_request_handler_rejects_entry_without_title(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "research.json"
            _write_json(artifact_path, {"entries": []})

            handler = partial(ResearchDashboardRequestHandler, artifact_path=artifact_path)
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base_url = f"http://127.0.0.1:{server.server_port}"
                request = urllib.request.Request(
                    f"{base_url}/api/research/entries",
                    data=json.dumps({"summary": "Missing title."}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    urllib.request.urlopen(request)
                self.assertEqual(ctx.exception.code, 400)
                payload = json.loads(ctx.exception.read().decode("utf-8"))
                self.assertEqual(payload["error"], "title is required.")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


class ResearchDashboardRelayTest(unittest.TestCase):
    def test_fetch_issue_comments_after_falls_back_to_full_comment_list(self):
        relay_config = _PaperclipRelayConfig(
            api_url="http://paperclip.test",
            api_key="token",
            run_id="run-1",
            issue_id="issue-1",
            agent_name="Research Scientist",
            timeout_seconds=5,
            poll_interval_seconds=0.01,
        )

        with mock.patch(
            "newton._src.tools.research_dashboard._paperclip_request_json",
            side_effect=[
                urllib.error.HTTPError(
                    url="http://paperclip.test/api/issues/issue-1/comments?after=comment-1&order=asc",
                    code=500,
                    msg="Internal Server Error",
                    hdrs=None,
                    fp=None,
                ),
                [
                    {"id": "comment-1", "body": "prompt"},
                    {"id": "comment-2", "body": "reply"},
                    {"id": "comment-3", "body": "follow-up"},
                ],
            ],
        ):
            comments = _fetch_issue_comments_after(relay_config, "comment-1")

        self.assertEqual([item["id"] for item in comments], ["comment-2", "comment-3"])


class ResearchDashboardCliTest(unittest.TestCase):
    def test_make_argument_parser_uses_repo_stable_default_artifact_path(self):
        args = make_argument_parser().parse_args([])

        self.assertEqual(Path(args.artifact_path), default_research_artifact_path())

    def test_make_argument_parser_uses_env_default_freshness_threshold(self):
        with mock.patch.dict(os.environ, {"NEWTON_RESEARCH_MAX_ARTIFACT_AGE_HOURS": "12"}):
            self.assertEqual(default_research_max_artifact_age_hours(), 12.0)
            args = make_argument_parser().parse_args([])

        self.assertEqual(args.max_artifact_age_hours, 12.0)


class ResearchDashboardHtmlTest(unittest.TestCase):
    def test_rendered_html_places_timeline_below_primary_content(self):
        html = _render_index_html()

        self.assertLess(html.index("<section class=\"content-layout\">"), html.index("<section class=\"panel timeline-shell\">"))
        self.assertLess(html.index("Research sections"), html.index("Research chat"))
        self.assertLess(html.index("Research chat"), html.index("Timeline"))

    def test_rendered_html_keeps_quote_escape_mapping_valid_for_browser_js(self):
        html = _render_index_html()

        self.assertIn('\'"\': "&quot;"', html)
