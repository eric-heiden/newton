# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Serve a lightweight dashboard for structured research artifacts."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from uuid import uuid4


_DEFAULT_CHAT_TIMEOUT_SECONDS = 20
_DEFAULT_CHAT_POLL_INTERVAL_SECONDS = 1
_DEFAULT_CHAT_CONTEXT_ENTRY_LIMIT = 6
_DEFAULT_MAX_ARTIFACT_AGE_HOURS = 36.0
_RESEARCH_CHAT_PROMPT_ISSUE_ID = "f5a6c132-0c32-4898-bd6e-100d4085c76c"
_RESEARCH_CHAT_PROMPT_AGENT_NAME = "Research Scientist"
_CHAT_REPLY_PREFIX = "CHAT-REPLY:"
_DEFAULT_ENTRY_LIMIT = 20
_ARTIFACT_LOCK = threading.Lock()


@dataclass(frozen=True)
class _PaperclipRelayConfig:
    api_url: str
    api_key: str
    run_id: str
    issue_id: str
    agent_name: str
    timeout_seconds: int
    poll_interval_seconds: float


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _utc_now_datetime() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def default_research_artifact_path() -> Path:
    """Return the default structured artifact path for the dashboard runtime."""
    configured = os.environ.get("NEWTON_RESEARCH_ARTIFACT_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[3]
    return (repo_root / "scripts" / "research_dashboard.json").resolve()


def default_research_max_artifact_age_hours() -> float:
    """Return the default maximum age for a served research artifact."""
    return _as_float(os.environ.get("NEWTON_RESEARCH_MAX_ARTIFACT_AGE_HOURS"), _DEFAULT_MAX_ARTIFACT_AGE_HOURS)


def _as_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _as_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _load_paperclip_relay_config() -> _PaperclipRelayConfig | None:
    api_url = _as_text(os.environ.get("PAPERCLIP_API_URL"))
    api_key = _as_text(os.environ.get("PAPERCLIP_API_KEY"))
    run_id = _as_text(os.environ.get("PAPERCLIP_RUN_ID"))
    if not api_url or not api_key or not run_id:
        return None
    return _PaperclipRelayConfig(
        api_url=api_url.rstrip("/"),
        api_key=api_key,
        run_id=run_id,
        issue_id=_as_text(os.environ.get("NEWTON_RESEARCH_CHAT_ISSUE_ID"), default=_RESEARCH_CHAT_PROMPT_ISSUE_ID),
        agent_name=_as_text(
            os.environ.get("NEWTON_RESEARCH_CHAT_AGENT_NAME"),
            default=_RESEARCH_CHAT_PROMPT_AGENT_NAME,
        ),
        timeout_seconds=_as_int(
            os.environ.get("NEWTON_RESEARCH_CHAT_TIMEOUT_SECONDS"),
            default=_DEFAULT_CHAT_TIMEOUT_SECONDS,
        ),
        poll_interval_seconds=_as_float(
            os.environ.get("NEWTON_RESEARCH_CHAT_POLL_INTERVAL_SECONDS"),
            default=_DEFAULT_CHAT_POLL_INTERVAL_SECONDS,
        ),
    )


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _as_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _slugify_identifier(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized or "research-entry"


def _dedupe_identifier(identifier: str, existing_ids: set[str]) -> str:
    if identifier not in existing_ids:
        return identifier
    suffix = 2
    while f"{identifier}-{suffix}" in existing_ids:
        suffix += 1
    return f"{identifier}-{suffix}"


def _normalize_link(value: Any) -> dict[str, str] | None:
    if isinstance(value, str):
        url = value.strip()
        if not url:
            return None
        return {"label": url, "url": url}
    if not isinstance(value, dict):
        return None
    url = _as_text(value.get("url"))
    if not url:
        return None
    return {
        "label": _as_text(value.get("label"), default=url),
        "url": url,
    }


def _normalize_comments(value: Any) -> list[dict[str, str]]:
    comments: list[dict[str, str]] = []
    for item in _as_list(value):
        if isinstance(item, str):
            body = item.strip()
            if body:
                comments.append({"author": "Agent", "body": body, "created_at": ""})
            continue
        if not isinstance(item, dict):
            continue
        body = _as_text(item.get("body") or item.get("comment"))
        if not body:
            continue
        comments.append(
            {
                "author": _as_text(item.get("author"), default="Agent"),
                "body": body,
                "created_at": _as_text(item.get("created_at") or item.get("date")),
            }
        )
    return comments


def _normalize_next_steps(value: Any) -> list[str]:
    steps: list[str] = []
    for item in _as_list(value):
        text = _as_text(item)
        if text:
            steps.append(text)
    return steps


def _normalize_tags(value: Any) -> list[str]:
    tags: list[str] = []
    for item in _as_list(value):
        text = _as_text(item)
        if text:
            tags.append(text)
    return tags


def _normalize_entry(raw: dict[str, Any], fallback_index: int) -> dict[str, Any]:
    source = _normalize_link(raw.get("source") or raw.get("link") or raw.get("url"))
    date = _as_text(raw.get("updated_at") or raw.get("published_at") or raw.get("date") or raw.get("discovered_at"))
    section = _as_text(raw.get("section") or raw.get("category"), default="Unsorted")
    title = _as_text(raw.get("title"), default=f"Research item {fallback_index}")
    summary = _as_text(raw.get("summary") or raw.get("abstract") or raw.get("description"))
    implementation_notes = _as_text(
        raw.get("implementation_notes") or raw.get("implementation_relevance") or raw.get("details") or raw.get("notes")
    )
    next_steps = _normalize_next_steps(raw.get("next_steps") or raw.get("actions"))
    comments = _normalize_comments(raw.get("comments") or raw.get("agent_comments"))
    tags = _normalize_tags(raw.get("tags"))
    kind = _as_text(raw.get("kind") or raw.get("type"), default="artifact")
    identifier = _as_text(raw.get("id"), default=f"entry-{fallback_index}")
    return {
        "id": identifier,
        "title": title,
        "summary": summary,
        "section": section,
        "kind": kind,
        "date": date,
        "source": source,
        "implementation_notes": implementation_notes,
        "next_steps": next_steps,
        "comments": comments,
        "tags": tags,
    }


def _load_raw_artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw_artifact = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_artifact, dict):
        raise ValueError("Research artifact root must be a JSON object.")
    return raw_artifact


def _write_raw_artifact(path: Path, raw_artifact: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(json.dumps(raw_artifact, indent=2) + "\n", encoding="utf-8")
    temp_path.replace(path)


def _normalize_timeline(
    raw_timeline: Any,
    entries: list[dict[str, Any]],
) -> list[dict[str, str]]:
    updates: list[dict[str, str]] = []
    for index, item in enumerate(_as_list(raw_timeline), start=1):
        if not isinstance(item, dict):
            continue
        title = _as_text(item.get("title"), default=f"Update {index}")
        summary = _as_text(item.get("summary") or item.get("body"))
        date = _as_text(item.get("date") or item.get("updated_at"))
        entry_id = _as_text(item.get("entry_id"))
        updates.append(
            {
                "id": _as_text(item.get("id"), default=f"timeline-{index}"),
                "title": title,
                "summary": summary,
                "date": date,
                "entry_id": entry_id,
            }
        )

    if updates:
        return updates

    for entry in entries:
        updates.append(
            {
                "id": f"timeline-{entry['id']}",
                "title": entry["title"],
                "summary": entry["summary"],
                "date": entry["date"],
                "entry_id": entry["id"],
            }
        )

    updates.sort(key=lambda item: (item["date"], item["title"]), reverse=True)
    return updates[:12]


def _normalize_sections(
    raw_sections: Any,
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    entry_map = {entry["id"]: entry for entry in entries}
    sections: list[dict[str, Any]] = []
    if isinstance(raw_sections, list):
        for index, item in enumerate(raw_sections, start=1):
            if not isinstance(item, dict):
                continue
            title = _as_text(item.get("title") or item.get("name"), default=f"Section {index}")
            identifier = _as_text(item.get("id"), default=f"section-{index}")
            entry_ids: list[str] = []
            for raw_entry in _as_list(item.get("entry_ids") or item.get("items")):
                if isinstance(raw_entry, str):
                    if raw_entry in entry_map:
                        entry_ids.append(raw_entry)
                    continue
                if isinstance(raw_entry, dict):
                    raw_id = _as_text(raw_entry.get("id"))
                    if raw_id in entry_map:
                        entry_ids.append(raw_id)
            if entry_ids:
                sections.append(
                    {
                        "id": identifier,
                        "title": title,
                        "description": _as_text(item.get("description")),
                        "entry_ids": entry_ids,
                    }
                )

    if sections:
        return sections

    grouped: dict[str, list[str]] = {}
    for entry in entries:
        grouped.setdefault(entry["section"], []).append(entry["id"])
    for index, (title, entry_ids) in enumerate(sorted(grouped.items()), start=1):
        sections.append(
            {
                "id": f"section-{index}",
                "title": title,
                "description": "",
                "entry_ids": entry_ids,
            }
        )
    return sections


def _normalize_chat(raw_chat: Any) -> dict[str, Any]:
    if not isinstance(raw_chat, dict):
        raw_chat = {}
    relay_config = _load_paperclip_relay_config()
    mode = "relay" if relay_config else _as_text(raw_chat.get("mode"), default="stub")
    return {
        "mode": mode,
        "agent_name": _as_text(
            raw_chat.get("agent_name"),
            default=relay_config.agent_name if relay_config else "Research Scientist",
        ),
        "placeholder": _as_text(
            raw_chat.get("placeholder"),
            default="Ask how this research should turn into Newton implementation work.",
        ),
        "status": _as_text(
            None if relay_config else raw_chat.get("status"),
            default="Live relay ready." if relay_config else "Awaiting upstream integration.",
        ),
    }


def _parse_iso_datetime(value: Any) -> datetime.datetime | None:
    text = _as_text(value)
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed.astimezone(datetime.timezone.utc)


def assess_research_artifact_freshness(
    artifact_path: str | Path,
    max_age_hours: float,
    *,
    now: datetime.datetime | None = None,
) -> dict[str, Any]:
    """Assess whether the research artifact is fresh enough to serve."""
    path = Path(artifact_path)
    checked_at = now.astimezone(datetime.timezone.utc) if now is not None else _utc_now_datetime()
    freshness: dict[str, Any] = {
        "checked_at": checked_at.isoformat(),
        "max_age_hours": float(max_age_hours),
        "artifact_timestamp": None,
        "age_hours": None,
        "is_stale": False,
        "source": None,
        "reason": "",
    }

    if not path.exists():
        freshness["reason"] = "Research artifact file does not exist."
        return freshness

    artifact_timestamp: datetime.datetime | None = None
    source = "mtime"
    try:
        raw_artifact = _load_raw_artifact(path)
    except (json.JSONDecodeError, ValueError):
        raw_artifact = {}

    semantic_candidates = [
        _parse_iso_datetime(raw_artifact.get("generated_at")),
        _parse_iso_datetime(raw_artifact.get("updated_at")),
        _parse_iso_datetime(raw_artifact.get("date")),
    ]
    for item in _as_list(raw_artifact.get("entries") or raw_artifact.get("items")):
        if not isinstance(item, dict):
            continue
        semantic_candidates.extend(
            [
                _parse_iso_datetime(item.get("updated_at")),
                _parse_iso_datetime(item.get("published_at")),
                _parse_iso_datetime(item.get("date")),
                _parse_iso_datetime(item.get("discovered_at")),
            ]
        )
    for item in _as_list(raw_artifact.get("timeline")):
        if not isinstance(item, dict):
            continue
        semantic_candidates.extend(
            [
                _parse_iso_datetime(item.get("updated_at")),
                _parse_iso_datetime(item.get("date")),
            ]
        )

    parsed_candidates = [item for item in semantic_candidates if item is not None]
    if parsed_candidates:
        artifact_timestamp = max(parsed_candidates)
        source = "artifact"
    else:
        artifact_timestamp = datetime.datetime.fromtimestamp(path.stat().st_mtime, tz=datetime.timezone.utc)

    age_hours = max((checked_at - artifact_timestamp).total_seconds() / 3600.0, 0.0)
    freshness["artifact_timestamp"] = artifact_timestamp.isoformat()
    freshness["age_hours"] = age_hours
    freshness["source"] = source
    freshness["is_stale"] = age_hours > float(max_age_hours)
    freshness["reason"] = (
        f"Research artifact is stale at {age_hours:.1f}h old; max allowed is {float(max_age_hours):.1f}h."
        if freshness["is_stale"]
        else f"Research artifact is fresh at {age_hours:.1f}h old."
    )
    return freshness


def build_research_dashboard_payload(artifact_path: str | Path) -> dict[str, Any]:
    """Build a UI-ready payload for the research dashboard."""
    path = Path(artifact_path)
    freshness = assess_research_artifact_freshness(path, default_research_max_artifact_age_hours())
    payload: dict[str, Any] = {
        "generated_at": _utc_now(),
        "artifact_path": str(path),
        "artifact_available": path.exists(),
        "freshness": freshness,
        "status": "ready",
        "message": "",
        "timeline": [],
        "sections": [],
        "entries": [],
        "chat": _normalize_chat(None),
        "stats": {
            "entry_count": 0,
            "section_count": 0,
            "timeline_count": 0,
            "comment_count": 0,
        },
    }

    if not path.exists():
        payload["status"] = "missing"
        payload["message"] = "Research artifact not found yet. Waiting for the first structured feed."
        return payload

    try:
        raw_artifact = _load_raw_artifact(path)
    except json.JSONDecodeError as error:
        payload["status"] = "invalid"
        payload["message"] = f"Research artifact JSON is invalid: {error.msg}."
        return payload
    except ValueError as error:
        payload["status"] = "invalid"
        payload["message"] = str(error)
        return payload

    entries = [
        _normalize_entry(item, index)
        for index, item in enumerate(_as_list(raw_artifact.get("entries") or raw_artifact.get("items")), start=1)
        if isinstance(item, dict)
    ]
    sections = _normalize_sections(raw_artifact.get("sections"), entries)
    timeline = _normalize_timeline(raw_artifact.get("timeline"), entries)
    chat = _normalize_chat(raw_artifact.get("chat"))

    payload["entries"] = entries
    payload["sections"] = sections
    payload["timeline"] = timeline
    payload["chat"] = chat
    payload["message"] = _as_text(raw_artifact.get("message"))
    payload["stats"] = {
        "entry_count": len(entries),
        "section_count": len(sections),
        "timeline_count": len(timeline),
        "comment_count": sum(len(entry["comments"]) for entry in entries),
    }

    if not entries:
        payload["status"] = "empty"
        if not payload["message"]:
            payload["message"] = "Research artifact loaded, but it does not contain any entries yet."

    return payload


def _build_entry_record(request_payload: dict[str, Any], existing_ids: set[str]) -> dict[str, Any]:
    title = _as_text(request_payload.get("title"))
    if not title:
        raise ValueError("title is required.")

    identifier = _as_text(request_payload.get("id"))
    if identifier:
        identifier = _slugify_identifier(identifier)
    else:
        identifier = _slugify_identifier(title)
    identifier = _dedupe_identifier(identifier, existing_ids)

    entry_date = _as_text(request_payload.get("date"), default=_utc_now())
    source = _normalize_link(request_payload.get("source") or request_payload.get("url"))
    comments = _normalize_comments(request_payload.get("comments"))
    author = _as_text(request_payload.get("author"))
    note = _as_text(request_payload.get("note") or request_payload.get("comment"))
    if author or note:
        comments.append(
            {
                "author": author or "Research Scientist",
                "body": note or "Added via research dashboard capture API.",
                "created_at": entry_date,
            }
        )

    return {
        "id": identifier,
        "section": _as_text(request_payload.get("section"), default="Unsorted"),
        "kind": _as_text(request_payload.get("kind"), default="note"),
        "title": title,
        "summary": _as_text(request_payload.get("summary")),
        "date": entry_date,
        "source": source,
        "implementation_notes": _as_text(request_payload.get("implementation_notes")),
        "next_steps": _normalize_next_steps(request_payload.get("next_steps")),
        "comments": comments,
        "tags": _normalize_tags(request_payload.get("tags")),
    }


def _append_entry_to_artifact(artifact_path: Path, request_payload: dict[str, Any]) -> dict[str, Any]:
    with _ARTIFACT_LOCK:
        raw_artifact = _load_raw_artifact(artifact_path)
        raw_entries = _as_list(raw_artifact.get("entries") or raw_artifact.get("items"))
        existing_ids = {
            _as_text(item.get("id"))
            for item in raw_entries
            if isinstance(item, dict) and _as_text(item.get("id"))
        }
        entry = _build_entry_record(request_payload, existing_ids)
        raw_entries.append(entry)
        raw_artifact["entries"] = raw_entries
        if "items" in raw_artifact:
            raw_artifact.pop("items")

        raw_timeline = _as_list(raw_artifact.get("timeline"))
        raw_timeline.insert(
            0,
            {
                "id": f"update-{entry['id']}",
                "title": f"Added {entry['title']}",
                "summary": entry["summary"] or entry["implementation_notes"] or "New research entry captured.",
                "date": entry["date"],
                "entry_id": entry["id"],
            },
        )
        raw_artifact["timeline"] = raw_timeline[:24]
        _write_raw_artifact(artifact_path, raw_artifact)
        return entry


def _filter_entries(
    entries: list[dict[str, Any]],
    query: str,
    section: str,
    kind: str,
    tag: str,
    limit: int,
) -> list[dict[str, Any]]:
    normalized_query = query.lower()
    normalized_section = section.lower()
    normalized_kind = kind.lower()
    normalized_tag = tag.lower()
    filtered: list[dict[str, Any]] = []
    for entry in entries:
        if normalized_section and _as_text(entry.get("section")).lower() != normalized_section:
            continue
        if normalized_kind and _as_text(entry.get("kind")).lower() != normalized_kind:
            continue
        entry_tags = [_as_text(item).lower() for item in entry.get("tags", []) if _as_text(item)]
        if normalized_tag and normalized_tag not in entry_tags:
            continue
        if normalized_query:
            haystack = " ".join(
                [
                    _as_text(entry.get("id")),
                    _as_text(entry.get("title")),
                    _as_text(entry.get("summary")),
                    _as_text(entry.get("implementation_notes")),
                    " ".join(entry_tags),
                ]
            ).lower()
            if normalized_query not in haystack:
                continue
        filtered.append(entry)
        if len(filtered) >= limit:
            break
    return filtered


def _summarize_chat_entries(entries: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for entry in entries[:_DEFAULT_CHAT_CONTEXT_ENTRY_LIMIT]:
        notes = _as_text(entry.get("implementation_notes"))
        next_steps = entry.get("next_steps") if isinstance(entry.get("next_steps"), list) else []
        line = (
            f"- [{_as_text(entry.get('section'), default='Unsorted')}] "
            f"{_as_text(entry.get('title'), default='Untitled')}: "
            f"{_as_text(entry.get('summary'), default='No summary provided.')}"
        )
        if notes:
            line += f" Implementation notes: {notes}"
        if next_steps:
            line += f" Next steps: {'; '.join(_as_text(step) for step in next_steps if _as_text(step))}."
        lines.append(line)
    return "\n".join(lines) if lines else "- No structured research entries are available yet."


def _build_chat_prompt(
    request_id: str,
    agent_name: str,
    message: str,
    dashboard_payload: dict[str, Any],
) -> str:
    artifact_status = _as_text(dashboard_payload.get("status"), default="unknown")
    artifact_message = _as_text(dashboard_payload.get("message"))
    context_summary = _summarize_chat_entries(
        dashboard_payload.get("entries") if isinstance(dashboard_payload.get("entries"), list) else []
    )
    return (
        "## Dashboard chat relay\n\n"
        f"@{agent_name} Please answer this dashboard chat request.\n\n"
        f"Reply on this issue with a comment whose first line is exactly `{_CHAT_REPLY_PREFIX} {request_id}`.\n"
        "Put the answer on the lines after that marker.\n\n"
        "Constraints:\n"
        "- Ground the answer in the structured artifact context below.\n"
        "- If the artifact does not support the answer, say so explicitly.\n"
        "- Keep the answer concise and implementation-oriented.\n"
        "- Do not mention Paperclip, issue workflow, or relay mechanics.\n\n"
        f"Question:\n{message}\n\n"
        "Artifact context:\n"
        f"- status: {artifact_status}\n"
        f"- message: {artifact_message or 'none'}\n"
        f"- artifact path: {_as_text(dashboard_payload.get('artifact_path'))}\n"
        f"- entry count: {dashboard_payload.get('stats', {}).get('entry_count', 0)}\n"
        f"{context_summary}\n"
    )


def _paperclip_headers(config: _PaperclipRelayConfig) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
        "X-Paperclip-Run-Id": config.run_id,
    }


def _paperclip_request_json(
    config: _PaperclipRelayConfig,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any] | list[Any]:
    request = urllib.request.Request(
        f"{config.api_url}{path}",
        data=(json.dumps(payload).encode("utf-8") if payload is not None else None),
        headers=_paperclip_headers(config),
        method=method,
    )
    with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_chat_prompt(
    config: _PaperclipRelayConfig,
    body: str,
) -> dict[str, Any]:
    response = _paperclip_request_json(
        config,
        "POST",
        f"/api/issues/{config.issue_id}/comments",
        {"body": body},
    )
    if not isinstance(response, dict):
        raise ValueError("Paperclip comment response must be a JSON object.")
    return response


def _fetch_issue_comments_after(
    config: _PaperclipRelayConfig,
    after_comment_id: str,
) -> list[dict[str, Any]]:
    try:
        response = _paperclip_request_json(
            config,
            "GET",
            f"/api/issues/{config.issue_id}/comments?after={urllib.parse.quote(after_comment_id)}&order=asc",
        )
    except urllib.error.HTTPError as error:
        if error.code < HTTPStatus.INTERNAL_SERVER_ERROR:
            raise
        response = _paperclip_request_json(
            config,
            "GET",
            f"/api/issues/{config.issue_id}/comments?order=asc",
        )
    if not isinstance(response, list):
        raise ValueError("Paperclip comments response must be a JSON array.")
    comments = [item for item in response if isinstance(item, dict)]
    if not comments:
        return comments

    after_index = -1
    for index, item in enumerate(comments):
        if _as_text(item.get("id")) == after_comment_id:
            after_index = index
            break
    if after_index < 0:
        return comments
    return comments[after_index + 1 :]


def _extract_chat_reply(comment_body: str, request_id: str) -> str | None:
    lines = comment_body.splitlines()
    expected_prefix = f"{_CHAT_REPLY_PREFIX} {request_id}"
    for index, line in enumerate(lines):
        if line.strip() != expected_prefix:
            continue
        reply = "\n".join(lines[index + 1 :]).strip()
        return reply or None
    return None


def _wait_for_chat_reply(
    config: _PaperclipRelayConfig,
    after_comment_id: str,
    request_id: str,
) -> dict[str, Any]:
    deadline = time.monotonic() + config.timeout_seconds
    while time.monotonic() < deadline:
        comments = _fetch_issue_comments_after(config, after_comment_id)
        for comment in comments:
            body = _as_text(comment.get("body"))
            reply = _extract_chat_reply(body, request_id)
            if reply:
                return {
                    "status": "ok",
                    "reply": reply,
                    "comment_id": _as_text(comment.get("id")),
                }
        time.sleep(config.poll_interval_seconds)
    return {
        "status": "timeout",
        "reply": "",
        "comment_id": "",
    }


def _build_chat_response(
    chat_config: dict[str, Any],
    message: str,
    dashboard_payload: dict[str, Any],
) -> tuple[HTTPStatus, dict[str, Any]]:
    agent_name = chat_config["agent_name"]
    relay_config = _load_paperclip_relay_config()
    if relay_config is None:
        return HTTPStatus.SERVICE_UNAVAILABLE, {
            "status": "error",
            "agent_name": agent_name,
            "error": "Dashboard chat relay is missing Paperclip runtime configuration.",
        }

    request_id = f"dashboard-chat-{uuid4().hex[:12]}"
    prompt = _build_chat_prompt(request_id, relay_config.agent_name, message, dashboard_payload)
    try:
        comment = _post_chat_prompt(relay_config, prompt)
        comment_id = _as_text(comment.get("id"))
        if not comment_id:
            raise ValueError("Paperclip did not return the created comment id.")
        relay_result = _wait_for_chat_reply(relay_config, comment_id, request_id)
    except TimeoutError:
        relay_result = {"status": "timeout", "reply": "", "comment_id": ""}
    except Exception as error:
        return HTTPStatus.BAD_GATEWAY, {
            "status": "error",
            "agent_name": agent_name,
            "error": f"Research chat relay failed: {error}",
        }

    if relay_result["status"] == "timeout":
        return HTTPStatus.GATEWAY_TIMEOUT, {
            "status": "error",
            "agent_name": agent_name,
            "error": (
                f"{agent_name} did not answer within {relay_config.timeout_seconds} seconds. "
                "Retry the question or inspect the backing issue thread."
            ),
        }

    return HTTPStatus.OK, {
        "status": "ok",
        "agent_name": agent_name,
        "reply": relay_result["reply"],
    }


def _render_index_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Newton Research Dashboard</title>
  <style>
    :root {
      --canvas: #f4efe4;
      --ink: #10222b;
      --muted: #58646a;
      --card: rgba(255, 251, 245, 0.88);
      --line: rgba(16, 34, 43, 0.12);
      --accent: #c55a31;
      --accent-deep: #8f3414;
      --accent-soft: rgba(197, 90, 49, 0.12);
      --teal: #1f6a70;
      --shadow: 0 22px 60px rgba(16, 34, 43, 0.12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at 10% 10%, rgba(197, 90, 49, 0.16), transparent 26rem),
        radial-gradient(circle at 90% 0%, rgba(31, 106, 112, 0.13), transparent 24rem),
        linear-gradient(180deg, #faf6ef, var(--canvas));
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
      min-height: 100vh;
    }
    main {
      width: min(1380px, calc(100vw - 2rem));
      margin: 0 auto;
      padding: 1.5rem 0 2rem;
    }
    .hero, .panel, .chat-shell, .detail-shell {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 1.5rem;
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }
    .hero {
      padding: 1.5rem;
      display: grid;
      gap: 1.25rem;
      grid-template-columns: 2fr 1fr;
      align-items: end;
    }
    .eyebrow, .meta, .badge, button, input, textarea {
      font-family: "IBM Plex Mono", "SFMono-Regular", Consolas, monospace;
    }
    .eyebrow {
      display: inline-flex;
      padding: 0.3rem 0.55rem;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent-deep);
      font-size: 0.76rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    h1, h2, h3 {
      margin: 0;
      font-weight: 600;
    }
    h1 {
      margin-top: 0.8rem;
      font-size: clamp(2.4rem, 6vw, 5rem);
      line-height: 0.94;
      max-width: 12ch;
    }
    p {
      margin: 0;
      color: var(--muted);
      line-height: 1.6;
    }
    .hero-copy {
      display: grid;
      gap: 0.9rem;
      max-width: 70ch;
    }
    .status-card {
      padding: 1rem;
      border-radius: 1.2rem;
      background: linear-gradient(180deg, rgba(16, 34, 43, 0.04), rgba(16, 34, 43, 0.01));
      border: 1px solid var(--line);
    }
    .stats {
      display: grid;
      gap: 1rem;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      margin-top: 1.2rem;
    }
    .stat {
      padding: 1rem;
      border: 1px solid var(--line);
      border-radius: 1.2rem;
      background: rgba(255, 255, 255, 0.6);
    }
    .stat-value {
      display: block;
      margin-top: 0.45rem;
      font-size: 2rem;
      line-height: 1;
    }
    .content-layout {
      display: grid;
      gap: 1rem;
      grid-template-columns: minmax(0, 1.8fr) minmax(320px, 0.95fr);
      margin-top: 1rem;
      align-items: start;
    }
    .sidebar-stack {
      display: grid;
      gap: 1rem;
      align-items: start;
    }
    .timeline-shell {
      margin-top: 1rem;
    }
    .panel, .chat-shell, .detail-shell {
      padding: 1rem;
    }
    .panel-header, .detail-header, .chat-header {
      display: flex;
      justify-content: space-between;
      gap: 1rem;
      align-items: center;
      margin-bottom: 0.9rem;
    }
    .scroll {
      max-height: 68vh;
      overflow: auto;
      padding-right: 0.2rem;
    }
    .timeline-item, .tile {
      padding: 0.95rem;
      border-radius: 1.1rem;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.64);
      cursor: pointer;
      transition: transform 140ms ease, border-color 140ms ease, background 140ms ease;
    }
    .timeline-item:hover, .tile:hover, .tile.is-selected {
      transform: translateY(-1px);
      border-color: rgba(197, 90, 49, 0.35);
      background: rgba(197, 90, 49, 0.08);
    }
    .timeline-list, .tile-grid {
      display: grid;
      gap: 0.8rem;
    }
    .sections-shell {
      display: grid;
      gap: 1rem;
    }
    .section-summary {
      display: grid;
      gap: 0.55rem;
      padding: 1rem 1.1rem;
      border-radius: 1.2rem;
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(197, 90, 49, 0.07), rgba(255, 255, 255, 0.72));
    }
    .section-block {
      padding: 1rem 1.05rem 1.05rem;
      border-radius: 1.25rem;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.56);
    }
    .section-block header {
      display: flex;
      justify-content: space-between;
      gap: 0.9rem;
      align-items: start;
      margin-bottom: 0.85rem;
    }
    .section-copy {
      display: grid;
      gap: 0.3rem;
    }
    .section-meta {
      text-align: right;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      min-height: 1.5rem;
      padding: 0.15rem 0.5rem;
      border-radius: 999px;
      background: rgba(16, 34, 43, 0.08);
      color: var(--muted);
      font-size: 0.72rem;
    }
    .meta {
      color: var(--muted);
      font-size: 0.8rem;
    }
    .tile-topline, .timeline-topline {
      display: flex;
      justify-content: space-between;
      gap: 0.75rem;
      align-items: center;
      margin-bottom: 0.45rem;
    }
    .detail-shell {
      position: sticky;
      top: 1rem;
    }
    .tile-grid {
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    }
    .detail-body {
      display: grid;
      gap: 1rem;
    }
    .detail-card {
      padding: 0.9rem;
      border-radius: 1rem;
      background: rgba(255, 255, 255, 0.65);
      border: 1px solid var(--line);
    }
    ul {
      margin: 0.6rem 0 0;
      padding-left: 1.2rem;
    }
    li + li {
      margin-top: 0.35rem;
    }
    .tags {
      display: flex;
      gap: 0.45rem;
      flex-wrap: wrap;
    }
    .chat-shell form {
      display: grid;
      gap: 0.8rem;
    }
    textarea, input {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 1rem;
      background: rgba(255, 255, 255, 0.86);
      color: var(--ink);
      padding: 0.9rem 1rem;
      font-size: 0.95rem;
    }
    textarea {
      min-height: 9rem;
      resize: vertical;
    }
    button {
      border: 0;
      border-radius: 999px;
      background: var(--ink);
      color: white;
      padding: 0.85rem 1rem;
      font-size: 0.86rem;
      cursor: pointer;
    }
    .chat-response {
      margin-top: 0.8rem;
      padding: 0.9rem;
      border-radius: 1rem;
      background: rgba(31, 106, 112, 0.08);
      border: 1px solid rgba(31, 106, 112, 0.18);
    }
    .empty {
      padding: 1rem;
      border-radius: 1rem;
      background: rgba(255, 255, 255, 0.6);
      color: var(--muted);
      border: 1px dashed var(--line);
    }
    a {
      color: var(--accent-deep);
    }
    @media (max-width: 1100px) {
      .hero, .content-layout {
        grid-template-columns: 1fr;
      }
      .detail-shell {
        position: static;
      }
      .scroll {
        max-height: none;
      }
    }
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <div class="hero-copy">
        <span class="eyebrow">Newton Research</span>
        <h1>Daily research, shaped into engineering next steps.</h1>
        <p id="hero-message">Loading structured findings, timeline updates, and implementation notes.</p>
      </div>
      <aside class="status-card">
        <div class="meta">Artifact status</div>
        <h2 id="artifact-status">Loading</h2>
        <p class="meta" id="artifact-path"></p>
      </aside>
    </section>

    <section class="stats" id="stats"></section>

    <section class="content-layout">
      <div class="panel">
        <div class="panel-header">
          <div>
            <h2>Research sections</h2>
            <p class="meta">Projects, papers, and articles grouped into reviewable sections with populated entries.</p>
          </div>
        </div>
        <div class="scroll" id="sections-root"></div>
      </div>

      <div class="sidebar-stack">
        <div class="detail-shell">
          <div class="detail-header">
            <div>
              <h2>Detail view</h2>
              <p class="meta">Implementation notes, comments, and next-step work items.</p>
            </div>
          </div>
          <div class="detail-body" id="detail-root">
            <div class="empty">Select a research tile to inspect its details.</div>
          </div>
        </div>

        <section class="chat-shell">
          <div class="chat-header">
            <div>
              <h2>Research chat</h2>
              <p class="meta" id="chat-status">Preparing Research Scientist integration.</p>
            </div>
          </div>
          <form id="chat-form">
            <textarea id="chat-input" placeholder="Ask what to implement next from the current research set."></textarea>
            <button type="submit">Send question</button>
          </form>
          <div class="chat-response" id="chat-response" hidden></div>
        </section>
      </div>
    </section>

    <section class="panel timeline-shell">
      <div class="panel-header">
        <div>
          <h2>Timeline</h2>
          <p class="meta">Recent research updates, moved below the main review surface for a cleaner populated-state layout.</p>
        </div>
      </div>
      <div class="scroll">
        <div class="timeline-list" id="timeline-root"></div>
      </div>
    </section>
  </main>

  <script>
    const statsRoot = document.getElementById("stats");
    const timelineRoot = document.getElementById("timeline-root");
    const sectionsRoot = document.getElementById("sections-root");
    const detailRoot = document.getElementById("detail-root");
    const heroMessage = document.getElementById("hero-message");
    const artifactStatus = document.getElementById("artifact-status");
    const artifactPath = document.getElementById("artifact-path");
    const chatStatus = document.getElementById("chat-status");
    const chatForm = document.getElementById("chat-form");
    const chatInput = document.getElementById("chat-input");
    const chatResponse = document.getElementById("chat-response");

    let dashboard = null;
    let selectedEntryId = null;

    const htmlEscapes = {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    };
    const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => htmlEscapes[char] || char);

    const findEntry = (entryId) => (dashboard?.entries || []).find((entry) => entry.id === entryId) || null;

    const renderStats = (summary) => {
      const stats = [
        { label: "Entries", value: summary.stats.entry_count, meta: "Research items available for review" },
        { label: "Sections", value: summary.stats.section_count, meta: "Grouped domains or source types" },
        { label: "Timeline", value: summary.stats.timeline_count, meta: "Recent updates highlighted in the feed" },
        { label: "Comments", value: summary.stats.comment_count, meta: "Agent notes embedded in the artifact" },
      ];
      statsRoot.innerHTML = stats.map((item) => `
        <article class="stat">
          <div class="meta">${escapeHtml(item.label)}</div>
          <span class="stat-value">${escapeHtml(item.value)}</span>
          <div class="meta">${escapeHtml(item.meta)}</div>
        </article>
      `).join("");
    };

    const renderTimeline = () => {
      const updates = dashboard.timeline || [];
      if (!updates.length) {
        timelineRoot.innerHTML = "<div class='empty'>No timeline updates are available yet.</div>";
        return;
      }
      timelineRoot.innerHTML = updates.map((item) => `
        <article class="timeline-item" data-entry-id="${escapeHtml(item.entry_id)}">
          <div class="timeline-topline">
            <span class="badge">${escapeHtml(item.date || "undated")}</span>
            ${item.entry_id ? `<span class="meta">${escapeHtml(item.entry_id)}</span>` : ""}
          </div>
          <h3>${escapeHtml(item.title)}</h3>
          <p>${escapeHtml(item.summary || "No summary provided.")}</p>
        </article>
      `).join("");

      timelineRoot.querySelectorAll("[data-entry-id]").forEach((node) => {
        node.addEventListener("click", () => {
          const entryId = node.getAttribute("data-entry-id");
          if (entryId) {
            selectEntry(entryId);
          }
        });
      });
    };

    const renderSections = () => {
      const sections = dashboard.sections || [];
      const entries = dashboard.entries || [];
      if (!sections.length) {
        sectionsRoot.innerHTML = "<div class='empty'>No research sections are available yet.</div>";
        return;
      }
      sectionsRoot.innerHTML = `
        <section class="sections-shell">
          <article class="section-summary">
            <span class="eyebrow">Populated artifact view</span>
            <h3>Review sections first, then drill into a single research item.</h3>
            <p>The dashboard now treats the populated state as the primary experience: grouped entries lead the page, the detail panel stays close by, and the timeline sits below as supporting context.</p>
          </article>
          ${sections.map((section) => {
        const tiles = (section.entry_ids || []).map((entryId) => entries.find((entry) => entry.id === entryId)).filter(Boolean);
        const tileMarkup = tiles.length ? tiles.map((entry) => `
          <article class="tile ${entry.id === selectedEntryId ? "is-selected" : ""}" data-entry-id="${escapeHtml(entry.id)}">
            <div class="tile-topline">
              <span class="badge">${escapeHtml(entry.kind)}</span>
              <span class="meta">${escapeHtml(entry.date || "undated")}</span>
            </div>
            <h3>${escapeHtml(entry.title)}</h3>
            <p>${escapeHtml(entry.summary || "No summary provided.")}</p>
          </article>
        `).join("") : "<div class='empty'>No items in this section yet.</div>";
        return `
          <section class="section-block">
            <header>
              <div class="section-copy">
                <h3>${escapeHtml(section.title)}</h3>
                ${section.description ? `<p class="meta">${escapeHtml(section.description)}</p>` : ""}
              </div>
              <div class="section-meta">
                <span class="badge">${escapeHtml(tiles.length)} entries</span>
              </div>
            </header>
            <div class="tile-grid">${tileMarkup}</div>
          </section>
        `;
      }).join("")}
        </section>
      `;

      sectionsRoot.querySelectorAll("[data-entry-id]").forEach((node) => {
        node.addEventListener("click", () => {
          const entryId = node.getAttribute("data-entry-id");
          if (entryId) {
            selectEntry(entryId);
          }
        });
      });
    };

    const renderDetail = () => {
      const entry = findEntry(selectedEntryId);
      if (!entry) {
        detailRoot.innerHTML = "<div class='empty'>Select a research tile to inspect its details.</div>";
        return;
      }

      const sourceMarkup = entry.source
        ? `<a href="${escapeHtml(entry.source.url)}" target="_blank" rel="noreferrer">${escapeHtml(entry.source.label)}</a>`
        : "<span class='meta'>No source linked</span>";
      const tagsMarkup = entry.tags.length
        ? `<div class="tags">${entry.tags.map((tag) => `<span class="badge">${escapeHtml(tag)}</span>`).join("")}</div>`
        : "<span class='meta'>No tags provided.</span>";
      const commentsMarkup = entry.comments.length
        ? `<ul>${entry.comments.map((item) => `<li><strong>${escapeHtml(item.author)}</strong>: ${escapeHtml(item.body)}</li>`).join("")}</ul>`
        : "<span class='meta'>No agent comments yet.</span>";
      const stepsMarkup = entry.next_steps.length
        ? `<ul>${entry.next_steps.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`
        : "<span class='meta'>No concrete next steps listed yet.</span>";

      detailRoot.innerHTML = `
        <section class="detail-card">
          <div class="tile-topline">
            <span class="badge">${escapeHtml(entry.section)}</span>
            <span class="meta">${escapeHtml(entry.date || "undated")}</span>
          </div>
          <h3>${escapeHtml(entry.title)}</h3>
          <p>${escapeHtml(entry.summary || "No summary provided.")}</p>
        </section>
        <section class="detail-card">
          <div class="meta">Source</div>
          <div style="margin-top: 0.4rem;">${sourceMarkup}</div>
        </section>
        <section class="detail-card">
          <div class="meta">Implementation notes</div>
          <p style="margin-top: 0.4rem;">${escapeHtml(entry.implementation_notes || "No implementation notes provided.")}</p>
        </section>
        <section class="detail-card">
          <div class="meta">Next steps</div>
          ${stepsMarkup}
        </section>
        <section class="detail-card">
          <div class="meta">Agent comments</div>
          ${commentsMarkup}
        </section>
        <section class="detail-card">
          <div class="meta">Tags</div>
          <div style="margin-top: 0.45rem;">${tagsMarkup}</div>
        </section>
      `;
    };

    const selectEntry = (entryId) => {
      selectedEntryId = entryId;
      renderSections();
      renderDetail();
    };

    const loadDashboard = async () => {
      const response = await fetch("/api/research");
      dashboard = await response.json();
      artifactStatus.textContent = dashboard.status;
      artifactPath.textContent = dashboard.artifact_path;
      heroMessage.textContent = dashboard.message || "Structured research feed loaded successfully.";
      chatStatus.textContent = `${dashboard.chat.agent_name}: ${dashboard.chat.status}`;
      chatInput.placeholder = dashboard.chat.placeholder;
      renderStats(dashboard);
      renderTimeline();
      selectedEntryId = dashboard.entries?.[0]?.id || null;
      renderSections();
      renderDetail();
    };

    chatForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      const message = chatInput.value.trim();
      if (!message) {
        return;
      }
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });
      const payload = await response.json();
      chatResponse.hidden = false;
      if (!response.ok) {
        chatResponse.innerHTML = `<strong>${escapeHtml(payload.agent_name || "Research chat")}</strong><p style="margin-top: 0.4rem;">${escapeHtml(payload.error || "Chat request failed.")}</p>`;
        return;
      }
      chatResponse.innerHTML = `<strong>${escapeHtml(payload.agent_name)}</strong><p style="margin-top: 0.4rem;">${escapeHtml(payload.reply)}</p>`;
    });

    loadDashboard().catch((error) => {
      heroMessage.textContent = `Failed to load dashboard: ${error.message}`;
      timelineRoot.innerHTML = "<div class='empty'>Dashboard payload could not be loaded.</div>";
      sectionsRoot.innerHTML = "<div class='empty'>Dashboard payload could not be loaded.</div>";
      detailRoot.innerHTML = "<div class='empty'>Dashboard payload could not be loaded.</div>";
    });
  </script>
</body>
</html>
"""


class ResearchDashboardRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler for the research dashboard."""

    server_version = "NewtonResearchDashboard/1.0"

    def __init__(self, *args: Any, artifact_path: Path, **kwargs: Any):
        self._artifact_path = artifact_path
        super().__init__(*args, **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._write_response(
                HTTPStatus.OK,
                _render_index_html().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if parsed.path == "/api/research":
            payload = build_research_dashboard_payload(self._artifact_path)
            self._write_response(
                HTTPStatus.OK,
                json.dumps(payload, indent=2).encode("utf-8"),
                "application/json; charset=utf-8",
            )
            return
        if parsed.path == "/api/research/entries":
            dashboard_payload = build_research_dashboard_payload(self._artifact_path)
            query = urllib.parse.parse_qs(parsed.query)
            filtered_entries = _filter_entries(
                dashboard_payload["entries"],
                query=_as_text(query.get("q", [""])[0]),
                section=_as_text(query.get("section", [""])[0]),
                kind=_as_text(query.get("kind", [""])[0]),
                tag=_as_text(query.get("tag", [""])[0]),
                limit=_as_int(query.get("limit", [_DEFAULT_ENTRY_LIMIT])[0], _DEFAULT_ENTRY_LIMIT),
            )
            self._write_response(
                HTTPStatus.OK,
                json.dumps(
                    {
                        "artifact_path": dashboard_payload["artifact_path"],
                        "count": len(filtered_entries),
                        "entries": filtered_entries,
                    },
                    indent=2,
                ).encode("utf-8"),
                "application/json; charset=utf-8",
            )
            return
        if parsed.path.startswith("/api/research/entries/"):
            entry_id = parsed.path.rsplit("/", 1)[-1]
            dashboard_payload = build_research_dashboard_payload(self._artifact_path)
            entry = next((item for item in dashboard_payload["entries"] if item["id"] == entry_id), None)
            if entry is None:
                self._write_response(
                    HTTPStatus.NOT_FOUND,
                    b'{"error": "Entry not found."}\n',
                    "application/json; charset=utf-8",
                )
                return
            self._write_response(
                HTTPStatus.OK,
                json.dumps(entry, indent=2).encode("utf-8"),
                "application/json; charset=utf-8",
            )
            return
        if parsed.path == "/api/healthz":
            self._write_response(
                HTTPStatus.OK,
                b'{"ok": true}\n',
                "application/json; charset=utf-8",
            )
            return
        self._write_response(HTTPStatus.NOT_FOUND, b"Not found\n", "text/plain; charset=utf-8")

    def do_POST(self):
        parsed = urllib.parse.urlsplit(self.path)
        raw_body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        try:
            request_payload = json.loads(raw_body.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._write_response(
                HTTPStatus.BAD_REQUEST,
                b'{"error": "Invalid JSON body."}\n',
                "application/json; charset=utf-8",
            )
            return

        if parsed.path == "/api/research/entries":
            if not isinstance(request_payload, dict):
                self._write_response(
                    HTTPStatus.BAD_REQUEST,
                    b'{"error": "Entry payload must be a JSON object."}\n',
                    "application/json; charset=utf-8",
                )
                return
            try:
                entry = _append_entry_to_artifact(self._artifact_path, request_payload)
            except (json.JSONDecodeError, ValueError) as error:
                self._write_response(
                    HTTPStatus.BAD_REQUEST,
                    json.dumps({"error": str(error)}, indent=2).encode("utf-8"),
                    "application/json; charset=utf-8",
                )
                return
            self._write_response(
                HTTPStatus.CREATED,
                json.dumps(entry, indent=2).encode("utf-8"),
                "application/json; charset=utf-8",
            )
            return

        if parsed.path != "/api/chat":
            self._write_response(HTTPStatus.NOT_FOUND, b"Not found\n", "text/plain; charset=utf-8")
            return

        message = _as_text(request_payload.get("message"))
        if not message:
            self._write_response(
                HTTPStatus.BAD_REQUEST,
                b'{"error": "message is required."}\n',
                "application/json; charset=utf-8",
            )
            return

        dashboard_payload = build_research_dashboard_payload(self._artifact_path)
        status, chat_payload = _build_chat_response(dashboard_payload["chat"], message, dashboard_payload)
        self._write_response(
            status,
            json.dumps(chat_payload, indent=2).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def log_message(self, format: str, *args: Any):
        return

    def _write_response(self, status: HTTPStatus, body: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def make_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the research dashboard server."""
    parser = argparse.ArgumentParser(description="Serve the Newton research dashboard for local artifacts.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=7070, help="Port to serve on.")
    parser.add_argument(
        "--artifact-path",
        default=str(default_research_artifact_path()),
        help="Path to the structured research artifact JSON.",
    )
    parser.add_argument(
        "--max-artifact-age-hours",
        type=float,
        default=default_research_max_artifact_age_hours(),
        help="Fail startup when the newest research evidence is older than this many hours.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the research dashboard HTTP server."""
    args = make_argument_parser().parse_args(argv)
    artifact_path = Path(args.artifact_path).resolve()
    freshness = assess_research_artifact_freshness(artifact_path, args.max_artifact_age_hours)
    if freshness["is_stale"]:
        print(freshness["reason"])
        print(f"Artifact path: {artifact_path}")
        return 2
    handler = partial(ResearchDashboardRequestHandler, artifact_path=artifact_path)

    with ThreadingHTTPServer((args.host, args.port), handler) as httpd:
        print(f"Serving research dashboard on http://{args.host}:{args.port}")
        print(f"Artifact path: {artifact_path}")
        httpd.serve_forever()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
