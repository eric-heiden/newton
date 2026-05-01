# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Serve a comparative dashboard for ASV benchmark artifacts."""

from __future__ import annotations

import argparse
import datetime
import itertools
import json
import math
import mimetypes
import os
import urllib.parse
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

_COMPARISON_PARAM_PRIORITY = ("solver", "solver_type", "integrator", "algorithm", "backend")
_DEFAULT_MAX_RESULT_AGE_HOURS = 36.0


def _isoformat_from_timestamp(timestamp_ms: int | float | None) -> str | None:
    if timestamp_ms is None:
        return None
    return datetime.datetime.fromtimestamp(float(timestamp_ms) / 1000.0, tz=datetime.timezone.utc).isoformat()


def _utc_now_datetime() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _short_commit(commit_hash: str | None) -> str | None:
    if not commit_hash:
        return None
    return commit_hash[:8]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_machine_metadata(machine_dir: Path) -> dict[str, Any]:
    machine_json = machine_dir / "machine.json"
    if not machine_json.exists():
        return {"machine": machine_dir.name, "display_name": machine_dir.name}

    metadata = _read_json(machine_json)
    machine_name = str(metadata.get("machine", machine_dir.name))
    cpu = metadata.get("cpu")
    gpu = metadata.get("gpu")
    summary_parts = [str(part) for part in (cpu, gpu) if part]
    return {
        "machine": machine_name,
        "display_name": machine_name,
        "summary": " | ".join(summary_parts) if summary_parts else None,
        "metadata": metadata,
    }


def _load_benchmark_metadata(results_dir: Path) -> dict[str, dict[str, Any]]:
    benchmark_json = results_dir / "benchmarks.json"
    if not benchmark_json.exists():
        return {}
    metadata = _read_json(benchmark_json)
    return metadata if isinstance(metadata, dict) else {}


def _decode_result_row(columns: list[str], row: list[Any]) -> dict[str, Any]:
    return {column: row[index] if index < len(row) else None for index, column in enumerate(columns)}


def _normalize_param_name(raw_name: Any, index: int) -> str:
    if raw_name in (None, ""):
        return f"param_{index + 1}"
    return str(raw_name)


def _make_case_variants(
    benchmark_name: str,
    benchmark_param_names: list[Any] | None,
    benchmark_params: list[list[str]] | None,
) -> list[dict[str, Any]]:
    if not benchmark_params:
        return [
            {
                "case_key": benchmark_name,
                "display_name": benchmark_name,
                "variant_label": benchmark_name,
                "param_names": [],
                "param_values": [],
                "param_map": {},
            }
        ]

    variants: list[dict[str, Any]] = []
    param_names = list(benchmark_param_names or [])
    for param_values in itertools.product(*benchmark_params):
        normalized_values = [str(value) for value in param_values]
        param_map = {
            _normalize_param_name(param_names[index] if index < len(param_names) else None, index): value
            for index, value in enumerate(normalized_values)
        }
        variant_label = ", ".join(f"{name}={value}" for name, value in param_map.items()) or benchmark_name
        display_name = f"{benchmark_name} ({variant_label})"
        variants.append(
            {
                "case_key": display_name,
                "display_name": display_name,
                "variant_label": variant_label,
                "param_names": list(param_map.keys()),
                "param_values": normalized_values,
                "param_map": param_map,
            }
        )
    return variants


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(numeric) else numeric


def _iter_result_records(
    result_path: Path,
    machine_info: dict[str, Any],
    benchmark_metadata: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = _read_json(result_path)
    columns = list(payload.get("result_columns", []))
    results = payload.get("results", {})
    env_name = payload.get("env_name") or "default"
    run_date = _isoformat_from_timestamp(payload.get("date"))
    run_summary = {
        "machine": machine_info["machine"],
        "machine_display_name": machine_info["display_name"],
        "machine_summary": machine_info.get("summary"),
        "env_name": env_name,
        "commit_hash": payload.get("commit_hash"),
        "commit_short": _short_commit(payload.get("commit_hash")),
        "date": run_date,
        "result_file": result_path.name,
        "benchmark_count": len(results),
        "benchmark_case_count": 0,
    }

    records: list[dict[str, Any]] = []
    for benchmark_name, row in results.items():
        if not isinstance(row, list):
            continue

        decoded = _decode_result_row(columns, row)
        benchmark_values = decoded.get("result")
        if benchmark_values is None:
            continue
        if not isinstance(benchmark_values, list):
            benchmark_values = [benchmark_values]

        metadata = benchmark_metadata.get(benchmark_name, {})
        benchmark_params = decoded.get("params") or []
        case_variants = _make_case_variants(benchmark_name, metadata.get("param_names"), benchmark_params)
        unit = metadata.get("unit")
        started_at = decoded.get("started_at")
        duration = decoded.get("duration")

        for index, value in enumerate(benchmark_values):
            numeric_value = _coerce_float(value)
            if numeric_value is None:
                continue

            if index < len(case_variants):
                case_variant = case_variants[index]
            else:
                case_key = f"{benchmark_name} [{index}]"
                case_variant = {
                    "case_key": case_key,
                    "display_name": case_key,
                    "variant_label": case_key,
                    "param_names": [],
                    "param_values": [],
                    "param_map": {},
                }

            records.append(
                {
                    **case_variant,
                    "benchmark_name": benchmark_name,
                    "machine": machine_info["machine"],
                    "machine_display_name": machine_info["display_name"],
                    "machine_summary": machine_info.get("summary"),
                    "env_name": env_name,
                    "unit": unit,
                    "value": numeric_value,
                    "started_at": _isoformat_from_timestamp(started_at),
                    "duration": _coerce_float(duration),
                    "date": run_date,
                    "commit_hash": payload.get("commit_hash"),
                    "commit_short": _short_commit(payload.get("commit_hash")),
                }
            )

    run_summary["benchmark_case_count"] = len(records)
    return run_summary, records


def _pick_series_param(cases: list[dict[str, Any]]) -> str | None:
    available_values: dict[str, set[str]] = {}
    for case in cases:
        for name, value in case.get("param_map", {}).items():
            available_values.setdefault(name, set()).add(value)

    for candidate in _COMPARISON_PARAM_PRIORITY:
        values = available_values.get(candidate)
        if values and len(values) > 1:
            return candidate

    for name, values in sorted(available_values.items()):
        if len(values) > 1:
            return name

    return None


def _format_param_items(param_items: tuple[tuple[str, str], ...]) -> str:
    if not param_items:
        return "All published cases"
    return " | ".join(f"{name}={value}" for name, value in param_items)


def _build_comparison_groups(case_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped_cases: dict[tuple[str, str, str, str | None], list[dict[str, Any]]] = {}
    for case in case_summaries:
        grouped_cases.setdefault(
            (
                case["benchmark_name"],
                case["machine"],
                case["env_name"],
                case.get("unit"),
            ),
            [],
        ).append(case)

    comparison_groups: list[dict[str, Any]] = []
    for (benchmark_name, machine, env_name, unit), cases in grouped_cases.items():
        series_param = _pick_series_param(cases)
        scenario_groups: dict[tuple[tuple[str, str], ...], list[dict[str, Any]]] = {}

        for case in cases:
            if series_param is None:
                scenario_key: tuple[tuple[str, str], ...] = ()
            else:
                scenario_key = tuple(
                    sorted((name, value) for name, value in case.get("param_map", {}).items() if name != series_param)
                )
            scenario_groups.setdefault(scenario_key, []).append(case)

        for scenario_items, scenario_cases in scenario_groups.items():
            series: list[dict[str, Any]] = []
            for case in sorted(
                scenario_cases,
                key=lambda item: (
                    item.get("param_map", {}).get(series_param) if series_param else item.get("variant_label"),
                    item["display_name"],
                ),
            ):
                if series_param:
                    label = case.get("param_map", {}).get(series_param, case["variant_label"])
                else:
                    label = case["variant_label"]

                series.append(
                    {
                        "key": case["case_key"],
                        "label": label,
                        "display_name": case["display_name"],
                        "value": case["value"],
                        "previous_value": case["previous_value"],
                        "delta_value": case["delta_value"],
                        "delta_pct": case["delta_pct"],
                        "status": case["status"],
                        "history": case["history"],
                        "history_min": case["history_min"],
                        "history_max": case["history_max"],
                        "commit_short": case["commit_short"],
                        "date": case["date"],
                        "param_map": case["param_map"],
                    }
                )

            latest_date = max((entry["date"] for entry in series if entry["date"]), default=None)
            title = (
                benchmark_name if not scenario_items else f"{benchmark_name} [{_format_param_items(scenario_items)}]"
            )
            subtitle_parts = [scenario_cases[0]["machine_display_name"], env_name]
            if series_param:
                subtitle_parts.append(f"compare by {series_param}")

            comparison_groups.append(
                {
                    "key": "::".join(
                        [
                            benchmark_name,
                            machine,
                            env_name,
                            unit or "",
                            series_param or "",
                            ",".join(f"{name}={value}" for name, value in scenario_items),
                        ]
                    ),
                    "title": title,
                    "subtitle": " | ".join(part for part in subtitle_parts if part),
                    "benchmark_name": benchmark_name,
                    "machine": machine,
                    "machine_display_name": scenario_cases[0]["machine_display_name"],
                    "env_name": env_name,
                    "unit": unit,
                    "series_param": series_param,
                    "scenario_label": _format_param_items(scenario_items),
                    "scenario_params": [{"name": name, "value": value} for name, value in scenario_items],
                    "series_count": len(series),
                    "latest_date": latest_date,
                    "max_value": max(item["value"] for item in series),
                    "min_value": min(item["value"] for item in series),
                    "status_counts": {
                        "regression": sum(1 for item in series if item["status"] == "regression"),
                        "improvement": sum(1 for item in series if item["status"] == "improvement"),
                        "new": sum(1 for item in series if item["status"] == "new"),
                        "stable": sum(1 for item in series if item["status"] == "stable"),
                    },
                    "series": series,
                }
            )

    comparison_groups.sort(
        key=lambda group: (
            group["title"],
            group["machine_display_name"],
            group["env_name"],
            group["scenario_label"],
        )
    )
    return comparison_groups


def default_benchmark_index_path() -> Path:
    configured = os.environ.get("NEWTON_BENCHMARK_INDEX_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[3] / "benchmarks" / "results" / "index.json"


def default_benchmark_max_result_age_hours() -> float:
    configured = os.environ.get("NEWTON_BENCHMARK_MAX_RESULT_AGE_HOURS")
    if configured is None:
        return _DEFAULT_MAX_RESULT_AGE_HOURS
    try:
        parsed = float(configured)
    except ValueError:
        return _DEFAULT_MAX_RESULT_AGE_HOURS
    return parsed if parsed > 0.0 else _DEFAULT_MAX_RESULT_AGE_HOURS


def _parse_iso_datetime(value: Any) -> datetime.datetime | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed.astimezone(datetime.timezone.utc)


def assess_benchmark_freshness(
    results_dir: str | Path,
    benchmark_index_path: str | Path | None,
    max_age_hours: float,
    *,
    now: datetime.datetime | None = None,
) -> dict[str, Any]:
    """Assess whether benchmark evidence is fresh enough to serve."""
    results_path = Path(results_dir)
    benchmark_index = Path(benchmark_index_path) if benchmark_index_path is not None else default_benchmark_index_path()
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

    semantic_candidates: list[datetime.datetime] = []
    if benchmark_index.exists():
        try:
            index_payload = _read_json(benchmark_index)
        except json.JSONDecodeError:
            index_payload = {}
        semantic_candidates.extend(
            item
            for item in [
                _parse_iso_datetime(index_payload.get("generated_at")),
                _parse_iso_datetime(index_payload.get("latest_run", {}).get("generated_at"))
                if isinstance(index_payload.get("latest_run"), dict)
                else None,
                _parse_iso_datetime(index_payload.get("latest_run", {}).get("started_at"))
                if isinstance(index_payload.get("latest_run"), dict)
                else None,
            ]
            if item is not None
        )
        for run_payload in _as_run_payloads(benchmark_index):
            semantic_candidates.extend(
                item
                for item in [
                    _parse_iso_datetime(run_payload.get("generated_at")),
                    _parse_iso_datetime(run_payload.get("started_at")),
                ]
                if item is not None
            )
    elif results_path.exists():
        for result_path in results_path.glob("*/*.json"):
            if result_path.name == "machine.json":
                continue
            try:
                payload = _read_json(result_path)
            except json.JSONDecodeError:
                continue
            run_date = _isoformat_from_timestamp(payload.get("date"))
            parsed = _parse_iso_datetime(run_date)
            if parsed is not None:
                semantic_candidates.append(parsed)

    if semantic_candidates:
        artifact_timestamp = max(semantic_candidates)
        source = "artifact"
    else:
        fallback_paths = [path for path in [benchmark_index, results_path] if path.exists()]
        if not fallback_paths:
            freshness["reason"] = "No benchmark artifact or result directory exists."
            return freshness
        artifact_timestamp = max(
            datetime.datetime.fromtimestamp(path.stat().st_mtime, tz=datetime.timezone.utc) for path in fallback_paths
        )
        source = "mtime"

    age_hours = max((checked_at - artifact_timestamp).total_seconds() / 3600.0, 0.0)
    freshness["artifact_timestamp"] = artifact_timestamp.isoformat()
    freshness["age_hours"] = age_hours
    freshness["source"] = source
    freshness["is_stale"] = age_hours > float(max_age_hours)
    freshness["reason"] = (
        f"Benchmark evidence is stale at {age_hours:.1f}h old; max allowed is {float(max_age_hours):.1f}h."
        if freshness["is_stale"]
        else f"Benchmark evidence is fresh at {age_hours:.1f}h old."
    )
    return freshness


def _as_run_payloads(benchmark_index: Path) -> list[dict[str, Any]]:
    runs_dir = benchmark_index.parent / "runs"
    return [_read_json(path) for path in sorted(runs_dir.glob("*.json"))] if runs_dir.exists() else []


def _make_summary(results_path: Path, html_path: Path, benchmark_index_path: Path) -> dict[str, Any]:
    return {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "results_dir": str(results_path),
        "html_dir": str(html_path),
        "benchmark_index_path": str(benchmark_index_path),
        "freshness": assess_benchmark_freshness(
            results_path,
            benchmark_index_path,
            default_benchmark_max_result_age_hours(),
        ),
        "results_available": results_path.exists(),
        "html_available": html_path.exists(),
        "benchmark_index_available": benchmark_index_path.exists(),
        "status": "empty",
        "machines": [],
        "latest_runs": [],
        "benchmark_cases": [],
        "comparison_groups": [],
        "filters": {
            "environments": [],
            "machines": [],
        },
        "highlights": {
            "recent_runs": [],
            "improvements": [],
            "regressions": [],
            "new_cases": [],
        },
        "stats": {
            "machine_count": 0,
            "environment_count": 0,
            "latest_run_count": 0,
            "benchmark_case_count": 0,
            "comparison_group_count": 0,
            "regression_count": 0,
            "improvement_count": 0,
            "new_case_count": 0,
            "latest_result_at": None,
        },
    }


def _finalize_summary(
    summary: dict[str, Any],
    grouped_records: dict[tuple[str, str, str], list[dict[str, Any]]],
    all_runs: list[dict[str, Any]],
    environments: set[str],
    latest_result_at: str | None,
) -> dict[str, Any]:
    all_runs.sort(
        key=lambda run: (
            run["date"] or "",
            run["machine"],
            run["env_name"],
            run["commit_short"] or "",
        ),
        reverse=True,
    )
    summary["latest_runs"] = all_runs
    summary["highlights"]["recent_runs"] = all_runs[:5]
    summary["stats"]["latest_run_count"] = len(all_runs)
    summary["stats"]["latest_result_at"] = latest_result_at
    summary["stats"]["environment_count"] = len(environments)
    summary["filters"]["environments"] = [{"value": env_name, "label": env_name} for env_name in sorted(environments)]
    summary["filters"]["machines"] = [
        {"value": machine["machine"], "label": machine["display_name"]}
        for machine in sorted(summary["machines"], key=lambda item: item["display_name"])
    ]

    case_summaries: list[dict[str, Any]] = []
    for records in grouped_records.values():
        records.sort(
            key=lambda record: (
                record["date"] or "",
                record["started_at"] or "",
                record["commit_short"] or "",
            ),
            reverse=True,
        )
        latest = records[0]
        previous = records[1] if len(records) > 1 else None

        previous_value = previous["value"] if previous is not None else None
        delta_value = None
        delta_pct = None
        if previous_value not in (None, 0.0):
            delta_value = latest["value"] - previous_value
            delta_pct = (delta_value / previous_value) * 100.0
        elif previous_value == 0.0:
            delta_value = latest["value"]

        if previous is None:
            case_status = "new"
        elif delta_pct is not None and delta_pct > 5.0:
            case_status = "regression"
        elif delta_pct is not None and delta_pct < -5.0:
            case_status = "improvement"
        else:
            case_status = "stable"

        history = [
            {
                "date": record["date"],
                "started_at": record["started_at"],
                "commit_short": record["commit_short"],
                "value": record["value"],
            }
            for record in records
        ]
        history_values = [record["value"] for record in records]

        case_summaries.append(
            {
                **latest,
                "previous_value": previous_value,
                "delta_value": delta_value,
                "delta_pct": delta_pct,
                "status": case_status,
                "history_length": len(records),
                "history": history,
                "history_min": min(history_values),
                "history_max": max(history_values),
            }
        )

    case_summaries.sort(key=lambda record: (record["machine"], record["env_name"], record["display_name"]))
    comparison_groups = _build_comparison_groups(case_summaries)

    summary["benchmark_cases"] = case_summaries
    summary["comparison_groups"] = comparison_groups
    summary["stats"]["benchmark_case_count"] = len(case_summaries)
    summary["stats"]["comparison_group_count"] = len(comparison_groups)
    summary["stats"]["regression_count"] = sum(1 for case in case_summaries if case["status"] == "regression")
    summary["stats"]["improvement_count"] = sum(1 for case in case_summaries if case["status"] == "improvement")
    summary["stats"]["new_case_count"] = sum(1 for case in case_summaries if case["status"] == "new")
    summary["highlights"]["improvements"] = [case for case in case_summaries if case["status"] == "improvement"][:5]
    summary["highlights"]["regressions"] = [case for case in case_summaries if case["status"] == "regression"][:5]
    summary["highlights"]["new_cases"] = [case for case in case_summaries if case["status"] == "new"][:5]

    if not case_summaries:
        summary["message"] = "No completed benchmark results were found."
    else:
        summary["status"] = "ready"

    return summary


def _build_asv_dashboard_summary(
    results_dir: str | Path,
    html_dir: str | Path,
    benchmark_index_path: str | Path,
) -> dict[str, Any]:
    """Build a UI-ready summary of ASV benchmark results."""
    results_path = Path(results_dir)
    html_path = Path(html_dir)
    benchmark_index = Path(benchmark_index_path)
    benchmark_metadata = _load_benchmark_metadata(results_path) if results_path.exists() else {}

    summary = _make_summary(results_path, html_path, benchmark_index)

    if not results_path.exists():
        summary["message"] = "ASV results directory not found."
        return summary

    machine_dirs = sorted(path for path in results_path.iterdir() if path.is_dir())
    machine_lookup = {machine_dir.name: _load_machine_metadata(machine_dir) for machine_dir in machine_dirs}
    for machine_info in machine_lookup.values():
        machine_info["run_count"] = 0
        machine_info["benchmark_case_count"] = 0
    summary["machines"] = list(machine_lookup.values())
    summary["stats"]["machine_count"] = len(machine_lookup)

    all_runs: list[dict[str, Any]] = []
    grouped_records: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    latest_result_at: str | None = None
    environments: set[str] = set()

    for machine_dir in machine_dirs:
        machine_info = machine_lookup[machine_dir.name]
        for result_path in sorted(machine_dir.glob("*.json")):
            if result_path.name == "machine.json":
                continue

            run_summary, records = _iter_result_records(result_path, machine_info, benchmark_metadata)
            all_runs.append(run_summary)
            machine_info["run_count"] += 1
            machine_info["benchmark_case_count"] += len(records)
            environments.add(run_summary["env_name"])
            run_date = run_summary.get("date")
            if run_date and (latest_result_at is None or run_date > latest_result_at):
                latest_result_at = run_date

            for record in records:
                grouped_records.setdefault((record["machine"], record["env_name"], record["case_key"]), []).append(
                    record
                )

    summary = _finalize_summary(summary, grouped_records, all_runs, environments, latest_result_at)
    if not summary["benchmark_cases"]:
        summary["message"] = "No completed ASV benchmark results were found."
    return summary


def _build_solver_matrix_dashboard_summary(
    results_dir: str | Path,
    html_dir: str | Path,
    benchmark_index_path: str | Path,
) -> dict[str, Any]:
    """Build a UI-ready summary of the solver benchmark matrix artifact."""
    results_path = Path(results_dir)
    html_path = Path(html_dir)
    benchmark_index = Path(benchmark_index_path)
    summary = _make_summary(results_path, html_path, benchmark_index)

    if not benchmark_index.exists():
        summary["message"] = "Solver benchmark matrix artifact not found."
        return summary

    index_payload = _read_json(benchmark_index)
    run_payloads = _as_run_payloads(benchmark_index)
    if not run_payloads and isinstance(index_payload.get("latest_run"), dict):
        run_payloads = [index_payload["latest_run"]]

    if not run_payloads:
        summary["message"] = "Solver benchmark matrix artifact has no runs."
        return summary

    machine_lookup: dict[str, dict[str, Any]] = {}
    all_runs: list[dict[str, Any]] = []
    grouped_records: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    environments: set[str] = set()
    latest_result_at: str | None = None

    for run_payload in run_payloads:
        machine_name = str(run_payload.get("host") or "benchmark-host")
        env_name = str(run_payload.get("device") or "default")
        machine_info = machine_lookup.setdefault(
            machine_name,
            {
                "machine": machine_name,
                "display_name": machine_name,
                "summary": f"Solver benchmark host | device {env_name}",
                "metadata": {"host": machine_name},
                "run_count": 0,
                "benchmark_case_count": 0,
            },
        )

        run_date = run_payload.get("generated_at") or run_payload.get("started_at")
        if run_date and (latest_result_at is None or run_date > latest_result_at):
            latest_result_at = run_date

        results = run_payload.get("results", [])
        run_summary = {
            "machine": machine_name,
            "machine_display_name": machine_name,
            "machine_summary": machine_info["summary"],
            "env_name": env_name,
            "commit_hash": None,
            "commit_short": None,
            "date": run_date,
            "result_file": f"runs/{run_payload.get('run_id', 'latest')}.json",
            "benchmark_count": len(results),
            "benchmark_case_count": len(results),
        }
        all_runs.append(run_summary)
        machine_info["run_count"] += 1
        machine_info["benchmark_case_count"] += len(results)
        environments.add(env_name)

        for result in results:
            scenario = str(result.get("scenario") or "scenario")
            capability = str(result.get("capability") or "uncategorized")
            solver_name = str(result.get("solver_name") or result.get("solver_id") or "solver")
            value = _coerce_float(result.get("steps_per_second"))
            if value is None:
                continue
            case_key = f"{scenario}:{solver_name}"
            grouped_records.setdefault((machine_name, env_name, case_key), []).append(
                {
                    "case_key": case_key,
                    "display_name": f"{scenario} ({solver_name})",
                    "variant_label": solver_name,
                    "param_names": ["solver"],
                    "param_values": [solver_name],
                    "param_map": {"solver": solver_name},
                    "benchmark_name": f"{scenario} [{capability}]",
                    "machine": machine_name,
                    "machine_display_name": machine_name,
                    "machine_summary": machine_info["summary"],
                    "env_name": env_name,
                    "unit": "steps/s",
                    "value": value,
                    "started_at": run_payload.get("started_at"),
                    "duration": _coerce_float(result.get("total_time_ms")),
                    "date": run_date,
                    "commit_hash": None,
                    "commit_short": None,
                }
            )

    summary["machines"] = list(machine_lookup.values())
    summary["stats"]["machine_count"] = len(summary["machines"])
    summary["notes"] = list(index_payload.get("latest_run", {}).get("notes", []))
    return _finalize_summary(summary, grouped_records, all_runs, environments, latest_result_at)


def build_dashboard_summary(
    results_dir: str | Path,
    html_dir: str | Path,
    benchmark_index_path: str | Path | None = None,
) -> dict[str, Any]:
    benchmark_index = Path(benchmark_index_path) if benchmark_index_path is not None else default_benchmark_index_path()
    if benchmark_index.exists():
        return _build_solver_matrix_dashboard_summary(results_dir, html_dir, benchmark_index)
    return _build_asv_dashboard_summary(results_dir, html_dir, benchmark_index)


def _render_index_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Newton Benchmark Dashboard</title>
  <style>
    :root {
      --paper: #f3efe5;
      --panel: rgba(255, 251, 245, 0.86);
      --panel-strong: rgba(255, 255, 255, 0.92);
      --ink: #182126;
      --muted: #59676e;
      --line: rgba(24, 33, 38, 0.12);
      --accent: #bb563f;
      --accent-soft: rgba(187, 86, 63, 0.14);
      --positive: #1f7a58;
      --negative: #a53628;
      --neutral: #65727a;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(187, 86, 63, 0.18), transparent 26rem),
        radial-gradient(circle at right center, rgba(24, 33, 38, 0.10), transparent 26rem),
        linear-gradient(180deg, #faf6ef, var(--paper));
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
    }
    main {
      max-width: 1280px;
      margin: 0 auto;
      padding: 44px 18px 88px;
    }
    h1 {
      margin: 0 0 10px;
      font-size: clamp(2.2rem, 5vw, 4.8rem);
      line-height: 0.95;
    }
    h2, h3 {
      margin: 0 0 10px;
      font-weight: 700;
    }
    p {
      margin: 0;
      color: var(--muted);
    }
    .hero {
      display: grid;
      gap: 16px;
      grid-template-columns: minmax(0, 1.4fr) minmax(280px, 0.8fr);
      align-items: start;
      margin-bottom: 24px;
    }
    .hero-copy {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 28px;
      padding: 28px;
      backdrop-filter: blur(18px);
    }
    .hero-copy p {
      max-width: 58rem;
      font-size: 1.02rem;
      line-height: 1.5;
    }
    .hero-side {
      background: linear-gradient(180deg, rgba(24, 33, 38, 0.92), rgba(46, 58, 64, 0.90));
      color: white;
      border-radius: 28px;
      padding: 24px;
      min-height: 100%;
    }
    .hero-side p {
      color: rgba(255, 255, 255, 0.74);
      line-height: 1.5;
    }
    .hero-side strong {
      display: block;
      font-size: 2rem;
      line-height: 1;
      margin-bottom: 8px;
    }
    .actions {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 20px;
    }
    a.button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      padding: 12px 16px;
      border-radius: 999px;
      text-decoration: none;
      border: 1px solid transparent;
      font-weight: 700;
    }
    a.button.primary {
      color: white;
      background: var(--accent);
    }
    a.button.secondary {
      color: var(--ink);
      border-color: var(--line);
      background: rgba(255, 255, 255, 0.6);
    }
    .grid {
      display: grid;
      gap: 16px;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      margin-bottom: 18px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 18px;
      backdrop-filter: blur(16px);
    }
    .metric {
      font-size: 2rem;
      font-weight: 700;
      margin-top: 10px;
    }
    .metric-label {
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 0.78rem;
      color: var(--muted);
    }
    .section-head {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: end;
      margin-bottom: 14px;
    }
    .toolbar {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin: 18px 0 20px;
    }
    .toolbar label {
      display: grid;
      gap: 6px;
      font-size: 0.84rem;
      color: var(--muted);
    }
    select {
      min-width: 180px;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: var(--panel-strong);
      color: var(--ink);
      font: inherit;
    }
    .comparison-grid {
      display: grid;
      gap: 16px;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    }
    .comparison-card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 18px;
      box-shadow: 0 18px 40px rgba(24, 33, 38, 0.06);
    }
    .comparison-card header {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: start;
      margin-bottom: 14px;
    }
    .eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 0.76rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 8px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 0.74rem;
      background: var(--accent-soft);
      color: var(--accent);
      white-space: nowrap;
    }
    .plot-frame {
      border-radius: 18px;
      background: linear-gradient(180deg, rgba(255, 255, 255, 0.86), rgba(255, 247, 239, 0.82));
      border: 1px solid rgba(24, 33, 38, 0.08);
      padding: 12px;
      margin-bottom: 14px;
    }
    .comparison-meta {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 14px;
      color: var(--muted);
      font-size: 0.88rem;
    }
    .series-list {
      display: grid;
      gap: 10px;
    }
    .series-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      align-items: center;
      padding-top: 10px;
      border-top: 1px solid rgba(24, 33, 38, 0.08);
    }
    .series-row:first-child {
      border-top: none;
      padding-top: 0;
    }
    .series-label {
      font-weight: 700;
    }
    .series-subtitle {
      margin-top: 4px;
      font-size: 0.84rem;
      color: var(--muted);
    }
    .series-value {
      text-align: right;
      font-family: ui-monospace, "SFMono-Regular", "SF Mono", Consolas, monospace;
      font-size: 0.92rem;
    }
    .series-sparkline {
      width: 96px;
      height: 26px;
    }
    .delta-positive { color: var(--positive); }
    .delta-negative { color: var(--negative); }
    .delta-neutral { color: var(--neutral); }
    .list-grid {
      display: grid;
      gap: 16px;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      margin-top: 16px;
    }
    .run-card, .change-card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 16px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border-radius: 20px;
      overflow: hidden;
      border: 1px solid var(--line);
    }
    th, td {
      text-align: left;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      font-size: 0.92rem;
      vertical-align: top;
    }
    th {
      background: rgba(24, 33, 38, 0.04);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      font-size: 0.78rem;
      color: var(--muted);
    }
    .empty {
      padding: 18px;
      border-radius: 20px;
      border: 1px dashed rgba(24, 33, 38, 0.22);
      color: var(--muted);
      background: rgba(255, 255, 255, 0.45);
    }
    .muted {
      color: var(--muted);
    }
    @media (max-width: 840px) {
      .hero {
        grid-template-columns: 1fr;
      }
      .section-head {
        display: block;
      }
      .comparison-card header {
        display: block;
      }
      .series-row {
        grid-template-columns: 1fr;
      }
      .series-value {
        text-align: left;
      }
    }
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <div class="hero-copy">
        <div class="eyebrow">Comparative plots first</div>
        <h1>Newton Benchmark Dashboard</h1>
        <p id="subtitle">Loading benchmark artifacts and comparison plots...</p>
        <div class="actions">
          <a class="button primary" href="/asv/">Open native ASV report</a>
          <a class="button secondary" href="/api/summary">View JSON summary</a>
        </div>
      </div>
      <aside class="hero-side">
        <strong id="hero-groups">0 plots</strong>
        <p>Each card groups the latest published benchmark values into comparison-ready plots so solver, scenario, and capability differences are visible before drilling into raw ASV output.</p>
      </aside>
    </section>

    <section id="stats" class="grid"></section>

    <section class="panel">
      <div class="section-head">
        <div>
          <div class="eyebrow">Filter</div>
          <h2>Comparative plots</h2>
          <p>Focus the hosted view on the machine or ASV environment you want to inspect.</p>
        </div>
      </div>
      <div class="toolbar">
        <label>
          Machine
          <select id="machine-filter">
            <option value="">All machines</option>
          </select>
        </label>
        <label>
          Environment
          <select id="env-filter">
            <option value="">All environments</option>
          </select>
        </label>
      </div>
      <div id="comparison-groups" class="comparison-grid"></div>
    </section>

    <section class="list-grid">
      <article class="panel">
        <div class="section-head">
          <div>
            <div class="eyebrow">Signal</div>
            <h2>What changed last</h2>
            <p>Recent improvements, regressions, and new benchmark cases from the latest published state.</p>
          </div>
        </div>
        <div id="changes" class="list-grid"></div>
      </article>
      <article class="panel">
        <div class="section-head">
          <div>
            <div class="eyebrow">Publishing</div>
            <h2>Recent runs</h2>
            <p>Latest ASV result files observed by the hosted dashboard.</p>
          </div>
        </div>
        <div id="runs" class="list-grid"></div>
      </article>
    </section>

    <section class="panel">
      <div class="section-head">
        <div>
          <div class="eyebrow">Reference</div>
          <h2>Latest benchmark cases</h2>
          <p>Raw latest-case rows remain available for exact value inspection.</p>
        </div>
      </div>
      <div style="overflow-x:auto">
        <table>
          <thead>
            <tr>
              <th>Case</th>
              <th>Machine</th>
              <th>Env</th>
              <th>Latest</th>
              <th>Delta</th>
              <th>Commit</th>
            </tr>
          </thead>
          <tbody id="cases"></tbody>
        </table>
      </div>
    </section>
  </main>
  <script>
    function escapeHtmlAttr(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    function formatNumber(value) {
      if (value === null || value === undefined) return "n/a";
      return Number(value).toFixed(4);
    }

    function formatDate(value) {
      if (!value) return "n/a";
      const date = new Date(value);
      if (Number.isNaN(date.getTime())) return value;
      return date.toLocaleString();
    }

    function deltaMeta(value) {
      if (value === null || value === undefined) {
        return { text: "n/a", cls: "delta-neutral" };
      }
      if (value > 5) return { text: `+${value.toFixed(1)}%`, cls: "delta-negative" };
      if (value < -5) return { text: `${value.toFixed(1)}%`, cls: "delta-positive" };
      const prefix = value > 0 ? "+" : "";
      return { text: `${prefix}${value.toFixed(1)}%`, cls: "delta-neutral" };
    }

    function buildBarChart(series) {
      if (!series.length) {
        return "<div class=\\"empty\\">No plot data available.</div>";
      }

      const width = 520;
      const rowHeight = 42;
      const labelWidth = 136;
      const chartWidth = width - labelWidth - 16;
      const maxValue = Math.max(...series.map((item) => Number(item.value) || 0), 1e-9);
      const height = Math.max(series.length * rowHeight + 12, 64);

      const rows = series.map((item, index) => {
        const y = index * rowHeight + 10;
        const barWidth = Math.max((Number(item.value) || 0) / maxValue * chartWidth, 2);
        const meta = deltaMeta(item.delta_pct);
        const fill = item.status === "regression"
          ? "#a53628"
          : item.status === "improvement"
            ? "#1f7a58"
            : item.status === "new"
              ? "#bb563f"
              : "#6f7c84";
        return `
          <text x="0" y="${y + 16}" font-size="12" fill="#59676e">${escapeHtmlAttr(item.label)}</text>
          <rect x="${labelWidth}" y="${y}" width="${chartWidth}" height="18" rx="9" fill="rgba(24,33,38,0.06)"></rect>
          <rect x="${labelWidth}" y="${y}" width="${barWidth}" height="18" rx="9" fill="${fill}"></rect>
          <text x="${labelWidth + Math.min(barWidth + 8, chartWidth - 2)}" y="${y + 14}" font-size="12" fill="#182126">${escapeHtmlAttr(formatNumber(item.value))}</text>
          <text x="${labelWidth}" y="${y + 34}" font-size="11" fill="${meta.cls === "delta-positive" ? "#1f7a58" : meta.cls === "delta-negative" ? "#a53628" : "#65727a"}">${escapeHtmlAttr(meta.text)}</text>
        `;
      }).join("");

      return `
        <svg viewBox="0 0 ${width} ${height}" width="100%" role="img" aria-label="Comparison plot">
          ${rows}
        </svg>
      `;
    }

    function buildSparkline(history) {
      if (!history || history.length <= 1) {
        return "<span class=\\"muted\\">single point</span>";
      }
      const width = 96;
      const height = 26;
      const values = history.map((item) => Number(item.value) || 0);
      const min = Math.min(...values);
      const max = Math.max(...values);
      const span = Math.max(max - min, 1e-9);
      const points = values.map((value, index) => {
        const x = values.length === 1 ? width / 2 : index / (values.length - 1) * width;
        const y = height - ((value - min) / span * (height - 6) + 3);
        return `${x.toFixed(2)},${y.toFixed(2)}`;
      }).join(" ");
      return `
        <svg class="series-sparkline" viewBox="0 0 ${width} ${height}" aria-hidden="true">
          <polyline fill="none" stroke="#bb563f" stroke-width="2" points="${points}"></polyline>
        </svg>
      `;
    }

    function renderComparisonGroups(payload) {
      const machineValue = document.getElementById("machine-filter").value;
      const envValue = document.getElementById("env-filter").value;
      const groups = payload.comparison_groups.filter((group) => {
        return (!machineValue || group.machine === machineValue) && (!envValue || group.env_name === envValue);
      });

      document.getElementById("hero-groups").textContent = `${groups.length} plots`;
      const container = document.getElementById("comparison-groups");
      if (!groups.length) {
        container.innerHTML = "<div class=\\"empty\\">No comparison groups match the current filters.</div>";
        return;
      }

      container.innerHTML = groups.map((group) => {
        const bars = buildBarChart(group.series);
        const seriesRows = group.series.map((item) => {
          const delta = deltaMeta(item.delta_pct);
          return `
            <div class="series-row">
              <div>
                <div class="series-label">${escapeHtmlAttr(item.label)}</div>
                <div class="series-subtitle">${escapeHtmlAttr(item.display_name)} | ${escapeHtmlAttr(item.commit_short || "n/a")} | ${escapeHtmlAttr(formatDate(item.date))}</div>
              </div>
              <div class="series-value">
                <div>${escapeHtmlAttr(formatNumber(item.value))} ${escapeHtmlAttr(group.unit || "")}</div>
                <div class="${delta.cls}">${escapeHtmlAttr(delta.text)}</div>
                <div>${buildSparkline(item.history)}</div>
              </div>
            </div>
          `;
        }).join("");

        return `
          <article class="comparison-card">
            <header>
              <div>
                <div class="eyebrow">${escapeHtmlAttr(group.series_param ? `Compare by ${group.series_param}` : "Latest values")}</div>
                <h3>${escapeHtmlAttr(group.title)}</h3>
                <p>${escapeHtmlAttr(group.subtitle)}</p>
              </div>
              <div class="pill">${escapeHtmlAttr(group.scenario_label)}</div>
            </header>
            <div class="comparison-meta">
              <span>${group.series_count} series</span>
              <span>${escapeHtmlAttr(group.unit || "unitless")}</span>
              <span>${escapeHtmlAttr(group.latest_date ? formatDate(group.latest_date) : "n/a")}</span>
            </div>
            <div class="plot-frame">${bars}</div>
            <div class="series-list">${seriesRows}</div>
          </article>
        `;
      }).join("");
    }

    function renderStats(payload) {
      const stats = [
        ["Machines", payload.stats.machine_count],
        ["Runs", payload.stats.latest_run_count],
        ["Cases", payload.stats.benchmark_case_count],
        ["Plots", payload.stats.comparison_group_count],
        ["Latest result", payload.stats.latest_result_at ? formatDate(payload.stats.latest_result_at) : "n/a"],
      ];
      document.getElementById("stats").innerHTML = stats.map(([label, value]) => `
        <article class="panel">
          <div class="metric-label">${label}</div>
          <div class="metric">${value}</div>
        </article>
      `).join("");
    }

    function renderChanges(payload) {
      const groups = [
        ["Regressions", payload.highlights.regressions],
        ["Improvements", payload.highlights.improvements],
        ["New cases", payload.highlights.new_cases],
      ];
      const cards = groups.map(([title, items]) => {
        const body = items.length
          ? items.map((item) => {
              const delta = deltaMeta(item.delta_pct);
              return `<div class="change-card"><strong>${escapeHtmlAttr(item.display_name)}</strong><p>${escapeHtmlAttr(item.machine)} | ${escapeHtmlAttr(item.env_name)} | <span class="${delta.cls}">${escapeHtmlAttr(delta.text)}</span></p></div>`;
            }).join("")
          : '<div class="empty">No entries.</div>';
        return `<div><h3>${escapeHtmlAttr(title)}</h3><div class="list-grid">${body}</div></div>`;
      }).join("");
      document.getElementById("changes").innerHTML = cards;
    }

    function renderRuns(payload) {
      const runs = payload.highlights.recent_runs;
      document.getElementById("runs").innerHTML = runs.length
        ? runs.map((run) => `
            <div class="run-card">
              <strong>${escapeHtmlAttr(run.machine_display_name)}</strong>
              <p>${escapeHtmlAttr(run.env_name)} | ${escapeHtmlAttr(run.commit_short || "n/a")}</p>
              <p>${escapeHtmlAttr(formatDate(run.date))}</p>
              <p>${escapeHtmlAttr(String(run.benchmark_case_count))} benchmark cases in ${escapeHtmlAttr(run.result_file)}</p>
            </div>
          `).join("")
        : '<div class="empty">No ASV runs found.</div>';
    }

    function renderCases(payload) {
      document.getElementById("cases").innerHTML = payload.benchmark_cases.slice(0, 24).map((item) => {
        const delta = deltaMeta(item.delta_pct);
        return `
          <tr>
            <td title="${escapeHtmlAttr(item.display_name)}">${item.display_name}</td>
            <td>${item.machine}</td>
            <td>${item.env_name}</td>
            <td>${formatNumber(item.value)} ${item.unit || ""}</td>
            <td class="${delta.cls}">${delta.text}</td>
            <td>${item.commit_short || "n/a"}</td>
          </tr>
        `;
      }).join("");
    }

    function populateFilter(selectId, options, emptyLabel) {
      const select = document.getElementById(selectId);
      const current = select.value;
      select.innerHTML = `<option value="">${emptyLabel}</option>` + options.map((option) => `
        <option value="${escapeHtmlAttr(option.value)}">${escapeHtmlAttr(option.label)}</option>
      `).join("");
      select.value = current;
    }

    async function load() {
      const response = await fetch("/api/summary");
      const payload = await response.json();
      document.getElementById("subtitle").textContent =
        payload.message ||
        `${payload.stats.comparison_group_count} comparison plots from ${payload.stats.benchmark_case_count} latest benchmark cases.`;

      populateFilter("machine-filter", payload.filters.machines, "All machines");
      populateFilter("env-filter", payload.filters.environments, "All environments");
      renderStats(payload);
      renderComparisonGroups(payload);
      renderChanges(payload);
      renderRuns(payload);
      renderCases(payload);

      document.getElementById("machine-filter").onchange = () => renderComparisonGroups(payload);
      document.getElementById("env-filter").onchange = () => renderComparisonGroups(payload);
    }

    load().catch((error) => {
      document.getElementById("subtitle").textContent = `Failed to load summary: ${error}`;
      document.getElementById("comparison-groups").innerHTML = '<div class="empty">Summary request failed.</div>';
    });
  </script>
</body>
</html>
"""


class BenchmarkDashboardRequestHandler(BaseHTTPRequestHandler):
    """Serve dashboard assets and a summarized JSON API."""

    server_version = "NewtonBenchmarkDashboard/1.0"

    def __init__(
        self,
        *args: Any,
        results_dir: Path,
        html_dir: Path,
        benchmark_index_path: Path,
        **kwargs: Any,
    ) -> None:
        self.results_dir = Path(results_dir)
        self.html_dir = Path(html_dir)
        self.benchmark_index_path = Path(benchmark_index_path)
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in {"", "/"}:
            self._send_html(_render_index_html())
            return
        if path == "/api/summary":
            self._send_json(build_dashboard_summary(self.results_dir, self.html_dir, self.benchmark_index_path))
            return
        if path.startswith("/asv"):
            relative = urllib.parse.unquote(path.removeprefix("/asv").lstrip("/"))
            self._serve_static(self.html_dir, relative)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, root: Path, relative_path: str) -> None:
        target = (root / relative_path).resolve() if relative_path else root.resolve()
        root_resolved = root.resolve()

        if target == root_resolved and root_resolved.is_dir():
            target = root_resolved / "index.html"
        elif target.is_dir():
            target = target / "index.html"

        try:
            target.relative_to(root_resolved)
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return

        if not target.exists() or not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        body = target.read_bytes()
        mime_type, _ = mimetypes.guess_type(str(target))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def make_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the Newton benchmark dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7000)
    parser.add_argument("--results-dir", default="asv/results")
    parser.add_argument("--html-dir", default="asv/html")
    parser.add_argument("--benchmark-index", default=str(default_benchmark_index_path()))
    parser.add_argument(
        "--max-result-age-hours",
        type=float,
        default=default_benchmark_max_result_age_hours(),
        help="Fail startup when the newest benchmark evidence is older than this many hours.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_argument_parser().parse_args(argv)
    freshness = assess_benchmark_freshness(args.results_dir, args.benchmark_index, args.max_result_age_hours)
    if freshness["is_stale"]:
        print(freshness["reason"])
        print(f"Benchmark index: {Path(args.benchmark_index).resolve()}")
        return 2
    handler = partial(
        BenchmarkDashboardRequestHandler,
        results_dir=Path(args.results_dir),
        html_dir=Path(args.html_dir),
        benchmark_index_path=Path(args.benchmark_index),
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
