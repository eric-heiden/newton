# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Aggregate visual-calibration trials into per-trial rows, group totals, and paired ratios."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def timing(workspace: Path, cli: str) -> dict:
    """Tool-execution intervals from event arrival times: union, summed duration, and remaining model time."""
    path = workspace / "agent.times.jsonl"
    if not path.exists():
        return {}
    times = [json.loads(line)["seconds"] for line in path.read_text().splitlines()]
    events = [json.loads(line) for line in (workspace / "agent.jsonl").read_text(errors="replace").splitlines()]
    intervals, pending = [], {}
    for event, t in zip(events, times, strict=False):
        if cli == "codex":
            item = event.get("item", {})
            if event.get("type") == "item.started" and item.get("type") in ("command_execution", "mcp_tool_call"):
                pending[item.get("id")] = t
            elif event.get("type") == "item.completed" and item.get("id") in pending:
                intervals.append((pending.pop(item["id"]), t, item["type"] == "mcp_tool_call"))
        elif event.get("type") == "assistant":
            for block in event.get("message", {}).get("content", []):
                if block.get("type") == "tool_use":
                    pending[block["id"]] = (t, block["name"].startswith("mcp__"))
        elif event.get("type") == "user":
            content = event.get("message", {}).get("content", [])
            for block in content if isinstance(content, list) else []:
                if block.get("type") == "tool_result" and block.get("tool_use_id") in pending:
                    start, is_mcp = pending.pop(block["tool_use_id"])
                    intervals.append((start, t, is_mcp))
    total = times[-1] if times else 0.0
    union, end = 0.0, -1.0
    for start, stop, _ in sorted(intervals):
        if stop > end:
            union += stop - max(start, end)
            end = stop
    summed = sum(stop - start for start, stop, _ in intervals)
    return {
        "event_seconds": round(total, 2),
        "tool_union_seconds": round(union, 2),
        "tool_summed_seconds": round(summed, 2),
        "mcp_summed_seconds": round(sum(stop - start for start, stop, m in intervals if m), 2),
        "model_seconds": round(total - union, 2),
        "tool_parallelism": round(summed / union, 3) if union else None,
    }


def rows(directory: Path) -> list[dict]:
    result = []
    for summary in sorted(directory.glob("*/summary.json")):
        if ".infra-failure-" in summary.parent.name:
            continue
        s = json.loads(summary.read_text())
        workspace = summary.parent
        anytime = workspace / "anytime.json"
        first = json.loads(anytime.read_text())["first_passing_seconds"] if anytime.exists() else None
        replicate = workspace.name.rsplit("-", 1)[-1]
        result.append(
            {
                "trial": workspace.name,
                "task": s["task"],
                # The spec's "model" holds the provider model ID; the short key is in the trial name.
                "model": {"claude-opus-5-5": "opus", "gpt-6-astra": "astra"}.get(s["model"], s["model"]),
                "model_id": s["model"],
                "condition": s["condition"],
                "replicate": replicate,
                "success": s["success"],
                "timed_out": s["timed_out"],
                "seconds": round(s["total_seconds"], 2),
                "input_tokens": s["usage"]["input_tokens"],
                "cached_input_tokens": s["usage"]["cached_input_tokens"],
                "output_tokens": s["usage"]["output_tokens"],
                "uncached_plus_output": s["usage"]["uncached_input_plus_output"],
                "cost_usd": s.get("cost_usd"),
                "tool_calls": s["tool_call_total"],
                "tool_errors": s["tool_errors"],
                "images_in_context": s["mcp_images_returned"],
                "parameter_sets": s["unique_parameter_sets"],
                "simulated_seconds": s["simulated_seconds"],
                "simulator_processes": s["simulator_processes"],
                "normalized_worst": s["verification"].get("normalized_worst"),
                "first_passing_seconds": first,
                **timing(workspace, s.get("cli", "claude")),
            }
        )
    return result


def _geomean_ratio(pairs, key, seed=20260929, draws=20000):
    ratios = [a[key] / b[key] for a, b in pairs if a[key] and b[key]]
    if not ratios:
        return None
    logs = [math.log(r) for r in ratios]
    estimate = math.exp(sum(logs) / len(logs))
    rng = random.Random(seed)
    samples = sorted(math.exp(sum(rng.choice(logs) for _ in logs) / len(logs)) for _ in range(draws))
    return {
        "ratio": round(estimate, 4),
        "ci95": [round(samples[int(0.025 * draws)], 4), round(samples[int(0.975 * draws) - 1], 4)],
        "n": len(ratios),
        "mcp_better": sum(r < 1 for r in ratios),
    }


def summarize(table: list[dict]) -> dict:
    groups = {}
    for row in table:
        key = (row["task"], row["model"], row["condition"])
        g = groups.setdefault(
            key,
            {
                "trials": 0,
                "successes": 0,
                "seconds": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "uncached_plus_output": 0,
                "tool_calls": 0,
            },
        )
        g["trials"] += 1
        g["successes"] += int(row["success"])
        for field in ("seconds", "input_tokens", "output_tokens", "uncached_plus_output", "tool_calls"):
            g[field] += row[field]
    index = {(r["task"], r["model"], r["condition"], r["replicate"]): r for r in table}
    paired = {}
    for scope in ("all", "opus", "astra", "cloth_drape", "push", "arm_offsets"):
        pairs = []
        for (task, model, condition, replicate), row in index.items():
            if condition != "mcp" or scope not in ("all", model, task):
                continue
            other = index.get((task, model, "restart", replicate))
            if other is not None:
                pairs.append((row, other))
        success_pairs = [(a, b) for a, b in pairs if a["success"] and b["success"]]
        paired[scope] = {
            "pairs": len(pairs),
            "mcp_successes": sum(a["success"] for a, _ in pairs),
            "restart_successes": sum(b["success"] for _, b in pairs),
            "all_pairs": {
                key: _geomean_ratio(pairs, key)
                for key in ("seconds", "input_tokens", "uncached_plus_output", "tool_calls")
            },
            "jointly_successful": {
                key: _geomean_ratio(success_pairs, key)
                for key in ("seconds", "input_tokens", "uncached_plus_output", "tool_calls")
            },
        }
    return {"groups": {"/".join(k): v for k, v in sorted(groups.items())}, "paired_mcp_over_restart": paired}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    table = rows(args.directory)
    result = {"trials": table, **summarize(table)}
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    for row in table:
        print(
            f"{row['trial']:32s} ok={int(row['success'])} t={row['seconds']:7.1f} in={row['input_tokens']:>9} out={row['output_tokens']:>6} "
            f"tools={row['tool_calls']:>3} img={row['images_in_context']:>2} sets={row['parameter_sets']:>4} worst={row['normalized_worst']}"
        )
    print(json.dumps(result["paired_mcp_over_restart"]["all"], indent=1))


if __name__ == "__main__":
    main()
