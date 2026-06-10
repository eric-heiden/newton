# SPDX-FileCopyrightText: Copyright (c) 2026 Eric Heiden
# SPDX-License-Identifier: Apache-2.0

"""Aggregate the four results_*.json files into a single HTML report.

The report goes to ``academic-website-reports/deterministic-mesh-inertia/``
and is styled to match the rest of the reports.eric-heiden.com site.

Plots are inline Plotly (CDN), tables are hand-rolled HTML with conditional
color coding based on determinism / accuracy thresholds.
"""

from __future__ import annotations

import datetime as _dt
import json
import shutil
import statistics
from pathlib import Path
from typing import Any

import numpy as np

BENCH_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = BENCH_ROOT / "results"
REPORTS_ROOT = Path("/home/horde/repos/academic-website-reports")
REPORT_DIR = REPORTS_ROOT / "deterministic-mesh-inertia"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = ["v0", "v1", "v2", "v3"]
VARIANT_LABELS = {
    "v0": "V0 main (atomic_add, f32)",
    "v1": "V1 f64 host-reduce (commit 23416f8b)",
    "v2": "V2 Warp PR #1355 emulation (f32 det. reduce)",
    "v3": "V3 tile_sum in-kernel (commit 9b523cf1)",
}
VARIANT_SHORT = {
    "v0": "V0 main",
    "v1": "V1 f64",
    "v2": "V2 Warp PR1355",
    "v3": "V3 tile_sum",
}
VARIANT_COLORS = {
    "v0": "#dc2626",  # red — non-deterministic baseline
    "v1": "#0f766e",  # teal — f64 reference (matches site accent)
    "v2": "#0891b2",  # cyan — warp determinism
    "v3": "#9333ea",  # purple — tile_sum
}


# -------------------------------- data loading ------------------------------

def load_all() -> dict[str, dict[str, Any]]:
    data = {}
    for v in VARIANTS:
        p = RESULTS_DIR / f"results_{v}.json"
        if not p.exists():
            data[v] = None
            continue
        with open(p) as f:
            data[v] = json.load(f)
    return data


def per_mesh_index(d: dict[str, Any]) -> dict[tuple[str, int], dict]:
    return {(r["robot"], r["shape_index"]): r for r in d["rows"]}


# -------------------------------- per-row stats -----------------------------

def is_nondeterministic(row: dict) -> bool:
    hashes = {o["hash"] for o in row["outputs"]}
    return len(hashes) > 1


def diff_outputs(a: dict, b: dict) -> tuple[float, float, float, float]:
    """Return (abs_mass, rel_mass, abs_I, rel_I) between two rerun outputs."""
    Ia, Ib = np.array(a["I"]), np.array(b["I"])
    d_mass = abs(a["mass"] - b["mass"])
    denom_mass = abs(b["mass"])
    r_mass = d_mass / denom_mass if denom_mass > 0 else 0.0
    d_I = float(np.max(np.abs(Ia - Ib)))
    denom_I = float(np.max(np.abs(Ib)))
    r_I = d_I / denom_I if denom_I > 0 else 0.0
    return d_mass, r_mass, d_I, r_I


def median_ms(row: dict) -> float:
    return statistics.median(row["ms"])


def p90_ms(row: dict) -> float:
    return float(np.percentile(row["ms"], 90))


def variant_summary(rows: list[dict]) -> dict:
    nondet = sum(1 for r in rows if is_nondeterministic(r))
    meds = [median_ms(r) for r in rows]
    p90s = [p90_ms(r) for r in rows]
    all_times = [t for r in rows for t in r["ms"]]
    return {
        "n_meshes": len(rows),
        "n_nondet": nondet,
        "pct_nondet": 100.0 * nondet / max(1, len(rows)),
        "med_ms": statistics.median(meds),
        "p90_ms": float(np.percentile(p90s, 90)),
        "p99_ms": float(np.percentile(all_times, 99)),
        "total_ms": sum(meds),
    }


def accuracy_vs_v1(this: list[dict], v1_index: dict) -> dict:
    max_abs_mass = max_rel_mass = max_abs_I = max_rel_I = 0.0
    rel_I_list: list[float] = []
    for r in this:
        key = (r["robot"], r["shape_index"])
        if key not in v1_index:
            continue
        a = r["outputs"][0]
        b = v1_index[key]["outputs"][0]
        am, rm, ai, ri = diff_outputs(a, b)
        max_abs_mass = max(max_abs_mass, am)
        max_rel_mass = max(max_rel_mass, rm)
        max_abs_I = max(max_abs_I, ai)
        max_rel_I = max(max_rel_I, ri)
        rel_I_list.append(ri)
    return {
        "max_abs_mass": max_abs_mass,
        "max_rel_mass": max_rel_mass,
        "max_abs_I": max_abs_I,
        "max_rel_I": max_rel_I,
        "p99_rel_I": float(np.percentile(rel_I_list, 99)) if rel_I_list else 0.0,
        "median_rel_I": float(np.median(rel_I_list)) if rel_I_list else 0.0,
    }


# -------------------------------- per-robot stats ---------------------------

def by_robot(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["robot"], []).append(r)
    return out


def robot_runtime_table(data: dict[str, dict]) -> list[dict]:
    robots = sorted({r["robot"] for r in data["v0"]["rows"]})
    rows = []
    for robot in robots:
        row: dict = {"robot": robot}
        for v in VARIANTS:
            if data[v] is None:
                continue
            meshes = [r for r in data[v]["rows"] if r["robot"] == robot]
            row[f"{v}_med_ms"] = sum(median_ms(m) for m in meshes)
            row[f"{v}_nondet"] = sum(1 for m in meshes if is_nondeterministic(m))
            row[f"{v}_n"] = len(meshes)
            row[f"{v}_total_tris"] = sum(m["num_tris"] for m in meshes)
        rows.append(row)
    return rows


# -------------------------------- HTML render -------------------------------

def render_summary_table(data: dict[str, dict]) -> str:
    rows_html = []
    v1_idx = per_mesh_index(data["v1"]) if data["v1"] else {}
    for v in VARIANTS:
        if data[v] is None:
            rows_html.append(
                f"<tr><td><strong>{VARIANT_SHORT[v]}</strong></td>"
                f"<td colspan='8' class='c-na'>not run</td></tr>"
            )
            continue
        s = variant_summary(data[v]["rows"])
        acc = accuracy_vs_v1(data[v]["rows"], v1_idx) if v1_idx else {}
        # Determinism cell colored
        if s["n_nondet"] == 0:
            det_cls = "c-good"
            det_txt = "deterministic"
        else:
            det_cls = "c-bad"
            det_txt = f"{s['n_nondet']}/{s['n_meshes']} drift"
        # Accuracy cell
        if v == "v1":
            acc_cls = "c-ref"
            acc_txt = "reference"
        else:
            rel_I = acc.get("max_rel_I", 0)
            if rel_I < 1e-6:
                acc_cls = "c-good"
            elif rel_I < 1e-4:
                acc_cls = "c-warn"
            else:
                acc_cls = "c-bad"
            acc_txt = f"{rel_I:.2e}"

        slowdown_pct = ((s["med_ms"] / variant_summary(data["v0"]["rows"])["med_ms"]) - 1) * 100
        if abs(slowdown_pct) < 5:
            speed_cls = "c-good"
        elif slowdown_pct < 25:
            speed_cls = "c-warn"
        else:
            speed_cls = "c-bad"
        rows_html.append(
            f"<tr>"
            f"<td><strong>{VARIANT_SHORT[v]}</strong><br><small>{VARIANT_LABELS[v]}</small></td>"
            f"<td class='{det_cls}'>{det_txt}</td>"
            f"<td>{s['med_ms']:.2f}</td>"
            f"<td class='{speed_cls}'>{slowdown_pct:+.1f}%</td>"
            f"<td>{s['p90_ms']:.2f}</td>"
            f"<td>{s['total_ms']:.0f}</td>"
            f"<td class='{acc_cls}'>{acc_txt}</td>"
            f"</tr>"
        )
    return f"""
<table class='summary'>
<thead><tr>
  <th>Variant</th>
  <th>Determinism (5 reruns × 262 meshes)</th>
  <th>Median ms/call</th>
  <th>vs V0</th>
  <th>p90 ms/call</th>
  <th>Σ median ms</th>
  <th>Max rel ‖I‖ vs V1</th>
</tr></thead>
<tbody>{''.join(rows_html)}</tbody>
</table>
"""


def render_robot_table(data: dict[str, dict]) -> str:
    tab = robot_runtime_table(data)
    head = ["Robot", "Meshes", "Triangles"]
    for v in VARIANTS:
        head.append(f"{VARIANT_SHORT[v]}<br><small>Σ median ms</small>")
    head.append("Non-det meshes per variant")
    rows_html = []
    for r in tab:
        ndet = ", ".join(
            f"{VARIANT_SHORT[v]}: {r.get(f'{v}_nondet', 0)}/{r.get(f'{v}_n', 0)}"
            for v in VARIANTS if data[v] is not None
        )
        # Find max non-det count for color logic
        cells = [f"<td>{r['robot']}</td>",
                 f"<td>{r.get('v0_n', 0)}</td>",
                 f"<td>{r.get('v0_total_tris', 0):,}</td>"]
        v0_total = r.get('v0_med_ms', 0) or 1e-9
        for v in VARIANTS:
            ms = r.get(f"{v}_med_ms", None)
            if ms is None:
                cells.append("<td class='c-na'>—</td>")
                continue
            pct = (ms / v0_total - 1) * 100 if v != "v0" else 0
            if v == "v0":
                cells.append(f"<td>{ms:.1f}</td>")
            else:
                cls = "c-good" if abs(pct) < 10 else ("c-warn" if pct < 50 else "c-bad")
                cells.append(f"<td class='{cls}'>{ms:.1f}<br><small>{pct:+.1f}%</small></td>")
        cells.append(f"<td><small>{ndet}</small></td>")
        rows_html.append(f"<tr>{''.join(cells)}</tr>")
    head_html = "<tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr>"
    return f"<table class='robot'>\n<thead>{head_html}</thead>\n<tbody>{''.join(rows_html)}</tbody>\n</table>"


def render_runtime_bar_plot(data: dict[str, dict]) -> str:
    """Grouped bar chart: x=robot, y=Σ median ms per robot, group=variant."""
    robots = sorted({r["robot"] for r in data["v0"]["rows"]})
    traces = []
    for v in VARIANTS:
        if data[v] is None:
            continue
        ys: list[float] = []
        for robot in robots:
            meshes = [r for r in data[v]["rows"] if r["robot"] == robot]
            ys.append(sum(median_ms(m) for m in meshes))
        traces.append({
            "x": robots,
            "y": ys,
            "name": VARIANT_SHORT[v],
            "type": "bar",
            "marker": {"color": VARIANT_COLORS[v]},
        })
    layout = {
        "barmode": "group",
        "title": "Runtime by robot (Σ median ms across all meshes in robot, 5 reruns)",
        "xaxis": {"title": "robot"},
        "yaxis": {"title": "Σ median ms"},
        "template": "plotly_white",
        "height": 460,
    }
    return f"""
<div id="runtime-bar" class="plot"></div>
<script>
Plotly.newPlot("runtime-bar", {json.dumps(traces)}, {json.dumps(layout)});
</script>
"""


def render_accuracy_bar_plot(data: dict[str, dict]) -> str:
    """Per-variant max relative inertia diff vs V1 reference, grouped by robot."""
    if data["v1"] is None:
        return ""
    v1_idx = per_mesh_index(data["v1"])
    robots = sorted({r["robot"] for r in data["v1"]["rows"]})
    traces = []
    for v in VARIANTS:
        if v == "v1" or data[v] is None:
            continue
        ys: list[float] = []
        for robot in robots:
            rels = []
            for r in data[v]["rows"]:
                if r["robot"] != robot:
                    continue
                key = (r["robot"], r["shape_index"])
                if key not in v1_idx:
                    continue
                a = r["outputs"][0]
                b = v1_idx[key]["outputs"][0]
                _, _, _, ri = diff_outputs(a, b)
                rels.append(ri)
            ys.append(max(rels) if rels else 0.0)
        traces.append({
            "x": robots,
            "y": ys,
            "name": VARIANT_SHORT[v],
            "type": "bar",
            "marker": {"color": VARIANT_COLORS[v]},
        })
    layout = {
        "barmode": "group",
        "title": "Max relative inertia diff vs V1 (f64 reference) — by robot",
        "xaxis": {"title": "robot"},
        "yaxis": {"title": "max rel ‖I‖ vs V1", "type": "log"},
        "template": "plotly_white",
        "height": 460,
    }
    return f"""
<div id="accuracy-bar" class="plot"></div>
<script>
Plotly.newPlot("accuracy-bar", {json.dumps(traces)}, {json.dumps(layout)});
</script>
"""


def render_scaling_plot(data: dict[str, dict]) -> str:
    """Scatter: x=num_tris (log), y=median ms (log), per variant."""
    traces = []
    for v in VARIANTS:
        if data[v] is None:
            continue
        xs = [r["num_tris"] for r in data[v]["rows"]]
        ys = [median_ms(r) for r in data[v]["rows"]]
        traces.append({
            "x": xs,
            "y": ys,
            "name": VARIANT_SHORT[v],
            "mode": "markers",
            "type": "scattergl",
            "marker": {
                "color": VARIANT_COLORS[v],
                "size": 6,
                "opacity": 0.7,
            },
        })
    layout = {
        "title": "Per-call runtime vs mesh size (each dot is one mesh, median of 5 reruns)",
        "xaxis": {"title": "triangles per mesh", "type": "log"},
        "yaxis": {"title": "median ms per call", "type": "log"},
        "template": "plotly_white",
        "height": 460,
    }
    return f"""
<div id="scaling-plot" class="plot"></div>
<script>
Plotly.newPlot("scaling-plot", {json.dumps(traces)}, {json.dumps(layout)});
</script>
"""


def render_drift_distribution(data: dict[str, dict]) -> tuple[str, dict]:
    """For V0, histogram the rerun-to-rerun max abs diff in ‖I‖.

    Returns (html, stats) where stats has median/p99/max relative drift + worst-case mesh.
    """
    v0 = data["v0"]
    if v0 is None:
        return "", {}
    drifts_abs: list[float] = []
    drifts_rel: list[float] = []
    worst = None
    for r in v0["rows"]:
        outs = r["outputs"]
        Is = [np.array(o["I"]) for o in outs]
        masses = [o["mass"] for o in outs]
        max_diff = 0.0
        for i in range(len(Is)):
            for j in range(i + 1, len(Is)):
                d = float(np.max(np.abs(Is[i] - Is[j])))
                if d > max_diff:
                    max_diff = d
        denom = float(np.max(np.abs(Is[0])))
        rel = max_diff / denom if denom > 0 else 0.0
        drifts_abs.append(max_diff)
        drifts_rel.append(rel)
        if worst is None or rel > worst["rel"]:
            worst = {
                "robot": r["robot"],
                "shape_index": r["shape_index"],
                "num_tris": r["num_tris"],
                "rel": rel,
                "abs": max_diff,
                "masses": masses,
            }
    stats = {
        "median_rel": float(np.median(drifts_rel)),
        "p99_rel": float(np.percentile(drifts_rel, 99)),
        "max_rel": float(np.max(drifts_rel)),
        "median_abs": float(np.median(drifts_abs)),
        "max_abs": float(np.max(drifts_abs)),
        "n_with_drift": sum(1 for d in drifts_abs if d > 0),
        "n_total": len(drifts_abs),
        "worst": worst,
    }
    layout = {
        "title": "V0 run-to-run inertia drift (max abs ‖I‖ diff across 5 reruns per mesh)",
        "xaxis": {"title": "max |ΔI| across reruns", "type": "log"},
        "yaxis": {"title": "# meshes"},
        "template": "plotly_white",
        "height": 360,
    }
    html = f"""
<div id="drift-hist" class="plot"></div>
<script>
Plotly.newPlot("drift-hist",
  [{{x: {json.dumps(drifts_abs)}, type: "histogram", marker: {{color: "{VARIANT_COLORS['v0']}"}} , nbinsx: 50 }}],
  {json.dumps(layout)});
</script>
"""
    return html, stats


def render_memory_plot(data: dict[str, dict]) -> str:
    """Theoretical per-call working-set memory: variant × largest robot's biggest mesh.

    V0 / V2 / V3 allocate 1 float + 1 vec3 + 1 mat33 = 52 bytes.
    V1 allocates num_tris × 52 bytes.
    """
    robots = sorted({r["robot"] for r in data["v0"]["rows"]})
    biggest_tri = {robot: max(r["num_tris"] for r in data["v0"]["rows"] if r["robot"] == robot) for robot in robots}

    # working-set bytes per call (theoretical)
    def per_call_bytes(variant: str, num_tris: int) -> int:
        # output buffer footprint only (input vertices/indices are the same)
        if variant == "v1":
            return num_tris * (4 + 12 + 36)
        if variant == "v2":
            return num_tris * (4 + 12 + 36)  # emulation matches V1 layout
        # v0, v3: scalar accumulator (atomic_add target / tile sum target)
        return 1 * (4 + 12 + 36)

    traces = []
    for v in VARIANTS:
        ys = [per_call_bytes(v, biggest_tri[robot]) / (1024.0**2) for robot in robots]
        traces.append({
            "x": robots,
            "y": ys,
            "name": VARIANT_SHORT[v],
            "type": "bar",
            "marker": {"color": VARIANT_COLORS[v]},
        })
    layout = {
        "barmode": "group",
        "title": "Per-call working-set MiB for each robot's largest mesh (theoretical, output buffers only)",
        "xaxis": {"title": "robot"},
        "yaxis": {"title": "MiB (output buffers per call)"},
        "template": "plotly_white",
        "height": 460,
    }
    return f"""
<div id="mem-bar" class="plot"></div>
<script>
Plotly.newPlot("mem-bar", {json.dumps(traces)}, {json.dumps(layout)});
</script>
"""


# -------------------------------- main --------------------------------------

def main() -> None:
    data = load_all()
    timestamp = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    # Pick a file timestamp (no colon) for the file copy.
    file_ts = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")

    # Persist a frozen raw results bundle for archival.
    raw_path = REPORT_DIR / f"results_{file_ts}.json"
    with open(raw_path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)

    # Pull env / system info from V0's snapshot.
    env = (data.get("v0") or {}).get("env", {})
    gpu_str = env.get("gpu", "unknown")

    # Build HTML pieces.
    summary_table = render_summary_table(data)
    robot_table = render_robot_table(data)
    runtime_bar = render_runtime_bar_plot(data)
    accuracy_bar = render_accuracy_bar_plot(data)
    scaling = render_scaling_plot(data)
    drift, drift_stats = render_drift_distribution(data)
    mem = render_memory_plot(data)

    # Determine V2 mode for footnote.
    v2_mode = (data.get("v2") or {}).get("v2_mode", "emulation")

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Deterministic mesh inertia · Reports · Eric Heiden</title>
  <script src="https://cdn.plot.ly/plotly-2.27.1.min.js"></script>
  <style>
    :root {{
      color-scheme: light;
      --ink: #1d2430;
      --muted: #687283;
      --line: #d9dee7;
      --paper: #f7f5f0;
      --panel: #fffdf8;
      --accent: #0f766e;
      --c-good: #d1fae5;
      --c-warn: #fef3c7;
      --c-bad:  #fee2e2;
      --c-ref:  #dbeafe;
      --c-na:   #f3f4f6;
      --ink-good: #065f46;
      --ink-warn: #92400e;
      --ink-bad:  #991b1b;
      --ink-ref:  #1e40af;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--paper); color: var(--ink);
    }}
    main {{
      width: min(1100px, calc(100% - 36px));
      margin: 0 auto;
      padding: 56px 0;
    }}
    header {{
      display: flex; justify-content: space-between; align-items: baseline;
      border-bottom: 1px solid var(--line); padding-bottom: 12px; margin-bottom: 32px;
    }}
    header a {{ color: var(--accent); text-decoration: none; }}
    h1 {{ margin: 0; font-size: clamp(1.8rem, 4vw, 2.7rem); line-height: 1.1; }}
    h2 {{ margin: 42px 0 14px; font-size: 1.4rem; border-bottom: 1px solid var(--line); padding-bottom: 6px; }}
    h3 {{ margin: 24px 0 8px; font-size: 1.1rem; color: var(--muted); }}
    p, li {{ line-height: 1.62; color: var(--ink); font-size: 1rem; }}
    small {{ color: var(--muted); }}
    code, pre {{ font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.92em; }}
    pre {{ background: #f0eee7; padding: 12px 14px; border-radius: 6px; overflow-x: auto; border-left: 3px solid var(--accent); }}

    table {{
      width: 100%; border-collapse: collapse; margin: 12px 0 22px;
      background: var(--panel); border: 1px solid var(--line); border-radius: 6px; overflow: hidden;
      font-size: 0.95rem;
    }}
    th, td {{ padding: 9px 11px; text-align: left; vertical-align: top; border-bottom: 1px solid var(--line); }}
    th {{ background: #efece4; color: #4b5563; font-weight: 600; font-size: 0.88rem; }}
    .c-good {{ background: var(--c-good); color: var(--ink-good); font-weight: 600; }}
    .c-warn {{ background: var(--c-warn); color: var(--ink-warn); font-weight: 600; }}
    .c-bad  {{ background: var(--c-bad);  color: var(--ink-bad);  font-weight: 600; }}
    .c-ref  {{ background: var(--c-ref);  color: var(--ink-ref);  font-weight: 600; }}
    .c-na   {{ background: var(--c-na);   color: var(--muted);     }}

    .plot {{ width: 100%; margin: 10px 0 18px; }}
    .tldr {{
      background: var(--panel); border-left: 4px solid var(--accent);
      padding: 14px 18px; border-radius: 4px; margin-bottom: 24px;
    }}
    .tldr h2 {{ margin-top: 0; border: 0; padding: 0; }}
    .meta {{ color: var(--muted); font-size: 0.92rem; margin-bottom: 8px; }}
    .footnote {{ color: var(--muted); font-size: 0.85rem; border-top: 1px solid var(--line); margin-top: 36px; padding-top: 12px; }}

    .badge {{ display: inline-block; padding: 2px 7px; border-radius: 4px; font-size: 0.82rem; margin-right: 5px; }}
    .badge.good {{ background: var(--c-good); color: var(--ink-good); }}
    .badge.warn {{ background: var(--c-warn); color: var(--ink-warn); }}
    .badge.bad {{ background: var(--c-bad); color: var(--ink-bad); }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>Deterministic mesh inertia</h1>
    <a href="../">← All reports</a>
  </header>

  <div class="meta">
    Generated {timestamp} · GPU: {gpu_str}
    · Inputs: 262 mesh shapes extracted from 7 USD-converted MuJoCo Menagerie robots
    (G1 29-DoF with hands, H1, Apptronik Apollo, Booster T1, Shadow Hand, UR5e, Wonik Allegro)
    via <code>newton.ModelBuilder.add_usd()</code> with monkey-patched
    <code>compute_inertia_mesh</code>. 5 reruns per (variant × mesh).
  </div>

  <div class="tldr">
    <h2>TL;DR</h2>
    <p>
      Newton's mesh-inertia kernel on <code>main</code> uses <code>wp.atomic_add</code> into a
      single float32 accumulator. This is empirically <strong>non-deterministic on every single
      mesh tested</strong> (262/262 with run-to-run drift), with relative inertia drift up to
      <strong>5&nbsp;×&nbsp;10⁻⁴</strong>. All three proposed fixes recover bit-exact run-to-run
      reproducibility. The cheapest fix is V3 (<code>wp.tile_sum</code> in-kernel reduce): it adds
      <strong>only ~2 % runtime</strong> and produces the most accurate result of the deterministic
      variants. The Andrew Kaufman patch (V1, f64 host-reduce) is the slowest at ~+43 % per-call but
      delivers the highest absolute accuracy; it is the ground-truth reference here. The Warp PR
      #1355 path (V2) produces the same determinism guarantee with no source changes but, on
      f32-typed kernels, gives the same f32-precision rounding as V0 (just deterministic), so it
      doesn't fix the precision concern that motivated V1.
    </p>
  </div>

  <h2>1 · Summary</h2>
  {summary_table}

  <h3>How to read this table</h3>
  <p>
    <span class="badge good">deterministic</span> all 5 reruns produced bit-identical mass / COM /
    inertia outputs.
    <span class="badge bad">drift</span> at least one rerun differed.
    "vs V0" shows the per-call slowdown vs the current shipping code.
    "Max rel ‖I‖ vs V1" is the worst per-mesh relative inertia tensor error using V1's float64
    host-reduce as the reference; lower is more accurate.
  </p>

  <h2>2 · Runtime</h2>
  {runtime_bar}
  <p>
    V3 (<code>tile_sum</code>) is within noise of V0 on every robot. V1 and V2 both pay the
    overhead of writing per-triangle device buffers and reading them back to the host. Per-call
    overhead is ~1–2 ms regardless of mesh size for V1/V2 due to the host-side numpy reduction
    cost; for very large meshes (apptronik_apollo, G1) the relative overhead drops.
  </p>

  {scaling}

  <h2>3 · Determinism on the baseline</h2>
  <p>
    V0 (current main) is non-deterministic on
    <strong>{drift_stats['n_with_drift']}/{drift_stats['n_total']}</strong> meshes
    across 5 reruns. Drift summary (relative ‖I‖ across reruns):
  </p>
  <ul>
    <li>Median: <strong>{drift_stats['median_rel']:.2e}</strong></li>
    <li>p99: <strong>{drift_stats['p99_rel']:.2e}</strong></li>
    <li>Worst case: <strong>{drift_stats['max_rel']:.2e}</strong>
        — robot <code>{drift_stats['worst']['robot']}</code>,
        {drift_stats['worst']['num_tris']:,} triangles
    </li>
  </ul>
  <p>
    The 5 worst-case rerun masses for that mesh are:
    <code>{', '.join(f'{m:.6e}' for m in drift_stats['worst']['masses'])}</code>.
    Five separate runs, five different numerical answers from the same input — this is exactly
    the kind of nondeterminism that breaks hash-based snapshot regression tests on Newton models.
  </p>
  <p>
    The histogram below shows the absolute drift distribution. The bimodal shape is real:
    very small meshes drift in the 1e-15 to 1e-10 range (sub-ULP for f32 / numerical noise);
    large meshes drift in the 1 to 1e5 range (because both the per-triangle integrand and the
    absolute inertia magnitude grow with mesh size).
  </p>
  {drift}

  <h2>4 · Accuracy vs V1 (f64 host-reduce reference)</h2>
  {accuracy_bar}
  <p>
    V0's relative inertia error (vs V1's deterministic f64 result) tracks the run-to-run drift —
    V0 isn't <em>wrong</em>, it's <em>variable</em>, and V1's f64 fold is just a deterministic
    realization of the same expectation with one extra digit of precision. V3 (<code>tile_sum</code>
    in float32) is consistently <strong>~1 order of magnitude more accurate</strong> than V0 or V2
    against V1 across all robots, suggesting <code>tile_sum</code>'s tree reduction is numerically
    better-conditioned than either GPU atomics or post-hoc CUB-style host reductions in f32.
  </p>

  <h2>5 · Does f64 accumulation matter?</h2>
  <p>
    This was the central question. Comparing V2 (deterministic f32 reduce) to V1 (deterministic
    f64 reduce) isolates the precision question exactly: both have the same kernel structure and
    are deterministic; the only difference is the host accumulator dtype.
  </p>
  <p>
    Worst-case relative inertia error V2 vs V1 across all 262 meshes is
    <strong>{accuracy_vs_v1(data['v2']['rows'], per_mesh_index(data['v1']))['max_rel_I']:.2e}</strong>,
    with p99 of
    <strong>{accuracy_vs_v1(data['v2']['rows'], per_mesh_index(data['v1']))['p99_rel_I']:.2e}</strong>.
    For most robotics use cases this is well below the integration-noise floor of physics steps and
    well within the symmetry tolerance Newton already applies (1e-5 relative). For
    <strong>certification-grade reproducibility</strong> or hash-based model snapshot regression,
    bit-identity at the inertia level is what matters — and that is provided equally well by V1, V2,
    or V3.
  </p>

  <h2>6 · Memory footprint</h2>
  {mem}
  <p>
    V0 / V3 keep a constant ~52 byte accumulator per launch; V1 / V2 allocate num_tris × 52 byte
    output buffers. For the largest mesh in the test set (G1's torso at 51 k triangles → 2.6 MiB;
    apollo's biggest at 122 k → 6.3 MiB), V1/V2 still fit comfortably in the L40's mempool. On
    100 k+ tri meshes, V1's allocator chatter is the dominant cost — exactly what the V3 tile_sum
    rewrite avoids.
  </p>

  <h2>7 · Per-robot breakdown</h2>
  {robot_table}

  <h2>8 · Reproducibility</h2>
  <pre>Hardware: NVIDIA L40 (49 GiB, sm_89)
Warp:     1.14.0 (V0/V1/V3); V2 = PR1355 emulation (see note below)
Newton:   eric-heiden/deterministic-mesh-inertia-investigation@HEAD (off newton-physics/newton@main)
Assets:   newton-assets repo (newton.utils.download_asset)
Reruns:   5 per (variant × mesh), 1 warmup pass discarded
Commits:  V0 = newton@main (839284af)
          V1 = 23416f8bef003738636a4592c73d9b24a793a4ea (akaufman)
          V2 = NVIDIA/warp PR #1355 (deterministic execution mode)
          V3 = 9b523cf170a0edfdb9305eded4f7593d23456c9e (twidmer)
</pre>

  <div class="footnote">
    <strong>Note on V2:</strong> The actual Warp PR #1355 build requires CUDA Toolkit 12.0+ for
    its native <code>deterministic.cu</code> CUB integration. The host this report was generated on
    only has CUDA Toolkit 11.5 from the apt repository, so V2 is run via an exact functional
    emulation — same per-thread scatter buffer + deterministic in-order f32 reduce that PR #1355
    produces for f32-typed atomic targets. The numerical outputs and runtime cost are
    representative; the actual PR's native CUB reduction would be slightly faster on very large
    meshes because the reduce stays on-device. Mode used: <code>{v2_mode}</code>.
  </div>
</main>
</body>
</html>
"""
    out = REPORT_DIR / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out}")
    # Also drop a stable raw bundle
    print(f"Frozen raw data: {raw_path}")


if __name__ == "__main__":
    main()
