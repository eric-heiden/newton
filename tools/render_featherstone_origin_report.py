# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render the Featherstone origin-offset investigation as a standalone HTML report."""

from __future__ import annotations

import html
import json
import math
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPORT_DIR = Path("reports/featherstone_origin_offset")
OUT_PATH = REPORT_DIR / "report.html"


def load_json(name: str) -> dict:
    """Load one benchmark result."""
    return json.loads((REPORT_DIR / name).read_text(encoding="utf-8"))


def finite(value: object) -> bool:
    """Return whether a value is a finite number."""
    return isinstance(value, int | float) and math.isfinite(float(value))


def fmt(value: object, unit: str = "", precision: int = 3) -> str:
    """Format a numeric value for HTML tables."""
    if value is None:
        return "none"
    if not finite(value):
        return "NaN"
    number = float(value)
    if abs(number) >= 1000.0 or (0.0 < abs(number) < 0.001):
        text = f"{number:.{precision}e}"
    else:
        text = f"{number:.{precision}f}"
    return f"{text} {unit}".rstrip()


def series(result: dict, key: str) -> list[tuple[float, float]]:
    """Extract finite ``(frame, value)`` points."""
    points = []
    for sample in result["samples"]:
        x = sample.get("frame")
        y = sample.get(key)
        if finite(x) and finite(y):
            points.append((float(x), float(y)))
    return points


def nice_bounds(values: Iterable[float], log: bool) -> tuple[float, float]:
    """Return plot bounds."""
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return (0.0, 1.0)
    lo = min(vals)
    hi = max(vals)
    if log:
        lo = max(lo, 1.0e-9)
        hi = max(hi, lo * 1.01)
        return (lo, hi)
    if lo == hi:
        pad = 1.0 if lo == 0.0 else abs(lo) * 0.1
        return (lo - pad, hi + pad)
    pad = (hi - lo) * 0.08
    return (lo - pad, hi + pad)


def svg_plot(
    title: str,
    ylabel: str,
    datasets: list[tuple[str, str, list[tuple[float, float]]]],
    *,
    log_y: bool = False,
    width: int = 860,
    height: int = 330,
) -> str:
    """Render a simple inline SVG line plot."""
    margin_left = 74
    margin_right = 24
    margin_top = 44
    margin_bottom = 54
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    all_x = [x for _, _, points in datasets for x, _ in points]
    all_y = [y for _, _, points in datasets for _, y in points]
    x0, x1 = nice_bounds(all_x, log=False)
    y0, y1 = nice_bounds(all_y, log=log_y)
    if log_y:
        ly0 = math.log10(y0)
        ly1 = math.log10(y1)

    def sx(x: float) -> float:
        return margin_left + (x - x0) / (x1 - x0) * plot_w

    def sy(y: float) -> float:
        if log_y:
            y_log = math.log10(max(y, y0))
            return margin_top + (ly1 - y_log) / (ly1 - ly0) * plot_h
        return margin_top + (y1 - y) / (y1 - y0) * plot_h

    grid = []
    for i in range(5):
        t = i / 4.0
        x = margin_left + t * plot_w
        y = margin_top + t * plot_h
        grid.append(f'<line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{margin_top + plot_h}" />')
        grid.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{margin_left + plot_w}" y2="{y:.1f}" />')

    lines = []
    legend = []
    for index, (label, color, points) in enumerate(datasets):
        if not points:
            continue
        d = " ".join(("M" if i == 0 else "L") + f"{sx(x):.1f},{sy(y):.1f}" for i, (x, y) in enumerate(points))
        lines.append(f'<path d="{d}" stroke="{color}" />')
        lx = margin_left + index * 230
        ly = height - 18
        legend.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 28}" y2="{ly}" stroke="{color}" class="legend-line" />')
        legend.append(f'<text x="{lx + 36}" y="{ly + 4}" class="legend">{html.escape(label)}</text>')

    x_labels = []
    for i in range(5):
        value = x0 + (x1 - x0) * i / 4.0
        x_labels.append(f'<text x="{sx(value):.1f}" y="{height - 34}" text-anchor="middle">{value:.0f}</text>')

    y_labels = []
    for i in range(5):
        if log_y:
            exponent = ly0 + (ly1 - ly0) * (4 - i) / 4.0
            value = 10.0**exponent
        else:
            value = y0 + (y1 - y0) * (4 - i) / 4.0
        y = margin_top + i / 4.0 * plot_h
        y_labels.append(f'<text x="{margin_left - 12}" y="{y + 4:.1f}" text-anchor="end">{fmt(value, precision=2)}</text>')

    return f"""
    <figure class="plot">
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
        <text x="{margin_left}" y="24" class="plot-title">{html.escape(title)}</text>
        <g class="grid">{''.join(grid)}</g>
        <line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" class="axis" />
        <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" class="axis" />
        <g class="labels">{''.join(x_labels)}{''.join(y_labels)}</g>
        <text x="{margin_left + plot_w / 2:.1f}" y="{height - 6}" text-anchor="middle" class="axis-title">frame</text>
        <text transform="translate(18 {margin_top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle" class="axis-title">{html.escape(ylabel)}</text>
        <g class="lines">{''.join(lines)}</g>
        <g>{''.join(legend)}</g>
      </svg>
    </figure>
    """


def summary_table(rows: list[tuple[str, dict, dict | None]]) -> str:
    """Render the main summary table."""
    body = []
    for label, long_run, perf in rows:
        samples = long_run["samples"]
        last_finite = next((sample for sample in reversed(samples) if sample.get("finite")), samples[-1])
        body.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{long_run['frames_completed']}</td>"
            f"<td>{long_run.get('first_nonfinite_frame') or 'none'}</td>"
            f"<td>{fmt(last_finite.get('far_J_max_abs'))}</td>"
            f"<td>{fmt(last_finite.get('far_M_max_abs'))}</td>"
            f"<td>{fmt(last_finite.get('root_delta_error_m'), 'm')}</td>"
            f"<td>{fmt(last_finite.get('local_pose_error_m'), 'm')}</td>"
            f"<td>{fmt(perf.get('fps') if perf else long_run.get('fps'), 'FPS')}</td>"
            "</tr>"
        )
    return """
    <table>
      <thead>
        <tr>
          <th>Code state</th>
          <th>Long-run frames</th>
          <th>First non-finite frame</th>
          <th>far max |J|</th>
          <th>far max |M|</th>
          <th>last finite root error</th>
          <th>last finite local error</th>
          <th>warm-cache speed</th>
        </tr>
      </thead>
      <tbody>
    """ + "\n".join(body) + """
      </tbody>
    </table>
    """


def main() -> None:
    """Build the report."""
    baseline = load_json("current_branch_baseline_2000.json")
    fixed = load_json("translated_root_com_2500.json")
    origin = load_json("origin_main_2000.json")
    baseline_perf = load_json("current_branch_baseline_perf_1000.json")
    fixed_perf = load_json("translated_root_com_perf_1000.json")
    origin_perf = load_json("origin_main_perf_1000.json")
    generated_at = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S %Z (PT)")

    rows = [
        ("origin/main a046d6131", origin, origin_perf),
        ("branch baseline 1cd4196dc", baseline, baseline_perf),
        ("patched working tree", fixed, fixed_perf),
    ]

    plot_datasets = [
        ("origin/main", "#b5483a", series(origin, "far_M_max_abs")),
        ("branch baseline", "#8657c8", series(baseline, "far_M_max_abs")),
        ("patched", "#178a64", series(fixed, "far_M_max_abs")),
    ]
    j_datasets = [
        ("origin/main", "#b5483a", series(origin, "far_J_max_abs")),
        ("branch baseline", "#8657c8", series(baseline, "far_J_max_abs")),
        ("patched", "#178a64", series(fixed, "far_J_max_abs")),
    ]
    err_datasets = [
        ("origin/main", "#b5483a", series(origin, "root_delta_error_m")),
        ("branch baseline", "#8657c8", series(baseline, "root_delta_error_m")),
        ("patched", "#178a64", series(fixed, "root_delta_error_m")),
    ]

    css = """
    :root {
      color-scheme: light;
      --ink: #18212a;
      --muted: #5e6b76;
      --line: #d8e0e7;
      --panel: #f7f9fb;
      --accent: #178a64;
      --warn: #b5483a;
    }
    body {
      margin: 0;
      font-family: Inter, "Segoe UI", Arial, sans-serif;
      color: var(--ink);
      background: #fff;
      line-height: 1.5;
    }
    main { max-width: 1080px; margin: 0 auto; padding: 42px 32px 72px; }
    h1 { font-size: 34px; line-height: 1.12; margin: 0 0 10px; letter-spacing: 0; }
    h2 { margin: 36px 0 12px; font-size: 22px; }
    h3 { margin: 24px 0 8px; font-size: 17px; }
    p { margin: 10px 0; }
    code { background: #eef3f6; padding: 1px 5px; border-radius: 4px; }
    .lede { color: var(--muted); font-size: 17px; max-width: 850px; }
    .stamp { color: var(--muted); font-size: 13px; margin-bottom: 28px; }
    .example-intro { margin: 26px 0 28px; }
    .example-intro h2 { margin-top: 0; }
    .example-video {
      display: block;
      width: 100%;
      max-width: 960px;
      aspect-ratio: 16 / 9;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #0f1720;
    }
    .callout {
      border: 1px solid var(--line);
      border-left: 5px solid var(--accent);
      background: var(--panel);
      padding: 14px 16px;
      border-radius: 8px;
      margin: 20px 0;
    }
    .grid-cards { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 18px 0 28px; }
    .card { border: 1px solid var(--line); border-radius: 8px; padding: 14px; background: #fff; }
    .metric { font-size: 24px; font-weight: 700; margin-top: 4px; }
    .label { color: var(--muted); font-size: 13px; }
    table { width: 100%; border-collapse: collapse; margin: 16px 0 22px; font-size: 14px; }
    th, td { border-bottom: 1px solid var(--line); padding: 9px 10px; text-align: left; vertical-align: top; }
    th { background: #f1f5f8; font-weight: 650; }
    ul { padding-left: 22px; }
    li { margin: 6px 0; }
    .plot { border: 1px solid var(--line); border-radius: 8px; padding: 8px; margin: 18px 0; }
    svg { width: 100%; height: auto; display: block; }
    .grid line { stroke: #e6edf2; stroke-width: 1; }
    .axis { stroke: #637381; stroke-width: 1.2; }
    .lines path { fill: none; stroke-width: 2.4; }
    .plot-title { font-size: 18px; font-weight: 700; fill: var(--ink); }
    .labels text, .axis-title, .legend { font-size: 12px; fill: var(--muted); }
    .legend-line { stroke-width: 3; }
    .good { color: var(--accent); font-weight: 650; }
    .bad { color: var(--warn); font-weight: 650; }
    """

    fixed_last = fixed["samples"][-1]
    baseline_bad = baseline.get("first_nonfinite_frame")
    origin_bad = origin.get("first_nonfinite_frame")

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Featherstone Origin-Offset Stability Report</title>
  <style>{css}</style>
</head>
<body>
<main>
  <h1>Featherstone Origin-Offset Stability Report</h1>
  <p class="lede">Investigation of the tricycle NaN triggered by world-origin-referenced Featherstone internals for floating-base articulations far from the origin.</p>
  <p class="stamp">Generated {generated_at} from CUDA runs on NVIDIA GeForce RTX 4090. Current branch baseline: <code>1cd4196dc</code>. Fresh <code>origin/main</code>: <code>a046d6131</code>.</p>

  <section class="example-intro">
    <h2>Example</h2>
    <p>The origin-offset probe drives a small floating-base tricycle near the origin while an identical copy follows the same controls 100 m away.</p>
    <video class="example-video" controls muted loop playsinline preload="metadata" src="assets/tricycle_origin_offset.mp4"></video>
  </section>

  <div class="grid-cards">
    <div class="card"><div class="label">Current branch first non-finite sample</div><div class="metric bad">{baseline_bad}</div></div>
    <div class="card"><div class="label">origin/main first non-finite sample</div><div class="metric bad">{origin_bad}</div></div>
    <div class="card"><div class="label">Patched frames completed</div><div class="metric good">{fixed["frames_completed"]}</div></div>
  </div>

  <div class="callout">
    <strong>Recommended fix:</strong> for articulations rooted by <code>FREE</code> or <code>DISTANCE</code> joints, express Featherstone-only scratch quantities in a translated solve frame whose origin is the root body COM and whose axes remain world-aligned. Public <code>State.body_qd</code>, <code>State.body_f</code>, <code>Control.joint_f</code>, and <code>State.body_parent_f</code> remain COM/world-frame quantities.
  </div>

  <h2>Reproduction</h2>
  <p>I first ran <code>test_final()</code> for <code>newton/examples/robot/example_robot_tricycle_origin_offset.py</code> much longer than the registered 20-frame example test. With <code>--far-offset 100</code> and 2000 frames, the current branch failed because <code>body_q</code> contained non-finite values. A per-frame probe found the first bad frame at 1594 in the exact example path; the reusable benchmark sampled the same failure at frame 1600.</p>
  <p>The failure is not that the public generalized mass matrix is visibly large in every branch. In this branch, changes around free-joint COM coordinates keep <code>H</code> small for longer, but the internal spatial Jacobian <code>J</code> and block spatial mass <code>M</code> still carry absolute world-origin moment arms. At frame 1575 on the branch baseline, far <code>|J|</code> is about 306 and far <code>|M|</code> is about 842,808 before becoming non-finite.</p>

  <h2>What Changed</h2>
  <ul>
    <li>Added a per-body scratch <code>body_solve_origin</code> used only by <code>SolverFeatherstone</code>.</li>
    <li>For floating-root articulations, compute the root COM once during <code>eval_rigid_id</code> and subtract it from internal anchor and COM positions.</li>
    <li>Build internal <code>joint_S_s</code>, <code>body_I_s</code>, <code>body_v_s</code>, <code>body_a_s</code>, <code>body_f_s</code>, external-force shifts, <code>J</code>, and <code>M</code> in that translated frame.</li>
    <li>Convert back at solver boundaries so public body twists and wrenches keep the current branch contract.</li>
    <li>Keep non-floating roots on the original world-origin path. I tried applying the translated frame to every articulation, but that slightly regressed a CPU descendant-free-joint contract test, so the final patch is deliberately narrower.</li>
  </ul>

  <h2>Results</h2>
  {summary_table(rows)}

  {svg_plot("Far internal spatial mass magnitude", "max |M|", plot_datasets, log_y=True)}
  {svg_plot("Far internal spatial Jacobian magnitude", "max |J|", j_datasets, log_y=True)}
  {svg_plot("Near-vs-far root displacement error", "error [m]", err_datasets)}

  <h2>Interpretation</h2>
  <p>The patch removes the numerical scale-up in the Featherstone internals for the floating tricycle. In the patched run, far <code>|J|</code> stays around 1 and far <code>|M|</code> stays around 9 through 2500 frames, even when the far root reaches {fmt(fixed_last.get("far_root_x_m"), "m")}.</p>
  <p>The long contact-rich trajectory is still not translation-invariant to the original <code>0.25 m</code> tolerance at every horizon. The root displacement error oscillates and later grows, while local body layout remains small ({fmt(fixed_last.get("local_pose_error_m"), "m")} at frame 2500). That remaining drift appears contact/trajectory sensitive rather than a renewed internal-matrix blow-up: the internal matrices remain bounded and finite.</p>

  <h2>Alternatives Considered</h2>
  <ul>
    <li><strong>Translate all articulations:</strong> numerically stable for the tricycle, but rejected because it changed fixed/revolute-root descendant-free behavior enough to fail a CPU tolerance.</li>
    <li><strong>Use articulation COM instead of root COM:</strong> likely similarly stable, but it needs an extra mass-weighted reduction/update per articulation. Root COM is cheaper, already available from the root FK, and enough to bound the floating-base moment arms.</li>
    <li><strong>Leave <code>H</code> only in local coordinates:</strong> insufficient because the bad scale enters earlier through <code>S</code>, <code>I</code>, bias forces, <code>J</code>, and <code>M</code>.</li>
  </ul>

  <h2>Verification</h2>
  <ul>
    <li><code>uv run --extra dev -m newton.tests -k test_robot.example_robot_tricycle_origin_offset</code>: pass on CPU and CUDA.</li>
    <li><code>uv run --extra dev -m newton.tests -k featherstone_free</code>: pass on CPU and CUDA.</li>
    <li><code>uv run --extra dev -m newton.tests -k test_parent_force</code>: pass, including Featherstone <code>body_parent_f</code> checks.</li>
    <li><code>uv run --extra dev -m newton.tests -k test_basic.example_basic_joints</code>: pass on CPU and CUDA.</li>
    <li>Long probe: patched working tree completed 2500 frames without non-finite state; branch baseline failed at sample frame 1600; <code>origin/main</code> failed at sample frame 1650.</li>
  </ul>

  <h2>Files</h2>
  <ul>
    <li><code>newton/_src/solvers/featherstone/kernels.py</code></li>
    <li><code>newton/_src/solvers/featherstone/solver_featherstone.py</code></li>
    <li><code>tools/featherstone_origin_offset_probe.py</code></li>
    <li><code>tools/render_featherstone_origin_report.py</code></li>
  </ul>
</main>
</body>
</html>
"""
    OUT_PATH.write_text(html_text, encoding="utf-8")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
