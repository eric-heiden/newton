# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Assemble the HRDexDB report (HTML + figures + videos) for gh-pages.

Reads evaluation summaries, tuning logs, figures from ``make_plots.py`` and
rendered overlay videos; writes a self-contained report directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
VIDEOS = HERE / "videos"

HAND_LABEL = {"allegro_v5": "Allegro V5", "inspire_f1": "Inspire F1"}


def load_summary(tag, hand, source):
    p = RESULTS / tag / f"{hand}_{source}" / "summary.json"
    if not p.exists():
        return {}
    return {k: v for k, v in json.loads(p.read_text()).items() if "error" not in v}


def agg(summary, key, exclude_outliers=True):
    vals = [
        v[key]
        for v in summary.values()
        if not (exclude_outliers and (v.get("calib_outlier") or v.get("sim_unstable")))
    ]
    vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return {}
    a = np.array(vals, dtype=float)
    return {"mean": a.mean(), "median": np.median(a), "p90": np.percentile(a, 90), "n": len(a)}


def fmt_cm(v):
    return f"{v * 100:.1f}"


def stats_row(tag, hand, source, train_objects):
    s = load_summary(tag, hand, source)
    if not s:
        return None
    valid = {
        k: v
        for k, v in s.items()
        if not v.get("calib_outlier") and not v.get("sim_unstable") and not np.isnan(v.get("add_rmse", np.nan))
    }
    hold = {k: v for k, v in valid.items() if k.split("/")[0] not in train_objects}
    add = agg(valid, "add_rmse")
    pos = agg(valid, "pos_rmse")
    rot = agg(valid, "rot_rmse_deg")
    lift = np.mean([v["lift_match"] for v in valid.values()]) if valid else 0
    add_h = agg(hold, "add_rmse")
    return {
        "tag": tag,
        "hand": hand,
        "n": add.get("n", 0),
        "add_mean": add.get("mean", np.nan),
        "add_median": add.get("median", np.nan),
        "add_p90": add.get("p90", np.nan),
        "add_holdout_median": add_h.get("median", np.nan),
        "pos_median": pos.get("median", np.nan),
        "rot_median": rot.get("median", np.nan),
        "lift_match": lift,
    }


def tuned_params_table(hand, source="cmd"):
    p = RESULTS / f"tuned_params_{hand}_{source}.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def tuning_stats(hand, source="cmd"):
    path = RESULTS / f"tune_{hand}_{source}.csv"
    if not path.exists():
        return {}
    gens = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            gens[int(row["gen"])].append(float(row["mean_add"]))
    if not gens:
        return {}
    g = sorted(gens)
    first = min(gens[g[0]])
    best = min(min(gens[i]) for i in g)
    return {"generations": len(g), "evals": sum(len(gens[i]) for i in g), "first_best": first, "best": best}


VIDEO_SECTIONS = "videos.json"  # written by render pipeline: list of dicts


def video_cards(report_dir: Path) -> str:
    meta_path = HERE / "videos" / "videos.json"
    if not meta_path.exists():
        return "<p>No videos rendered.</p>"
    cards = []
    for m in json.loads(meta_path.read_text()):
        src = VIDEOS / m["file"]
        if not src.exists():
            continue
        dst = report_dir / "videos" / m["file"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        cards.append(
            f"""<figure class="video-card">
  <video controls muted loop playsinline preload="metadata" src="videos/{m["file"]}"></video>
  <figcaption><strong>{m["title"]}</strong> — {m["caption"]}</figcaption>
</figure>"""
        )
    return "\n".join(cards)


def copy_figures(report_dir: Path) -> list[str]:
    out = report_dir / "data"
    out.mkdir(parents=True, exist_ok=True)
    copied = []
    for png in sorted(FIGURES.glob("*.png")):
        shutil.copy2(png, out / png.name)
        copied.append(png.name)
    return copied


def img(name, caption):
    return f"""<figure>
  <img src="data/{name}" alt="{caption}" loading="lazy">
  <figcaption>{caption}</figcaption>
</figure>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(Path.home() / "repos/academic-website-reports/HRDexDB"))
    args = parser.parse_args()
    report_dir = Path(args.out)
    report_dir.mkdir(parents=True, exist_ok=True)

    from tune import PARAM_SPACE, TRAIN_EPISODES

    train_objects = {h: [o for o, _ in TRAIN_EPISODES[h]] for h in HAND_LABEL}

    figures = copy_figures(report_dir)
    videos_html = video_cards(report_dir)

    # ---------------- stats ----------------
    rows = []
    for hand in HAND_LABEL:
        for tag in ("default", "tuned"):
            r = stats_row(tag, hand, "cmd", train_objects[hand])
            if r:
                rows.append(r)

    def results_table():
        head = (
            "<tr><th>Hand</th><th>Params</th><th>Episodes</th><th>ADD RMSE median [cm]</th>"
            "<th>ADD RMSE mean [cm]</th><th>p90 [cm]</th><th>holdout median [cm]</th>"
            "<th>pos RMSE median [cm]</th><th>rot RMSE median [°]</th><th>lift match</th></tr>"
        )
        body = ""
        for r in rows:
            body += (
                f"<tr><td>{HAND_LABEL[r['hand']]}</td><td>{r['tag']}</td><td>{r['n']}</td>"
                f"<td><strong>{fmt_cm(r['add_median'])}</strong></td><td>{fmt_cm(r['add_mean'])}</td>"
                f"<td>{fmt_cm(r['add_p90'])}</td><td>{fmt_cm(r['add_holdout_median'])}</td>"
                f"<td>{fmt_cm(r['pos_median'])}</td><td>{r['rot_median']:.1f}</td>"
                f"<td>{r['lift_match']:.0%}</td></tr>"
            )
        return f"<table>{head}{body}</table>"

    def pilot_table():
        out = "<tr><th>Hand</th><th>Target source</th><th>Episodes</th><th>ADD RMSE median [cm]</th><th>lift match</th></tr>"
        for hand in HAND_LABEL:
            for source, label in [("cmd", "recorded commands"), ("meas", "measured joints")]:
                s = load_summary("pilot", hand, source)
                if not s:
                    continue
                a = agg(s, "add_rmse")
                lift = np.mean([v["lift_match"] for v in s.values()])
                out += (
                    f"<tr><td>{HAND_LABEL[hand]}</td><td>{label}</td><td>{a['n']}</td>"
                    f"<td>{fmt_cm(a['median'])}</td><td>{lift:.0%}</td></tr>"
                )
        return f"<table>{out}</table>"

    def params_table():
        head = "<tr><th>Parameter</th><th>Search range</th>"
        body_rows = {n: f"<tr><td><code>{n}</code></td><td>[{lo:g}, {hi:g}]</td>" for n, lo, hi in PARAM_SPACE}
        for hand in HAND_LABEL:
            tp = tuned_params_table(hand)
            head += f"<th>{HAND_LABEL[hand]} (tuned)</th>"
            for n, _, _ in PARAM_SPACE:
                body_rows[n] += f"<td>{tp.get(n, float('nan')):.4g}</td>" if tp else "<td>—</td>"
        head += "</tr>"
        body = "".join(v + "</tr>" for v in body_rows.values())
        return f"<table>{head}{body}</table>"

    tuning_notes = []
    for hand in HAND_LABEL:
        ts = tuning_stats(hand)
        if ts:
            tuning_notes.append(
                f"{HAND_LABEL[hand]}: {ts['generations']} generations, {ts['evals']} rollout sets, "
                f"training ADD RMSE {fmt_cm(ts['first_best'])} → {fmt_cm(ts['best'])} cm"
            )

    # Hero metrics from tuned (fallback default) runs.
    tuned_rows = [r for r in rows if r["tag"] == "tuned"] or rows
    n_eps = sum(r["n"] for r in tuned_rows)
    objects = set()
    for hand in HAND_LABEL:
        for tag in ("tuned", "default"):
            for k in load_summary(tag, hand, "cmd"):
                objects.add(k.split("/")[0])
    med = np.nanmedian([r["add_median"] for r in tuned_rows]) if tuned_rows else float("nan")
    lift = np.nanmean([r["lift_match"] for r in tuned_rows]) if tuned_rows else float("nan")
    hero = f"""<div class="metric-grid">
    <div class="metric"><span>Episodes replayed</span><strong>{n_eps}</strong><small>{len(objects)} objects, 2 hands, fully dynamic</small></div>
    <div class="metric"><span>Median object ADD RMSE</span><strong>{fmt_cm(med)} cm</strong><small>tuned params, all valid episodes</small></div>
    <div class="metric"><span>Lift outcome match</span><strong>{lift:.0%}</strong><small>sim lifts iff the real robot lifted</small></div>
    <div class="metric"><span>Control interface</span><strong>joint_target_q</strong><small>PD targets only — object 100% passive</small></div>
  </div>"""

    # Revalidation table (winner's-curse check)
    reval_path = RESULTS / "revalidation.json"
    reval_html = "<p>(revalidation pending)</p>"
    n_evals = "160"
    if reval_path.exists():
        reval = json.loads(reval_path.read_text())
        rt = (
            "<tr><th>Hand</th><th>CMA-ES reported best [cm]</th>"
            "<th>Tuned, revalidated [cm]</th><th>Defaults [cm]</th></tr>"
        )
        for hand, v in reval.items():
            rt += (
                f"<tr><td>{HAND_LABEL[hand]}</td><td>{fmt_cm(v['cma_reported_best'])}</td>"
                f"<td>{fmt_cm(v['tuned_train_mean'])}</td><td>{fmt_cm(v['default_train_mean'])}</td></tr>"
            )
        reval_html = f"<table>{rt}</table>"

    # Repeatability sentence
    repeat_path = RESULTS / "repeatability.json"
    repeat_txt = "see repeatability study"
    if repeat_path.exists():
        rep = json.loads(repeat_path.read_text())
        spreads, flip = [], []
        for v in rep.values():
            a = np.array([x for x in v["add_rmse"] if not np.isnan(x)])
            if len(a):
                spreads.append(a.std())
            flip.append(0 < np.mean(v["sim_lifted"]) < 1)
        repeat_txt = (
            f"{len(rep)} episodes × 8 runs: within-instance ADD std ≤ {max(spreads) * 100:.2f} cm, "
            f"{sum(flip)} episode(s) with non-deterministic lift outcome"
        )

    html = (HERE / "report_template.html").read_text()
    html = html.replace("{{REVALIDATION}}", reval_html)
    html = html.replace("{{N_EVALS}}", n_evals)
    html = html.replace("{{REPEATABILITY}}", repeat_txt)
    html = html.replace("{{HERO_METRICS}}", hero)
    html = html.replace("{{N_EPISODES}}", str(n_eps))
    html = html.replace("{{N_OBJECTS}}", str(len(objects)))
    html = html.replace("{{POPSIZE}}", "8")
    html = html.replace("{{RESULTS_TABLE}}", results_table())
    html = html.replace("{{PILOT_TABLE}}", pilot_table())
    html = html.replace("{{PARAMS_TABLE}}", params_table())
    html = html.replace("{{TUNING_NOTES}}", " · ".join(tuning_notes) if tuning_notes else "tuning in progress")
    html = html.replace("{{VIDEOS}}", videos_html)
    (report_dir / "index.html").write_text(html)
    print(f"report -> {report_dir} ({len(figures)} figures)")


if __name__ == "__main__":
    main()
