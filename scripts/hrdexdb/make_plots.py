# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Generate all report figures from evaluation results.

Reads ``results/<tag>/<hand>_<source>/`` summaries + per-episode npz files and
``results/tune_*.csv`` CMA-ES logs; writes PNGs into the report data dir.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).parent / "results"

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

HAND_LABEL = {"allegro_v5": "Allegro V5", "inspire_f1": "Inspire F1"}
C_SIM = "#2456a6"
C_GT = "#166534"
C_TUNED = "#9a3412"


def load_summary(tag: str, hand: str, source: str) -> dict:
    p = RESULTS / tag / f"{hand}_{source}" / "summary.json"
    if not p.exists():
        return {}
    return {
        k: v
        for k, v in json.loads(p.read_text()).items()
        if "error" not in v
        and not v.get("calib_outlier")
        and not v.get("sim_unstable")
        and not np.isnan(v.get("add_rmse", np.nan))
    }


def fig_convergence(out: Path, hands=("allegro_v5", "inspire_f1"), source="cmd"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
    for ax, hand in zip(axes, hands):
        path = RESULTS / f"tune_{hand}_{source}.csv"
        if not path.exists():
            continue
        gens = defaultdict(list)
        with open(path) as f:
            for row in csv.DictReader(f):
                gens[int(row["gen"])].append(float(row["mean_add"]))
        g = sorted(gens)
        best = np.minimum.accumulate([min(gens[i]) for i in g])
        mean = [np.mean(gens[i]) for i in g]
        lo = [np.min(gens[i]) for i in g]
        hi = [np.max(gens[i]) for i in g]
        ax.fill_between(g, lo, hi, alpha=0.18, color=C_SIM, label="population range")
        ax.plot(g, mean, color=C_SIM, lw=1.2, label="population mean")
        ax.plot(g, best, color=C_TUNED, lw=2, label="best so far")
        ax.set_title(HAND_LABEL[hand])
        ax.set_xlabel("generation")
        ax.set_ylabel("mean object ADD RMSE [m]")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("CMA-ES convergence (training episodes)", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "tuning_convergence.png", bbox_inches="tight")
    plt.close(fig)


def fig_episode_trajectory(npz_path: Path, out_png: Path, title: str):
    d = np.load(npz_path, allow_pickle=True)
    t = d["t"]
    ps, pg = d["obj_pos_sim"], d["obj_pos_gt"]
    from replay import quat_geodesic_deg

    rot_err = quat_geodesic_deg(d["obj_quat_sim"], d["obj_quat_gt"])
    pos_err = np.linalg.norm(ps - pg, axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    labels = ["x", "y", "z"]
    for i in range(3):
        axes[0].plot(t, pg[:, i], color=C_GT, lw=1.6, alpha=[0.45, 0.7, 1.0][i], label=f"GT {labels[i]}")
        axes[0].plot(t, ps[:, i], color=C_SIM, lw=1.2, ls="--", alpha=[0.45, 0.7, 1.0][i], label=f"sim {labels[i]}")
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("object position [m]")
    axes[0].legend(frameon=False, fontsize=6.5, ncol=2)
    axes[1].plot(t, pos_err * 100, color=C_SIM, lw=1.5)
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("position error [cm]")
    axes[2].plot(t, rot_err, color=C_TUNED, lw=1.5)
    axes[2].set_xlabel("time [s]")
    axes[2].set_ylabel("rotation error [deg]")
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def fig_per_object(out: Path, tag_a: str, tag_b: str | None, hand: str, source: str, metric="add_rmse"):
    sa = load_summary(tag_a, hand, source)
    sb = load_summary(tag_b, hand, source) if tag_b else {}
    per_obj_a, per_obj_b = defaultdict(list), defaultdict(list)
    for k, v in sa.items():
        per_obj_a[k.split("/")[0]].append(v[metric])
    for k, v in sb.items():
        per_obj_b[k.split("/")[0]].append(v[metric])
    objs = sorted(per_obj_a, key=lambda o: np.mean(per_obj_a[o]))
    x = np.arange(len(objs))
    fig, ax = plt.subplots(figsize=(max(7, len(objs) * 0.32), 3.8))
    w = 0.4 if sb else 0.7
    ax.bar(x - (w / 2 if sb else 0), [np.mean(per_obj_a[o]) for o in objs], w, color=C_SIM, label=tag_a)
    if sb:
        ax.bar(x + w / 2, [np.mean(per_obj_b.get(o, [np.nan])) for o in objs], w, color=C_TUNED, label=tag_b)
    ax.set_xticks(x)
    ax.set_xticklabels([o.replace("_", " ") for o in objs], rotation=75, ha="right", fontsize=7)
    ax.set_ylabel(f"{metric.replace('_', ' ')} [m]")
    ax.set_title(f"{HAND_LABEL[hand]} — per-object tracking error")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / f"per_object_{hand}_{source}.png", bbox_inches="tight")
    plt.close(fig)


def fig_pilot(out: Path, hand: str, tag="pilot"):
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    data, ticks = [], []
    for source, label in [("cmd", "commands"), ("meas", "measured")]:
        s = load_summary(tag, hand, source)
        if s:
            data.append([v["add_rmse"] for v in s.values()])
            ticks.append(f"{label}\n(n={len(s)})")
    if not data:
        plt.close(fig)
        return
    bp = ax.boxplot(data, tick_labels=ticks, showmeans=True, patch_artist=True)
    for patch, c in zip(bp["boxes"], [C_SIM, C_TUNED]):
        patch.set_facecolor(c)
        patch.set_alpha(0.35)
    ax.set_ylabel("object ADD RMSE [m]")
    ax.set_title(f"{HAND_LABEL[hand]} — PD target source pilot")
    fig.tight_layout()
    fig.savefig(out / f"pilot_{hand}.png", bbox_inches="tight")
    plt.close(fig)


def fig_default_vs_tuned(out: Path, hand: str, source: str, train_objects: list[str]):
    sd = load_summary("default", hand, source)
    st = load_summary("tuned", hand, source)
    if not sd or not st:
        return
    common = sorted(set(sd) & set(st))
    train = [k for k in common if k.split("/")[0] in train_objects]
    hold = [k for k in common if k.split("/")[0] not in train_objects]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    groups = [
        ("default\ntrain", [sd[k]["add_rmse"] for k in train], C_SIM),
        ("tuned\ntrain", [st[k]["add_rmse"] for k in train], C_TUNED),
        ("default\nholdout", [sd[k]["add_rmse"] for k in hold], C_SIM),
        ("tuned\nholdout", [st[k]["add_rmse"] for k in hold], C_TUNED),
    ]
    bp = ax.boxplot(
        [g[1] for g in groups],
        tick_labels=[f"{g[0]}\n(n={len(g[1])})" for g in groups],
        showmeans=True,
        patch_artist=True,
    )
    for patch, g in zip(bp["boxes"], groups):
        patch.set_facecolor(g[2])
        patch.set_alpha(0.35)
    ax.set_ylabel("object ADD RMSE [m]")
    ax.set_title(f"{HAND_LABEL[hand]} — default vs CMA-ES-tuned ({source})")
    fig.tight_layout()
    fig.savefig(out / f"default_vs_tuned_{hand}_{source}.png", bbox_inches="tight")
    plt.close(fig)


def fig_scatter_lift(out: Path, tags_hands: list[tuple[str, str, str]]):
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    markers = {"allegro_v5": "o", "inspire_f1": "^"}
    for tag, hand, source in tags_hands:
        s = load_summary(tag, hand, source)
        if not s:
            continue
        zerr = [v["z_peak_err"] for v in s.values()]
        add = [v["add_rmse"] for v in s.values()]
        ax.scatter(zerr, add, s=22, alpha=0.65, marker=markers[hand], label=f"{HAND_LABEL[hand]} ({tag})")
    ax.axvline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("peak-height error sim − GT [m]")
    ax.set_ylabel("object ADD RMSE [m]")
    ax.set_title("Lift fidelity vs tracking error")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "scatter_lift.png", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(Path(__file__).parent / "figures"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from tune import TRAIN_EPISODES

    fig_convergence(out)
    for hand in ("allegro_v5", "inspire_f1"):
        fig_pilot(out, hand)
        train_objects = [o for o, _ in TRAIN_EPISODES[hand]]
        for source in ("cmd",):
            for tag in ("default", "tuned"):
                if (RESULTS / tag / f"{hand}_{source}").exists():
                    fig_per_object(out, "default", "tuned" if (RESULTS / "tuned").exists() else None, hand, source)
            fig_default_vs_tuned(out, hand, source, train_objects)
    fig_scatter_lift(out, [("tuned", h, "cmd") for h in ("allegro_v5", "inspire_f1")])

    # Representative per-episode trajectory plots (best/median/worst by add_rmse)
    for hand in ("allegro_v5", "inspire_f1"):
        s = load_summary("tuned", hand, "cmd") or load_summary("default", hand, "cmd")
        tag = "tuned" if load_summary("tuned", hand, "cmd") else "default"
        if not s:
            continue
        ranked = sorted(s.items(), key=lambda kv: kv[1]["add_rmse"])
        picks = {"best": ranked[0], "median": ranked[len(ranked) // 2], "worst": ranked[-1]}
        for label, (k, v) in picks.items():
            obj, scene = k.split("/")
            npz = RESULTS / tag / f"{hand}_cmd" / f"{obj}_{scene}.npz"
            if npz.exists():
                fig_episode_trajectory(
                    npz,
                    out / f"traj_{hand}_{label}.png",
                    f"{HAND_LABEL[hand]} {obj}/{scene} ({label}, ADD RMSE {v['add_rmse'] * 100:.1f} cm)",
                )
    print(f"figures -> {out}")


if __name__ == "__main__":
    main()
