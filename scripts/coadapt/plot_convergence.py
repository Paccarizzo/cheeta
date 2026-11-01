"""Plots of a convergence check: how long to train before the design ranking settles.

A convergence check is recheck_designs.py with a longer budget (--iterations,
--random-episode-start, --out-name). This script compares its jobs in
<run-dir>/<name>/ with the same designs and seeds at the run's own budget
(the evolution's candidate for the run's seed, <run-dir>/<baseline>/ for the
others). Writes to <run-dir>/<name>/plots/:
  * convergence_curves.png  - training return of every design and seed, both budgets
  * convergence_ranking.png - seed-mean training return of every design over training
  * convergence_scores.png  - evaluation scores per design and seed, both budgets
  * convergence.csv         - one row per design, seed and budget

Needs matplotlib and numpy only (no Isaac Sim):
    python scripts/coadapt/plot_convergence.py --run-dir logs/coadapt/<run> --name converge_800
"""

import argparse
import csv
import os
import statistics

import numpy as np

from plot_evolution import (AQUA, BLUE, DOT, GRID, INK, INK_2, ORANGE, SURFACE, _fmt, _load, legend, plt, style,
                            titles, training_return)

parser = argparse.ArgumentParser(description="Plot a convergence check of an evolve.py run.")
parser.add_argument("--run-dir", type=str, required=True)
parser.add_argument("--name", type=str, default="converge_800", help="recheck_designs.py --out-name of the check")
parser.add_argument("--baseline", type=str, default="recheck", help="recheck_designs.py output at the run's budget")
args = parser.parse_args()

DESIGN_COLORS = [BLUE, ORANGE, AQUA]  # top designs in rank order; the unscaled cheetah is the ink reference


def job(path: str) -> dict | None:
    result = _load(os.path.join(path, "result.json"))
    return {"result": result, "metrics": _load(os.path.join(path, "metrics.json"))} if result else None


def short_job(run_dir: str, label: str, seed: int, run_seed: int) -> dict | None:
    """The design and seed at the run's budget: the evolution's candidate for the run's seed, else the recheck."""
    if label != "identity" and seed == run_seed:
        gen, cand = label.split("_g")[1].split("c")
        return job(os.path.join(run_dir, f"gen_{gen}", f"cand_{cand}"))
    return job(os.path.join(run_dir, args.baseline, f"{label}_s{seed}"))


def display(label: str) -> str:
    if label == "identity":
        return "unscaled cheetah"
    top, rest = label.split("_g")
    gen, cand = rest.split("c")
    return f"{top.replace('top', 'top ')} (gen {int(gen)}, cand {int(cand)})"


def color(label: str, labels: list[str]) -> str:
    return INK_2 if label == "identity" else DESIGN_COLORS[labels.index(label) % len(DESIGN_COLORS)]


def at(curve, iteration: int) -> float:
    """Training return over the episode window ending at `iteration` (nan if the run is shorter)."""
    it, val = curve
    i = int(np.searchsorted(it, iteration))
    return float(val[i]) if i < len(it) and it[i] == iteration else float("nan")


def seed_mean(curves: list) -> tuple[np.ndarray, np.ndarray] | None:
    curves = [c for c in curves if c is not None]
    if not curves:
        return None
    n = min(len(c[0]) for c in curves)
    return curves[0][0][:n], np.mean([c[1][:n] for c in curves], axis=0)


def checkpoints(iterations: int) -> list[int]:
    step = 100 if iterations <= 1000 else 200
    return sorted({200, *range(step, iterations + 1, step)})


def plot_curves(data, labels, short_it, long_it, long_desc, out):
    fig, axes = plt.subplots(1, len(labels), figsize=(2.9 * len(labels) + 0.6, 4.1), sharey=True, facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.71, bottom=0.14, wspace=0.08)
    for ax, label in zip(axes, labels):
        for d in data[label]:
            for budget, c, w, z in (("short", DOT, 1.2, 2), ("long", color(label, labels), 1.4, 3)):
                curve = d[f"{budget}_curve"]
                if curve is not None:
                    ax.plot(*curve, color=c, linewidth=w, zorder=z, solid_capstyle="round")
        ax.axvline(short_it, color=GRID, linewidth=1, zorder=1)
        style(ax, xlabel="PPO iteration")
        ax.set_xlim(0, long_it)
        ax.set_title(display(label), fontsize=9.5, color=INK, loc="left")
    axes[0].set_ylabel("training return per episode", color=INK_2, fontsize=9)
    axes[0].plot([], [], color=DOT, linewidth=1.2, label=f"run's budget: {short_it} it, synchronised starts")
    axes[0].plot([], [], color=INK_2, linewidth=1.4, label=long_desc + " (colour = design)")
    handles, names = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, names, frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0.004, 0.86),
                     ncol=2, handlelength=1.6)
    for text in leg.get_texts():
        text.set_color(INK_2)
    titles(fig, "Training return per design and seed",
           "One line per seed; stochastic training policy, reward averaged over the last episode. "
           f"Vertical line: the evolution's budget ({short_it} it).")
    fig.savefig(os.path.join(out, "convergence_curves.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_ranking(means, labels, short_it, long_it, long_desc, out):
    fig, ax = plt.subplots(figsize=(8.6, 5.0), facecolor=SURFACE)
    fig.subplots_adjust(left=0.09, right=0.78, top=0.77, bottom=0.11)
    ends = []
    for label in labels:
        if means[label] is None:
            continue
        it, val = means[label]
        ax.plot(it, val, color=color(label, labels), linewidth=2, zorder=3, label=display(label),
                solid_capstyle="round")
        ends.append([float(val[-1]), label])
    ax.axvline(short_it, color=GRID, linewidth=1, zorder=1)
    ax.annotate(f"evolution budget\n({short_it} it)", (short_it, 1.0), xycoords=("data", "axes fraction"),
                xytext=(4, -2), textcoords="offset points", va="top", fontsize=8, color=INK_2)
    # direct labels at the line ends, pushed apart so they don't overlap
    lo, hi = ax.get_ylim()
    gap = 0.045 * (hi - lo)
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    for y, label in ends:
        ax.annotate(f"{display(label)}  {float(means[label][1][-1]):.0f}", (1.0, y), xycoords=("axes fraction", "data"),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK_2)
    style(ax, ylabel="training return per episode (mean over seeds)", xlabel="PPO iteration")
    ax.set_xlim(0, long_it)
    legend(ax, loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2, columnspacing=1.4)
    titles(fig, "Does the design ranking settle with longer training?", long_desc + "; mean over the seeds.")
    fig.savefig(os.path.join(out, "convergence_ranking.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_scores(data, labels, short_it, long_desc, out):
    height = 0.75 * len(labels) + 1.9
    fig, ax = plt.subplots(figsize=(8.0, height), facecolor=SURFACE)
    fig.subplots_adjust(left=0.24, right=0.97, top=1 - 1.05 / height, bottom=0.55 / height + 0.06)
    for y, label in enumerate(reversed(labels)):
        for budget, dy, c in (("short", -0.17, DOT), ("long", 0.17, color(label, labels))):
            scores = [d[f"{budget}_score"] for d in data[label] if d[f"{budget}_score"] is not None]
            if not scores:
                continue
            ax.plot([min(scores), max(scores)], [y + dy, y + dy], color=GRID, linewidth=2, zorder=1)
            ax.scatter(scores, [y + dy] * len(scores), s=46, color=c, edgecolor=SURFACE, linewidth=2, zorder=3)
            mean = statistics.mean(scores)
            ax.plot([mean, mean], [y + dy - 0.12, y + dy + 0.12], color=INK, linewidth=2, zorder=4)
            # label away from the other budget's row
            ax.annotate(f"{mean:.0f}", (mean, y + dy), xytext=(0, 9 if dy > 0 else -9), textcoords="offset points",
                        ha="center", va="bottom" if dy > 0 else "top", fontsize=7.5, color=INK_2)
    ax.scatter([], [], s=46, color=DOT, label=f"{short_it} it, synchronised starts")
    ax.scatter([], [], s=46, color=INK_2, label=long_desc + " (colour = design)")
    ax.plot([], [], color=INK, linewidth=2, label="mean over seeds")
    style(ax, xlabel="score (episode return)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([display(lab) for lab in reversed(labels)])
    ax.set_ylim(-0.6, len(labels) - 0.3)
    handles, names = ax.get_legend_handles_labels()
    leg = fig.legend(handles, names, frameon=False, fontsize=8.5, loc="upper left",
                     bbox_to_anchor=(0.004, 1 - 0.62 / height), ncol=3, handlelength=1.6)
    for text in leg.get_texts():
        text.set_color(INK_2)
    titles(fig, "Evaluation score per design and seed, both budgets",
           "Deterministic policy, one episode in each of the 1024 envs; one dot per seed.")
    fig.savefig(os.path.join(out, "convergence_scores.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    run_dir = os.path.abspath(args.run_dir)
    run_cfg = _load(os.path.join(run_dir, "config.json"))
    check_cfg = _load(os.path.join(run_dir, args.name, "config.json"))
    run_seed, short_it, long_it = run_cfg["seed"], run_cfg["iterations"], check_cfg["iterations"]
    seeds = list(dict.fromkeys([run_seed, *check_cfg["seeds"]]))
    labels = [t["label"] for t in check_cfg["targets"]] + (["identity"] if check_cfg["identity"] else [])
    long_desc = f"{long_it} it" + (", random episode starts" if check_cfg.get("random_episode_start") else "")
    out = os.path.join(run_dir, args.name, "plots")
    os.makedirs(out, exist_ok=True)

    data = {}
    for label in labels:
        data[label] = []
        for seed in seeds:
            d = {"seed": seed}
            for budget, j in (("short", short_job(run_dir, label, seed, run_seed)),
                              ("long", job(os.path.join(run_dir, args.name, f"{label}_s{seed}")))):
                d[f"{budget}_score"] = j["result"]["score"] if j else None
                d[f"{budget}_distance"] = j["result"]["distance_x_m"] if j else None
                d[f"{budget}_curve"] = training_return(j) if j else None
            data[label].append(d)
    means = {label: seed_mean([d["long_curve"] for d in data[label]]) for label in labels}

    marks = checkpoints(long_it)
    with open(os.path.join(out, "convergence.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "seed", "budget", "score", "distance_x_m", *[f"train_return_it{m}" for m in marks],
                    "train_return_max"])
        for label in labels:
            for d in data[label]:
                for budget, iters in (("short", short_it), ("long", long_it)):
                    if d[f"{budget}_score"] is None:
                        continue
                    curve = d[f"{budget}_curve"]
                    w.writerow([label, d["seed"], f"{iters} it", _fmt(d[f"{budget}_score"]),
                                _fmt(d[f"{budget}_distance"]),
                                *[_fmt(at(curve, m)) if curve else "" for m in marks],
                                _fmt(float(curve[1].max())) if curve else ""])

    plot_curves(data, labels, short_it, long_it, long_desc, out)
    plot_ranking(means, labels, short_it, long_it, long_desc, out)
    plot_scores(data, labels, short_it, long_desc, out)

    # console summary: scores per budget, seed-mean training return at the checkpoints, ranking per checkpoint
    for label in labels:
        for budget, iters in (("short", short_it), ("long", long_it)):
            scores = [d[f"{budget}_score"] for d in data[label] if d[f"{budget}_score"] is not None]
            if scores:
                sd = statistics.stdev(scores) if len(scores) > 1 else float("nan")
                print(f"[conv] {label:14s} {iters:4d} it: score mean {statistics.mean(scores):6.1f} "
                      f"sd {sd:6.1f} over {len(scores)} seeds  {[round(s) for s in scores]}")
    for m in marks:
        row = {label: at(means[label], m) for label in labels if means[label] is not None}
        ranked = sorted((v, k) for k, v in row.items() if v == v)[::-1]
        if ranked:
            print(f"[conv] it {m:4d}: " + "  ".join(f"{k} {v:.0f}" for v, k in ranked))
    print(f"[conv] wrote {out}")


if __name__ == "__main__":
    main()
