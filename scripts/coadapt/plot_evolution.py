"""Plots and tables of an evolve.py run: how the designs perform over generations.

Reads <run-dir>/history.csv, the per-candidate result.json / metrics.json and,
if present, <run-dir>/recheck/recheck.csv (from recheck_designs.py). Writes to
<run-dir>/plots/:
  * score_by_generation.png   - every new candidate, best design, mean of the
                                top --parents, median of the new candidates
  * designs_by_generation.png - the six scale factors of every candidate, best design highlighted
  * learning_curves.png       - training reward per PPO iteration, one panel per generation
  * seed_check.png            - recheck scores per design and seed (only with recheck.csv)
  * summary_by_generation.csv, candidates.csv

Needs matplotlib and numpy only (no Isaac Sim):
    python scripts/coadapt/plot_evolution.py --run-dir logs/coadapt/<run>
"""

import argparse
import csv
import json
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DESIGN_NAMES = ["bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot"]
DESIGN_BOUNDS = (0.8, 2.0)

# reference palette (light mode): chart chrome + the first three categorical slots
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
DOT = "#b9b8b1"  # de-emphasised candidates
_fonts = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = next((f for f in ("Segoe UI", "Helvetica", "Arial") if f in _fonts), "DejaVu Sans")

parser = argparse.ArgumentParser(description="Plot an evolve.py run.")
parser.add_argument("--run-dir", type=str, required=True)
parser.add_argument("--parents", type=int, default=None, help="top designs that parent the next generation "
                    "(default: --elites of the run)")


def style(ax, ylabel: str | None = None, xlabel: str | None = None) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(1)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=9, length=0)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=9)


def titles(fig, title: str, subtitle: str | None = None) -> None:
    height = fig.get_figheight()  # place by inches so short figures don't crowd the title
    fig.text(0.012, 1 - 0.14 / height, title, ha="left", va="top", fontsize=13, fontweight="semibold", color=INK)
    if subtitle:
        fig.text(0.012, 1 - 0.44 / height, subtitle, ha="left", va="top", fontsize=9, color=INK_2)


def legend(ax, **kwargs) -> None:
    leg = ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, handlelength=1.6, **kwargs)
    for text in leg.get_texts():
        text.set_color(INK_2)


def load_rows(run_dir: str) -> list[dict]:
    with open(os.path.join(run_dir, "history.csv")) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["generation"] = int(r["generation"])
        r["trained"] = r["trained"] == "1"
        r["score"] = float(r["score"])
        r["distance_x_m"] = float(r["distance_x_m"]) if r["distance_x_m"] else float("nan")
        r["design"] = [float(r[n]) for n in DESIGN_NAMES]
        r["new"] = r["role"] != "elite"  # trained in this generation (elites carry an earlier score)
        job = os.path.join(run_dir, f"gen_{r['generation']:02d}", r["id"])
        r["result"] = _load(os.path.join(job, "result.json")) if r["new"] else None
        r["metrics"] = _load(os.path.join(job, "metrics.json")) if r["new"] else None
    return rows


def _load(path: str):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def summarise(rows: list[dict], n_parents: int) -> list[dict]:
    by_id = {(r["generation"], r["id"]): r for r in rows}
    summary = []
    for gen in sorted({r["generation"] for r in rows}):
        pop = sorted((r for r in rows if r["generation"] == gen and r["trained"]), key=lambda r: -r["score"])
        new = [r for r in pop if r["new"]]
        best = pop[0]
        # children that beat the parent they were mutated from
        beat = [r["score"] > by_id[(gen - 1, r["parent"])]["score"] for r in new
                if r["parent"] and (gen - 1, r["parent"]) in by_id]
        summary.append({
            "generation": gen,
            "best_score": best["score"],
            "best_distance_x_m": best["distance_x_m"],
            "best_id": f"gen_{gen:02d}/{best['id']}" if best["new"] else f"gen_{gen:02d}/{best['id']} (kept)",
            f"top{n_parents}_mean": statistics.mean(r["score"] for r in pop[:n_parents]),
            "new_n": len(new),
            "new_median": statistics.median(r["score"] for r in new) if new else float("nan"),
            "new_mean": statistics.mean(r["score"] for r in new) if new else float("nan"),
            "new_max": max((r["score"] for r in new), default=float("nan")),
            "new_min": min((r["score"] for r in new), default=float("nan")),
            "children_beating_parent": f"{sum(beat)}/{len(beat)}" if beat else "",
            "new_in_next_parents": "",
            **{f"best_{n}": x for n, x in zip(DESIGN_NAMES, best["design"])},
        })
    # how many of a generation's new candidates made it into the parent set
    for s, nxt in zip(summary, summary[1:]):
        pop = sorted((r for r in rows if r["generation"] == s["generation"] and r["trained"]), key=lambda r: -r["score"])
        s["new_in_next_parents"] = f"{sum(r['new'] for r in pop[:n_parents])}/{n_parents}"
    return summary


def write_tables(rows: list[dict], summary: list[dict], out: str) -> None:
    with open(os.path.join(out, "summary_by_generation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        for s in summary:
            w.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in s.items()})
    fields = ["generation", "id", "role", "parent", "score", "score_std", "distance_x_m", "train_return_last_episode",
              "train_return_max", "policy_std_last", "train_time_s", *DESIGN_NAMES]
    with open(os.path.join(out, "candidates.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in rows:
            if not r["new"] or not r["trained"]:
                continue
            res, tm = r["result"] or {}, (r["result"] or {}).get("train_metrics") or {}
            curve = training_return(r)
            w.writerow([r["generation"], r["id"], r["role"], r["parent"], f"{r['score']:.4f}",
                        _fmt(res.get("score_std")), _fmt(r["distance_x_m"]),
                        _fmt(float(curve[1][-1])) if curve else "", _fmt(float(curve[1].max())) if curve else "",
                        _fmt(tm.get("policy_std_last")), _fmt(res.get("train_time_s"), 1),
                        *[f"{x:.4f}" for x in r["design"]]])


def _fmt(v, digits: int = 4) -> str:
    return "" if v is None or v != v else f"{v:.{digits}f}"


def jitter(n: int, width: float = 0.16) -> np.ndarray:
    return np.linspace(-width, width, n) if n > 1 else np.zeros(n)


def plot_scores(rows, summary, n_parents, baseline, out):
    gens = [s["generation"] for s in summary]
    fig, ax = plt.subplots(figsize=(8.6, 5.2), facecolor=SURFACE)
    fig.subplots_adjust(left=0.09, right=0.85, top=0.79, bottom=0.11)
    for gen in gens:
        new = sorted((r["score"] for r in rows if r["generation"] == gen and r["new"] and r["trained"]))
        ax.scatter(gen + jitter(len(new)), new, s=26, color=DOT, edgecolor=SURFACE, linewidth=1, zorder=2,
                   label="new candidate" if gen == gens[0] else None)
    series = [("best design", "best_score", BLUE), (f"mean of top {n_parents} (parents)", f"top{n_parents}_mean", ORANGE),
              ("median of new candidates", "new_median", AQUA)]
    for label, key, color in series:
        ys = [s[key] for s in summary]
        ax.plot(gens, ys, color=color, linewidth=2, solid_capstyle="round", solid_joinstyle="round", zorder=3,
                label=label)
        ax.scatter(gens, ys, s=40, color=color, edgecolor=SURFACE, linewidth=2, zorder=4)
    best = summary[-1]["best_score"]
    ax.annotate(f"best {best:.0f}", (gens[-1], best), xytext=(0, 9), textcoords="offset points", ha="center",
                va="bottom", fontsize=9, color=INK)
    if baseline:
        # the unscaled cheetah over several seeds: mean line + seed range band
        mean = statistics.mean(baseline)
        ax.axhspan(min(baseline), max(baseline), color=INK_2, alpha=0.07, linewidth=0, zorder=0)
        ax.axhline(mean, color=INK_2, linewidth=1, zorder=1)
        ax.annotate(f"unscaled cheetah\nmean {mean:.0f} ({len(baseline)} seeds)\nband: {min(baseline):.0f}–"
                    f"{max(baseline):.0f}", (1.0, mean), xycoords=("axes fraction", "data"), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=INK_2, linespacing=1.3)
    style(ax, ylabel="score (episode return)", xlabel="generation")
    ax.set_xticks(gens)
    ax.set_xlim(gens[0] - 0.4, gens[-1] + 0.4)
    legend(ax, loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=4, columnspacing=1.4)
    titles(fig, "Design score per generation",
           "Every candidate trained once from scratch (seed 42, same budget); parents carry their score. "
           "Score ≈ 6 × forward distance in m over a 16.6 s episode.")
    fig.savefig(os.path.join(out, "score_by_generation.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_designs(rows, summary, out):
    gens = [s["generation"] for s in summary]
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.6), sharex=True, sharey=True, facecolor=SURFACE)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.10, hspace=0.32, wspace=0.12)
    for k, (ax, name) in enumerate(zip(axes.flat, DESIGN_NAMES)):
        for gen in gens:
            vals = [r["design"][k] for r in rows if r["generation"] == gen and r["new"]]
            ax.scatter(gen + jitter(len(vals)), vals, s=16, color=DOT, edgecolor=SURFACE, linewidth=0.8, zorder=2,
                       label="new candidate" if gen == gens[0] else None)
        best = [s[f"best_{name}"] for s in summary]
        ax.plot(gens, best, color=BLUE, linewidth=2, zorder=3, label="best design")
        ax.scatter(gens, best, s=30, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=4)
        style(ax)
        ax.set_title(name, fontsize=9.5, color=INK, loc="left")
        ax.set_ylim(DESIGN_BOUNDS[0] - 0.08, DESIGN_BOUNDS[1] + 0.08)
        ax.set_xticks(gens)
    for ax in axes[1]:
        ax.set_xlabel("generation", color=INK_2, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("scale factor", color=INK_2, fontsize=9)
    legend(axes[0, 2], loc="lower right", bbox_to_anchor=(1.0, 1.12), ncol=2)
    titles(fig, "Leg segment scale factors per generation",
           f"Design box [{DESIGN_BOUNDS[0]}, {DESIGN_BOUNDS[1]}]; 1.0 = original segment length.")
    fig.savefig(os.path.join(out, "designs_by_generation.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)


def training_return(r: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """Training reward per step averaged over the last episode, scaled to an episode return.

    Without --random-episode-start all envs start together and end only by timeout, so they
    reset in sync and the raw per-iteration reward is a sawtooth with the episode's period;
    a rolling mean over one episode removes it and puts the curve in the units of the score."""
    pts = r["metrics"]["series"].get("reward_step", []) if r["metrics"] else []
    if not pts:
        return None
    episode_steps = (r["result"] or {}).get("eval_steps", 998) + 2
    window = max(1, round(episode_steps / r["metrics"]["rollouts"]))
    it, val = np.array(pts).T
    if len(val) < window:
        return None
    rolling = np.convolve(val, np.ones(window) / window, mode="valid") * episode_steps
    return it[window - 1:], rolling


def plot_learning_curves(rows, summary, out):
    gens = [s["generation"] for s in summary]
    fig, axes = plt.subplots(1, len(gens), figsize=(2.3 * len(gens) + 0.8, 4.0), sharey=True, facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.72, bottom=0.15, wspace=0.08)
    for ax, gen in zip(axes, gens):
        new = sorted((r for r in rows if r["generation"] == gen and r["new"] and r["metrics"]), key=lambda r: r["score"])
        for i, r in enumerate(new):
            curve = training_return(r)
            if curve is None:
                continue
            it, val = curve
            top = i == len(new) - 1
            ax.plot(it, val, color=BLUE if top else DOT, linewidth=2 if top else 1.1, zorder=3 if top else 2,
                    label=("best of generation" if top else ("other candidates" if i == 0 else None))
                    if gen == gens[0] else None)
        style(ax, xlabel="PPO iteration")
        ax.set_title(f"generation {gen}", fontsize=9.5, color=INK, loc="left")
    axes[0].set_ylabel("training return per episode", color=INK_2, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0.004, 0.86),
                     ncol=2, handlelength=1.6)
    for text in leg.get_texts():
        text.set_color(INK_2)
    titles(fig, "Learning curves of the newly trained candidates",
           "Stochastic training policy, reward averaged over the last episode (envs reset in sync); "
           "best = highest evaluation score.")
    fig.savefig(os.path.join(out, "learning_curves.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)


def load_recheck(run_dir: str) -> list[dict] | None:
    path = os.path.join(run_dir, "recheck", "recheck.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["score"] = float(r["score"])
    return rows


def plot_seed_check(recheck: list[dict], run_seed: int, out: str) -> None:
    labels = list(dict.fromkeys(r["label"] for r in recheck))
    height = 0.62 * len(labels) + 1.9
    fig, ax = plt.subplots(figsize=(8.0, height), facecolor=SURFACE)
    fig.subplots_adjust(left=0.2, right=0.97, top=1 - 0.95 / height, bottom=0.55 / height + 0.06)
    for y, label in enumerate(reversed(labels)):
        scores = [r for r in recheck if r["label"] == label]
        mean = statistics.mean(r["score"] for r in scores)
        ax.plot([min(r["score"] for r in scores), max(r["score"] for r in scores)], [y, y], color=GRID, linewidth=2,
                zorder=1)
        for r in scores:
            run = int(r["seed"]) == run_seed  # drawn on top so overlapping new-seed dots can't hide it
            ax.scatter(r["score"], y, s=46, color=BLUE if run else DOT, edgecolor=SURFACE, linewidth=2,
                       zorder=5 if run else 3)
        ax.plot([mean, mean], [y - 0.22, y + 0.22], color=INK, linewidth=2, zorder=4)
        ax.annotate(f"mean {mean:.0f}", (mean, y + 0.26), ha="center", va="bottom", fontsize=8, color=INK_2)
    ax.scatter([], [], s=46, color=BLUE, label=f"seed {run_seed} (the run's seed)")
    ax.scatter([], [], s=46, color=DOT, label="new seeds")
    ax.plot([], [], color=INK, linewidth=2, label="mean over seeds")
    style(ax, xlabel="score (episode return)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(["unscaled cheetah" if lab == "identity" else lab for lab in reversed(labels)])
    ax.set_ylim(-0.6, len(labels) - 0.3)
    legend(ax, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3)
    titles(fig, "Seed check of the best designs",
           "Same design and budget, retrained from scratch with other seeds: the spread is training noise.")
    fig.savefig(os.path.join(out, "seed_check.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    args = parser.parse_args()  # parsed here so plot_convergence.py can import the helpers
    run_dir = os.path.abspath(args.run_dir)
    config = _load(os.path.join(run_dir, "config.json")) or {}
    n_parents = args.parents or config.get("elites", 4)
    out = os.path.join(run_dir, "plots")
    os.makedirs(out, exist_ok=True)
    rows = load_rows(run_dir)
    summary = summarise(rows, n_parents)
    write_tables(rows, summary, out)
    recheck = load_recheck(run_dir)
    baseline = [r["score"] for r in recheck or [] if r["label"] == "identity"]
    plot_scores(rows, summary, n_parents, baseline, out)
    plot_designs(rows, summary, out)
    plot_learning_curves(rows, summary, out)
    if recheck:
        plot_seed_check(recheck, config.get("seed", 42), out)
    for s in summary:
        print(f"[plot] gen {s['generation']}: best {s['best_score']:.2f} ({s['best_id']}), "
              f"top{n_parents} mean {s[f'top{n_parents}_mean']:.2f}, new median {s['new_median']:.2f}, "
              f"children beating parent {s['children_beating_parent'] or '-'}")
    print(f"[plot] wrote {out}")


if __name__ == "__main__":
    main()
