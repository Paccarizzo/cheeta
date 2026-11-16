"""Plots and tables of a compare_methods.py comparison (shared policy vs from scratch).

Reads <out>/compare.json and the two evolution runs in <out>/shared and
<out>/scratch. Writes to <out>/plots/:
  * shared_evolution.png  - shared run: every design's score per generation, mean of the
                            parents, reference designs; below, parents minus references
  * scratch_evolution.png - from-scratch run: every trained candidate, best so far, mean
                            of the elites
  * final_test.png        - the final designs trained from scratch (common judge) next to
                            their zero-shot score under the shared policy
  * designs.png           - the six scale factors of the final designs
  * rank_check.png        - shared-policy score vs from-scratch score of shared designs
  * final_curves.png      - training return of the final-test jobs over PPO iterations
  * budget.csv            - training transitions, wall-clock and design evaluations per method

Needs matplotlib and numpy only (no Isaac Sim):
    python scripts/coadapt/plot_compare.py --out logs/coadapt/compare
"""

import argparse
import csv
import json
import os
import statistics

import matplotlib.transforms
import numpy as np

from plot_evolution import (BASELINE, BLUE, DESIGN_BOUNDS, DESIGN_NAMES, DOT, GRID, INK_2, MUTED, ORANGE, SURFACE,
                            _load, legend, plt, style, titles, training_return)

METHOD_COLOR = {"shared": BLUE, "shared-rank": BLUE, "scratch": ORANGE, "reference": INK_2}
METHOD_NAME = {"shared": "shared policy", "scratch": "from scratch", "reference": "unscaled cheetah"}

parser = argparse.ArgumentParser(description="Plot a compare_methods.py comparison.")
parser.add_argument("--out", type=str, required=True, help="comparison directory")


def grid(ax, axis: str) -> None:
    """Gridlines on another axis than style()'s default y."""
    ax.grid(False)
    ax.grid(axis=axis, color=GRID, linewidth=0.8)


def end_label(ax, x, y, text: str) -> None:
    """Direct label at the end of a line (text in ink, not the series colour)."""
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK_2)


def save(fig, plots: str, name: str, top: float) -> None:
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout(rect=(0, 0, 1, top))
    path = os.path.join(plots, f"{name}.png")
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"[plot] {path}")


def read_csv(path: str) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _float(v) -> float:
    return float(v) if v not in (None, "") else float("nan")


def plot_shared(out: str, plots: str, n_parents: int) -> None:
    summary = read_csv(os.path.join(out, "shared", "summary.csv"))
    hist = read_csv(os.path.join(out, "shared", "history.csv"))
    gens = [int(s["generation"]) for s in summary]
    col = {k: [_float(s[k]) for s in summary] for k in ("parents_mean", "identity_score", "refs_mean",
                                                        "parents_minus_refs")}
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.4, 6.6), sharex=True, gridspec_kw={"height_ratios": [2.3, 1]})
    rng = np.random.default_rng(0)
    for g in gens:
        s = np.array([float(r["score"]) for r in hist if int(r["generation"]) == g and r["role"] != "reference"])
        ax.scatter(g + rng.uniform(-0.3, 0.3, len(s)), s, s=3, color=DOT, alpha=0.4, linewidths=0, zorder=1)
    ax.scatter([], [], s=12, color=DOT, label="individual designs")
    ax.plot(gens, col["parents_mean"], color=BLUE, lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5, zorder=3,
            label=f"mean of the top {n_parents} (parents)")
    ax.plot(gens, col["identity_score"], color=INK_2, lw=2, marker="s", ms=5, mec=SURFACE, mew=1.5, zorder=3,
            label="unscaled cheetah")
    ax.plot(gens, col["refs_mean"], color=MUTED, lw=2, marker="D", ms=4.5, mec=SURFACE, mew=1.5, zorder=3,
            label="reference designs, mean")
    for key, text in (("parents_mean", "parents"), ("identity_score", "unscaled"), ("refs_mean", "references")):
        end_label(ax, gens[-1], col[key][-1], text)
    style(ax, "score (shared policy, deterministic)")
    legend(ax, loc="lower right", ncols=2)  # below the cloud, which never reaches 0
    ax2.axhline(0, color=BASELINE, lw=1)
    ax2.plot(gens, col["parents_minus_refs"], color=BLUE, lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5)
    end_label(ax2, gens[-1], col["parents_minus_refs"][-1], f"{col['parents_minus_refs'][-1]:+.0f}")
    style(ax2, "parents - references", "generation")
    ax2.set_xticks(gens)
    titles(fig, "Shared-policy evolution",
           "All scores come from the policy of that generation, so they rise as it trains; "
           "the lower panel removes that drift.")
    save(fig, plots, "shared_evolution", top=0.92)


def plot_scratch(out: str, plots: str, n_elites: int) -> None:
    rows = [r for r in read_csv(os.path.join(out, "scratch", "history.csv")) if r["trained"] == "1"]
    gens = sorted({int(r["generation"]) for r in rows})
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    rng = np.random.default_rng(0)
    best_so_far, elite_mean, best = [], [], -np.inf
    for g in gens:
        pop = sorted((float(r["score"]) for r in rows if int(r["generation"]) == g), reverse=True)
        new = [float(r["score"]) for r in rows if int(r["generation"]) == g and r["role"] != "elite"]
        ax.scatter(g + rng.uniform(-0.12, 0.12, len(new)), new, s=22, color=DOT, edgecolor=SURFACE, linewidth=1, zorder=2)
        best = max(best, pop[0])
        best_so_far.append(best)
        elite_mean.append(statistics.mean(pop[:n_elites]))
    ax.scatter([], [], s=22, color=DOT, label="new candidates (trained this generation)")
    ax.plot(gens, best_so_far, color=ORANGE, lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5, zorder=3, label="best so far")
    ax.plot(gens, elite_mean, color=ORANGE, lw=1.2, alpha=0.55, marker="o", ms=4, mec=SURFACE, mew=1.2, zorder=3,
            label=f"mean of the top {n_elites} (elites)")
    end_label(ax, gens[-1], best_so_far[-1], f"{best_so_far[-1]:.0f}")
    style(ax, "score (trained from scratch)", "generation")
    ax.set_xticks(gens)
    legend(ax, loc="lower right")
    titles(fig, "From-scratch evolution", "Every candidate gets the same budget from a fresh policy, "
           "so scores are comparable across generations.")
    save(fig, plots, "scratch_evolution", top=0.88)


def plot_final(targets: list[dict], plots: str, identity: dict | None) -> None:
    order = targets[::-1]  # first target at the top
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 0.42 * len(order) + 1.6), sharey=True)
    panels = (("final_score", "trained from scratch (common judge)"), ("zeroshot_score", "zero-shot, shared policy"))
    for ax, (key, label) in zip(axes, panels):
        if identity and identity.get(key) is not None:  # dashed so it doesn't read as a gridline
            ax.axvline(identity[key], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
        for yi, t in zip(y, order):
            if t.get(key) is not None:
                ax.scatter(t[key], yi, s=60, color=METHOD_COLOR[t["method"]], edgecolor=SURFACE, linewidth=2, zorder=3)
        style(ax, xlabel=f"score, {label}")
        grid(ax, "x")
    axes[0].set_yticks(y, [METHOD_NAME["reference"] if t["method"] == "reference" else t["label"].replace("_", " ")
                           for t in order])
    for method in ("shared", "scratch", "reference"):
        axes[1].scatter([], [], s=60, color=METHOD_COLOR[method], label=METHOD_NAME[method])
    legend(axes[1], loc="lower right")
    titles(fig, "Final test of the best designs",
           "Left: each design trained alone from scratch, one seed. Right: the shared run's final policy, "
           "no training. Dashed line: unscaled cheetah.")
    save(fig, plots, "final_test", top=1 - 0.75 / fig.get_figheight())


def plot_designs(targets: list[dict], plots: str) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    x = np.arange(len(DESIGN_NAMES))
    for bound in DESIGN_BOUNDS:
        ax.axhline(bound, color=BASELINE, lw=1)
    for t in targets:
        top1 = t["label"].endswith("top1")
        lw, alpha = (2.4, 1.0) if top1 or t["method"] == "reference" else (1.1, 0.45)
        ax.plot(x, t["design"], color=METHOD_COLOR[t["method"]], lw=lw, alpha=alpha, marker="o",
                ms=5 if top1 else 3, mec=SURFACE, mew=1, zorder=3 if top1 else 2)
    for method in ("shared", "scratch", "reference"):
        ax.plot([], [], color=METHOD_COLOR[method], lw=2, label=METHOD_NAME[method])
    style(ax, "scale factor")
    ax.set_xticks(x, DESIGN_NAMES)
    ax.set_ylim(DESIGN_BOUNDS[0] - 0.05, DESIGN_BOUNDS[1] + 0.05)
    legend(ax, loc="upper left", ncols=3)
    titles(fig, "Designs in the final test", "Six leg-segment scale factors; thick lines: best design of "
           "each method. Grey rules: design bounds.")
    save(fig, plots, "designs", top=0.88)


def plot_ranks(shared: list[dict], plots: str, rho: float, identity: dict | None) -> None:
    pts = [t for t in shared if t.get("final_score") is not None]
    if len(pts) < 3:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    top = [t for t in pts if t["method"] == "shared"]
    spread = [t for t in pts if t["method"] != "shared"]
    ax.scatter([t["run_score"] for t in top], [t["final_score"] for t in top], s=60, color=BLUE,
               edgecolor=SURFACE, linewidth=2, zorder=3, label=f"top {len(top)} (final test)")
    ax.scatter([t["run_score"] for t in spread], [t["final_score"] for t in spread], s=48, color=SURFACE,
               edgecolor=BLUE, linewidth=1.8, zorder=3, label="spread over the ranking")
    if identity and identity.get("final_score") is not None:
        ax.axhline(identity["final_score"], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.annotate("unscaled cheetah", (0, identity["final_score"]), xycoords=("axes fraction", "data"),
                    xytext=(4, 4), textcoords="offset points", fontsize=8.5, color=INK_2)
    style(ax, "score, trained from scratch", "score, shared policy (last generation)")
    grid(ax, "both")
    legend(ax, loc="upper left")
    titles(fig, "Does the shared policy rank designs correctly?",
           f"Shared designs across the final ranking (#rank). Spearman {rho:.2f} over {len(pts)} designs.")
    fig.tight_layout(rect=(0, 0, 1, 0.86))  # final geometry before the labels are placed
    place_labels(ax, [(t["run_score"], t["final_score"]) for t in pts], [f"#{t['shared_rank']}" for t in pts])
    save(fig, plots, "rank_check", top=0.86)


def place_labels(ax, xy: list[tuple], texts: list[str]) -> None:
    """Point labels, each on the side around its point that overlaps other labels and points the least."""
    ax.figure.canvas.draw()  # settle the autoscaled limits
    renderer = ax.figure.canvas.get_renderer()
    r = 5  # marker radius in pixels, roughly
    taken = [matplotlib.transforms.Bbox.from_extents(px - r, py - r, px + r, py + r)
             for px, py in ax.transData.transform(xy)]
    sides = ((6, 0, "left", "center"), (-6, 0, "right", "center"), (0, 6, "center", "bottom"), (0, -6, "center", "top"),
             (5, 5, "left", "bottom"), (-5, 5, "right", "bottom"), (5, -5, "left", "top"), (-5, -5, "right", "top"))
    for i, ((x, y), text) in enumerate(zip(xy, texts)):
        others = taken[:i] + taken[i + 1:]  # its own point never counts
        best = None
        for dx, dy, ha, va in sides:
            label = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=8,
                                color=MUTED)
            box = label.get_window_extent(renderer)
            hits = sum(max(0, min(box.x1, t.x1) - max(box.x0, t.x0)) * max(0, min(box.y1, t.y1) - max(box.y0, t.y0))
                       for t in others)  # overlap area, so grazing a corner beats covering a point
            if best is None or hits < best[0]:
                if best:
                    best[1].remove()
                best = (hits, label, box)
            else:
                label.remove()
            if hits == 0:
                break
        taken.append(best[2])


def plot_curves(out: str, targets: list[dict], plots: str) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for t in targets:
        job = os.path.join(out, "final", t["label"])
        curve = training_return({"result": _load(os.path.join(job, "result.json")),
                                 "metrics": _load(os.path.join(job, "metrics.json"))})
        if curve is not None:
            ref = t["method"] == "reference"
            ax.plot(*curve, color=METHOD_COLOR[t["method"]], lw=2 if ref else 1.2, alpha=1.0 if ref else 0.75,
                    zorder=3 if ref else 2)
    for method in ("shared", "scratch", "reference"):
        ax.plot([], [], color=METHOD_COLOR[method], lw=2, label=METHOD_NAME[method])
    style(ax, "training return per episode", "PPO iteration")
    legend(ax, loc="lower right")
    titles(fig, "Training curves of the final test", "One line per design, trained from scratch; stochastic "
           "training policy, reward averaged over the last episode.")
    save(fig, plots, "final_curves", top=0.88)


def write_budget(state: dict, plots: str) -> None:
    budget, wall = state.get("budget", {}), state.get("wall_time_s", {})
    rows = []
    for method in ("shared", "scratch"):
        b = budget.get(method, {})
        rows.append({"method": METHOD_NAME[method], "training_transitions": b.get("transitions"),
                     "wall_time_h": round(wall.get(method, 0.0) / 3600, 2), "worker_jobs": b.get("jobs"),
                     "design_evaluations": b.get("design_evaluations")})
    path = os.path.join(plots, "budget.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[plot] {path}")


def main():
    args = parser.parse_args()
    out = os.path.abspath(args.out)
    plots = os.path.join(out, "plots")
    os.makedirs(plots, exist_ok=True)
    with open(os.path.join(out, "compare.json")) as f:
        state = json.load(f)
    cfg = state["config"]
    targets, ranks = state.get("targets", []), state.get("ranks", [])
    identity = next((t for t in targets if t["method"] == "reference"), None)

    plot_shared(out, plots, cfg["shared_parents"])
    plot_scratch(out, plots, cfg["scratch_elites"])
    if targets:
        plot_final(targets, plots, identity)
        plot_designs(targets, plots)
        plot_curves(out, targets, plots)
        plot_ranks([t for t in targets + ranks if t["method"] in ("shared", "shared-rank")], plots,
                   state.get("rank_check_spearman", float("nan")), identity)
    write_budget(state, plots)


if __name__ == "__main__":
    main()

