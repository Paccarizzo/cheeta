"""Shared-policy vs from-scratch evolution at a matched training budget, followed
by a common final test of the designs both runs found.

Stages (finished jobs are skipped, so an interrupted comparison resumes):
  1. shared   evolve_shared.py: one design-conditioned policy for the whole population
  2. scratch  evolve.py --inherit scratch --elite-mode keep --random-episode-start: every
              candidate trained from scratch. --scratch-iterations defaults to the value
              that matches the shared run's training transitions.
  3. final    the --top best distinct designs of each run and the unscaled cheetah, each
              trained from scratch with the same budget (--final-iterations, random
              episode starts, one seed): the common judge
  4. zeroshot the same designs scored by the shared run's final policy, no training
  5. ranks    --rank-samples designs spread over the shared run's last ranking, trained
              like stage 3, to test how well the shared policy ranks designs

Output in <out>: shared/ and scratch/ (the two evolution runs), final/<label>/ (one
job per design), zeroshot/, compare.json (budgets, timings and every evaluated design
with its scores) and final.csv. Plots and tables: plot_compare.py.

Run with the Isaac python (drivers and workers inherit the interpreter):
    python scripts/coadapt/compare_methods.py --out logs/coadapt/compare
"""

import argparse
import csv
import os
import subprocess
import sys
import time

from _common import (REPO_ROOT, add_wandb_args, designs, load_json, load_result, run_worker, save_json, spearman,
                     timestamp, wandb_worker_args, worker_command)

parser = argparse.ArgumentParser(description="Compare shared-policy and from-scratch evolution at a matched budget.")
parser.add_argument("--out", type=str, default=None, help="comparison directory (set to an old one to resume)")
group = parser.add_argument_group("shared run")
group.add_argument("--shared-generations", type=int, default=10)
group.add_argument("--shared-population", type=int, default=1000)
group.add_argument("--shared-parents", type=int, default=100)
group.add_argument("--shared-immigrants", type=int, default=50)
group.add_argument("--shared-references", type=int, default=10)
group.add_argument("--envs-per-design", type=int, default=4)
group.add_argument("--shared-iterations-first", type=int, default=1000)
group.add_argument("--shared-iterations", type=int, default=300)
group = parser.add_argument_group("from-scratch run")
group.add_argument("--scratch-generations", type=int, default=5)
group.add_argument("--scratch-population", type=int, default=10)
group.add_argument("--scratch-elites", type=int, default=4)
group.add_argument("--scratch-num-envs", type=int, default=1024)
group.add_argument("--scratch-iterations", type=int, default=None,
                   help="PPO iterations per candidate (default: match the shared run's training transitions)")
group = parser.add_argument_group("evaluation")
group.add_argument("--top", type=int, default=5, help="best distinct designs of each run in the final test")
group.add_argument("--final-iterations", type=int, default=800)
group.add_argument("--final-num-envs", type=int, default=1024)
group.add_argument("--zeroshot-envs-per-design", type=int, default=16)
group.add_argument("--rank-samples", type=int, default=9, help="extra shared designs for the rank check (0 = skip)")
parser.add_argument("--sigma", type=float, default=0.15, help="mutation std dev (both runs)")
parser.add_argument("--sigma-decay", type=float, default=0.95, help="sigma multiplier per generation (both runs)")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", type=str, default=None, help="forwarded to the workers, e.g. cuda:0")
add_wandb_args(parser)
parser.set_defaults(wandb_project="cheeta-shared")
args = parser.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL_COLUMNS = ["label", "method", "source", "run_score", "final_score", "final_score_std", "final_distance_x_m",
                 "zeroshot_score", *designs.DESIGN_NAMES]


def wandb_cli() -> list[str]:
    """wandb flags for the evolution drivers (they name and group their own runs)."""
    if not args.wandb:
        return []
    cli = ["--wandb", "--wandb-project", args.wandb_project, *(["--wandb-tags", *args.wandb_tags] if args.wandb_tags else [])]
    if args.wandb_entity:
        cli += ["--wandb-entity", args.wandb_entity]
    if args.wandb_mode:
        cli += ["--wandb-mode", args.wandb_mode]
    return cli


def run_driver(script: str, run_dir: str, cli: list[str], label: str) -> None:
    """An evolution driver as a subprocess; its own resume logic skips finished jobs."""
    cmd = [sys.executable, "-u", os.path.join(HERE, script), "--run-dir", run_dir, *cli, *wandb_cli()]
    print(f"\n[compare] === {label} ===\n[compare] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    if proc.returncode != 0:
        print(f"[compare] WARNING: {label} driver exited with code {proc.returncode}")


def job_transitions(job_dir: str) -> int:
    """Training transitions of one worker job: iterations x envs x rollout length."""
    result = load_result(job_dir)
    metrics_path = os.path.join(job_dir, "metrics.json")
    if not result or not result.get("iterations") or not os.path.exists(metrics_path):
        return 0
    return result["iterations"] * result["num_envs"] * load_json(metrics_path)["rollouts"]


def run_jobs(run_dir: str, generations: int) -> list[str]:
    """Worker job directories of an evolution run (generation dirs, or their candidate subdirs)."""
    jobs = []
    for gen in range(generations):
        gen_dir = os.path.join(run_dir, f"gen_{gen:02d}")
        if os.path.exists(os.path.join(gen_dir, "result.json")):
            jobs.append(gen_dir)
        elif os.path.isdir(gen_dir):
            jobs += [os.path.join(gen_dir, d) for d in sorted(os.listdir(gen_dir)) if d.startswith("cand_")]
    return jobs


def read_history(run_dir: str) -> list[dict]:
    with open(os.path.join(run_dir, "history.csv")) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["generation"] = int(r["generation"])
        r["score"] = float(r["score"])
        r["design"] = [float(r[n]) for n in designs.DESIGN_NAMES]
    return rows


def distinct_top(rows: list[dict], n: int) -> list[dict]:
    """The n best rows with distinct designs (a kept elite repeats its design and score)."""
    picked, seen = [], set()
    for r in sorted(rows, key=lambda r: r["score"], reverse=True):
        if tuple(r["design"]) not in seen:
            seen.add(tuple(r["design"]))
            picked.append(r)
        if len(picked) == n:
            break
    return picked


def pick_targets(shared_dir: str, scratch_dir: str) -> tuple[list[dict], list[dict]]:
    """Final-test designs (top of both runs + unscaled cheetah) and rank-check designs."""
    shared_rows = read_history(shared_dir)
    last = max(r["generation"] for r in shared_rows)
    last_pop = sorted((r for r in shared_rows if r["generation"] == last and r["role"] != "reference"),
                      key=lambda r: r["score"], reverse=True)
    scratch_rows = [r for r in read_history(scratch_dir) if r["trained"] == "1"]
    targets = []
    for i, r in enumerate(distinct_top(last_pop, args.top), 1):
        targets.append({"label": f"shared_top{i}", "method": "shared", "source": f"gen_{last:02d}/{r['id']}",
                        "design": r["design"], "run_score": r["score"], "shared_rank": last_pop.index(r) + 1})
    for i, r in enumerate(distinct_top(scratch_rows, args.top), 1):
        targets.append({"label": f"scratch_top{i}", "method": "scratch", "source": f"gen_{r['generation']:02d}/{r['id']}",
                        "design": r["design"], "run_score": r["score"]})
    targets.append({"label": "identity", "method": "reference", "source": "unscaled cheetah",
                    "design": list(designs.IDENTITY_DESIGN), "run_score": None})
    ranks = []
    m, n, k = args.rank_samples, len(last_pop), args.top
    if m > 0 and n > k:
        # ranks spread evenly over the rest of the last generation (1-based, below the top-k)
        positions = sorted({round(k + i * (n - 1 - k) / max(m - 1, 1)) for i in range(m)})
        for pos in positions:
            r = last_pop[pos]
            ranks.append({"label": f"shared_rank{pos + 1:04d}", "method": "shared-rank", "source": f"gen_{last:02d}/{r['id']}",
                          "design": r["design"], "run_score": r["score"], "shared_rank": pos + 1})
    return targets, ranks


def train_final(targets: list[dict], out: str, name: str) -> None:
    for t in targets:
        cmd = worker_command(os.path.join(out, "final", t["label"]), num_envs=args.final_num_envs,
                             iterations=args.final_iterations, seed=args.seed, design=t["design"], device=args.device,
                             random_episode_start=True,
                             extra=wandb_worker_args(args, name=f"{name}-final-{t['label']}", group=f"compare-{name}",
                                                     job_type="final", tags=[t["method"]]))
        print(f"[compare] final test {t['label']}: design {t['design']}")
        run_worker(cmd, t["label"])


def main():
    out = os.path.abspath(args.out or os.path.join(REPO_ROOT, "logs", "coadapt", f"compare_{timestamp()}"))
    os.makedirs(out, exist_ok=True)
    name = os.path.basename(os.path.normpath(out))
    shared_dir, scratch_dir = os.path.join(out, "shared"), os.path.join(out, "scratch")
    state_path = os.path.join(out, "compare.json")
    state = load_json(state_path) if os.path.exists(state_path) else {}
    state["config"] = vars(args)
    timings = state.setdefault("wall_time_s", {})

    def timed(stage, fn):
        # accumulated over resumed invocations (a finished stage only adds a few seconds)
        t0 = time.time()
        fn()
        timings[stage] = round(timings.get(stage, 0.0) + time.time() - t0, 1)
        save_json(state, state_path)

    timed("shared", lambda: run_driver("evolve_shared.py", shared_dir, [
        "--generations", str(args.shared_generations), "--population", str(args.shared_population),
        "--parents", str(args.shared_parents), "--immigrants", str(args.shared_immigrants),
        "--references", str(args.shared_references), "--envs-per-design", str(args.envs_per_design),
        "--iterations-first", str(args.shared_iterations_first), "--iterations", str(args.shared_iterations),
        "--sigma", str(args.sigma), "--sigma-decay", str(args.sigma_decay), "--seed", str(args.seed),
        *(["--device", args.device] if args.device else [])], "stage 1: shared-policy evolution"))
    shared_jobs = run_jobs(shared_dir, args.shared_generations)
    if not shared_jobs or not os.path.exists(os.path.join(shared_dir, "policy.pt")):
        sys.exit("[compare] shared run produced no results, stopping")
    shared_transitions = sum(job_transitions(j) for j in shared_jobs)

    scratch_jobs_planned = args.scratch_population + (args.scratch_generations - 1) * (
        args.scratch_population - args.scratch_elites)
    rollouts = load_json(os.path.join(shared_jobs[0], "metrics.json"))["rollouts"]
    scratch_iterations = args.scratch_iterations or max(1, round(
        shared_transitions / (scratch_jobs_planned * args.scratch_num_envs * rollouts)))
    print(f"\n[compare] shared run: {shared_transitions:.3e} training transitions; from-scratch run: "
          f"{scratch_jobs_planned} jobs x {args.scratch_num_envs} envs x {scratch_iterations} iterations")
    timed("scratch", lambda: run_driver("evolve.py", scratch_dir, [
        "--generations", str(args.scratch_generations), "--population", str(args.scratch_population),
        "--elites", str(args.scratch_elites), "--immigrants", "0", "--inherit", "scratch", "--elite-mode", "keep",
        "--no-identity", "--random-episode-start", "--num_envs", str(args.scratch_num_envs),
        "--iterations", str(scratch_iterations), "--sigma", str(args.sigma), "--sigma-decay", str(args.sigma_decay),
        "--seed", str(args.seed), *(["--device", args.device] if args.device else [])],
        "stage 2: from-scratch evolution"))
    scratch_jobs = run_jobs(scratch_dir, args.scratch_generations)
    if not scratch_jobs:
        sys.exit("[compare] from-scratch run produced no results, stopping")

    targets, ranks = pick_targets(shared_dir, scratch_dir)
    state["budget"] = {
        "shared": {"transitions": shared_transitions, "jobs": len(shared_jobs),
                   "design_evaluations": sum(len(load_result(j)["per_design"]) for j in shared_jobs)},
        "scratch": {"transitions": sum(job_transitions(j) for j in scratch_jobs), "jobs": len(scratch_jobs),
                    "design_evaluations": len(scratch_jobs), "iterations": scratch_iterations},
    }
    state["targets"], state["ranks"] = targets, ranks
    save_json(state, state_path)

    print("\n[compare] === stage 3: final test (every design trained from scratch) ===")
    timed("final", lambda: train_final(targets, out, name))

    print("\n[compare] === stage 4: zero-shot scores under the shared policy ===")
    everything = targets + ranks
    zs_dir = os.path.join(out, "zeroshot")
    timed("zeroshot", lambda: run_worker(worker_command(
        zs_dir, num_envs=args.zeroshot_envs_per_design * len(everything), iterations=0, seed=args.seed,
        design_set=[t["design"] for t in everything], checkpoint=os.path.join(shared_dir, "policy.pt"),
        device=args.device), "zero-shot"))

    if ranks:
        print("\n[compare] === stage 5: rank check (shared designs across the ranking, trained from scratch) ===")
        timed("ranks", lambda: train_final(ranks, out, name))

    # collect every score into compare.json and final.csv
    zs = load_result(zs_dir)
    for i, t in enumerate(everything):
        result = load_result(os.path.join(out, "final", t["label"]))
        t["final_score"] = result["score"] if result else None
        t["final_score_std"] = result["score_std"] if result else None
        t["final_distance_x_m"] = result["distance_x_m"] if result else None
        t["zeroshot_score"] = zs["per_design"][i]["score"] if zs else None
    shared_scored = [t for t in everything if t["method"] in ("shared", "shared-rank") and t["final_score"] is not None]
    state["rank_check_spearman"] = spearman([t["run_score"] for t in shared_scored],
                                            [t["final_score"] for t in shared_scored])
    save_json(state, state_path)
    with open(os.path.join(out, "final.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FINAL_COLUMNS)
        for t in everything:
            w.writerow([t["label"], t["method"], t["source"], *(_fmt(t.get(k)) for k in (
                "run_score", "final_score", "final_score_std", "final_distance_x_m", "zeroshot_score")),
                *[f"{x:.4f}" for x in t["design"]]])

    print("\n[compare] final test (trained from scratch, "
          f"{args.final_iterations} iterations):")
    for t in targets:
        print(f"    {t['label']:<14} final {_fmt(t['final_score'], 1):>7}  zero-shot {_fmt(t['zeroshot_score'], 1):>7}  "
              f"run score {_fmt(t['run_score'], 1):>7}  {t['design']}")
    print(f"[compare] rank check: spearman(shared score, from-scratch score) = {state['rank_check_spearman']:.2f} "
          f"over {len(shared_scored)} shared designs")
    print(f"[compare] results: {state_path}\n[compare] plots: python scripts/coadapt/plot_compare.py --out {out}")


def _fmt(v, digits: int = 4) -> str:
    return "" if v is None else f"{v:.{digits}f}"


if __name__ == "__main__":
    main()
