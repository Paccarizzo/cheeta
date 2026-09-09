"""Nevergrad-driven co-adaptation of half-cheetah morphology and PPO policy.

Alternative driver to evolve.py: candidate designs are chosen by a nevergrad
optimizer (CMA-ES by default) instead of the hand-rolled GA. Each "generation"
is one ask/tell batch:

  1. ask the optimizer for --population candidate designs
  2. train and score each with scripts/coadapt/train_design.py (one Isaac Sim
     session per candidate, exactly as in evolve.py)
  3. tell the optimizer the scores; it updates its search distribution and the
     next batch is sampled from the updated state

Policy inheritance is decoupled from the optimizer: every candidate
warm-starts from the checkpoint of the nearest previously evaluated design
(Euclidean distance in design space), so "the policy almost works already"
holds no matter which algorithm proposed the design.

Useful --optimizer values: CMA (default), TBPSA (built for noisy scores),
PSO (what Luck et al. used), NGOpt (nevergrad's auto-selector),
RandomSearch (baseline). Any name from nevergrad's registry works.

This driver needs no Isaac Sim itself; run it with the env_isaaclab python so
the workers inherit the right interpreter:
    python scripts/coadapt/evolve_ng.py --generations 10 --population 8 ^
        --num_envs 256 --iterations 100

Interrupted runs resume: re-run with --run-dir pointing at the old run;
finished candidates are skipped (their result.json is reused) and completed
generations are replayed into the optimizer via tell.
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from datetime import datetime

import nevergrad as ng
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKER = os.path.join(REPO_ROOT, "scripts", "coadapt", "train_design.py")
DESIGN_BOUNDS = (0.8, 2.0)  # paper Sec. 7.1
DESIGN_DIM = 6
# loss handed to the optimizer when a candidate could not be scored (rank-based
# optimizers only need it to be finite and worse than everything real)
FAILED_LOSS = 1.0e9

parser = argparse.ArgumentParser(description="Evolve half-cheetah designs with nevergrad + per-design PPO training.")
parser.add_argument("--optimizer", type=str, default="CMA",
                    help="nevergrad optimizer name, e.g. CMA, TBPSA, PSO, NGOpt, RandomSearch")
parser.add_argument("--generations", type=int, default=10, help="ask/tell batches")
parser.add_argument("--population", type=int, default=8, help="candidates per batch")
parser.add_argument("--sigma", type=float, default=0.15, help="initial mutation std dev on the scale factors")
parser.add_argument("--no-warm-start", action="store_true",
                    help="train every candidate from scratch instead of the nearest evaluated design's checkpoint")
parser.add_argument("--num_envs", type=int, default=256, help="parallel envs per candidate")
parser.add_argument("--iterations", type=int, default=100, help="PPO iterations per candidate")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run-dir", type=str, default=None, help="run directory (set to an old run to resume)")
parser.add_argument("--device", type=str, default=None, help="forwarded to the worker, e.g. cuda:0")
args = parser.parse_args()


def nearest_checkpoint(design: list[float], archive: list[dict]) -> str | None:
    """Checkpoint of the closest already-evaluated design (None if archive is empty)."""
    best_ckpt, best_dist = None, math.inf
    for entry in archive:
        dist = math.dist(design, entry["design"])
        if dist < best_dist:
            best_ckpt, best_dist = entry["checkpoint"], dist
    return best_ckpt


def run_generation(candidates: list[dict], gen_dir: str) -> None:
    """One worker process per candidate; finished candidates are skipped."""
    for cand in candidates:
        out_dir = os.path.join(gen_dir, cand["id"])
        if os.path.exists(os.path.join(out_dir, "result.json")):
            print(f"[evolve-ng] {cand['id']}: already done, skipping")
            continue
        cmd = [
            sys.executable, "-u", WORKER,
            "--design", *[str(x) for x in cand["design"]],
            "--out", out_dir,
            "--num_envs", str(args.num_envs),
            "--iterations", str(args.iterations),
            "--seed", str(args.seed),
            "--headless",
        ]
        if cand.get("checkpoint"):
            cmd += ["--checkpoint", cand["checkpoint"]]
        if args.device:
            cmd += ["--device", args.device]
        print(f"[evolve-ng] {cand['id']}: training design {cand['design']}"
              + (f" (warm start: {cand['checkpoint']})" if cand.get("checkpoint") else ""))
        proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
        if proc.returncode != 0:
            print(f"[evolve-ng] WARNING: worker for {cand['id']} exited with code {proc.returncode}")


def collect_results(gen_dir: str, candidates: list[dict]) -> list[dict]:
    results = []
    for cand in candidates:
        path = os.path.join(gen_dir, cand["id"], "result.json")
        if os.path.exists(path):
            with open(path) as f:
                result = json.load(f)
            result["id"] = cand["id"]
        else:
            print(f"[evolve-ng] WARNING: no result for {cand['id']}, scoring -inf")
            result = {"id": cand["id"], "design": cand["design"], "score": float("-inf"),
                      "distance_x_m": float("nan"), "checkpoint": None}
        results.append(result)
    return results


def main():
    run_dir = args.run_dir or os.path.join(REPO_ROOT, "logs", "coadapt", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(run_dir, exist_ok=True)
    history_path = os.path.join(run_dir, "history.csv")
    new_history = not os.path.exists(history_path)
    history = open(history_path, "a", newline="")
    writer = csv.writer(history)
    if new_history:
        writer.writerow(["generation", "id", "score", "distance_x_m",
                         "bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot"])

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # identity design as the search's starting point; sigma matches the GA's
    # mutation scale so early batches stay near evaluated designs
    param = ng.p.Array(init=np.ones(DESIGN_DIM)).set_mutation(sigma=args.sigma).set_bounds(*DESIGN_BOUNDS)
    param.random_state = np.random.RandomState(args.seed)
    optimizer = ng.optimizers.registry[args.optimizer](
        parametrization=param,
        budget=args.generations * args.population,
        num_workers=args.population,
    )
    # make sure the unmodified cheetah is scored as the baseline
    optimizer.suggest(np.ones(DESIGN_DIM))

    archive: list[dict] = []  # every scored candidate: {design, score, checkpoint}
    best: dict | None = None

    for gen in range(args.generations):
        gen_dir = os.path.join(run_dir, f"gen_{gen:02d}")
        os.makedirs(gen_dir, exist_ok=True)
        designs_file = os.path.join(gen_dir, "designs.json")

        if os.path.exists(designs_file):
            # resuming: reuse the stored candidate set (designs + warm starts);
            # ng_cands are re-created so their scores can be told to the optimizer
            with open(designs_file) as f:
                candidates = json.load(f)
            ng_cands = [optimizer.parametrization.spawn_child(new_value=np.array(c["design"]))
                        for c in candidates]
            print(f"[evolve-ng] gen {gen}: reusing existing {designs_file}")
        else:
            candidates, ng_cands = [], []
            for i in range(args.population):
                cand = optimizer.ask()
                design = [round(float(x), 4) for x in cand.value]
                checkpoint = None if args.no_warm_start else nearest_checkpoint(design, archive)
                candidates.append({"id": f"cand_{i:02d}", "design": design, "checkpoint": checkpoint})
                ng_cands.append(cand)
            with open(designs_file, "w") as f:
                json.dump(candidates, f, indent=2)

        print(f"\n[evolve-ng] === generation {gen} ({len(candidates)} candidates, {args.optimizer}) ===")
        run_generation(candidates, gen_dir)
        results = collect_results(gen_dir, candidates)
        if any(r["score"] == float("-inf") for r in results):
            # a worker may have crashed; one retry per unfinished candidate
            print("[evolve-ng] retrying unfinished candidate(s)")
            run_generation(candidates, gen_dir)
            results = collect_results(gen_dir, candidates)

        for ng_cand, result in zip(ng_cands, results):
            score = result["score"]
            optimizer.tell(ng_cand, -score if math.isfinite(score) else FAILED_LOSS)
            if math.isfinite(score) and result.get("checkpoint"):
                archive.append({"design": result["design"], "score": score,
                                "checkpoint": result["checkpoint"]})

        results.sort(key=lambda r: r["score"], reverse=True)
        for r in results:
            writer.writerow([gen, r["id"], f"{r['score']:.4f}", f"{r.get('distance_x_m', float('nan')):.4f}"]
                            + [f"{x:.4f}" for x in r["design"]])
        history.flush()

        top = results[0]
        if best is None or top["score"] > best["score"]:
            best = {"generation": gen, **{k: top[k] for k in ("id", "design", "score", "distance_x_m", "checkpoint")}}
            with open(os.path.join(run_dir, "best.json"), "w") as f:
                json.dump(best, f, indent=2)
        print(f"[evolve-ng] gen {gen} best: {top['id']} score={top['score']:.2f} "
              f"dist={top.get('distance_x_m', float('nan')):.2f} m design={top['design']}")

    history.close()
    recommendation = [round(float(x), 4) for x in optimizer.provide_recommendation().value]
    print("\n[evolve-ng] done.")
    print(f"[evolve-ng] best evaluated (gen {best['generation']}): score={best['score']:.2f} design={best['design']}")
    print(f"[evolve-ng] optimizer recommendation (may be unevaluated): {recommendation}")
    print(f"[evolve-ng] history: {history_path}")
    design_str = " ".join(str(x) for x in best["design"])
    print("[evolve-ng] watch the best evaluated design with:")
    print(f"    python scripts/coadapt/play_design.py --design {design_str} --checkpoint {best['checkpoint']}")


if __name__ == "__main__":
    main()
