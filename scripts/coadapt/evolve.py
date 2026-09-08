"""Evolutionary co-adaptation of half-cheetah morphology and PPO policy.

Loop per generation:
  1. every candidate design is trained with skrl PPO and scored by
     scripts/coadapt/train_design.py (one Isaac Sim session per candidate;
     the scene cannot be rebuilt reliably within one session)
  2. the top --elites designs survive; children are Gaussian mutations of the
     elites, clipped to the paper's design bounds [0.8, 2.0]; --immigrants
     fresh random designs keep exploration alive
  3. children warm-start their policy from the parent's checkpoint and elites
     keep training their own policy, so design and behaviour improve together

This driver needs no Isaac Sim itself; run it with the env_isaaclab python so
the workers inherit the right interpreter:
    python scripts/coadapt/evolve.py --generations 10 --population 8 ^
        --num_envs 256 --iterations 100

Interrupted runs resume: re-run with --run-dir pointing at the old run and
finished candidates are skipped (their result.json is reused).
"""

import argparse
import csv
import json
import os
import random
import subprocess
import sys
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKER = os.path.join(REPO_ROOT, "scripts", "coadapt", "train_design.py")
DESIGN_BOUNDS = (0.8, 2.0)  # paper Sec. 7.1
DESIGN_DIM = 6

parser = argparse.ArgumentParser(description="Evolve half-cheetah designs with per-design PPO training.")
parser.add_argument("--generations", type=int, default=10)
parser.add_argument("--population", type=int, default=8, help="candidates per generation")
parser.add_argument("--elites", type=int, default=2, help="top designs kept unchanged")
parser.add_argument("--immigrants", type=int, default=1, help="fresh random designs per generation")
parser.add_argument("--sigma", type=float, default=0.15, help="mutation std dev on the scale factors")
parser.add_argument("--sigma-decay", type=float, default=0.95, help="sigma multiplier per generation")
parser.add_argument("--num_envs", type=int, default=256, help="parallel envs per candidate")
parser.add_argument("--iterations", type=int, default=100, help="PPO iterations per candidate per generation")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run-dir", type=str, default=None, help="run directory (set to an old run to resume)")
parser.add_argument("--device", type=str, default=None, help="forwarded to the worker, e.g. cuda:0")
args = parser.parse_args()


def random_design(rng: random.Random) -> list[float]:
    lo, hi = DESIGN_BOUNDS
    return [round(rng.uniform(lo, hi), 4) for _ in range(DESIGN_DIM)]


def mutate(design: list[float], sigma: float, rng: random.Random) -> list[float]:
    lo, hi = DESIGN_BOUNDS
    return [round(min(hi, max(lo, xi + rng.gauss(0.0, sigma))), 4) for xi in design]


def run_generation(candidates: list[dict], gen_dir: str) -> None:
    """One worker process per candidate; finished candidates are skipped."""
    for cand in candidates:
        out_dir = os.path.join(gen_dir, cand["id"])
        if os.path.exists(os.path.join(out_dir, "result.json")):
            print(f"[evolve] {cand['id']}: already done, skipping")
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
        print(f"[evolve] {cand['id']}: training design {cand['design']}"
              + (f" (warm start: {cand['checkpoint']})" if cand.get("checkpoint") else ""))
        proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
        if proc.returncode != 0:
            print(f"[evolve] WARNING: worker for {cand['id']} exited with code {proc.returncode}")


def collect_results(gen_dir: str, candidates: list[dict]) -> list[dict]:
    results = []
    for cand in candidates:
        path = os.path.join(gen_dir, cand["id"], "result.json")
        if os.path.exists(path):
            with open(path) as f:
                result = json.load(f)
            result["id"] = cand["id"]
        else:
            print(f"[evolve] WARNING: no result for {cand['id']}, scoring -inf")
            result = {"id": cand["id"], "design": cand["design"], "score": float("-inf"),
                      "distance_x_m": float("nan"), "checkpoint": cand.get("checkpoint")}
        results.append(result)
    return results


def main():
    rng = random.Random(args.seed)
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

    survivors: list[dict] = []  # [{design, score, checkpoint}]
    sigma = args.sigma
    best: dict | None = None

    for gen in range(args.generations):
        gen_dir = os.path.join(run_dir, f"gen_{gen:02d}")
        os.makedirs(gen_dir, exist_ok=True)
        designs_file = os.path.join(gen_dir, "designs.json")

        if os.path.exists(designs_file):
            # resuming an interrupted run: reuse the exact candidate set
            with open(designs_file) as f:
                candidates = json.load(f)
            print(f"[evolve] gen {gen}: reusing existing {designs_file}")
        else:
            candidates = []
            if gen == 0:
                # identity design + random population
                candidates.append({"id": "cand_00", "design": [1.0] * DESIGN_DIM, "checkpoint": None})
                while len(candidates) < args.population:
                    candidates.append({"id": f"cand_{len(candidates):02d}",
                                       "design": random_design(rng), "checkpoint": None})
            else:
                # elites keep their design and continue training their own policy
                for elite in survivors[: args.elites]:
                    candidates.append({"id": f"cand_{len(candidates):02d}",
                                       "design": elite["design"], "checkpoint": elite["checkpoint"]})
                # immigrants: fresh design, fresh policy
                for _ in range(args.immigrants):
                    if len(candidates) < args.population:
                        candidates.append({"id": f"cand_{len(candidates):02d}",
                                           "design": random_design(rng), "checkpoint": None})
                # children: mutated elite design, policy warm-started from the parent
                i = 0
                while len(candidates) < args.population:
                    parent = survivors[i % min(args.elites, len(survivors))]
                    candidates.append({"id": f"cand_{len(candidates):02d}",
                                       "design": mutate(parent["design"], sigma, rng),
                                       "checkpoint": parent["checkpoint"]})
                    i += 1
            with open(designs_file, "w") as f:
                json.dump(candidates, f, indent=2)

        print(f"\n[evolve] === generation {gen} ({len(candidates)} candidates, sigma={sigma:.3f}) ===")
        run_generation(candidates, gen_dir)
        results = collect_results(gen_dir, candidates)
        missing = [r for r in results if r["score"] == float("-inf")]
        if missing:
            # a worker may have crashed; one retry per unfinished candidate
            print(f"[evolve] retrying {len(missing)} unfinished candidate(s)")
            run_generation(candidates, gen_dir)
            results = collect_results(gen_dir, candidates)

        results.sort(key=lambda r: r["score"], reverse=True)
        survivors = results
        for r in results:
            writer.writerow([gen, r["id"], f"{r['score']:.4f}", f"{r.get('distance_x_m', float('nan')):.4f}"]
                            + [f"{x:.4f}" for x in r["design"]])
        history.flush()

        top = results[0]
        if best is None or top["score"] > best["score"]:
            best = {"generation": gen, **{k: top[k] for k in ("id", "design", "score", "distance_x_m", "checkpoint")}}
            with open(os.path.join(run_dir, "best.json"), "w") as f:
                json.dump(best, f, indent=2)
        print(f"[evolve] gen {gen} best: {top['id']} score={top['score']:.2f} "
              f"dist={top.get('distance_x_m', float('nan')):.2f} m design={top['design']}")
        sigma *= args.sigma_decay

    history.close()
    print("\n[evolve] done.")
    print(f"[evolve] best over run (gen {best['generation']}): score={best['score']:.2f} design={best['design']}")
    print(f"[evolve] history: {history_path}")
    design_str = " ".join(str(x) for x in best["design"])
    print("[evolve] watch it with:")
    print(f"    python scripts/coadapt/play_design.py --design {design_str} --checkpoint {best['checkpoint']}")


if __name__ == "__main__":
    main()
