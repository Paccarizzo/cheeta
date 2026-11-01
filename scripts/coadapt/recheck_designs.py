"""Seed check of an evolve.py run: retrain its best designs with new seeds.

Every candidate of an evolution run is trained once, with one seed, so the
winner may be lucky. This driver takes the --top best distinct trained designs
of a run and retrains each with every seed in --seeds, using the run's own
budget (num_envs, iterations, inherit=scratch). With --identity the unscaled
cheetah is trained too, with the run's seed and every --seeds seed, as the
reference the evolved designs should beat.

    python scripts/coadapt/recheck_designs.py --run-dir logs/coadapt/<run> --top 3 --seeds 43 44 --identity

With another budget (--iterations, --random-episode-start) the run's own
scores are not comparable, so every design is retrained with the run's seed
too; use --out-name to keep it apart from the plain recheck, e.g. a
convergence check:

    python scripts/coadapt/recheck_designs.py --run-dir logs/coadapt/<run> --identity ^
        --iterations 800 --random-episode-start --out-name converge_800

Output: <run-dir>/<out-name>/<label>_s<seed>/result.json per job and
<run-dir>/<out-name>/recheck.csv (one row per design and seed; with the run's
budget the run's original score is the run seed's row). Finished jobs are skipped.
"""

import argparse
import csv
import os

from _common import designs, load_json, load_result, run_worker, save_json, worker_command

parser = argparse.ArgumentParser(description="Retrain the best designs of an evolve.py run with new seeds.")
parser.add_argument("--run-dir", type=str, required=True, help="evolve.py run directory")
parser.add_argument("--top", type=int, default=3, help="number of best distinct designs to retrain")
parser.add_argument("--seeds", type=int, nargs="+", default=[43, 44], help="new training seeds")
parser.add_argument("--identity", action="store_true", help="also train the unscaled cheetah")
parser.add_argument("--iterations", type=int, default=None, help="PPO iterations (default: the run's)")
parser.add_argument("--random-episode-start", action="store_true", help="forwarded to the worker")
parser.add_argument("--out-name", type=str, default="recheck", help="output subdirectory of the run")
parser.add_argument("--device", type=str, default=None, help="forwarded to the worker, e.g. cuda:0")
args = parser.parse_args()


def best_designs(history_path: str, n: int) -> list[dict]:
    """The n best distinct trained designs of the run, with where they were first trained."""
    with open(history_path) as f:
        rows = [r for r in csv.DictReader(f) if r["trained"] == "1" and r["role"] != "elite"]
    rows.sort(key=lambda r: float(r["score"]), reverse=True)
    picked, seen = [], set()
    for r in rows:
        design = [float(r[name]) for name in designs.DESIGN_NAMES]
        if tuple(design) in seen:
            continue
        seen.add(tuple(design))
        picked.append({"label": f"top{len(picked) + 1}_g{int(r['generation']):02d}{r['id'].replace('cand_', 'c')}",
                       "design": design, "score": float(r["score"]), "distance_x_m": float(r["distance_x_m"])})
        if len(picked) == n:
            break
    return picked


def main():
    run_dir = os.path.abspath(args.run_dir)
    config = load_json(os.path.join(run_dir, "config.json"))
    out_root = os.path.join(run_dir, args.out_name)
    iterations = args.iterations or config["iterations"]
    # the run trained with synchronised episode starts
    same_budget = iterations == config["iterations"] and not args.random_episode_start
    seeds = list(dict.fromkeys([config["seed"], *args.seeds]))
    targets = best_designs(os.path.join(run_dir, "history.csv"), args.top)
    jobs = [(t, seed) for t in targets for seed in seeds if not (same_budget and seed == config["seed"])]
    if args.identity:
        identity = {"label": "identity", "design": list(designs.IDENTITY_DESIGN), "score": None}
        jobs += [(identity, seed) for seed in seeds]
    save_json({"targets": targets, "seeds": args.seeds, "identity": args.identity, "iterations": iterations,
               "num_envs": config["num_envs"], "random_episode_start": args.random_episode_start},
              os.path.join(out_root, "config.json"))

    for target, seed in jobs:
        label = f"{target['label']}_s{seed}"
        cmd = worker_command(os.path.join(out_root, label), num_envs=config["num_envs"],
                             iterations=iterations, seed=seed, design=target["design"],
                             device=args.device, random_episode_start=args.random_episode_start)
        print(f"[recheck] {label}: design {target['design']}")
        run_worker(cmd, label)

    rows = []
    for target in targets if same_budget else []:
        rows.append({"label": target["label"], "seed": config["seed"], "source": "evolution",
                     "score": target["score"], "distance_x_m": target["distance_x_m"], "design": target["design"]})
    for target, seed in jobs:
        result = load_result(os.path.join(out_root, f"{target['label']}_s{seed}"))
        if result is None:
            print(f"[recheck] WARNING: no result for {target['label']} seed {seed}")
            continue
        rows.append({"label": target["label"], "seed": seed, "source": "recheck", "score": result["score"],
                     "distance_x_m": result["distance_x_m"], "design": target["design"]})
    csv_path = os.path.join(out_root, "recheck.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "seed", "source", "score", "distance_x_m", *designs.DESIGN_NAMES])
        for r in rows:
            writer.writerow([r["label"], r["seed"], r["source"], f"{r['score']:.4f}", f"{r['distance_x_m']:.4f}",
                             *[f"{x:.4f}" for x in r["design"]]])

    for label in dict.fromkeys(r["label"] for r in rows):
        scores = [r["score"] for r in rows if r["label"] == label]
        mean = sum(scores) / len(scores)
        print(f"[recheck] {label:16s} mean {mean:7.2f} over {len(scores)} seeds "
              f"(min {min(scores):.2f}, max {max(scores):.2f})")
    print(f"[recheck] table: {csv_path}")


if __name__ == "__main__":
    main()
