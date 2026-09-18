"""Which cheap RL metrics predict a design's final score?

For N random designs: fine-tune the general policy on each design for
--iterations PPO iterations (the "ground truth" score), while recording the
cheap signals available early:
  * zero_shot_score, value_s0 - from the general policy before training
  * reward_step_at_{10,25,50}pct, reward_step_slope, reward_step_gain,
    value_loss_mean, policy_std_last, learning_rate_min, ... - from the
    learning curve (see MetricRecorder in train_design.py)

Then every metric is rank-correlated (Spearman) with the final score. The
first --repeat designs are trained a second time with another seed, which
gives the noise floor: how well the final score predicts *itself* across
seeds. A metric close to that floor is as good as it gets.

Proof of concept (~15 min):
    python scripts/coadapt/benchmark_metrics.py --general-checkpoint <general.pt> ^
        --designs 6 --iterations 60 --repeat 2 --num_envs 256 --wandb
Overnight:
    python scripts/coadapt/benchmark_metrics.py --general-checkpoint <general.pt> ^
        --designs 40 --iterations 400 --repeat 6 --num_envs 256 --wandb

--analyze-only recomputes the tables from an existing --run-dir.
"""

import argparse
import csv
import math
import os

from _common import (REPO_ROOT, add_wandb_args, designs, load_result, pearson, run_worker, save_json, spearman,
                     timestamp, wandb_init_driver, wandb_worker_args, worker_command)

parser = argparse.ArgumentParser(description="Correlate cheap RL metrics with the final design score.")
parser.add_argument("--general-checkpoint", type=str, required=True, help="design-conditioned policy (warm start)")
parser.add_argument("--designs", type=int, default=8, help="number of random designs")
parser.add_argument("--repeat", type=int, default=2, help="designs also trained with a second seed (noise floor)")
parser.add_argument("--iterations", type=int, default=200, help="PPO iterations per design (ground truth budget)")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--run-dir", type=str, default=None)
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--analyze-only", action="store_true", help="only recompute the correlation tables")
add_wandb_args(parser)
args = parser.parse_args()

PROBE_METRICS = ["zero_shot_score", "value_s0"]


def job_rows(run_dir: str, design_list: list[list[float]]) -> list[dict]:
    rows = []
    for i, design in enumerate(design_list):
        for rep in range(2 if i < args.repeat else 1):
            out_dir = os.path.join(run_dir, f"design_{i:02d}" + ("" if rep == 0 else f"_rep{rep}"))
            rows.append({"design_index": i, "rep": rep, "design": design, "out_dir": out_dir,
                         "seed": args.seed + rep})
    return rows


def run_jobs(rows: list[dict], run_name: str, group: str) -> None:
    for job in rows:
        label = os.path.basename(job["out_dir"])
        cmd = worker_command(job["out_dir"], num_envs=args.num_envs, iterations=args.iterations, seed=job["seed"],
                             design=job["design"], checkpoint=args.general_checkpoint, device=args.device,
                             extra=wandb_worker_args(args, name=f"{run_name}-{label}", group=group,
                                                     job_type="benchmark", tags=[f"rep{job['rep']}"]))
        print(f"[bench] {label}: design {job['design']} seed {job['seed']}")
        run_worker(cmd, label)


def analyze(rows: list[dict], run_dir: str, wb) -> None:
    records = []
    for job in rows:
        result = load_result(job["out_dir"])
        if result is None:
            print(f"[bench] WARNING: missing result for {job['out_dir']}")
            continue
        rec = {"id": os.path.basename(job["out_dir"]), "design_index": job["design_index"], "rep": job["rep"],
               "final_score": result["score"], "final_distance_x_m": result["distance_x_m"]}
        probe = result.get("probe") or {}
        for m in PROBE_METRICS:
            rec[m] = probe.get(m, float("nan"))
        for k, v in (result.get("train_metrics") or {}).items():
            rec[k] = v
        for name, x in zip(designs.DESIGN_NAMES, job["design"]):
            rec[f"design_{name}"] = x
        records.append(rec)
    if not records:
        print("[bench] nothing to analyze")
        return

    primary = [r for r in records if r["rep"] == 0]
    metric_names = [k for k in records[0] if k not in ("id", "design_index", "rep", "final_score", "final_distance_x_m")]
    final = [r["final_score"] for r in primary]
    correlations = []
    for m in metric_names:
        vals = [r.get(m, float("nan")) for r in primary]
        rho, pr = spearman(vals, final), pearson(vals, final)
        correlations.append({"metric": m, "spearman": rho, "pearson": pr, "n": len(primary)})
    correlations.sort(key=lambda c: -abs(c["spearman"]) if not math.isnan(c["spearman"]) else 1.0)

    # noise floor: final score of seed A vs seed B on the same designs
    reps = {r["design_index"]: r for r in records if r["rep"] == 1}
    pairs = [(r["final_score"], reps[r["design_index"]]["final_score"]) for r in primary if r["design_index"] in reps]
    noise = None
    if len(pairs) >= 3:
        a, b = [p[0] for p in pairs], [p[1] for p in pairs]
        noise = {"n_pairs": len(pairs), "spearman_seed_vs_seed": spearman(a, b),
                 "mean_abs_diff": sum(abs(x - y) for x, y in pairs) / len(pairs),
                 "mean_score": sum(a + b) / (2 * len(pairs))}
    elif pairs:
        a, b = [p[0] for p in pairs], [p[1] for p in pairs]
        noise = {"n_pairs": len(pairs), "spearman_seed_vs_seed": float("nan"),
                 "mean_abs_diff": sum(abs(x - y) for x, y in pairs) / len(pairs),
                 "mean_score": sum(a + b) / (2 * len(pairs))}

    # write tables
    with open(os.path.join(run_dir, "table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    with open(os.path.join(run_dir, "correlations.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "spearman", "pearson", "n"])
        w.writeheader()
        w.writerows(correlations)
    save_json({"noise_floor": noise, "correlations": correlations}, os.path.join(run_dir, "summary.json"))

    print(f"\n[bench] {len(primary)} designs, {args.iterations} iterations each; final score "
          f"min {min(final):.1f} max {max(final):.1f}")
    print(f"[bench] {'metric':32s} {'spearman':>9s} {'pearson':>9s}")
    for c in correlations:
        print(f"[bench] {c['metric']:32s} {c['spearman']:9.2f} {c['pearson']:9.2f}")
    if noise:
        print(f"[bench] noise floor over {noise['n_pairs']} repeated designs: spearman(seed vs seed)="
              f"{noise['spearman_seed_vs_seed']:.2f}, mean |score diff|={noise['mean_abs_diff']:.1f} "
              f"(mean score {noise['mean_score']:.1f})")
    print(f"[bench] tables: {os.path.join(run_dir, 'correlations.csv')}, {os.path.join(run_dir, 'table.csv')}")

    if wb is not None:
        import wandb

        table = wandb.Table(columns=list(records[0].keys()))
        for r in records:
            table.add_data(*[r.get(k) for k in records[0].keys()])
        corr_table = wandb.Table(columns=["metric", "spearman", "pearson"])
        for c in correlations:
            corr_table.add_data(c["metric"], c["spearman"], c["pearson"])
        log = {"designs": table, "correlations": corr_table,
               "spearman_bar": wandb.plot.bar(corr_table, "metric", "spearman", title="Spearman vs final score")}
        log.update({f"spearman/{c['metric']}": c["spearman"] for c in correlations})
        if noise:
            log.update({f"noise/{k}": v for k, v in noise.items()})
        wandb.log(log)


def main():
    run_dir = args.run_dir or os.path.join(REPO_ROOT, "logs", "coadapt", "benchmark", timestamp())
    os.makedirs(run_dir, exist_ok=True)
    run_name = os.path.basename(os.path.normpath(run_dir))
    group = args.wandb_group or f"benchmark-{run_name}"
    args.general_checkpoint = os.path.abspath(args.general_checkpoint)
    design_file = os.path.join(run_dir, "designs.json")
    if os.path.exists(design_file):
        from _common import load_json

        design_list = load_json(design_file)
    else:
        design_list = designs.sample_designs(args.designs, args.seed, include_identity=True)
        save_json(design_list, design_file)
    save_json(vars(args), os.path.join(run_dir, "config.json"))
    rows = job_rows(run_dir, design_list)
    wb = wandb_init_driver(args, name=f"{run_name}-driver", group=group, job_type="benchmark-driver", config=vars(args))
    if not args.analyze_only:
        run_jobs(rows, run_name, group)
    analyze(rows, run_dir, wb)
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
