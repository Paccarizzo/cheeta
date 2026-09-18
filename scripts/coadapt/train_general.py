"""Train the general (design-conditioned) half-cheetah policy.

The policy observes the design vector, so one network serves every morphology
(the paper's population network). Training runs in phases; each phase is one
worker process with a fresh batch of random designs spread over the parallel
envs (env i gets design i % N), warm-started from the previous phase. The
latest checkpoint is copied to <run-dir>/general.pt after every phase.

This driver needs no Isaac Sim itself; run it with the env_isaaclab python:
    python scripts/coadapt/train_general.py --phases 4 --designs-per-phase 64 ^
        --num_envs 512 --iterations 300 --wandb

Interrupted runs resume with --run-dir pointing at the old run (finished
phases are skipped).
"""

import argparse
import os
import shutil

from _common import (REPO_ROOT, add_wandb_args, load_result, run_worker, save_json, timestamp,
                     wandb_init_driver, wandb_worker_args, worker_command)

parser = argparse.ArgumentParser(description="Train a design-conditioned half-cheetah policy across random designs.")
parser.add_argument("--phases", type=int, default=4, help="training phases (fresh design batch each)")
parser.add_argument("--designs-per-phase", type=int, default=64, help="distinct designs per phase")
parser.add_argument("--num_envs", type=int, default=512, help="parallel envs (spread over the designs)")
parser.add_argument("--iterations", type=int, default=300, help="PPO iterations per phase")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--checkpoint", type=str, default=None, help="continue from this checkpoint")
parser.add_argument("--run-dir", type=str, default=None, help="run directory (set to an old run to resume)")
parser.add_argument("--device", type=str, default=None, help="forwarded to the worker, e.g. cuda:0")
add_wandb_args(parser)
args = parser.parse_args()


def main():
    run_dir = os.path.abspath(args.run_dir or os.path.join(REPO_ROOT, "logs", "coadapt", "general", timestamp()))
    os.makedirs(run_dir, exist_ok=True)
    run_name = os.path.basename(os.path.normpath(run_dir))
    group = args.wandb_group or f"general-{run_name}"
    save_json(vars(args), os.path.join(run_dir, "config.json"))
    wb = wandb_init_driver(args, name=f"{run_name}-driver", group=group, job_type="general-driver",
                           config=vars(args))

    checkpoint = os.path.abspath(args.checkpoint) if args.checkpoint else None
    general_path = os.path.join(run_dir, "general.pt")
    for phase in range(args.phases):
        phase_dir = os.path.join(run_dir, f"phase_{phase:02d}")
        label = f"phase {phase}"
        if load_result(phase_dir) is None:
            cmd = worker_command(
                phase_dir,
                num_envs=args.num_envs,
                iterations=args.iterations,
                seed=args.seed + phase,
                random_designs=args.designs_per_phase,
                design_seed=args.seed * 1000 + phase,
                checkpoint=checkpoint,
                device=args.device,
                extra=wandb_worker_args(args, name=f"{run_name}-phase{phase:02d}", group=group,
                                        job_type="general-train", tags=[f"phase{phase}"]),
            )
            print(f"[general] {label}: {args.designs_per_phase} designs, {args.iterations} iterations"
                  + (f" (warm start: {checkpoint})" if checkpoint else ""))
            run_worker(cmd, label)
        result = load_result(phase_dir)
        if result is None:
            print(f"[general] {label} failed twice, stopping")
            break
        checkpoint = result["checkpoint"]
        shutil.copyfile(checkpoint, general_path)
        scores = [d["score"] for d in result["per_design"]]
        print(f"[general] {label}: mean score {result['score']:.2f} over {len(scores)} designs "
              f"(min {min(scores):.2f}, max {max(scores):.2f}, dist {result['distance_x_m']:.2f} m)")
        if wb is not None:
            import wandb

            log = {"phase": phase, "score_mean": result["score"], "score_min": min(scores), "score_max": max(scores),
                   "distance_x_m": result["distance_x_m"], "train_time_s": result["train_time_s"]}
            if result.get("probe"):
                log["zero_shot_score_before_phase"] = result["probe"]["zero_shot_score"]
            if result.get("train_metrics"):
                log.update({f"train_metrics/{k}": v for k, v in result["train_metrics"].items()})
            wandb.log(log, step=phase)
    save_json({"checkpoint": general_path, "phases": args.phases, "source": checkpoint},
              os.path.join(run_dir, "general.json"))
    print(f"\n[general] done. general policy: {general_path}")
    print("[general] screen / evolve designs with it:")
    print(f"    python scripts/coadapt/evolve.py --general-checkpoint {general_path}")
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
