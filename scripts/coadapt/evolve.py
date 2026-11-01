"""Evolutionary co-adaptation of half-cheetah morphology and PPO policy.

Loop per generation:
  1. propose candidates: the top --elites designs survive, --immigrants fresh
     random designs are added, the rest are Gaussian mutations of elites
     (clipped to the paper's design bounds [0.8, 2.0])
  2. screen (needs --general-checkpoint): every candidate is scored zero-shot
     by the general, design-conditioned policy in ONE worker process (all
     designs in one scene, no training); the critic's value at the initial
     state V(s0) is recorded as well
  3. specialise: elites keep training their own policy (--elite-mode train) or
     carry their score unchanged (--elite-mode keep); of the new candidates
     the --train-top best by --select-by are fine-tuned, each in its own
     worker process, warm-started from the parent's policy (--inherit parent),
     from the general policy (--inherit general) or from scratch (--inherit scratch)
  4. the fine-tuned score is the fitness; screened-only candidates keep their
     zero-shot score and cannot become elites

Without --general-checkpoint every candidate is trained (the original loop).
For scores that are comparable across generations, use --inherit scratch
--elite-mode keep: every design gets the same training budget from a fresh
policy, so a rising best score means better designs, not longer training.
Screening metrics of the trained candidates are compared with their trained
scores every generation (Spearman rank correlation, logged to history/wandb),
which is the cheapest test of how useful a selection metric is.

Run with the env_isaaclab python (workers inherit the interpreter):
    python scripts/coadapt/evolve.py --general-checkpoint logs/coadapt/general/<run>/general.pt ^
        --generations 10 --population 12 --train-top 3 --num_envs 256 --iterations 100 --wandb

Interrupted runs resume: re-run with --run-dir pointing at the old run and
finished jobs are skipped (their result.json is reused).
"""

import argparse
import csv
import math
import os
import random

from _common import (REPO_ROOT, add_wandb_args, designs, load_json, load_result, run_worker, save_json,
                     spearman, timestamp, wandb_init_driver, wandb_worker_args, worker_command)

SELECT_METRICS = ["zero_shot", "value_s0", "random", "none"]

parser = argparse.ArgumentParser(description="Evolve half-cheetah designs with zero-shot screening + per-design PPO.")
parser.add_argument("--generations", type=int, default=10)
parser.add_argument("--population", type=int, default=8, help="candidates per generation (incl. elites)")
parser.add_argument("--elites", type=int, default=2, help="top trained designs kept unchanged")
parser.add_argument("--immigrants", type=int, default=1, help="fresh random designs per generation")
parser.add_argument("--sigma", type=float, default=0.15, help="mutation std dev on the scale factors")
parser.add_argument("--sigma-decay", type=float, default=0.95, help="sigma multiplier per generation")
parser.add_argument("--general-checkpoint", type=str, default=None,
                    help="design-conditioned policy used for zero-shot screening and as default warm start")
parser.add_argument("--train-top", type=int, default=None,
                    help="new candidates fine-tuned per generation (default: all); elites always continue")
parser.add_argument("--select-by", type=str, default="zero_shot", choices=SELECT_METRICS,
                    help="screening metric that picks the candidates to fine-tune")
parser.add_argument("--inherit", type=str, default="parent", choices=["parent", "general", "scratch"],
                    help="warm start of children: parent's specialised policy, the general policy or none")
parser.add_argument("--elite-mode", type=str, default="train", choices=["train", "keep"],
                    help="elites continue training their policy (train) or keep their score without training (keep)")
parser.add_argument("--no-identity", action="store_true",
                    help="generation 0 is all random designs (default: the first one is the unscaled cheetah)")
parser.add_argument("--screen-envs", type=int, default=None, help="envs for screening (default: num_envs)")
parser.add_argument("--num_envs", type=int, default=256, help="parallel envs per fine-tuned candidate")
parser.add_argument("--iterations", type=int, default=100, help="PPO iterations per candidate per generation")
parser.add_argument("--no-probe", action="store_true", help="skip the per-candidate probe before fine-tuning")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run-dir", type=str, default=None, help="run directory (set to an old run to resume)")
parser.add_argument("--device", type=str, default=None, help="forwarded to the worker, e.g. cuda:0")
add_wandb_args(parser)
args = parser.parse_args()

HISTORY_COLUMNS = ["generation", "id", "role", "parent", "trained", "score", "zero_shot_score", "value_s0",
                   "distance_x_m", "reward_step_slope", *designs.DESIGN_NAMES]


def propose(gen: int, survivors: list[dict], rng: random.Random, sigma: float) -> list[dict]:
    """Candidate set of a generation. ``checkpoint`` is the warm start (None = general policy)."""
    cands = []

    def add(design, role, parent=None, checkpoint=None):
        cands.append({"id": f"cand_{len(cands):02d}", "design": list(design), "role": role,
                      "parent": parent, "checkpoint": checkpoint})

    if gen == 0 or not survivors:
        if not args.no_identity:
            add(designs.IDENTITY_DESIGN, "seed")
        while len(cands) < args.population:
            add(designs.random_design(rng), "immigrant")
        return cands
    elites = survivors[: args.elites]
    for elite in elites:
        add(elite["design"], "elite", parent=elite["id"], checkpoint=elite["checkpoint"])
        if args.elite_mode == "keep":
            # stored in designs.json, so a resumed run carries the same score
            cands[-1]["kept"] = {k: elite.get(k) for k in ("score", "distance_x_m", "checkpoint", "train_metrics")}
    for _ in range(args.immigrants):
        if len(cands) < args.population:
            add(designs.random_design(rng), "immigrant")
    i = 0
    while len(cands) < args.population:
        parent = elites[i % len(elites)]
        add(designs.mutate(parent["design"], sigma, rng), "child", parent=parent["id"],
            checkpoint=parent["checkpoint"] if args.inherit == "parent" else None)
        i += 1
    return cands


def screen(candidates: list[dict], gen_dir: str, gen: int, run_name: str, group: str) -> None:
    """Zero-shot scores + V(s0) of all candidates with the general policy (one process)."""
    screen_dir = os.path.join(gen_dir, "screen")
    cmd = worker_command(
        screen_dir,
        num_envs=args.screen_envs or args.num_envs,
        iterations=0,
        seed=args.seed,
        design_set=[c["design"] for c in candidates],
        checkpoint=args.general_checkpoint,
        device=args.device,
        extra=wandb_worker_args(args, name=f"{run_name}-g{gen:02d}-screen", group=group, job_type="screen"),
    )
    print(f"[evolve] gen {gen}: screening {len(candidates)} designs with the general policy")
    run_worker(cmd, f"gen {gen} screen")
    result = load_result(screen_dir)
    if result is None:
        print("[evolve] WARNING: screening failed; candidates get no zero-shot score")
        return
    for cand, row in zip(candidates, result["per_design"]):
        cand["zero_shot_score"] = row["score"]
        cand["zero_shot_distance_x_m"] = row["distance_x_m"]
        cand["value_s0"] = row.get("value_s0")


def choose_training_set(candidates: list[dict], rng: random.Random) -> list[dict]:
    """Elites train (unless kept); the best --train-top new candidates by --select-by join them."""
    elites = [c for c in candidates if c["role"] == "elite" and not c.get("kept")]
    new = [c for c in candidates if c["role"] != "elite"]
    if args.train_top is None or args.train_top >= len(new) or args.select_by == "none":
        return elites + new
    if args.select_by == "random":
        picked = rng.sample(new, args.train_top)
    else:
        key = "zero_shot_score" if args.select_by == "zero_shot" else "value_s0"
        ranked = sorted(new, key=lambda c: c.get(key) if c.get(key) is not None else -math.inf, reverse=True)
        picked = ranked[: args.train_top]
    return elites + picked


def train(candidates: list[dict], gen_dir: str, gen: int, run_name: str, group: str) -> None:
    for cand in candidates:
        out_dir = os.path.join(gen_dir, cand["id"])
        checkpoint = cand.get("checkpoint") or (args.general_checkpoint if args.inherit != "scratch" else None)
        extra = list(wandb_worker_args(args, name=f"{run_name}-g{gen:02d}-{cand['id']}", group=group,
                                       job_type="candidate", tags=[cand["role"], f"gen{gen}"]))
        if args.no_probe:
            extra.append("--no-probe")
        cmd = worker_command(out_dir, num_envs=args.num_envs, iterations=args.iterations, seed=args.seed,
                             design=cand["design"], checkpoint=checkpoint, device=args.device, extra=extra)
        print(f"[evolve] {cand['id']} ({cand['role']}): training design {cand['design']}"
              + (f" (warm start: {checkpoint})" if checkpoint else ""))
        run_worker(cmd, cand["id"])


def collect(candidates: list[dict], trained: set[str], gen_dir: str) -> list[dict]:
    rows = []
    for cand in candidates:
        row = {**cand, "trained": False, "score": cand.get("zero_shot_score", -math.inf),
               "distance_x_m": cand.get("zero_shot_distance_x_m", float("nan")), "checkpoint": None,
               "train_metrics": None}
        if cand.get("kept"):
            row.update({"trained": True, **cand["kept"]})
        elif cand["id"] in trained:
            result = load_result(os.path.join(gen_dir, cand["id"]))
            if result is None:
                print(f"[evolve] WARNING: no result for {cand['id']}, keeping its screening score")
            else:
                row.update({"trained": True, "score": result["score"], "distance_x_m": result["distance_x_m"],
                            "checkpoint": result["checkpoint"], "train_metrics": result.get("train_metrics")})
                probe = result.get("probe")
                if probe and row.get("zero_shot_score") is None:
                    row["zero_shot_score"], row["value_s0"] = probe["zero_shot_score"], probe["value_s0"]
        rows.append(row)
    return rows


def history_row(gen: int, r: dict) -> list:
    tm = r.get("train_metrics") or {}
    return [gen, r["id"], r["role"], r.get("parent") or "", int(r["trained"]), f"{r['score']:.4f}",
            _fmt(r.get("zero_shot_score")), _fmt(r.get("value_s0")), _fmt(r.get("distance_x_m")),
            _fmt(tm.get("reward_step_slope"), 6)] + [f"{x:.4f}" for x in r["design"]]


def _fmt(v, digits: int = 4) -> str:
    return "" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{digits}f}"


def main():
    rng = random.Random(args.seed)
    run_dir = args.run_dir or os.path.join(REPO_ROOT, "logs", "coadapt", timestamp())
    os.makedirs(run_dir, exist_ok=True)
    run_name = os.path.basename(os.path.normpath(run_dir))
    group = args.wandb_group or f"evolve-{run_name}"
    if args.general_checkpoint:
        args.general_checkpoint = os.path.abspath(args.general_checkpoint)
    save_json(vars(args), os.path.join(run_dir, "config.json"))
    wb = wandb_init_driver(args, name=f"{run_name}-driver", group=group, job_type="evolve-driver", config=vars(args))

    # rewritten from scratch: a resumed run re-collects every generation from the job results
    history_path = os.path.join(run_dir, "history.csv")
    history = open(history_path, "w", newline="")
    writer = csv.writer(history)
    writer.writerow(HISTORY_COLUMNS)

    survivors: list[dict] = []
    sigma = args.sigma
    best: dict | None = None
    best_path = os.path.join(run_dir, "best.json")
    if os.path.exists(best_path):
        best = load_json(best_path)

    for gen in range(args.generations):
        gen_dir = os.path.join(run_dir, f"gen_{gen:02d}")
        os.makedirs(gen_dir, exist_ok=True)
        designs_file = os.path.join(gen_dir, "designs.json")
        if os.path.exists(designs_file):
            candidates = load_json(designs_file)
            print(f"[evolve] gen {gen}: reusing existing {designs_file}")
        else:
            candidates = propose(gen, survivors, rng, sigma)
            save_json(candidates, designs_file)
        print(f"\n[evolve] === generation {gen} ({len(candidates)} candidates, sigma={sigma:.3f}) ===")

        if args.general_checkpoint:
            screen(candidates, gen_dir, gen, run_name, group)
            save_json(candidates, designs_file)
        to_train = choose_training_set(candidates, rng)
        print(f"[evolve] gen {gen}: fine-tuning {[c['id'] for c in to_train]}")
        train(to_train, gen_dir, gen, run_name, group)

        results = collect(candidates, {c["id"] for c in to_train}, gen_dir)
        results.sort(key=lambda r: (r["trained"], r["score"]), reverse=True)
        survivors = [r for r in results if r["trained"] and r["checkpoint"]]
        for r in results:
            writer.writerow(history_row(gen, r))
        history.flush()

        top = survivors[0] if survivors else results[0]
        if best is None or top["score"] > best["score"]:
            best = {"generation": gen, **{k: top.get(k) for k in ("id", "design", "score", "distance_x_m", "checkpoint")},
                    "general_checkpoint": args.general_checkpoint}
            save_json(best, best_path)
        trained_rows = [r for r in results if r["trained"] and not r.get("kept")]  # trained this generation
        zs = [r.get("zero_shot_score") for r in trained_rows]
        vs = [r.get("value_s0") for r in trained_rows]
        sc = [r["score"] for r in trained_rows]
        corr_zs = spearman([z if z is not None else float("nan") for z in zs], sc)
        corr_vs = spearman([v if v is not None else float("nan") for v in vs], sc)
        print(f"[evolve] gen {gen} best: {top['id']} score={top['score']:.2f} "
              f"dist={top.get('distance_x_m', float('nan')):.2f} m design={top['design']}")
        if trained_rows and not math.isnan(corr_zs):
            print(f"[evolve] gen {gen} screening vs trained score: spearman(zero_shot)={corr_zs:.2f} "
                  f"spearman(V(s0))={corr_vs:.2f} over {len(trained_rows)} trained candidates")
        if wb is not None:
            import wandb

            log = {"generation": gen, "sigma": sigma, "best_score": top["score"], "best_score_so_far": best["score"],
                   "n_trained": len(trained_rows),
                   "trained_score_mean": sum(sc) / len(sc) if sc else float("nan"),
                   "screen/spearman_zero_shot": corr_zs, "screen/spearman_value_s0": corr_vs}
            zs_all = [r["zero_shot_score"] for r in results if r.get("zero_shot_score") is not None]
            if zs_all:
                log.update({"screen/zero_shot_max": max(zs_all), "screen/zero_shot_mean": sum(zs_all) / len(zs_all)})
            table = wandb.Table(columns=HISTORY_COLUMNS)
            for r in results:
                table.add_data(*history_row(gen, r))
            log["candidates"] = table
            wandb.log(log, step=gen)
        sigma *= args.sigma_decay

    history.close()
    print("\n[evolve] done.")
    print(f"[evolve] best over run (gen {best['generation']}): score={best['score']:.2f} design={best['design']}")
    print(f"[evolve] history: {history_path}")
    design_str = " ".join(str(x) for x in best["design"])
    print("[evolve] watch it with:")
    print(f"    python scripts/coadapt/play_design.py --design {design_str} --checkpoint {best['checkpoint']}")
    if wb is not None:
        wb.summary.update({"best_score": best["score"], "best_design": best["design"], "best_generation": best["generation"]})
        wb.finish()


if __name__ == "__main__":
    main()
