"""Evolution of half-cheetah designs with one shared, design-conditioned policy.

Per generation ONE worker process trains a single PPO policy on the whole
population at once (env i gets design i % N, the normalised design vector is
part of the observation), warm-started from the previous generation's policy,
and then scores every design with it: deterministic episodes,
--envs-per-design per design. The top --parents designs survive into the next
generation and are scored again there with the updated policy; the rest of the
population is --immigrants fresh random designs plus Gaussian mutations of the
parents (every parent gets the same number of children).

Raw scores rise across generations even if the designs do not, because the
policy keeps training. Every generation therefore also contains fixed
reference designs: the unscaled cheetah and --references random designs drawn
once. They are trained on like the rest but never selected; design progress is
the parents' score relative to them (summary.csv: parents_minus_refs).

The shared policy is a compromise over the population: its scores rank the
designs, but understate designs that need a gait of their own. Check the final
top designs by training each on its own, with several seeds.

Run with the Isaac python (workers inherit the interpreter):
    python scripts/coadapt/evolve_shared.py --generations 10 --population 1000 --parents 100 ^
        --iterations-first 1000 --iterations 300 --wandb

With --wandb the driver and every generation's worker log to the wandb
project "cheeta-shared" (not "cheeta"); --wandb-project overrides it.

Output in <run-dir>: gen_XX/ (population.json and the worker's result.json,
metrics.json, agent.pt), history.csv (one row per design and generation),
summary.csv (one row per generation), best.json (top designs of the latest
generation) and policy.pt (the latest shared policy; it also works as
evolve.py --general-checkpoint). Interrupted runs resume with --run-dir
pointing at the old run (finished generations are reused).
"""

import argparse
import csv
import math
import os
import random
import shutil
import statistics
import time

from _common import (REPO_ROOT, add_wandb_args, designs, load_json, load_result, run_worker, save_json,
                     spearman, timestamp, wandb_init_driver, wandb_worker_args, worker_command)

parser = argparse.ArgumentParser(description="Evolve half-cheetah designs with one shared design-conditioned policy.")
parser.add_argument("--generations", type=int, default=10)
parser.add_argument("--population", type=int, default=1000, help="designs per generation (incl. parents, without references)")
parser.add_argument("--parents", type=int, default=100, help="top designs that survive and parent the next generation")
parser.add_argument("--immigrants", type=int, default=50, help="fresh random designs per generation")
parser.add_argument("--references", type=int, default=10, help="fixed random reference designs (plus the unscaled cheetah)")
parser.add_argument("--sigma", type=float, default=0.15, help="mutation std dev on the scale factors (generation 1)")
parser.add_argument("--sigma-decay", type=float, default=0.95, help="sigma multiplier per generation")
parser.add_argument("--envs-per-design", type=int, default=4, help="parallel envs, and scoring episodes, per design")
parser.add_argument("--iterations-first", type=int, default=1000, help="PPO iterations of generation 0")
parser.add_argument("--iterations", type=int, default=300, help="PPO iterations of every later generation")
parser.add_argument("--checkpoint", type=str, default=None, help="warm start of generation 0 (e.g. a general policy)")
parser.add_argument("--synced-episodes", action="store_true",
                    help="no random episode start: all envs reset on the same step (the old behaviour)")
parser.add_argument("--no-probe", action="store_true", help="skip the zero-shot score of the population before training")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run-dir", type=str, default=None, help="run directory (set to an old run to resume)")
parser.add_argument("--device", type=str, default=None, help="forwarded to the worker, e.g. cuda:0")
add_wandb_args(parser)
# own wandb project (driver and workers), apart from the evolve.py / train_general.py runs in "cheeta"
parser.set_defaults(wandb_project="cheeta-shared")
args = parser.parse_args()
if not 0 < args.parents <= args.population - args.immigrants:
    parser.error("need 0 < --parents <= --population - --immigrants")

HISTORY_COLUMNS = ["generation", "id", "role", "parent", "score", "score_std", "zero_shot_score", "value_s0",
                   "distance_x_m", *designs.DESIGN_NAMES]
SUMMARY_COLUMNS = ["generation", "sigma", "iterations", "num_envs", "best_id", "best_score", "parents_mean",
                   "population_median", "identity_score", "refs_mean", "parents_minus_refs", "best_minus_identity",
                   "children_beating_parent", "children_in_parents", "immigrants_in_parents",
                   "spearman_zero_shot", "build_time_s", "probe_time_s", "train_time_s", "wall_time_s",
                   *[f"best_{n}" for n in designs.DESIGN_NAMES]]


def propose(gen: int, parents: list[dict], rng: random.Random, sigma: float) -> list[dict]:
    """Population of a generation: the parents (scored again), immigrants and children of the parents."""
    pop = []

    def add(design, role, parent=None):
        pop.append({"id": f"d{len(pop):04d}", "design": list(design), "role": role, "parent": parent})

    if gen == 0 or not parents:
        while len(pop) < args.population:
            add(designs.random_design(rng), "immigrant")
        return pop
    for p in parents:
        add(p["design"], "elite", parent=p["id"])
    for _ in range(args.immigrants):
        add(designs.random_design(rng), "immigrant")
    i = 0
    while len(pop) < args.population:
        parent = parents[i % len(parents)]
        add(designs.mutate(parent["design"], sigma, rng), "child", parent=parent["id"])
        i += 1
    return pop


def reference_designs() -> list[dict]:
    """The unscaled cheetah and --references random designs, the same in every generation."""
    refs = designs.sample_designs(args.references + 1, seed=args.seed + 7919, include_identity=True)
    return [{"id": "identity" if i == 0 else f"ref{i:02d}", "design": d, "role": "reference", "parent": None}
            for i, d in enumerate(refs)]


def attach_scores(members: list[dict], result: dict) -> None:
    per_design = result["per_design"]
    if len(per_design) != len(members):
        raise RuntimeError(f"worker scored {len(per_design)} designs, expected {len(members)}")
    probe = (result.get("probe") or {}).get("per_design") or [None] * len(members)
    for m, row, pr in zip(members, per_design, probe):
        m.update(score=row["score"], score_std=row["score_std"], distance_x_m=row["distance_x_m"],
                 zero_shot_score=pr["score"] if pr else None, value_s0=pr.get("value_s0") if pr else None)


def summarise(gen: int, sigma: float, iterations: int, num_envs: int, population: list[dict], refs: list[dict],
              result: dict, wall_time: float | None) -> dict:
    ranked = sorted(population, key=lambda m: m["score"], reverse=True)
    parents = ranked[: args.parents]
    best = ranked[0]
    parents_mean = statistics.mean(m["score"] for m in parents)
    refs_mean = statistics.mean(m["score"] for m in refs)
    identity = refs[0]["score"]
    # children against their parent, both scored by this generation's policy
    parent_score = {m["parent"]: m["score"] for m in population if m["role"] == "elite"}
    beat = [m["score"] > parent_score[m["parent"]] for m in population
            if m["role"] == "child" and m["parent"] in parent_score]
    zs = [m["zero_shot_score"] for m in population]
    probe = result.get("probe") or {}
    return {
        "generation": gen, "sigma": sigma, "iterations": iterations, "num_envs": num_envs,
        "best_id": best["id"], "best_score": best["score"], "parents_mean": parents_mean,
        "population_median": statistics.median(m["score"] for m in population),
        "identity_score": identity, "refs_mean": refs_mean,
        "parents_minus_refs": parents_mean - refs_mean, "best_minus_identity": best["score"] - identity,
        "children_beating_parent": f"{sum(beat)}/{len(beat)}" if beat else "",
        "children_in_parents": sum(m["role"] == "child" for m in parents),
        "immigrants_in_parents": sum(m["role"] == "immigrant" for m in parents),
        "spearman_zero_shot": spearman(zs, [m["score"] for m in population]) if None not in zs else float("nan"),
        "build_time_s": result.get("build_time_s"), "probe_time_s": probe.get("probe_time_s"),
        "train_time_s": result.get("train_time_s"), "wall_time_s": round(wall_time, 1) if wall_time else None,
        **{f"best_{n}": x for n, x in zip(designs.DESIGN_NAMES, best["design"])},
    }


def history_row(gen: int, m: dict) -> list:
    return [gen, m["id"], m["role"], m.get("parent") or "", _fmt(m["score"]), _fmt(m["score_std"]),
            _fmt(m.get("zero_shot_score")), _fmt(m.get("value_s0")), _fmt(m.get("distance_x_m"))] + \
        [f"{x:.4f}" for x in m["design"]]


def _fmt(v, digits: int = 4) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return ""
    return f"{v:.{digits}f}" if isinstance(v, float) else str(v)


def main():
    run_dir = os.path.abspath(args.run_dir or os.path.join(REPO_ROOT, "logs", "coadapt", f"shared_{timestamp()}"))
    os.makedirs(run_dir, exist_ok=True)
    run_name = os.path.basename(os.path.normpath(run_dir))
    group = args.wandb_group or f"shared-{run_name}"
    save_json(vars(args), os.path.join(run_dir, "config.json"))
    wb = wandb_init_driver(args, name=f"{run_name}-driver", group=group, job_type="shared-driver", config=vars(args))

    # rewritten from scratch: a resumed run re-collects every generation from the job results
    history = open(os.path.join(run_dir, "history.csv"), "w", newline="")
    hist_writer = csv.writer(history)
    hist_writer.writerow(HISTORY_COLUMNS)
    summary_file = open(os.path.join(run_dir, "summary.csv"), "w", newline="")
    sum_writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_COLUMNS)
    sum_writer.writeheader()

    refs = reference_designs()
    checkpoint = os.path.abspath(args.checkpoint) if args.checkpoint else None
    policy_path = os.path.join(run_dir, "policy.pt")
    parents: list[dict] = []
    for gen in range(args.generations):
        gen_dir = os.path.join(run_dir, f"gen_{gen:02d}")
        os.makedirs(gen_dir, exist_ok=True)
        # per-generation rng and sigma, so a resumed run proposes the same designs
        rng = random.Random(args.seed * 1000 + gen)
        sigma = args.sigma * args.sigma_decay ** max(gen - 1, 0)
        pop_file = os.path.join(gen_dir, "population.json")
        if os.path.exists(pop_file):
            population = load_json(pop_file)
            print(f"[shared] gen {gen}: reusing existing {pop_file}")
        else:
            population = propose(gen, parents, rng, sigma)
            save_json(population, pop_file)
        members = population + [dict(r) for r in refs]
        iterations = args.iterations_first if gen == 0 else args.iterations
        num_envs = args.envs_per_design * len(members)
        print(f"\n[shared] === generation {gen}: {len(population)} designs + {len(refs)} references over "
              f"{num_envs} envs, {iterations} iterations, sigma={sigma:.3f} ===")

        extra = list(wandb_worker_args(args, name=f"{run_name}-g{gen:02d}", group=group, job_type="shared-train",
                                       tags=[f"gen{gen}"]))
        if args.no_probe:
            extra.append("--no-probe")
        cmd = worker_command(gen_dir, num_envs=num_envs, iterations=iterations, seed=args.seed + gen,
                             design_set=[m["design"] for m in members], checkpoint=checkpoint, device=args.device,
                             random_episode_start=not args.synced_episodes, extra=extra)
        done_before = load_result(gen_dir) is not None
        t0 = time.time()
        run_worker(cmd, f"gen {gen}")
        wall_time = None if done_before else time.time() - t0
        result = load_result(gen_dir)
        if result is None:
            print(f"[shared] gen {gen} failed twice, stopping")
            break
        checkpoint = result["checkpoint"]
        shutil.copyfile(checkpoint, policy_path)

        attach_scores(members, result)
        population, gen_refs = members[: len(population)], members[len(population):]
        for m in members:
            hist_writer.writerow(history_row(gen, m))
        history.flush()
        s = summarise(gen, sigma, iterations, num_envs, population, gen_refs, result, wall_time)
        sum_writer.writerow({k: _fmt(v) for k, v in s.items()})
        summary_file.flush()

        ranked = sorted(population, key=lambda m: m["score"], reverse=True)
        parents = ranked[: args.parents]
        save_json({"generation": gen, "checkpoint": policy_path, "identity_score": s["identity_score"],
                   "refs_mean": s["refs_mean"],
                   "top": [{k: m.get(k) for k in ("id", "role", "design", "score", "score_std", "distance_x_m")}
                           for m in ranked[:10]]},
                  os.path.join(run_dir, "best.json"))
        print(f"[shared] gen {gen}: best {s['best_id']} {s['best_score']:.1f} | top-{args.parents} mean "
              f"{s['parents_mean']:.1f} | median {s['population_median']:.1f} | unscaled {s['identity_score']:.1f} | "
              f"refs mean {s['refs_mean']:.1f} | parents - refs {s['parents_minus_refs']:+.1f}")
        print(f"[shared] gen {gen}: children beating parent {s['children_beating_parent'] or '-'}, new parents: "
              f"{s['children_in_parents']} children + {s['immigrants_in_parents']} immigrants, "
              f"spearman(zero-shot, score) {s['spearman_zero_shot']:.2f}, best design {ranked[0]['design']}")
        if wb is not None:
            import wandb

            log = {k: v for k, v in s.items() if isinstance(v, (int, float)) and v is not None}
            table = wandb.Table(columns=HISTORY_COLUMNS)
            for m in parents + gen_refs:
                table.add_data(*history_row(gen, m))
            log["parents_and_refs"] = table
            wandb.log(log, step=gen)

    history.close()
    summary_file.close()
    if parents:
        best = parents[0]
        print(f"\n[shared] done. top design of the last generation: {best['design']} (score {best['score']:.1f})")
        print(f"[shared] summary: {os.path.join(run_dir, 'summary.csv')}")
        print("[shared] watch it with:")
        print(f"    python scripts/coadapt/play_design.py --design {' '.join(str(x) for x in best['design'])} "
              f"--checkpoint {policy_path}")
    if wb is not None:
        if parents:
            wb.summary.update({"best_score": parents[0]["score"], "best_design": parents[0]["design"]})
        wb.finish()


if __name__ == "__main__":
    main()
