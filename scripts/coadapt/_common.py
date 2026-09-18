"""Helpers shared by the co-adaptation drivers (evolve.py, train_general.py,
benchmark_metrics.py). Drivers need no Isaac Sim; they launch
scripts/coadapt/train_design.py as a subprocess per job.
"""

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKER = os.path.join(REPO_ROOT, "scripts", "coadapt", "train_design.py")


def _load_designs_module():
    """cheeta.designs without importing the cheeta package (which needs Isaac Sim)."""
    path = os.path.join(REPO_ROOT, "source", "cheeta", "cheeta", "designs.py")
    spec = importlib.util.spec_from_file_location("cheeta_designs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


designs = _load_designs_module()


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def add_wandb_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("wandb")
    group.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    group.add_argument("--wandb-project", type=str, default="cheeta", help="wandb project")
    group.add_argument("--wandb-entity", type=str, default=None, help="wandb entity (team/user)")
    group.add_argument("--wandb-group", type=str, default=None, help="wandb group (default: run directory name)")
    group.add_argument("--wandb-tags", type=str, nargs="*", default=[], help="extra wandb tags")
    group.add_argument("--wandb-mode", type=str, default=None, help="wandb mode: online, offline, disabled")


def wandb_worker_args(args: argparse.Namespace, name: str, group: str, job_type: str, tags: Sequence[str] = ()) -> list[str]:
    """CLI arguments forwarding the driver's wandb settings to a worker run."""
    if not args.wandb:
        return []
    cmd = ["--wandb", "--wandb-project", args.wandb_project, "--wandb-group", group,
           "--wandb-name", name, "--wandb-job-type", job_type]
    if args.wandb_entity:
        cmd += ["--wandb-entity", args.wandb_entity]
    if args.wandb_mode:
        cmd += ["--wandb-mode", args.wandb_mode]
    all_tags = list(args.wandb_tags) + list(tags)
    if all_tags:
        cmd += ["--wandb-tags", *all_tags]
    return cmd


def wandb_init_driver(args: argparse.Namespace, name: str, group: str, job_type: str, config: dict):
    """Start a wandb run for the driver process itself (None when --wandb is off)."""
    if not args.wandb:
        return None
    import wandb

    kwargs = dict(project=args.wandb_project, group=group, name=name, job_type=job_type,
                  tags=list(args.wandb_tags) or None, config=config, resume="allow", id=_run_id(name))
    if args.wandb_entity:
        kwargs["entity"] = args.wandb_entity
    if args.wandb_mode:
        kwargs["mode"] = args.wandb_mode
    return wandb.init(**kwargs)


def _run_id(name: str) -> str:
    """Stable wandb run id derived from the run name (so a resumed driver reuses its run)."""
    import hashlib

    return hashlib.md5(name.encode()).hexdigest()[:12]


def worker_command(
    out_dir: str,
    *,
    num_envs: int,
    iterations: int,
    seed: int,
    design: Sequence[float] | None = None,
    design_set: Sequence[Sequence[float]] | None = None,
    random_designs: int = 0,
    design_seed: int = 0,
    checkpoint: str | None = None,
    device: str | None = None,
    extra: Sequence[str] = (),
) -> list[str]:
    """Command line for one worker run (train + score, or eval-only with iterations=0).

    Exactly one design source: ``design`` (one design), ``design_set`` (several
    designs in one scene) or ``random_designs`` (sampled by the worker).
    """
    cmd = [sys.executable, "-u", WORKER, "--out", out_dir, "--num_envs", str(num_envs),
           "--iterations", str(iterations), "--seed", str(seed), "--headless"]
    if design_set is not None:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "design_set.json")
        with open(path, "w") as f:
            json.dump([list(d) for d in design_set], f)
        cmd += ["--design-set", path]
    elif design is not None:
        cmd += ["--design", *[str(x) for x in design]]
    elif random_designs > 0:
        cmd += ["--random-designs", str(random_designs), "--design-seed", str(design_seed), "--include-identity"]
    else:
        raise ValueError("one of design, design_set or random_designs is required")
    if checkpoint:
        cmd += ["--checkpoint", checkpoint]
    if device:
        cmd += ["--device", device]
    cmd += list(extra)
    return cmd


def run_worker(cmd: list[str], label: str, retries: int = 1) -> bool:
    """Run a worker, retrying once on failure; True when result.json exists afterwards."""
    out_dir = cmd[cmd.index("--out") + 1]
    result_path = os.path.join(out_dir, "result.json")
    if os.path.exists(result_path):
        print(f"[driver] {label}: already done, skipping")
        return True
    for attempt in range(retries + 1):
        if attempt:
            print(f"[driver] {label}: retry {attempt}")
        proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
        if proc.returncode != 0:
            print(f"[driver] WARNING: worker for {label} exited with code {proc.returncode}")
        if os.path.exists(result_path):
            return True
    print(f"[driver] WARNING: no result for {label}")
    return False


def load_result(out_dir: str) -> dict | None:
    path = os.path.join(out_dir, "result.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0  # average rank for ties
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    n = len(x)
    if n < 3:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(sxx * syy)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation (nan for fewer than 3 finite pairs)."""
    pairs = [(a, b) for a, b in zip(x, y) if math.isfinite(a) and math.isfinite(b)]
    if len(pairs) < 3:
        return float("nan")
    xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
    return pearson(_ranks(xs), _ranks(ys))
