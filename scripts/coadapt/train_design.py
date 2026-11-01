"""Worker for design evaluation: train a skrl PPO policy on one or more
half-cheetah designs and score them. The drivers (evolve.py, train_general.py,
benchmark_metrics.py) launch one of these per job.

Modes (all write <out>/result.json):
  * one design, train + score          --design XI*6 --iterations N
  * several designs in one scene       --design-set designs.json  (env i gets design i % N)
                                       or --random-designs N --design-seed S
  * eval only (zero-shot screening)    --iterations 0 --checkpoint general.pt

Steps: (1) build the env, (2) optionally warm-start from a checkpoint,
(3) probe the warm-started policy on the designs (zero-shot return and the
critic's value at the initial state), (4) train PPO for --iterations rollout
cycles while recording learning-curve metrics, (5) evaluate deterministically
(one episode per env, aggregated per design), (6) write result.json,
metrics.json and agent.pt. One job per process: Isaac Sim does not reliably
support rebuilding the scene after env.close().

Example (repo root, conda env env_isaaclab):
    python scripts/coadapt/train_design.py --headless --num_envs 256 --iterations 100 ^
        --design 1.0 1.0 1.0 1.0 1.0 1.0 --out logs/coadapt/manual/design_00
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train and score half-cheetah designs with skrl PPO.")
parser.add_argument("--task", type=str, default="Template-Cheeta-Direct-v0", help="task id")
parser.add_argument("--design", type=float, nargs=6, default=None, metavar="XI",
                    help="6 design scale factors in [0.8, 2.0]")
parser.add_argument("--design-set", type=str, default=None, help="json file with a list of designs")
parser.add_argument("--random-designs", type=int, default=0, help="number of random designs to sample")
parser.add_argument("--design-seed", type=int, default=0, help="seed for --random-designs")
parser.add_argument("--include-identity", action="store_true", help="make the first random design the unscaled cheetah")
parser.add_argument("--checkpoint", type=str, default=None, help="warm-start checkpoint")
parser.add_argument("--out", type=str, required=True, help="output directory")
parser.add_argument("--num_envs", type=int, default=256, help="parallel envs")
parser.add_argument("--iterations", type=int, default=100, help="PPO iterations (rollout cycles); 0 = evaluate only")
parser.add_argument("--seed", type=int, default=42, help="seed")
parser.add_argument("--no-design-obs", action="store_true", help="hide the design vector from the policy")
parser.add_argument("--no-probe", action="store_true", help="skip the zero-shot / critic probe before training")
parser.add_argument("--random-episode-start", action="store_true",
                    help="start training with random episode lengths so the envs do not all time out on the same step")
group = parser.add_argument_group("wandb")
group.add_argument("--wandb", action="store_true", help="log this run to Weights & Biases")
group.add_argument("--wandb-project", type=str, default="cheeta")
group.add_argument("--wandb-entity", type=str, default=None)
group.add_argument("--wandb-group", type=str, default=None)
group.add_argument("--wandb-name", type=str, default=None)
group.add_argument("--wandb-job-type", type=str, default="train")
group.add_argument("--wandb-tags", type=str, nargs="*", default=[])
group.add_argument("--wandb-mode", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import collections  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import skrl  # noqa: E402
import torch  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402

import cheeta.tasks  # noqa: F401, E402
from cheeta.designs import sample_designs, validate_design  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402


def resolve_designs() -> list[list[float]]:
    if args_cli.design_set:
        with open(args_cli.design_set) as f:
            designs = [validate_design(d) for d in json.load(f)]
    elif args_cli.random_designs > 0:
        designs = sample_designs(args_cli.random_designs, args_cli.design_seed, args_cli.include_identity)
    elif args_cli.design is not None:
        designs = [validate_design(args_cli.design)]
    else:
        parser.error("one of --design, --design-set or --random-designs is required")
    if not designs:
        parser.error("empty design list")
    return designs


def _raw(env):
    return env.unwrapped if hasattr(env, "unwrapped") else env._unwrapped


def hard_reset(env):
    """Reset every env for real. skrl's Isaac Lab wrapper resets the env only on
    its first reset() call (and reseeds it then); later calls return cached
    observations, which would start an evaluation mid-episode."""
    env._reset_once = True
    env._seed = None
    return env.reset()


def per_design_stats(raw, returns: torch.Tensor, distance: torch.Tensor) -> list[dict]:
    """Mean/std of the per-env returns and distances for every distinct design."""
    stats = []
    for i, design in enumerate(raw.design_list):
        mask = raw.design_ids == i
        r, d = returns[mask], distance[mask]
        stats.append({
            "design": design,
            "score": r.mean().item(),
            "score_std": r.std().item() if r.numel() > 1 else 0.0,
            "distance_x_m": d.mean().item(),
            "distance_x_m_std": d.std().item() if d.numel() > 1 else 0.0,
            "n_envs": int(mask.sum().item()),
        })
    return stats


def evaluate(env, runner) -> dict:
    """One deterministic episode in every parallel env; mean/std scores overall and per design."""
    raw = _raw(env)
    runner.agent.enable_training_mode(False, apply_to_models=True)
    obs, _ = hard_reset(env)
    states = env.state()
    returns = torch.zeros(raw.num_envs, device=raw.device)
    # stop 2 steps short of the timeout so no env auto-resets mid-measurement
    num_steps = raw.max_episode_length - 2
    # no_grad, not inference_mode: inference tensors would break the env's later in-place resets
    with torch.no_grad():
        for _ in range(num_steps):
            outputs = runner.agent.act(obs, states, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, reward, _, _, _ = env.step(actions)
            states = env.state()
            returns += reward.flatten()
    distance = raw.distance_x
    return {
        "score": returns.mean().item(),
        "score_std": returns.std().item(),
        "distance_x_m": distance.mean().item(),
        "distance_x_m_std": distance.std().item(),
        "eval_episodes": raw.num_envs,
        "eval_steps": num_steps,
        "per_design": per_design_stats(raw, returns, distance),
    }


def value_at_start(env, runner) -> dict:
    """Critic estimate V(s0) of the current policy, overall and per design (the
    paper's fitness proxy: no rollout needed)."""
    raw = _raw(env)
    agent = runner.agent
    obs, _ = hard_reset(env)
    states = env.state()
    with torch.no_grad():
        inputs = {
            "observations": agent._observation_preprocessor(obs),
            "states": agent._state_preprocessor(states),
        }
        values, _ = agent.value.act(inputs, role="value")
        values = agent._value_preprocessor(values, inverse=True).flatten()
    return {
        "value_s0": values.mean().item(),
        "per_design": [v.item() for v in raw.per_design_mean(values)],
    }


class MetricRecorder:
    """Hooks into the skrl agent to keep per-iteration learning-curve metrics
    (what skrl writes to TensorBoard, plus the mean per-step training reward)."""

    TAGS = {
        "Loss / Policy loss": "policy_loss",
        "Loss / Value loss": "value_loss",
        "Policy / Standard deviation": "policy_std",
        "Learning / Learning rate": "learning_rate",
        "Reward / Total reward (mean)": "episode_reward",
        "Info / episode_return": "episode_return",
        "Info / distance_x_m": "distance_x_m",
    }

    def __init__(self, agent, rollouts: int):
        self.rollouts = rollouts
        self.series: dict[str, list[tuple[int, float]]] = collections.defaultdict(list)
        self._buf: dict[str, list[float]] = collections.defaultdict(list)
        self._reward_sum: torch.Tensor | None = None
        self._reward_n = 0
        self._orig_track = agent.track_data
        self._orig_write = agent.write_tracking_data
        self._orig_record = agent.record_transition
        agent.track_data = self._track
        agent.write_tracking_data = self._write
        agent.record_transition = self._record

    def _track(self, tag: str, value: float) -> None:
        if tag in self.TAGS:
            self._buf[self.TAGS[tag]].append(float(value))
        self._orig_track(tag, value)

    def _record(self, **kwargs) -> None:
        rewards = kwargs["rewards"]
        mean = rewards.float().mean()
        self._reward_sum = mean if self._reward_sum is None else self._reward_sum + mean
        self._reward_n += 1
        self._orig_record(**kwargs)

    def _write(self, *, timestep: int, timesteps: int) -> None:
        iteration = timestep // self.rollouts
        for name, values in self._buf.items():
            self.series[name].append((iteration, float(np.mean(values))))
        self._buf.clear()
        if self._reward_n:
            self.series["reward_step"].append((iteration, (self._reward_sum / self._reward_n).item()))
            self._reward_sum, self._reward_n = None, 0
        self._orig_write(timestep=timestep, timesteps=timesteps)

    def summary(self) -> dict:
        """Scalar features of the learning curves (candidates for design selection)."""
        out = {}

        def window(vals, lo, hi):
            lo_i, hi_i = int(lo * len(vals)), max(int(hi * len(vals)), int(lo * len(vals)) + 1)
            return float(np.mean(vals[lo_i:hi_i]))

        rs = self.series.get("reward_step", [])
        if len(rs) >= 2:
            it = np.array([p[0] for p in rs], dtype=float)
            val = np.array([p[1] for p in rs], dtype=float)
            out["reward_step_first"] = window(val, 0.0, 0.05)
            out["reward_step_last"] = window(val, 0.95, 1.0)
            out["reward_step_mean"] = float(val.mean())
            out["reward_step_slope"] = float(np.polyfit(it, val, 1)[0]) if it.ptp() > 0 else 0.0
            for frac in (0.1, 0.25, 0.5):
                out[f"reward_step_at_{int(frac * 100)}pct"] = window(val, frac - 0.03, frac + 0.03)
            out["reward_step_gain"] = out["reward_step_last"] - out["reward_step_first"]
        for name in ("policy_loss", "value_loss", "policy_std", "learning_rate", "episode_return", "distance_x_m"):
            pts = self.series.get(name, [])
            if pts:
                val = np.array([p[1] for p in pts], dtype=float)
                out[f"{name}_mean"] = float(val.mean())
                out[f"{name}_last"] = window(val, 0.95, 1.0)
                if name == "learning_rate":
                    out["learning_rate_min"] = float(val.min())
        return out


def setup_wandb_cfg(agent_cfg: dict, designs: list[list[float]]) -> None:
    if not args_cli.wandb:
        return
    kwargs = {
        "project": args_cli.wandb_project,
        "job_type": args_cli.wandb_job_type,
        "name": args_cli.wandb_name or os.path.basename(os.path.abspath(args_cli.out)),
        "config": {
            "designs": designs,
            "num_designs": len(designs),
            "num_envs": args_cli.num_envs,
            "iterations": args_cli.iterations,
            "warm_start": args_cli.checkpoint,
            "seed": args_cli.seed,
        },
    }
    for key in ("entity", "group", "mode"):
        value = getattr(args_cli, f"wandb_{key}")
        if value:
            kwargs[key] = value
    if args_cli.wandb_tags:
        kwargs["tags"] = list(args_cli.wandb_tags)
    agent_cfg["agent"]["experiment"]["wandb"] = True
    agent_cfg["agent"]["experiment"]["wandb_kwargs"] = kwargs


def log_wandb_result(result: dict) -> None:
    if not args_cli.wandb:
        return
    import wandb

    flat = {f"result/{k}": v for k, v in result.items() if isinstance(v, (int, float))}
    if result.get("probe"):
        flat.update({f"probe/{k}": v for k, v in result["probe"].items() if isinstance(v, (int, float))})
    if result.get("train_metrics"):
        flat.update({f"train_metrics/{k}": v for k, v in result["train_metrics"].items()})
    wandb.log(flat)
    wandb.summary.update(flat)
    columns = ["bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot", "score", "score_std", "distance_x_m"]
    table = wandb.Table(columns=columns)
    for row in result["per_design"]:
        table.add_data(*row["design"], row["score"], row["score_std"], row["distance_x_m"])
    wandb.log({"per_design": table})
    wandb.finish()


def main() -> dict:
    designs = resolve_designs()
    out_dir = os.path.abspath(args_cli.out)
    os.makedirs(out_dir, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if len(designs) == 1:
        env_cfg.design_vector = list(designs[0])
    else:
        env_cfg.design_set = [list(d) for d in designs]
    env_cfg.observe_design = not args_cli.no_design_obs
    env_cfg.seed = args_cli.seed

    agent_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")
    rollouts = agent_cfg["agent"]["rollouts"]
    agent_cfg["seed"] = args_cli.seed
    agent_cfg["trainer"]["timesteps"] = max(args_cli.iterations, 1) * rollouts
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["directory"] = out_dir
    agent_cfg["agent"]["experiment"]["experiment_name"] = "skrl"
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # final agent.pt is saved manually
    # one TensorBoard point per PPO iteration (also the resolution of the recorded learning curves)
    agent_cfg["agent"]["experiment"]["write_interval"] = rollouts if args_cli.iterations > 0 else 0
    setup_wandb_cfg(agent_cfg, designs)

    t_build = time.time()
    env = gym.make(args_cli.task, cfg=env_cfg)
    # skrl's wrapper reseeds the env on its first reset with this key (default 0,
    # captured at wrapping time), which would make every run identical
    skrl.config.torch.key = args_cli.seed
    env = SkrlVecEnvWrapper(env)
    runner = Runner(env, agent_cfg)
    build_time = time.time() - t_build
    print(f"[worker] {len(designs)} design(s) over {args_cli.num_envs} envs built in {build_time:.1f}s")

    warm_start = os.path.abspath(args_cli.checkpoint) if args_cli.checkpoint else None
    if warm_start:
        print(f"[worker] warm start from: {warm_start}")
        runner.agent.load(warm_start)

    result = {
        "designs": [list(d) for d in designs],
        "design": list(designs[0]) if len(designs) == 1 else None,
        "warm_start": warm_start,
        "iterations": args_cli.iterations,
        "num_envs": args_cli.num_envs,
        "seed": args_cli.seed,
        "observe_design": env_cfg.observe_design,
        "random_episode_start": args_cli.random_episode_start,
        "build_time_s": round(build_time, 1),
        "probe": None,
        "train_metrics": None,
    }

    # probe the warm-started policy before training: zero-shot return + critic value at s0
    if warm_start and (args_cli.iterations == 0 or not args_cli.no_probe):
        t0 = time.time()
        probe = evaluate(env, runner)
        critic = value_at_start(env, runner)
        for row, v in zip(probe["per_design"], critic["per_design"]):
            row["value_s0"] = v
        result["probe"] = {
            "zero_shot_score": probe["score"],
            "zero_shot_score_std": probe["score_std"],
            "zero_shot_distance_x_m": probe["distance_x_m"],
            "value_s0": critic["value_s0"],
            "per_design": probe["per_design"],
            "probe_time_s": round(time.time() - t0, 1),
        }
        print(f"[worker] probe: zero-shot score {probe['score']:.2f} (dist {probe['distance_x_m']:.2f} m), "
              f"V(s0) {critic['value_s0']:.2f}")

    if args_cli.iterations > 0:
        recorder = MetricRecorder(runner.agent, rollouts)
        if args_cli.random_episode_start:
            # as rsl_rl's init_at_random_ep_len: episodes only end by timeout, so without
            # this every env resets on the same step and each rollout sees one episode phase.
            # env.reset() is the wrapper's first (seeded) reset; the trainer's reset() then
            # returns these observations. evaluate() resets every env, so scoring is unaffected.
            env.reset()
            raw = _raw(env)
            raw.episode_length_buf = torch.randint_like(raw.episode_length_buf, high=int(raw.max_episode_length))
        runner.agent.enable_training_mode(True, apply_to_models=True)
        start = time.time()
        runner.run()
        train_time = time.time() - start
        ckpt_path = os.path.join(out_dir, "agent.pt")
        runner.agent.save(ckpt_path)
        result["train_metrics"] = recorder.summary()
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump({"rollouts": rollouts, "series": dict(recorder.series)}, f)
        final = evaluate(env, runner)
        result.update(final)
        result.update({"checkpoint": ckpt_path, "train_time_s": round(train_time, 1)})
    else:
        # evaluation only: the score is the warm-started policy's zero-shot performance
        if not warm_start:
            final = evaluate(env, runner)
            result.update(final)
        else:
            probe = result["probe"]
            result.update({
                "score": probe["zero_shot_score"],
                "score_std": probe["zero_shot_score_std"],
                "distance_x_m": probe["zero_shot_distance_x_m"],
                "per_design": probe["per_design"],
                "eval_episodes": args_cli.num_envs,
            })
        result.update({"checkpoint": warm_start, "train_time_s": 0.0})

    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    log_wandb_result(result)
    env.close()
    print(f"[worker] {len(designs)} design(s) -> score {result['score']:.2f} "
          f"(dist {result['distance_x_m']:.2f} m) in {result['train_time_s']:.0f}s training")
    return result


if __name__ == "__main__":
    main()
    simulation_app.close()
