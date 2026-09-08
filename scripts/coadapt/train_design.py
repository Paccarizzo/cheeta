"""Worker for design evaluation: train a skrl PPO policy on one half-cheetah
design and score it. evolve.py launches one of these per candidate.

It (1) builds the env with the given design vector, (2) optionally warm-starts
the policy from a parent checkpoint, (3) trains PPO for --iterations rollout
cycles, (4) runs one deterministic evaluation episode in every parallel env,
and (5) writes result.json with the scores plus agent.pt with the trained
policy. One design per process: Isaac Sim does not reliably support rebuilding
the scene after env.close(), so each design gets a fresh app session.

Example (repo root, conda env env_isaaclab):
    python scripts/coadapt/train_design.py --headless --num_envs 256 --iterations 100 ^
        --design 1.0 1.0 1.0 1.0 1.0 1.0 --out logs/coadapt/manual/design_00
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train and score half-cheetah designs with skrl PPO.")
parser.add_argument("--task", type=str, default="Template-Cheeta-Direct-v0", help="task id")
parser.add_argument("--design", type=float, nargs=6, required=True, metavar="XI",
                    help="6 design scale factors in [0.8, 2.0]")
parser.add_argument("--checkpoint", type=str, default=None, help="warm-start checkpoint")
parser.add_argument("--out", type=str, required=True, help="output directory")
parser.add_argument("--num_envs", type=int, default=256, help="parallel envs")
parser.add_argument("--iterations", type=int, default=100, help="PPO iterations (rollout cycles)")
parser.add_argument("--seed", type=int, default=42, help="seed")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402

import cheeta.tasks  # noqa: F401, E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402


def evaluate(env, runner) -> dict:
    """One deterministic episode in every parallel env; returns mean/std scores."""
    raw = env.unwrapped if hasattr(env, "unwrapped") else env._unwrapped
    runner.agent.enable_training_mode(False, apply_to_models=True)
    obs, _ = env.reset()
    states = env.state()
    returns = torch.zeros(raw.num_envs, device=raw.device)
    # stop 2 steps short of the timeout so no env auto-resets mid-measurement
    num_steps = raw.max_episode_length - 2
    with torch.inference_mode():
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
    }


def train_and_score(design: list[float], checkpoint: str | None, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.design_vector = list(design)
    env_cfg.seed = args_cli.seed

    agent_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")
    agent_cfg["seed"] = args_cli.seed
    agent_cfg["trainer"]["timesteps"] = args_cli.iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["directory"] = os.path.abspath(out_dir)
    agent_cfg["agent"]["experiment"]["experiment_name"] = "skrl"
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # final agent.pt is saved manually

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env)
    runner = Runner(env, agent_cfg)
    if checkpoint:
        print(f"[worker] warm start from: {checkpoint}")
        runner.agent.load(os.path.abspath(checkpoint))

    start = time.time()
    runner.run()
    train_time = time.time() - start

    ckpt_path = os.path.abspath(os.path.join(out_dir, "agent.pt"))
    runner.agent.save(ckpt_path)

    result = evaluate(env, runner)
    result.update({
        "design": list(design),
        "checkpoint": ckpt_path,
        "warm_start": checkpoint,
        "iterations": args_cli.iterations,
        "num_envs": args_cli.num_envs,
        "seed": args_cli.seed,
        "train_time_s": round(train_time, 1),
    })
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    env.close()
    print(f"[worker] design {[f'{v:.3f}' for v in design]} -> "
          f"score {result['score']:.2f} (dist {result['distance_x_m']:.2f} m) in {train_time:.0f}s")
    return result


if __name__ == "__main__":
    train_and_score(args_cli.design, args_cli.checkpoint, args_cli.out)
    simulation_app.close()
