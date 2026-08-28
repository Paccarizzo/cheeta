"""Run the Half-Cheetah env with random joint torques and report forward distance.

Run from the repo root (conda env env_isaaclab):
    python scripts/run_random_cheeta.py --headless --num_envs 4 --steps 600

Drop --headless to watch the cheetahs in the viewer.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Random-action rollout for the Half-Cheetah task.")
parser.add_argument("--num_envs", type=int, default=4, help="number of parallel envs")
parser.add_argument("--steps", type=int, default=600, help="control steps to simulate")
parser.add_argument("--task", type=str, default="Template-Cheeta-Direct-v0", help="task id")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import cheeta.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    raw = env.unwrapped
    print(f"[info] observation space: {env.observation_space}")
    print(f"[info] action space: {env.action_space}")

    obs, _ = env.reset()
    print(f"[info] first observation shape: {obs['policy'].shape}")

    total_reward = torch.zeros(raw.num_envs, device=raw.device)
    num_actions = env.action_space.shape[-1]
    for step in range(args_cli.steps):
        actions = torch.empty((raw.num_envs, num_actions), device=raw.device).uniform_(-1.0, 1.0)
        obs, reward, _, _, _ = env.step(actions)
        total_reward += reward
        if (step + 1) % 100 == 0:
            dist = raw.distance_x
            print(f"[step {step + 1:4d}] distance x [m]: {[f'{v:+.2f}' for v in dist.tolist()]}")

    dist = raw.distance_x
    print(f"[result] after {args_cli.steps} random control steps:")
    print(f"[result]   distance x [m] per env: {[f'{v:+.3f}' for v in dist.tolist()]}")
    print(f"[result]   mean distance x [m]: {dist.mean().item():+.3f}")
    print(f"[result]   cumulative reward per env: {[f'{v:.3f}' for v in total_reward.tolist()]}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
