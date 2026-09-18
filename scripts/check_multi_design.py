"""Smoke test for several designs in one scene: spawn N designs over the envs,
run random actions and print per-design distances and the observation shape.

    python scripts/check_multi_design.py --headless --num_envs 8 --designs 4
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--designs", type=int, default=4, help="number of random designs (0 = single design)")
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import cheeta.tasks  # noqa: F401, E402
from cheeta.designs import sample_designs  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main():
    env_cfg = parse_env_cfg("Template-Cheeta-Direct-v0", device=args_cli.device, num_envs=args_cli.num_envs)
    if args_cli.designs > 0:
        env_cfg.design_set = sample_designs(args_cli.designs, args_cli.seed, include_identity=True)
    t0 = time.time()
    env = gym.make("Template-Cheeta-Direct-v0", cfg=env_cfg)
    raw = env.unwrapped
    print(f"[check] env built in {time.time() - t0:.1f}s, replicate_physics={raw.cfg.scene.replicate_physics}")
    print(f"[check] designs: {raw.design_list}")
    print(f"[check] design_ids: {raw.design_ids.tolist()}")
    obs, _ = env.reset()
    print(f"[check] obs shape: {tuple(obs['policy'].shape)} (expected {args_cli.num_envs} x {raw.cfg.observation_space})")
    print(f"[check] design obs of env 0..3:\n{obs['policy'][:4, -6:]}")
    # standing check: initial height and body heights per env after reset
    print(f"[check] rootz after reset: {raw.robot.data.joint_pos[:, raw._rootz_id[0]].tolist()}")
    root_z = raw.robot.data.root_pos_w[:, 2]
    print(f"[check] root z (world link) per env: {[round(v, 3) for v in root_z.tolist()]}")
    # passive settle then random actions
    with torch.inference_mode():
        for _ in range(60):
            obs, *_ = env.step(torch.zeros(args_cli.num_envs, 6, device=raw.device))
        print(f"[check] rootz after 60 passive steps: {[round(v, 3) for v in raw.robot.data.joint_pos[:, raw._rootz_id[0]].tolist()]}")
        for _ in range(args_cli.steps):
            actions = torch.rand(args_cli.num_envs, 6, device=raw.device) * 2 - 1
            obs, rew, term, trunc, info = env.step(actions)
    dist = raw.distance_x
    print(f"[check] distance per env: {[round(v, 2) for v in dist.tolist()]}")
    print(f"[check] distance per design: {[round(v, 2) for v in raw.per_design_mean(dist).tolist()]}")
    # verify design-specific geometry: body positions differ across designs
    body_z = raw.robot.data.body_pos_w[:, :, 2]
    print(f"[check] mean body z per env: {[round(v, 3) for v in body_z.mean(dim=1).tolist()]}")
    env.close()
    print("[check] OK")


if __name__ == "__main__":
    main()
    simulation_app.close()
