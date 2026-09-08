"""Watch a trained policy run on a given half-cheetah design.

Typical use, straight from the line evolve.py prints at the end:
    python scripts/coadapt/play_design.py --design 1.2 0.9 1.0 1.4 1.0 0.8 ^
        --checkpoint logs/coadapt/<run>/gen_04/cand_02/agent.pt

Without --checkpoint the cheetah runs with zero actions (passive dynamics).
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained policy on a half-cheetah design.")
parser.add_argument("--task", type=str, default="Template-Cheeta-Direct-v0", help="task id")
parser.add_argument("--design", type=float, nargs=6, default=[1.0] * 6, metavar="XI",
                    help="6 design scale factors in [0.8, 2.0]")
parser.add_argument("--checkpoint", type=str, default=None, help="agent checkpoint (.pt) to load")
parser.add_argument("--num_envs", type=int, default=4, help="parallel envs")
parser.add_argument("--real-time", action="store_true", help="slow down to real time")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402

import cheeta.tasks  # noqa: F401, E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.design_vector = list(args_cli.design)
    env = gym.make(args_cli.task, cfg=env_cfg)
    raw = env.unwrapped
    dt = raw.step_dt
    env = SkrlVecEnvWrapper(env)

    runner = None
    if args_cli.checkpoint:
        agent_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")
        agent_cfg["trainer"]["close_environment_at_exit"] = False
        agent_cfg["agent"]["experiment"]["write_interval"] = 0
        agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
        runner = Runner(env, agent_cfg)
        print(f"[play] loading checkpoint: {os.path.abspath(args_cli.checkpoint)}")
        runner.agent.load(os.path.abspath(args_cli.checkpoint))
        runner.agent.enable_training_mode(False, apply_to_models=True)

    obs, _ = env.reset()
    states = env.state()
    step = 0
    while simulation_app.is_running():
        start = time.time()
        with torch.inference_mode():
            if runner is not None:
                outputs = runner.agent.act(obs, states, timestep=0, timesteps=0)
                actions = outputs[-1].get("mean_actions", outputs[0])
            else:
                actions = torch.zeros((raw.num_envs, raw.cfg.action_space), device=raw.device)
            obs, _, _, _, _ = env.step(actions)
            states = env.state()
        step += 1
        if step % 100 == 0:
            print(f"[play] distance x [m]: {[f'{v:+.2f}' for v in raw.distance_x.tolist()]}")
        if args_cli.real_time:
            sleep = dt - (time.time() - start)
            if sleep > 0:
                time.sleep(sleep)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
