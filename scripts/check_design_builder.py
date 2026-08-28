"""Standalone check for desin_builder: build the asset, spawn it, and verify the
joints hold their commanded initial angles.

Run from the repo root with the Isaac Lab python:
    C:\\Users\\knorbaeck\\isaacsim\\python.bat scripts\\check_design_builder.py --headless

Drop --headless to watch the chain in the viewer.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Check desin_builder asset generation.")
parser.add_argument("--steps", type=int, default=300, help="simulation steps to run")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# everything below needs the app running
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from desin_builder import build_isaaclab_asset  # noqa: E402


def main():
    limb_lengths = [0.4, 0.35, 0.3, 0.25, 0.2, 0.15]
    joint_angles = [0.3, -0.5, 0.4, -0.4, 0.5, -0.3]

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120))
    sim.set_camera_view(eye=(3.0, 3.0, 2.0), target=(0.0, 0.0, 1.0))

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/GroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
    light_cfg.func("/World/Light", light_cfg)

    print("[check] building asset from limb_lengths + joint_angles ...")
    robot_cfg = build_isaaclab_asset(limb_lengths, joint_angles, prim_path="/World/Robot")
    print(f"[check] USD generated at: {robot_cfg.spawn.usd_path}")

    robot = Articulation(robot_cfg)
    sim.reset()

    print(f"[check] joint names: {robot.joint_names}")
    print(f"[check] bodies:      {robot.body_names}")

    expected = torch.tensor(joint_angles, device=robot.device)
    start_err = (robot.data.joint_pos[0] - expected).abs().max().item()
    print(f"[check] initial joint pos: {robot.data.joint_pos[0].tolist()}")
    print(f"[check] max initial error vs joint_angles: {start_err:.4f} rad")

    for _ in range(args_cli.steps):
        robot.set_joint_position_target(robot.data.default_joint_pos)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())

    end_err = (robot.data.joint_pos[0] - expected).abs().max().item()
    print(f"[check] joint pos after {args_cli.steps} steps: {robot.data.joint_pos[0].tolist()}")
    print(f"[check] max deviation from targets after settling: {end_err:.4f} rad")

    ok = len(robot.joint_names) == len(limb_lengths) and start_err < 1e-3 and end_err < 0.25
    print("[check] RESULT:", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    sys.exit(code)
