"""Build half-cheetah assets (URDF + USD) with custom limb lengths.

The design vector holds six scale factors, one per leg segment
(bthigh, bshin, bfoot, fthigh, fshin, ffoot), each in [0.8, 2.0].
Base lengths are (0.29, 0.30, 0.188, 0.29, 0.30, 0.188) m.

Examples (repo root, conda env env_isaaclab):
    # default design, print asset paths only
    python scripts/make_cheeta.py --headless

    # long back legs, view it standing in the viewer
    python scripts/make_cheeta.py --design 1.6 1.6 1.2 1.0 1.0 1.0 --preview

    # segment lengths in meters instead of scale factors
    python scripts/make_cheeta.py --lengths 0.35 0.40 0.20 0.29 0.30 0.188 --headless

    # compare 5 random designs side by side
    python scripts/make_cheeta.py --random 5 --preview
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Build half-cheetah assets with custom limb lengths.")
parser.add_argument("--design", type=float, nargs=6, action="append", metavar="XI",
                    help="6 scale factors in [0.8, 2.0]; repeat the flag for several designs")
parser.add_argument("--lengths", type=float, nargs=6, action="append", metavar="L",
                    help="6 segment lengths in meters (converted to scale factors)")
parser.add_argument("--random", type=int, default=0, metavar="N", help="add N random designs")
parser.add_argument("--seed", type=int, default=0, help="seed for --random")
parser.add_argument("--out", type=str, default=None, help="output directory for URDF/USD (default: temp assets dir)")
parser.add_argument("--preview", action="store_true", help="spawn the designs side by side and simulate")
parser.add_argument("--preview-steps", type=int, default=0, help="preview steps to run (0 = until window closed)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os  # noqa: E402
import random  # noqa: E402

from cheeta.design_builder import (  # noqa: E402
    BASE_SEGMENT_LENGTHS,
    DESIGN_BOUNDS,
    LEG_JOINT_NAMES,
    build_half_cheetah_asset,
    segment_lengths_from_design,
)


def collect_designs() -> list[list[float]]:
    designs = [list(d) for d in (args_cli.design or [])]
    for lengths in args_cli.lengths or []:
        designs.append([length / base for length, base in zip(lengths, BASE_SEGMENT_LENGTHS)])
    rng = random.Random(args_cli.seed)
    lo, hi = DESIGN_BOUNDS
    for _ in range(args_cli.random):
        designs.append([round(rng.uniform(lo, hi), 3) for _ in range(6)])
    if not designs:
        designs.append([1.0] * 6)
    for design in designs:
        for name, xi in zip(LEG_JOINT_NAMES, design):
            if not lo <= xi <= hi:
                print(f"[warn] {name} scale {xi:.3f} outside paper bounds [{lo}, {hi}]")
    return designs


def main():
    designs = collect_designs()
    configs = []
    for i, design in enumerate(designs):
        out_dir = os.path.join(args_cli.out, f"design_{i:02d}") if args_cli.out else None
        cfg = build_half_cheetah_asset(design, prim_path=f"/World/Cheeta_{i:02d}/Robot", output_dir=out_dir)
        configs.append(cfg)
        lengths = segment_lengths_from_design(design)
        print(f"[design {i:02d}] xi = {[f'{v:.3f}' for v in design]}")
        print(f"[design {i:02d}] segment lengths [m] = {[f'{v:.3f}' for v in lengths]}")
        print(f"[design {i:02d}] usd = {cfg.spawn.usd_path}")

    if not args_cli.preview:
        return

    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120))
    n = len(configs)
    sim.set_camera_view(eye=(0.0, -3.0 * max(n, 2), 2.0), target=(0.0, 3.0 * (n - 1) / 2, 0.5))
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/ground", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    robots = []
    for i, cfg in enumerate(configs):
        # one row of cheetahs, 3 m apart along y
        cfg.init_state.pos = (0.0, 3.0 * i, cfg.init_state.pos[2])
        robots.append(Articulation(cfg))
    sim.reset()

    print(f"[preview] simulating {n} design(s); close the window or Ctrl+C to stop")
    step = 0
    while simulation_app.is_running():
        for robot in robots:
            robot.set_joint_position_target(robot.data.default_joint_pos)
            robot.write_data_to_sim()
        sim.step()
        for robot in robots:
            robot.update(sim.get_physics_dt())
        step += 1
        if args_cli.preview_steps and step >= args_cli.preview_steps:
            break


if __name__ == "__main__":
    main()
    simulation_app.close()
