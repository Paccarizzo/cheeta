# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Half-Cheetah locomotion task from Luck, Ben Amor & Calandra,
"Data-efficient Co-Adaptation of Morphology and Behaviour with Deep RL"
(CoRL 2019), Sec. 7.1:

- state (17): rel. height, pitch, 6 joint positions, horizontal speed,
  vertical speed, angular velocity, 6 joint velocities
- actions (6): torques on the leg joints
- reward: r(s) = max(v_x / 10, 0)
- design vector xi in [0.8, 2.0]^6 scales the six leg segment lengths
"""

from cheeta.design_builder import JOINT_GEARS, LEG_JOINT_NAMES, build_half_cheetah_asset

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class CheetaEnvCfg(DirectRLEnvCfg):
    # env: physics at 120 Hz, control at 60 Hz, ~1000-step episodes as in PyBullet
    decimation = 2
    episode_length_s = 16.67
    # - spaces definition
    action_space = 6
    observation_space = 17
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # design vector xi (paper Sec. 7.1): scale factors of the base segment
    # lengths, one per leg segment, each in [0.8, 2.0]; ones = original design
    design_vector: list[float] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # built from design_vector in __post_init__ unless provided explicitly
    robot_cfg: ArticulationCfg | None = None

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=6.0, replicate_physics=True)

    # actions are joint torques: tau = gear * action, action in [-1, 1]
    gears: list[float] = [JOINT_GEARS[name] for name in LEG_JOINT_NAMES]
    # reward r(s) = max(v_x * reward_speed_scale, 0)
    reward_speed_scale = 0.1
    # uniform noise added to the leg joint angles on reset [rad]
    reset_noise = 0.1
    # ground friction (MuJoCo half cheetah uses 0.4)
    ground_friction = 0.4

    def __post_init__(self):
        parent_post_init = getattr(super(), "__post_init__", None)
        if parent_post_init is not None:
            parent_post_init()
        if self.robot_cfg is None:
            self.robot_cfg = build_half_cheetah_asset(self.design_vector, prim_path="/World/envs/env_.*/Robot")
