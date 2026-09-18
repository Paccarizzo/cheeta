# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Half-Cheetah locomotion task from Luck, Ben Amor & Calandra,
"Data-efficient Co-Adaptation of Morphology and Behaviour with Deep RL"
(CoRL 2019), Sec. 7.1:

- state (17): rel. height, pitch, 6 joint positions, horizontal speed,
  vertical speed, angular velocity, 6 joint velocities
- design observation (+6, optional): the design vector xi mapped to [-1, 1],
  so one policy can be trained across morphologies (the paper's population
  network conditioned on xi)
- actions (6): torques on the leg joints
- reward: r(s) = max(v_x / 10, 0)
- design vector xi in [0.8, 2.0]^6 scales the six leg segment lengths
"""

from cheeta.design_builder import JOINT_GEARS, LEG_JOINT_NAMES

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

BASE_OBS_DIM = 17
DESIGN_OBS_DIM = 6


@configclass
class CheetaEnvCfg(DirectRLEnvCfg):
    # env: physics at 120 Hz, control at 60 Hz, ~1000-step episodes as in PyBullet
    decimation = 2
    episode_length_s = 16.67
    # - spaces definition (observation_space is finalised by the env from observe_design)
    action_space = 6
    observation_space = BASE_OBS_DIM + DESIGN_OBS_DIM
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # design vector xi (paper Sec. 7.1): scale factors of the base segment
    # lengths, one per leg segment, each in [0.8, 2.0]; ones = original design
    design_vector: list[float] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # several designs in one scene: env i gets design_set[i % len(design_set)];
    # overrides design_vector and disables physics replication (heterogeneous envs)
    design_set: list[list[float]] | None = None
    # append the normalised design vector to the observation (design-conditioned policy)
    observe_design: bool = True
    # built from design_vector / design_set on scene setup unless provided explicitly;
    # kept out of __post_init__ so hydra/CLI overrides of design_vector take effect
    robot_cfg: ArticulationCfg | None = None

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=6.0, replicate_physics=True)

    # actions are joint torques: tau = gear * action, action in [-1, 1]
    gears: list[float] = [JOINT_GEARS[name] for name in LEG_JOINT_NAMES]
    # reward: speed term + control cost
    #   r = max(v_x * reward_speed_scale, 0) - ctrl_cost_weight * |a|^2   (clip_speed_reward=True)
    #   r = v_x * reward_speed_scale         - ctrl_cost_weight * |a|^2   (clip_speed_reward=False)
    # defaults reproduce the paper's design-independent reward r(s) = max(v_x / 10, 0);
    # clip_speed_reward=False, ctrl_cost_weight=0.1 gives a Gym-HalfCheetah-style reward
    reward_speed_scale = 0.1
    clip_speed_reward = True
    ctrl_cost_weight = 0.0
    # uniform noise added to the leg joint angles on reset [rad]
    reset_noise = 0.1
    # ground friction (MuJoCo half cheetah uses 0.4)
    ground_friction = 0.4
