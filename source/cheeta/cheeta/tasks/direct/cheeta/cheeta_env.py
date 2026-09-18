# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from cheeta.design_builder import LEG_JOINT_NAMES, build_half_cheetah_asset
from cheeta.designs import normalize_design, validate_design
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .cheeta_env_cfg import BASE_OBS_DIM, DESIGN_OBS_DIM, CheetaEnvCfg


class CheetaEnv(DirectRLEnv):
    """Planar half cheetah: torque-controlled legs, reward for forward speed.

    One design for all envs (``cfg.design_vector``) or a set of designs spread
    over the envs (``cfg.design_set``, env i gets design i % N). With
    ``cfg.observe_design`` the normalised design vector is part of the
    observation so a single policy can serve every morphology.
    """

    cfg: CheetaEnvCfg

    def __init__(self, cfg: CheetaEnvCfg, render_mode: str | None = None, **kwargs):
        # resolve the designs before the scene is built
        if cfg.design_set:
            self._design_list = [validate_design(d) for d in cfg.design_set]
            # heterogeneous envs: every env holds its own copy of the asset
            cfg.scene.replicate_physics = False
        else:
            self._design_list = [validate_design(cfg.design_vector)]
        cfg.observation_space = BASE_OBS_DIM + (DESIGN_OBS_DIM if cfg.observe_design else 0)
        super().__init__(cfg, render_mode, **kwargs)

        self._leg_ids, _ = self.robot.find_joints(LEG_JOINT_NAMES, preserve_order=True)
        self._rootx_id, _ = self.robot.find_joints("rootx")
        self._rootz_id, _ = self.robot.find_joints("rootz")
        self._rooty_id, _ = self.robot.find_joints("rooty")
        self._gears = torch.tensor(self.cfg.gears, device=self.device)
        self.actions = torch.zeros((self.num_envs, len(self._leg_ids)), device=self.device)
        # per-env cumulative reward of the running episode (the design's score)
        self._episode_return = torch.zeros(self.num_envs, device=self.device)

        # per-env design bookkeeping: design index and (normalised) design vector
        n = len(self._design_list)
        self.design_ids = torch.arange(self.num_envs, device=self.device) % n
        designs = torch.tensor(self._design_list, dtype=torch.float32, device=self.device)
        self.designs = designs[self.design_ids]  # (num_envs, 6) raw scale factors
        normalised = torch.tensor([normalize_design(d) for d in self._design_list], dtype=torch.float32, device=self.device)
        self._design_obs = normalised[self.design_ids]

    @property
    def num_designs(self) -> int:
        return len(self._design_list)

    @property
    def design_list(self) -> list[list[float]]:
        """The distinct designs in the scene, indexed by ``design_ids``."""
        return [list(d) for d in self._design_list]

    def per_design_mean(self, values: torch.Tensor) -> torch.Tensor:
        """Mean of a per-env tensor for each distinct design -> (num_designs,)."""
        n = self.num_designs
        sums = torch.zeros(n, device=values.device, dtype=torch.float32).index_add_(0, self.design_ids, values.float())
        counts = torch.zeros(n, device=values.device, dtype=torch.float32).index_add_(
            0, self.design_ids, torch.ones_like(values, dtype=torch.float32)
        )
        return sums / counts.clamp(min=1.0)

    def _setup_scene(self):
        if self.cfg.robot_cfg is None:
            design = self._design_list if len(self._design_list) > 1 else self._design_list[0]
            self.cfg.robot_cfg = build_half_cheetah_asset(design, prim_path="/World/envs/env_.*/Robot")
        self.robot = Articulation(self.cfg.robot_cfg)
        # ground plane with the task's friction
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=self.cfg.ground_friction,
                    dynamic_friction=self.cfg.ground_friction,
                )
            ),
        )
        if self.cfg.scene.replicate_physics:
            # clone env_0 into the other envs (identical assets)
            self.scene.clone_environments(copy_from_source=False)
        else:
            # heterogeneous envs were cloned as empty xforms by the scene and filled
            # per env by the multi-asset spawner; only inter-env collisions remain
            self.scene.filter_collisions(global_prim_paths=[])
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu" and self.cfg.scene.replicate_physics:
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    @property
    def distance_x(self) -> torch.Tensor:
        """Forward distance [m] of each env since its last reset (rootx joint position)."""
        return self.robot.data.joint_pos[:, self._rootx_id[0]]

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        self.robot.set_joint_effort_target(self.actions * self._gears, joint_ids=self._leg_ids)

    def _get_observations(self) -> dict:
        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel
        parts = [
            joint_pos[:, self._rootz_id],  # relative height
            joint_pos[:, self._rooty_id],  # pitch
            joint_pos[:, self._leg_ids],  # 6 leg joint positions
            joint_vel[:, self._rootx_id],  # horizontal speed
            joint_vel[:, self._rootz_id],  # vertical speed
            joint_vel[:, self._rooty_id],  # angular velocity
            joint_vel[:, self._leg_ids],  # 6 leg joint velocities
        ]
        if self.cfg.observe_design:
            parts.append(self._design_obs)  # 6 normalised design scales
        return {"policy": torch.cat(parts, dim=-1)}

    def _get_rewards(self) -> torch.Tensor:
        # default: r(s) = max(v_x / 10, 0), design-independent forward reward (paper Sec. 7.1)
        forward_speed = self.robot.data.joint_vel[:, self._rootx_id[0]]
        reward = forward_speed * self.cfg.reward_speed_scale
        if self.cfg.clip_speed_reward:
            reward = torch.clamp(reward, min=0.0)
        if self.cfg.ctrl_cost_weight > 0.0:
            reward -= self.cfg.ctrl_cost_weight * self.actions.square().sum(dim=-1)
        self._episode_return += reward
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        no_termination = torch.zeros_like(time_out)
        return no_termination, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        # log score (episodic return) and distance of the finished episodes
        self.extras["log"] = {
            "distance_x_m": self.robot.data.joint_pos[env_ids, self._rootx_id[0]].mean().item(),
            "episode_return": self._episode_return[env_ids].mean().item(),
        }
        self._episode_return[env_ids] = 0.0
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_pos[:, self._leg_ids] += sample_uniform(
            -self.cfg.reset_noise,
            self.cfg.reset_noise,
            (len(env_ids), len(self._leg_ids)),
            joint_pos.device,
        )
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
