# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from cheeta.design_builder import LEG_JOINT_NAMES
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .cheeta_env_cfg import CheetaEnvCfg


class CheetaEnv(DirectRLEnv):
    """Planar half cheetah: torque-controlled legs, reward for forward speed."""

    cfg: CheetaEnvCfg

    def __init__(self, cfg: CheetaEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._leg_ids, _ = self.robot.find_joints(LEG_JOINT_NAMES, preserve_order=True)
        self._rootx_id, _ = self.robot.find_joints("rootx")
        self._rootz_id, _ = self.robot.find_joints("rootz")
        self._rooty_id, _ = self.robot.find_joints("rooty")
        self._gears = torch.tensor(self.cfg.gears, device=self.device)
        self.actions = torch.zeros((self.num_envs, len(self._leg_ids)), device=self.device)

    def _setup_scene(self):
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
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
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
        obs = torch.cat(
            (
                joint_pos[:, self._rootz_id],  # relative height
                joint_pos[:, self._rooty_id],  # pitch
                joint_pos[:, self._leg_ids],  # 6 leg joint positions
                joint_vel[:, self._rootx_id],  # horizontal speed
                joint_vel[:, self._rootz_id],  # vertical speed
                joint_vel[:, self._rooty_id],  # angular velocity
                joint_vel[:, self._leg_ids],  # 6 leg joint velocities
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # r(s) = max(v_x / 10, 0), design-independent forward reward (paper Sec. 7.1)
        forward_speed = self.robot.data.joint_vel[:, self._rootx_id[0]]
        return torch.clamp(forward_speed * self.cfg.reward_speed_scale, min=0.0)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        no_termination = torch.zeros_like(time_out)
        return no_termination, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        # log how far these envs ran during the finished episode
        self.extras["log"] = {"distance_x_m": self.robot.data.joint_pos[env_ids, self._rootx_id[0]].mean().item()}
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
