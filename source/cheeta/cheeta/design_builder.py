"""Procedural Half-Cheetah asset generation for the co-adaptation task.

Morphology follows the Half-Cheetah task of Luck, Ben Amor & Calandra,
"Data-efficient Co-Adaptation of Morphology and Behaviour with Deep
Reinforcement Learning" (CoRL 2019), Sec. 7.1: the design vector xi scales the
lengths of the six leg segments (bthigh, bshin, bfoot, fthigh, fshin, ffoot)
with base lengths (0.29, 0.30, 0.188, 0.29, 0.30, 0.188) m and xi_i in [0.8, 2.0].

The robot is restricted to the x-z plane MuJoCo-style: a fixed world link
connects to the torso through two passive prismatic joints (rootx, rootz) and
one passive revolute joint (rooty). Only the six leg joints are actuated.

Pipeline: design vector -> URDF -> USD (Isaac Lab UrdfConverter) -> ArticulationCfg.
`half_cheetah_urdf` is dependency-free; `build_half_cheetah_asset` needs the
Isaac Sim app to be running.
"""

import math
import os
import tempfile
from collections.abc import Sequence

LEG_JOINT_NAMES = ["bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot"]
ROOT_JOINT_NAMES = ["rootx", "rootz", "rooty"]

# base segment lengths scaled by the design vector (paper Sec. 7.1)
BASE_SEGMENT_LENGTHS = (0.29, 0.30, 0.188, 0.29, 0.30, 0.188)
DESIGN_BOUNDS = (0.8, 2.0)

# passive joint springs/dampers and actuator gears from the MuJoCo/PyBullet
# half_cheetah model the paper builds on
JOINT_STIFFNESS = {"bthigh": 240.0, "bshin": 180.0, "bfoot": 120.0, "fthigh": 180.0, "fshin": 120.0, "ffoot": 60.0}
JOINT_DAMPING = {"bthigh": 6.0, "bshin": 4.5, "bfoot": 3.0, "fthigh": 4.5, "fshin": 3.0, "ffoot": 1.5}
JOINT_GEARS = {"bthigh": 120.0, "bshin": 90.0, "bfoot": 60.0, "fthigh": 120.0, "fshin": 60.0, "ffoot": 30.0}
JOINT_RANGES = {
    "bthigh": (-0.52, 1.05),
    "bshin": (-0.785, 0.785),
    "bfoot": (-0.4, 0.785),
    "fthigh": (-1.0, 0.7),
    "fshin": (-1.2, 0.87),
    "ffoot": (-0.5, 0.5),
}
# rest pitch of each segment relative to its parent, baked into the URDF joint
# frames so that joint angle 0 = standing pose (zigzag legs, cheetah-like)
REST_ANGLES = {"bthigh": -0.55, "bshin": 1.1, "bfoot": -0.6, "fthigh": 0.5, "fshin": -0.9, "ffoot": 0.45}

TORSO_LENGTH = 1.0
LIMB_RADIUS = 0.046
DENSITY = 1000.0


def _small_link(name: str) -> str:
    return (
        f'  <link name="{name}">\n'
        "    <inertial>\n"
        '      <mass value="0.01"/>\n'
        '      <inertia ixx="1e-5" iyy="1e-5" izz="1e-5" ixy="0" ixz="0" iyz="0"/>\n'
        "    </inertial>\n"
        "  </link>"
    )


def _segment_link(name: str, length: float) -> str:
    """Solid cylinder along -Z of the link frame (hangs from its joint)."""
    r = LIMB_RADIUS
    mass = DENSITY * math.pi * r * r * length
    i_trans = mass * (3 * r * r + length * length) / 12.0
    i_axial = mass * r * r / 2.0
    geom = f'<geometry><cylinder radius="{r}" length="{length}"/></geometry>'
    return (
        f'  <link name="{name}">\n'
        f"    <inertial>\n"
        f'      <origin xyz="0 0 {-length / 2}" rpy="0 0 0"/>\n'
        f'      <mass value="{mass:.4f}"/>\n'
        f'      <inertia ixx="{i_trans:.6f}" iyy="{i_trans:.6f}" izz="{i_axial:.6f}" ixy="0" ixz="0" iyz="0"/>\n'
        f"    </inertial>\n"
        f'    <visual><origin xyz="0 0 {-length / 2}" rpy="0 0 0"/>{geom}</visual>\n'
        f'    <collision><origin xyz="0 0 {-length / 2}" rpy="0 0 0"/>{geom}</collision>\n'
        f"  </link>"
    )


def _leg_joint(name: str, parent: str, xyz: tuple[float, float, float]) -> str:
    lo, hi = JOINT_RANGES[name]
    rest = REST_ANGLES[name]
    gear = JOINT_GEARS[name]
    return (
        f'  <joint name="{name}" type="revolute">\n'
        f'    <parent link="{parent}"/>\n'
        f'    <child link="{name}_link"/>\n'
        f'    <origin xyz="{xyz[0]} {xyz[1]} {xyz[2]}" rpy="0 {rest} 0"/>\n'
        f'    <axis xyz="0 1 0"/>\n'
        f'    <limit lower="{lo}" upper="{hi}" effort="{gear}" velocity="50"/>\n'
        f"  </joint>"
    )


def _root_joint(name: str, jtype: str, parent: str, child: str, axis: str, limit: float) -> str:
    return (
        f'  <joint name="{name}" type="{jtype}">\n'
        f'    <parent link="{parent}"/>\n'
        f'    <child link="{child}"/>\n'
        f'    <origin xyz="0 0 0" rpy="0 0 0"/>\n'
        f'    <axis xyz="{axis}"/>\n'
        f'    <limit lower="{-limit}" upper="{limit}" effort="1e6" velocity="1000"/>\n'
        f"  </joint>"
    )


def segment_lengths_from_design(design: Sequence[float]) -> list[float]:
    if len(design) != 6:
        raise ValueError(f"design vector must have 6 entries, got {len(design)}")
    return [x * base for x, base in zip(design, BASE_SEGMENT_LENGTHS)]


def standing_height(segment_lengths: Sequence[float]) -> float:
    """Vertical distance from hip axis to toe tip in the rest pose."""

    def drop(lengths: Sequence[float], names: Sequence[str]) -> float:
        angle, z = 0.0, 0.0
        for length, name in zip(lengths, names):
            angle += REST_ANGLES[name]
            z += length * math.cos(angle)
        return z

    return max(drop(segment_lengths[0:3], LEG_JOINT_NAMES[0:3]), drop(segment_lengths[3:6], LEG_JOINT_NAMES[3:6]))


def half_cheetah_urdf(segment_lengths: Sequence[float]) -> str:
    """URDF for the planar half cheetah with the given six segment lengths [m]."""
    lengths = dict(zip(LEG_JOINT_NAMES, segment_lengths))
    r = LIMB_RADIUS
    torso_mass = DENSITY * math.pi * r * r * TORSO_LENGTH
    t_axial = torso_mass * r * r / 2.0
    t_trans = torso_mass * (3 * r * r + TORSO_LENGTH * TORSO_LENGTH) / 12.0
    torso_geom = f'<geometry><cylinder radius="{r}" length="{TORSO_LENGTH}"/></geometry>'

    parts = ['<robot name="half_cheetah">']
    # planar root: world -(rootx)-> slide_x -(rootz)-> slide_z -(rooty)-> torso
    parts.append(_small_link("world"))
    parts.append(_root_joint("rootx", "prismatic", "world", "slide_x", "1 0 0", 1000.0))
    parts.append(_small_link("slide_x"))
    parts.append(_root_joint("rootz", "prismatic", "slide_x", "slide_z", "0 0 1", 1000.0))
    parts.append(_small_link("slide_z"))
    parts.append(_root_joint("rooty", "revolute", "slide_z", "torso", "0 1 0", 20 * math.pi))
    # torso: horizontal cylinder along +x (nose at +x)
    parts.append(
        f'  <link name="torso">\n'
        f"    <inertial>\n"
        f'      <mass value="{torso_mass:.4f}"/>\n'
        f'      <inertia ixx="{t_axial:.6f}" iyy="{t_trans:.6f}" izz="{t_trans:.6f}" ixy="0" ixz="0" iyz="0"/>\n'
        f"    </inertial>\n"
        f'    <visual><origin xyz="0 0 0" rpy="0 {math.pi / 2} 0"/>{torso_geom}</visual>\n'
        f'    <collision><origin xyz="0 0 0" rpy="0 {math.pi / 2} 0"/>{torso_geom}</collision>\n'
        f"  </link>"
    )
    # head, merged into the torso on conversion (fixed joint)
    parts.append(
        '  <joint name="head_joint" type="fixed">\n'
        '    <parent link="torso"/>\n'
        '    <child link="head"/>\n'
        f'    <origin xyz="{TORSO_LENGTH / 2} 0 0.03" rpy="0 -2.6 0"/>\n'
        "  </joint>"
    )
    parts.append(_segment_link("head", 0.28))
    # back leg at the rear of the torso, front leg at the nose
    parts.append(_leg_joint("bthigh", "torso", (-TORSO_LENGTH / 2, 0.0, 0.0)))
    parts.append(_segment_link("bthigh_link", lengths["bthigh"]))
    parts.append(_leg_joint("bshin", "bthigh_link", (0.0, 0.0, -lengths["bthigh"])))
    parts.append(_segment_link("bshin_link", lengths["bshin"]))
    parts.append(_leg_joint("bfoot", "bshin_link", (0.0, 0.0, -lengths["bshin"])))
    parts.append(_segment_link("bfoot_link", lengths["bfoot"]))
    parts.append(_leg_joint("fthigh", "torso", (TORSO_LENGTH / 2, 0.0, 0.0)))
    parts.append(_segment_link("fthigh_link", lengths["fthigh"]))
    parts.append(_leg_joint("fshin", "fthigh_link", (0.0, 0.0, -lengths["fthigh"])))
    parts.append(_segment_link("fshin_link", lengths["fshin"]))
    parts.append(_leg_joint("ffoot", "fshin_link", (0.0, 0.0, -lengths["fshin"])))
    parts.append(_segment_link("ffoot_link", lengths["ffoot"]))
    parts.append("</robot>")
    return "\n".join(parts)


def build_half_cheetah_asset(
    design: Sequence[float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    prim_path: str = "/World/envs/env_.*/Robot",
    output_dir: str | None = None,
):
    """Create the ArticulationCfg for a half cheetah with design vector ``design``.

    ``design`` are the six scale factors xi of the paper; segment lengths are
    xi_i * base_i. Requires the Isaac Sim app to be running (URDF -> USD).
    """
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    lengths = segment_lengths_from_design(design)

    if output_dir is None:
        key = "_".join(f"{v:.4g}" for v in design)
        output_dir = os.path.join(tempfile.gettempdir(), "cheeta_assets", "half_cheetah", key)
    os.makedirs(output_dir, exist_ok=True)
    urdf_path = os.path.join(output_dir, "half_cheetah.urdf")
    with open(urdf_path, "w") as f:
        f.write(half_cheetah_urdf(lengths))

    converter_cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=output_dir,
        usd_file_name="half_cheetah.usd",
        fix_base=True,  # the world link is fixed; motion happens in the root joints
        merge_fixed_joints=True,
        force_usd_conversion=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
    )
    usd_path = UrdfConverter(converter_cfg).usd_path

    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=10.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        # joint angle 0 = rest pose; drop the toes just above the ground
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, standing_height(lengths) + 0.05),
            joint_pos={".*": 0.0},
        ),
        actuators={
            # leg joints: passive spring/damper (MuJoCo-style) + feedforward torques
            "legs": ImplicitActuatorCfg(
                joint_names_expr=list(LEG_JOINT_NAMES),
                stiffness=dict(JOINT_STIFFNESS),
                damping=dict(JOINT_DAMPING),
                effort_limit_sim=dict(JOINT_GEARS),
            ),
            # planar root joints: fully passive
            "root": ImplicitActuatorCfg(
                joint_names_expr=list(ROOT_JOINT_NAMES),
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )
