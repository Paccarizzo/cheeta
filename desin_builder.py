"""Procedurally build an Isaac Lab articulation asset from limb lengths and joint angles.

Pipeline:
    limb_lengths -> URDF (serial chain of cylinder links, revolute joints)
                 -> USD via Isaac Lab's UrdfConverter
                 -> ArticulationCfg with joint_angles as the initial joint positions

`build_urdf` has no dependencies, so the geometry can be generated and inspected
without Isaac Sim installed. `build_isaaclab_asset` requires Isaac Lab and must be
run inside an Isaac Sim python environment (after AppLauncher has started the app).
"""

import math
import os
import tempfile

joint_angles = [0, 0, 0, 0, 0, 0]
limb_lengths = [1.27, 0.87, 1.17, 0.84, 0.80, 0.82]


def build_urdf(
    limb_lengths: list[float],
    limb_radius: float = 0.05,
    density: float = 1000.0,
    joint_axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
    joint_limit: float = math.pi,
    effort_limit: float = 100.0,
    velocity_limit: float = 20.0,
    robot_name: str = "cheeta",
) -> str:
    """Return a URDF string for a serial chain: one revolute joint per limb.

    Link i is a solid cylinder of length ``limb_lengths[i]`` hanging along -Z from
    joint i, which sits at the tip of the previous link and rotates about ``joint_axis``.
    Joints are named ``joint_0 .. joint_{n-1}``.
    """
    ax = " ".join(str(a) for a in joint_axis)
    parts = [f'<robot name="{robot_name}">']
    parts.append(
        '  <link name="base_link">\n'
        "    <inertial>\n"
        '      <mass value="0.01"/>\n'
        '      <inertia ixx="1e-5" iyy="1e-5" izz="1e-5" ixy="0" ixz="0" iyz="0"/>\n'
        "    </inertial>\n"
        "  </link>"
    )

    parent = "base_link"
    parent_length = 0.0
    for i, length in enumerate(limb_lengths):
        mass = density * math.pi * limb_radius**2 * length
        # solid cylinder about its own center of mass
        i_trans = mass * (3 * limb_radius**2 + length**2) / 12.0
        i_axial = mass * limb_radius**2 / 2.0

        parts.append(
            f'  <joint name="joint_{i}" type="revolute">\n'
            f'    <parent link="{parent}"/>\n'
            f'    <child link="link_{i}"/>\n'
            f'    <origin xyz="0 0 {-parent_length}" rpy="0 0 0"/>\n'
            f'    <axis xyz="{ax}"/>\n'
            f'    <limit lower="{-joint_limit}" upper="{joint_limit}" '
            f'effort="{effort_limit}" velocity="{velocity_limit}"/>\n'
            f"  </joint>"
        )
        parts.append(
            f'  <link name="link_{i}">\n'
            f"    <inertial>\n"
            f'      <origin xyz="0 0 {-length / 2.0}" rpy="0 0 0"/>\n'
            f'      <mass value="{mass}"/>\n'
            f'      <inertia ixx="{i_trans}" iyy="{i_trans}" izz="{i_axial}" '
            f'ixy="0" ixz="0" iyz="0"/>\n'
            f"    </inertial>\n"
            f"    <visual>\n"
            f'      <origin xyz="0 0 {-length / 2.0}" rpy="0 0 0"/>\n'
            f'      <geometry><cylinder radius="{limb_radius}" length="{length}"/></geometry>\n'
            f"    </visual>\n"
            f"    <collision>\n"
            f'      <origin xyz="0 0 {-length / 2.0}" rpy="0 0 0"/>\n'
            f'      <geometry><cylinder radius="{limb_radius}" length="{length}"/></geometry>\n'
            f"    </collision>\n"
            f"  </link>"
        )
        parent = f"link_{i}"
        parent_length = length

    parts.append("</robot>")
    return "\n".join(parts)


def build_isaaclab_asset(
    limb_lengths: list[float],
    joint_angles: list[float],
    prim_path: str = "{ENV_REGEX_NS}/Robot",
    output_dir: str | None = None,
    fix_base: bool = True,
    stiffness: float = 100.0,
    damping: float = 10.0,
    **urdf_kwargs,
):
    """Create an Isaac Lab ArticulationCfg for a chain defined by limb_lengths.

    ``joint_angles[i]`` (radians) becomes the initial position of ``joint_i``.
    Returns the ArticulationCfg; the generated USD is written to ``output_dir``
    (a temp directory keyed by the design parameters if not given).
    """
    if len(joint_angles) != len(limb_lengths):
        raise ValueError(
            f"got {len(limb_lengths)} limb lengths but {len(joint_angles)} joint angles"
        )

    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    if output_dir is None:
        # key the cache dir on the design so different designs don't clobber each other
        key = "_".join(f"{v:.4g}" for v in limb_lengths)
        output_dir = os.path.join(tempfile.gettempdir(), "cheeta_assets", key)
    os.makedirs(output_dir, exist_ok=True)

    urdf_path = os.path.join(output_dir, "robot.urdf")
    with open(urdf_path, "w") as f:
        f.write(build_urdf(limb_lengths, **urdf_kwargs))

    converter_cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=output_dir,
        usd_file_name="robot.usd",
        fix_base=fix_base,
        merge_fixed_joints=True,
        force_usd_conversion=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=stiffness, damping=damping
            ),
        ),
    )
    usd_path = UrdfConverter(converter_cfg).usd_path

    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, sum(limb_lengths) if fix_base else 0.05),
            joint_pos={f"joint_{i}": angle for i, angle in enumerate(joint_angles)},
        ),
        actuators={
            "joints": ImplicitActuatorCfg(
                joint_names_expr=["joint_.*"],
                stiffness=stiffness,
                damping=damping,
            ),
        },
    )


if __name__ == "__main__":
    # URDF generation works without Isaac Sim; print it as a sanity check.
    print(build_urdf(limb_lengths))
