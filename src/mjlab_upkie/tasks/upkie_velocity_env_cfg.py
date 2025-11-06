"""Upkie velocity task environment configuration."""

import math
from dataclasses import dataclass, field
import torch
import numpy as np

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.manager_term_config import (
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewardTerm,
    TerminationTermCfg as DoneTerm,
    EventTermCfg as EventTerm,
    CurriculumTermCfg as CurrTerm,
    term,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig
from mjlab_upkie.robot.upkie_constants import UPKIE_CFG
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from mjlab.third_party.isaaclab.isaaclab.utils.math import sample_uniform
from mjlab.envs import mdp, ManagerBasedRlEnv

from mjlab.tasks.velocity import mdp as mdp_vel
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab.scene import SceneCfg, Scene
from mjlab.terrains import TerrainImporterCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG

from typing import cast, TypedDict

SCENE_CFG = SceneCfg(
    terrain=TerrainImporterCfg(
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=0,
    ),
    num_envs=1,
    extent=2.0,
)

VIEWER_CONFIG = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    asset_name="robot",
    body_name="trunk",
    distance=3.0,
    elevation=10.0,
    azimuth=90.0,
)

SIM_CFG = SimulationCfg(
    mujoco=MujocoCfg(
        timestep=0.005,
        iterations=1,
    ),
)

POS_CTRL_JOINT_NAMES = ["left_hip", "left_knee", "right_hip", "right_knee"]
VEL_CTRL_JOINT_NAMES = ["left_wheel", "right_wheel"]

LEFT_HIP = 0
LEFT_KNEE = 1
LEFT_WHEEL = 2
RIGHT_HIP = 3
RIGHT_KNEE = 4
RIGHT_WHEEL = 5

# Joint indices (without floating base)
POSITION_JOINTS = np.array([LEFT_HIP, LEFT_KNEE, RIGHT_HIP, RIGHT_KNEE])
VELOCITY_JOINTS = np.array([LEFT_WHEEL, RIGHT_WHEEL])

# Having different action scales for the wheels and legs seems to have no effect
UPKIE_ACTION_SCALE: dict[str, float] = {
    "left_hip": 1.0,
    "left_knee": 1.0,
    "right_hip": 1.0,
    "right_knee": 1.0,
    "left_wheel": 1.0,
    "right_wheel": 1.0,
}

@dataclass
class ActionCfg:
    joint_pos: mdp.JointPositionActionCfg = term(
        mdp.JointPositionActionCfg,
        asset_name="robot",
        actuator_names=[".*"],
        scale=UPKIE_ACTION_SCALE,
        use_default_offset=False,
    )
    

@dataclass
class CommandsCfg:
    twist: mdp_vel.UniformVelocityCommandCfg = term(
        mdp_vel.UniformVelocityCommandCfg,
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.3,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp_vel.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.5, 0.5),
            heading=(-math.pi, math.pi),
        ),
    )


@dataclass
class ObservationCfg:
    @dataclass
    class PolicyCfg(ObsGroup):
        joints_pos: ObsTerm = term(
            ObsTerm, func=lambda env: env.sim.data.qpos[:, POSITION_JOINTS + 7]
        )  # Position of the POS_CTRL_JOINT_NAMES
        joints_vel: ObsTerm = term(
            ObsTerm, func=lambda env: env.sim.data.qvel[:, VELOCITY_JOINTS + 6]
        )  # Velocity of the VEL_CTRL_JOINT_NAMES
        trunk_imu: ObsTerm = term(ObsTerm, func=lambda env: env.sim.data.qpos[:, 3:7])  # Quaternion of the trunk
        trunk_gyro: ObsTerm = term(
            ObsTerm, func=lambda env: env.sim.data.qvel[:, 3:6]
        )  # Angular velocity of the trunk

        actions: ObsTerm = term(ObsTerm, func=mdp.last_action)
        command: ObsTerm = term(ObsTerm, func=mdp.generated_commands, params={"command_name": "twist"})

        def __post_init__(self):
            self.enable_corruption = True

    @dataclass
    class CriticCfg(PolicyCfg):
        trunk_lin_vel: ObsTerm = term(
            ObsTerm, func=lambda env: env.sim.data.qvel[:, 0:3]
        )  # Linear velocity of the trunk

        def __post_init__(self):
            super().__post_init__()
            self.enable_corruption = False

    policy: PolicyCfg = field(default_factory=PolicyCfg)
    critic: CriticCfg = field(default_factory=CriticCfg)


def straight_legs(
    env: ManagerBasedRlEnv,
    std: float,
) -> torch.Tensor:
    """Reward straightening the legs (robot standing upright)."""
    joints_pos = env.sim.data.qpos[:, POSITION_JOINTS + 7]
    error = torch.sum(torch.square(joints_pos), dim=1)
    return torch.exp(-error / std**2)

POS_CTRL_JOINTS_DEFAULT: dict[str, float] = {
    "left_hip": -0.8,
    "left_knee": 1.0,
    "right_hip": -0.8,
    "right_knee": 1.0,
}

def backward_legs(
    env: ManagerBasedRlEnv,
    std: float,
) -> torch.Tensor:
    """Reward having a backward knee angle (for style only)."""
    joints_pos = env.sim.data.qpos[:, POSITION_JOINTS + 7]
    target_pos = torch.tensor([
        POS_CTRL_JOINTS_DEFAULT["left_hip"],
        POS_CTRL_JOINTS_DEFAULT["left_knee"],
        POS_CTRL_JOINTS_DEFAULT["right_hip"],
        POS_CTRL_JOINTS_DEFAULT["right_knee"],
    ], device=env.device)
    error = torch.sum(torch.square(joints_pos - target_pos), dim=1)
    return torch.exp(-error / std**2)

@dataclass
class RewardCfg:
    track_linear_velocity: RewardTerm = term(
        RewardTerm,
        func=mdp_vel.track_linear_velocity,
        weight=2.0,
        params={"command_name": "twist", "std": math.sqrt(0.1)},
    )
    track_angular_velocity: RewardTerm = term(
        RewardTerm,
        func=mdp_vel.track_angular_velocity,
        weight=2.5,
        params={"command_name": "twist", "std": math.sqrt(0.1)},
    )
    upright: RewardTerm = term(
        RewardTerm,
        func=mdp_vel.flat_orientation,
        weight=1.0,
        params={
            "std": math.sqrt(0.2),
            "asset_cfg": SceneEntityCfg("robot", body_names=[]),  # Override in robot cfg.
        },
    )
    action_rate_l2: RewardTerm = term(RewardTerm, func=mdp.action_rate_l2, weight=-0.1)
    pose: RewardTerm = term(
        RewardTerm,
        func=straight_legs,
        weight=0.1,
        params={"std": math.sqrt(0.1)},
    )


@dataclass
class EventCfg:
    reset_base: EventTerm = term(
        EventTerm,
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
        },
    )
    reset_robot_joints: EventTerm = term(
        EventTerm,
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
        },
    )
    foot_friction: EventTerm = term(
        EventTerm,
        mode="startup",
        func=mdp.randomize_field,
        params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=[]),  # Override in robot cfg.
        "operation": "abs",
        "field": "geom_friction",
        "ranges": (0.8, 1.2),
        },
    )

    # print_debug: EventTerm = term(
    #     EventTerm,
    #     func= lambda env, env_ids: print(f"{env.sim.data.qpos[:, :3]}"),
    #     mode="interval",
    #     interval_range_s=(0.0, 0.0),
    # )


@dataclass
class EventCfgWithPushes(EventCfg):
    push_robot: EventTerm | None = term(
        EventTerm,
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (0.0, 0.0)}},
    )


@dataclass
class TerminationCfg:
    time_out: DoneTerm = term(DoneTerm, func=mdp.time_out, time_out=True)
    fell_over: DoneTerm = term(DoneTerm, func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)})
    illegal_contact: DoneTerm | None = term(
        DoneTerm,
        func=mdp_vel.illegal_contact,
        params={"sensor_name": "nonfoot_ground_touch"},
    )


class PushIntensityStage(TypedDict):
  step: int
  velocity_range_x: tuple[float, float] | None
  velocity_range_y: tuple[float, float] | None

# def increase_push_intensity(
#     env: ManagerBasedRlEnv,
#     env_ids: torch.Tensor,
#     event_name: str,
#     intensities: list[PushIntensityStage],
#     ) -> None:
#     event_cfg = env.event_manager.get_term_cfg(event_name)
#     assert event_cfg is not None
#     cfg = cast(UpkieVelocityEnvCfg, event_cfg)
#     for stage in intensities:
#         if env.common_step_counter > stage["step"]:
#         if "velocity_range_x" in stage and stage["velocity_range_x"] is not None:
#             cfg.ranges.velocity_range_x = stage["velocity_range_x"]
#         if "velocity_range_y" in stage and stage["velocity_range_y"] is not None:
#             cfg.ranges.velocity_range_y = stage["velocity_range_y"]

@dataclass
class CurriculumCfg:
    pass
    # push_intensity: CurrTerm | None = term(
    #     CurrTerm, func=increase_push_intensity, params={"event_name": "push_robot"}
    # )
    # terrain_levels: CurrTerm | None = term(
    #     CurrTerm, func=mdp.terrain_levels_vel, params={"command_name": "twist"}
    # )
    # command_vel: CurrTerm | None = term(
    #     CurrTerm,
    #     func=mdp_vel.commands_vel,
    #     params={
    #     "command_name": "twist",
    #     "velocity_stages": [
    #         {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
    #         {"step": 5000 * 24, "lin_vel_x": (-1.5, 2.0), "ang_vel_z": (-0.7, 0.7)},
    #         {"step": 10000 * 24, "lin_vel_x": (-2.0, 3.0)},
    #     ],
    #     },
    # )


@dataclass
class UpkieVelocityEnvCfg(ManagerBasedRlEnvCfg):
    scene: SceneCfg = field(default_factory=lambda: SCENE_CFG)
    viewer: ViewerConfig = field(default_factory=lambda: VIEWER_CONFIG)
    sim: SimulationCfg = field(default_factory=lambda: SIM_CFG)
    actions: ActionCfg = field(default_factory=ActionCfg)
    commands: CommandsCfg = field(default_factory=CommandsCfg)
    observations: ObservationCfg = field(default_factory=ObservationCfg)
    rewards: RewardCfg = field(default_factory=RewardCfg)
    events: EventCfgWithPushes = field(default_factory=EventCfgWithPushes)
    terminations: TerminationCfg = field(default_factory=TerminationCfg)
    # curriculum: CurriculumCfg = field(default_factory=CurriculumCfg)
    decimation: int = 4
    episode_length_s: float = 20.0

    def __post_init__(self):
        self.events.reset_base.params["pose_range"]["z"] = (0.58, 0.6)

        self.scene.entities = {"robot": UPKIE_CFG}

        nonfoot_ground_cfg = ContactSensorCfg(
            name="nonfoot_ground_touch",
            primary=ContactMatch(
                mode="geom",
                entity="robot",
                # Grab all collision geoms
                pattern=r".*_collision\d*$",
                # Except for the foot geoms.
                exclude=tuple(["left_foot_collision", "right_foot_collision"]),
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found",),
            reduce="none",
            num_slots=1,
        )
        self.scene.sensors = (nonfoot_ground_cfg,)

        self.events.foot_friction.params["asset_cfg"].geom_names = [
            "left_foot_collision",
            "right_foot_collision",
        ]

        self.actions.joint_pos.scale = 0.75

        self.viewer.body_name = "trunk"
        self.commands.twist.viz.z_offset = 0.75

        # Flat terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        self.sim.nconmax = 256
        self.sim.njmax = 512


@dataclass
class UpkieVelocityEnvNoPushCfg(UpkieVelocityEnvCfg):
    events: EventCfg = field(default_factory=EventCfg)  # Override to remove pushes


@dataclass
class UpkieVelocityEnvLegsBackwardCfg(UpkieVelocityEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.pose.func = backward_legs  # Modify the pose reward to favor backward legs


@dataclass
class UpkieVelocityEnvCfg_PLAY(UpkieVelocityEnvCfg):
    episode_length_s: float = 1e9  # Very long episodes for PLAY mode


@dataclass
class UpkieCfg(RslRlOnPolicyRunnerCfg):
    policy: RslRlPpoActorCriticCfg = field(
        default_factory=lambda: RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_obs_normalization=False,
            critic_obs_normalization=False,
            actor_hidden_dims=(512, 256, 128),
            critic_hidden_dims=(512, 256, 128),
            activation="elu",
        )
    )
    algorithm: RslRlPpoAlgorithmCfg = field(
        default_factory=lambda: RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        )
    )
    wandb_project: str = "mjlab_upkie"
    experiment_name: str = "upkie_velocity"
    save_interval: int = 250
    num_steps_per_env: int = 24
    max_iterations: int = 30_000
