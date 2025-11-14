"""Upkie velocity task environment configuration."""

import math
from dataclasses import dataclass, field
from copy import deepcopy
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
from mjlab_upkie.robot.upkie_constants import (
    DEFAULT_UPKIE_CFG,
    RK_UPKIE_CFG,
    DEFAULT_HEIGHT,
    RK_HEIGHT,
    DEFAULT_POSE,
    RK_POS,
    POS_CTRL_JOINT_NAMES,
    VEL_CTRL_JOINT_NAMES,
    POS_CTRL_JOINT_IDS,
    VEL_CTRL_JOINT_IDS,
)
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.third_party.isaaclab.isaaclab.utils.math import sample_uniform
from mjlab.envs import mdp, ManagerBasedRlEnv
from mjlab.envs.manager_based_env import ManagerBasedEnv

from mjlab.tasks.velocity import mdp as mdp_vel
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab.tasks.velocity.velocity_env_cfg import create_velocity_env_cfg
from mjlab.utils.retval import retval

from mjlab.scene import SceneCfg, Scene
from mjlab.terrains import TerrainImporterCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG

from typing import cast, TypedDict

from mjlab.third_party.isaaclab.isaaclab.utils.math import (
    quat_apply_inverse,
    sample_uniform,
)

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


@dataclass
class ActionCfg:
    joint_pos: mdp.JointPositionActionCfg = term(
        mdp.JointPositionActionCfg,
        asset_name="robot",
        actuator_names=[".*"],
        scale=0.75,
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
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
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


# def straight_legs(
#     env: ManagerBasedRlEnv,
#     std: float,
# ) -> torch.Tensor:
#     """Reward straightening the legs (robot standing upright)."""
#     joints_pos = env.sim.data.qpos[:, POSITION_JOINTS + 7]
#     error = torch.sum(torch.square(joints_pos), dim=1)
#     return torch.exp(-error / std**2)


# def backward_legs(
#     env: ManagerBasedRlEnv,
#     std: float,
# ) -> torch.Tensor:
#     """Reward having a backward knee angle (for style only)."""
#     joints_pos = env.sim.data.qpos[:, POSITION_JOINTS + 7]
#     target_pos = torch.tensor(
#         [
#             BACKWARD_LEGS_POS["left_hip"],
#             BACKWARD_LEGS_POS["left_knee"],
#             BACKWARD_LEGS_POS["right_hip"],
#             BACKWARD_LEGS_POS["right_knee"],
#         ],
#         device=env.device,
#     )
#     error = torch.sum(torch.square(joints_pos - target_pos), dim=1)
#     return torch.exp(-error / std**2)


@dataclass
class RewardCfg:
    track_linear_velocity: RewardTerm = term(
        RewardTerm,
        func=mdp_vel.track_linear_velocity,
        weight=2.0,  # Overridden in env cfg
        params={"command_name": "twist", "std": math.sqrt(0.1)},
    )
    track_angular_velocity: RewardTerm = term(
        RewardTerm,
        func=mdp_vel.track_angular_velocity,
        weight=2.0,  # Overridden in env cfg
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


def push_by_setting_velocity(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    intensity: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    asset: mdp_vel.Entity = env.scene[asset_cfg.name]
    vel_w = asset.data.root_link_vel_w[env_ids]
    quat_w = asset.data.root_link_quat_w[env_ids]
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=env.device)
    vel_w += sample_uniform(
        intensity * ranges[:, 0],
        intensity * ranges[:, 1],
        vel_w.shape,
        device=env.device,
    )
    vel_w[:, 3:] = quat_apply_inverse(quat_w, vel_w[:, 3:])
    asset.write_root_link_velocity_to_sim(vel_w, env_ids=env_ids)


@dataclass
class EventCfg:
    reset_base: EventTerm = term(
        EventTerm,
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0, 0), "y": (0, 0), "yaw": (-math.pi, math.pi)},
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
    push_robot: EventTerm | None = term(
        EventTerm,
        func=push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={
            "velocity_range": {"x": (-0.0, 0.0), "y": (0.0, 0.0)},  # Overridden in cfg
            "intensity": 0.0,
        },
    )
    # print_debug: EventTerm = term(
    #     EventTerm,
    #     func= lambda env, env_ids: print(f"{env.sim.data.qpos[:, :3]}"),
    #     mode="interval",
    #     interval_range_s=(0.0, 0.0),
    # )


@dataclass
class TerminationCfg:
    time_out: DoneTerm = term(DoneTerm, func=mdp.time_out, time_out=True)
    fell_over: DoneTerm = term(DoneTerm, func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)})
    illegal_contact: DoneTerm | None = term(
        DoneTerm,
        func=mdp_vel.illegal_contact,
        params={"sensor_name": "nonfoot_ground_touch"},
    )


def increase_push_intensity(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    intensities: list[tuple[float, float]],
) -> None:
    push_event_cfg = env.event_manager.get_term_cfg("push_robot")
    for step_threshold, intensity in intensities:
        if env.common_step_counter >= step_threshold:
            push_event_cfg.params["intensity"] = intensity


@dataclass
class CurriculumCfg:
    command_vel: CurrTerm | None = term(
        CurrTerm,
        func=mdp_vel.commands_vel,
        params={
            "command_name": "twist",
            "velocity_stages": [],  # Override in cfg
        },
    )
    push_intensity: CurrTerm | None = term(
        CurrTerm,
        func=increase_push_intensity,
        params={
            "intensities": [],  # Override in cfg
        },
    )
    # linear_reward_weights: CurrTerm | None = term(
    #     CurrTerm,
    #     func=mdp_vel.reward_weight,
    #     params={
    #         "reward_name": "track_linear_velocity",
    #         "weight_stages": [{"step": 0, "weight": 0.0}, {"step": 1000 * 24, "weight": 2.0}],
    #     },
    # )
    # angular_reward_weights: CurrTerm | None = term(
    #     CurrTerm,
    #     func=mdp_vel.reward_weight,
    #     params={
    #         "reward_name": "track_angular_velocity",
    #         "weight_stages": [{"step": 0, "weight": 0.0}, {"step": 1000 * 24, "weight": 2.5}],
    #     },
    # )
    # terrain_levels: CurrTerm | None = term(
    #     CurrTerm, func=mdp_vel.terrain_levels_vel, params={"command_name": "twist"}
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
    events: EventCfg = field(default_factory=EventCfg)
    terminations: TerminationCfg = field(default_factory=TerminationCfg)
    curriculum: CurriculumCfg = field(default_factory=CurriculumCfg)
    decimation: int = 4
    episode_length_s: float = 20.0

    def __post_init__(self):
        self.scene.entities = {"robot": UPKIE_CFG}
        self.actions.joint_pos.scale = 0.75
        self.viewer.body_name = "trunk"
        self.commands.twist.viz.z_offset = 0.75
        self.sim.nconmax = 256
        self.sim.njmax = 512

        # Add non-foot ground contact sensor
        nonfoot_ground_cfg = ContactSensorCfg(
            name="nonfoot_ground_touch",
            primary=ContactMatch(
                mode="geom",
                entity="robot",
                pattern=r".*_collision\d*$",
                exclude=tuple(["left_foot_collision", "right_foot_collision"]),
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found",),
            reduce="none",
            num_slots=1,
        )
        self.scene.sensors = (nonfoot_ground_cfg,)

        # Set correct geom names for foot friction randomization
        self.events.foot_friction.params["asset_cfg"].geom_names = [
            "left_foot_collision",
            "right_foot_collision",
        ]

        # Flat terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # Reset height
        self.events.reset_base.params["pose_range"]["z"] = (
            DEFAULT_HEIGHT,
            DEFAULT_HEIGHT,
        )

        # Set tracking reward weights
        self.rewards.track_linear_velocity.weight = 2.0
        self.rewards.track_angular_velocity.weight = 2.5
        self.rewards.pose.weight = 0.2

        # Set curriculum velocity stages
        self.curriculum.command_vel.params["velocity_stages"] = [
            {"step": 0, "lin_vel_x": (-0.5, 0.5), "ang_vel_z": (-0.5, 0.5)},
            {"step": 6001 * 24, "lin_vel_x": (-0.75, 0.75), "ang_vel_z": (-1.0, 1.0)},
            {"step": 12001 * 24, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-1.5, 1.5)},
        ]

        self.commands.twist.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.twist.ranges.ang_vel_z = (-1.5, 1.5)


@retval
def UpkieVelocityEnvCfg() -> ManagerBasedRlEnvCfg:
    site_names = ["left_foot", "right_foot"]

    feet_sensor_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_foot_link|right_foot_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    foot_frictions_geom_names = (
        "Left_Foot_collision",
        "Right_Foot_collision",
    )

    cfg: ManagerBasedRlEnvCfg = create_velocity_env_cfg(
        viewer_body_name="Trunk",
        robot_cfg=BOOSTER_K1_ROBOT_CFG,
        action_scale=0.75,
        posture_std_standing={".*": 0.05},
        posture_std_walking=std_walking,
        posture_std_running=std_walking,
        site_names=site_names,
        feet_sensor_cfg=feet_sensor_cfg,
        self_collision_sensor_cfg=self_collision_cfg,
        foot_friction_geom_names=foot_frictions_geom_names,
        body_ang_vel_weight=-0.05,
        angular_momentum_weight=-0.02,
        self_collision_weight=-1.0,
        air_time_weight=0.0,
    )

    # Removing base lin velocity observation
    del cfg.observations["policy"].terms["base_lin_vel"]

    #   def log_debug(env: ManagerBasedRlEnv, _):
    #     print("Sensor")
    #     print(env.scene.sensors["feet_ground_contact_left_foot_link_force"].data)
    #   cfg.events["log_debug"] = EventTermCfg(mode="interval", func=log_debug, interval_range_s=(0.0, 0.0))

    cfg.actions["joint_pos"].actuator_names = (r".*(?<!Head_Yaw)(?<!Head_Pitch)$",)

    cfg.events["reset_base"].params["pose_range"]["z"] = (0.5, 0.6)

    cfg.commands["twist"].viz.z_offset = 1.0

    # Walking on plane only
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # Disabling curriculum
    del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    cfg.sim.nconmax = 256
    cfg.sim.njmax = 512

    cfg.events["push_robot"].params["velocity_range"] = {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
    }

    # Slightly increased L2 action rate penalty
    cfg.rewards["action_rate_l2"].weight = -0.1

    # More standing env, disabling heading envs
    command: UniformVelocityCommandCfg = cfg.commands["twist"]
    command.rel_standing_envs = 0.25
    command.rel_heading_envs = 0.0

    return cfg


@retval
def UpkieVelocityEnvCfg_PLAY() -> ManagerBasedRlEnvCfg:
    cfg: ManagerBasedRlEnvCfg = deepcopy(UpkieVelocityEnvCfg)
    return cfg


# @dataclass
# class UpkieVelocityEnvWithPushCfg(UpkieVelocityEnvCfg):
#     def __post_init__(self):
#         super().__post_init__()

#         # Setting push event parameters
#         self.events.push_robot.params["velocity_range"] = {
#             "x": (-0.5, 0.5),
#             "y": (-0.5, 0.5),
#         }
#         self.curriculum.push_intensity.params["intensities"] = [
#             (30001 * 24, 1.0),
#             (60001 * 24, 1.7),
#             (90001 * 24, 2.0),
#         ]


# @dataclass
# class UpkieVelocityEnvStaticPushCfg(UpkieVelocityEnvCfg):
#     def __post_init__(self):
#         super().__post_init__()

#         # No movement
#         self.commands.twist.ranges.lin_vel_x = (0.0, 0.0)
#         self.commands.twist.ranges.ang_vel_z = (0.0, 0.0)
#         self.curriculum.command_vel.params["velocity_stages"] = []
#         self.rewards.track_linear_velocity.weight = 0.0
#         self.rewards.track_angular_velocity.weight = 0.0

#         # Setting push event parameters
#         self.events.push_robot.params["velocity_range"] = {
#             "x": (-0.5, 0.5),
#             "y": (-0.3, 0.3),
#         }
#         self.curriculum.push_intensity.params["intensities"] = [
#             (2001 * 24, 1.0),
#             (12001 * 24, 2.0),
#             (35001 * 24, 3.0),
#         ]


# @dataclass
# class UpkieVelocityEnvLegsBackwardCfg(UpkieVelocityEnvCfg):
#     def __post_init__(self):
#         super().__post_init__()

#         # Set tracking reward weights
#         self.rewards.track_linear_velocity.weight = 2.5
#         self.rewards.track_angular_velocity.weight = 2.0

#         # Reset robot in default pose (with legs backward)
#         self.events.reset_base.params["pose_range"]["z"] = (
#             RK_HEIGHT,
#             RK_HEIGHT,
#         )
#         self.events.reset_robot_joints.func = reset_legs_backward
#         self.events.reset_robot_joints.params = {}

#         # Modify the pose reward to favor backward legs
#         self.rewards.pose.func = backward_legs
#         self.rewards.pose.weight = 0.3
#         self.rewards.pose.params = {"std": math.sqrt(0.5)}


# @dataclass
# class UpkieVelocityEnvLegsBackwardWithPushCfg(UpkieVelocityEnvLegsBackwardCfg):
#     def __post_init__(self):
#         super().__post_init__()

#         # Setting push event parameters
#         self.events.push_robot.params["velocity_range"] = {
#             "x": (-0.3, 0.3),
#             "y": (0.0, 0.0),
#         }
#         self.curriculum.push_intensity.params["intensities"] = [
#             (20001 * 24, 1.0),
#             (30001 * 24, 2.0),
#         ]


# @dataclass
# class UpkieVelocityEnvCfg_PLAY(UpkieVelocityEnvCfg):
#     episode_length_s: float = 1e9  # Very long episodes for PLAY mode

#     def __post_init__(self):
#         super().__post_init__()

#         # Set only one velocity stage for PLAY mode
#         self.curriculum.command_vel.params["velocity_stages"] = []
#         self.commands.twist.ranges.lin_vel_x = (-1, 1)
#         self.commands.twist.ranges.ang_vel_z = (-1.5, 1.5)


# @dataclass
# class UpkieVelocityEnvLegsBackwardCfg_PLAY(UpkieVelocityEnvLegsBackwardCfg):
#     episode_length_s: float = 1e9  # Very long episodes for PLAY mode

#     def __post_init__(self):
#         super().__post_init__()

#         # Set only one velocity stage for PLAY mode
#         self.curriculum.command_vel.params["velocity_stages"] = []
#         self.commands.twist.ranges.lin_vel_x = (-1, 1)
#         self.commands.twist.ranges.ang_vel_z = (-1.5, 1.5)


# @dataclass
# class UpkieVelocityEnvStaticPushCfg_PLAY(UpkieVelocityEnvStaticPushCfg):
#     episode_length_s: float = 1e9  # Very long episodes for PLAY mode

#     def __post_init__(self):
#         super().__post_init__()

#         # Set only one push intensity for PLAY mode
#         self.curriculum.push_intensity.params["intensities"] = []
#         self.events.push_robot.params["intensity"] = 3.0


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
    save_interval: int = 1000
    num_steps_per_env: int = 24
    max_iterations: int = 30_000
