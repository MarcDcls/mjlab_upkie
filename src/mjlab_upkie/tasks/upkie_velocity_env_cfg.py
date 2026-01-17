# Copyright 2025 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Upkie velocity task environment configuration."""

import math
from dataclasses import dataclass, field
from copy import deepcopy
import torch

from mjlab.envs import mdp, ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.manager_term_config import (
    ActionTermCfg,
    CommandTermCfg,
    CurriculumTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.scene import SceneCfg
from mjlab.terrains import TerrainImporterCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from mjlab.utils.lab_api.math import sample_uniform, quat_apply_inverse
from mjlab.tasks.velocity import mdp as mdp_vel
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_upkie.robot.upkie_constants import (
    DEFAULT_UPKIE_CFG,
    DEFAULT_POSE,
    POS_CTRL_JOINT_NAMES,
    VEL_CTRL_JOINT_NAMES,
    POS_CTRL_JOINT_IDS,
    VEL_CTRL_JOINT_IDS,
    VIEWER_CONFIG,
    SIM_CFG,
)

SCENE_CFG = SceneCfg(
    terrain=TerrainImporterCfg(
        terrain_type="plane",
        terrain_generator=None,
        # terrain_type="generator",
        # terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=0,
    ),
    num_envs=1,
    extent=2.0,
)


def upkie_velocity_env_cfg(play: bool = False, static: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Upkie velocity environment configuration."""

    ################# Observations #################

    policy_terms = {
        "joint_pos": ObservationTermCfg(
            func=lambda env: env.sim.data.qpos[:, POS_CTRL_JOINT_IDS + 7],
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=lambda env: env.sim.data.qvel[:, VEL_CTRL_JOINT_IDS + 6],
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "trunk_imu": ObservationTermCfg(
            func=lambda env: env.sim.data.qpos[:, 3:7],
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "trunk_gyro": ObservationTermCfg(
            func=lambda env: env.sim.data.qvel[:, 3:6],
            noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
        "command": ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "twist"},
        ),
    }

    critic_terms = {
        **policy_terms,
        "base_lin_vel": ObservationTermCfg(
            func=lambda env: env.sim.data.qvel[:, 0:3],
        ),
    }

    observations = {
        "policy": ObservationGroupCfg(
            terms=policy_terms,
            concatenate_terms=True,
            enable_corruption=False if play else True,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    #################### Actions ###################

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            asset_name="robot",
            actuator_names=tuple(POS_CTRL_JOINT_NAMES),
            scale=1.0,
            use_default_offset=True,
        ),
        "joint_vel": JointVelocityActionCfg(
            asset_name="robot",
            actuator_names=tuple(VEL_CTRL_JOINT_NAMES),
            scale=100.0,
            use_default_offset=True,
        ),
    }

    ################### Commands ###################

    commands: dict[str, CommandTermCfg] = {
        "twist": UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(3.0, 8.0),
            rel_standing_envs=1.0 if static else 0.1,
            rel_heading_envs=0.0,
            heading_command=False,
            debug_vis=True,
            viz=UniformVelocityCommandCfg.VizCfg(z_offset=0.75),
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-1.5, 1.5),
            ),  # Overridden in curriculum if not play mode
        )
    }

    #################### Events ####################

    def push_by_setting_velocity(
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor,
        velocity_range: dict[str, tuple[float, float]],
        intensity: float = 1.0,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> None:
        """Push the robot by setting its base velocity directly."""
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

    events = {
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "yaw": (-3.14, 3.14),
                },
                "velocity_range": {},
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
        "push_robot": EventTermCfg(
            func=push_by_setting_velocity,
            mode="interval",
            interval_range_s=(1.0, 3.0),
            params={
                "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)},
                "intensity": 0.0,
            },  # Overridden in curriculum if not play mode
        ),
        "foot_friction": EventTermCfg(
            mode="startup",
            func=mdp.randomize_field,
            domain_randomization=True,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=("left_foot_collision", "right_foot_collision")),
                "operation": "abs",
                "field": "geom_friction",
                "ranges": (0.8, 1.2),
            },
        ),
        # "print_debug": EventTermCfg(
        #     func= lambda env, env_ids: print(f"Print debug event at step {env.common_step_counter}"),
        #     mode="interval",
        #     interval_range_s=(0.0, 0.0),
        # ),
    }

    #################### Rewards ###################

    def pose_reward(env: ManagerBasedRlEnv, std: float, target_pose) -> torch.Tensor:
        """Reward aiming for the target pose."""
        targets = torch.tensor([target_pose[name] for name in POS_CTRL_JOINT_NAMES], device=env.device)
        joints = env.sim.data.qpos[:, POS_CTRL_JOINT_IDS + 7]
        error = torch.sum(torch.square(joints - targets), dim=1)
        return torch.exp(-error / std**2)

    rewards = {
        "track_linear_velocity": RewardTermCfg(
            func=mdp_vel.track_linear_velocity,
            weight=2.0,
            params={
                "command_name": "twist",
                "std": math.sqrt(0.1),
            },
        ),
        "track_angular_velocity": RewardTermCfg(
            func=mdp_vel.track_angular_velocity,
            weight=2.0,
            params={
                "command_name": "twist",
                "std": math.sqrt(0.1),
            },
        ),
        "upright": RewardTermCfg(
            func=mdp_vel.flat_orientation,
            weight=1.0,
            params={
                "std": math.sqrt(0.2),
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk",)),
            },
        ),
        "pose": RewardTermCfg(
            func=pose_reward,
            weight=0.5,
            params={
                "std": math.sqrt(0.5),
                "target_pose": DEFAULT_POSE,
            },
        ),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
    }

    ################# Terminations #################

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=mdp.bad_orientation,
            params={"limit_angle": math.radians(70.0)},
        ),
        "illegal_contact": TerminationTermCfg(
            func=mdp_vel.illegal_contact,
            params={"sensor_name": "nonfoot_ground_touch"},
        ),
    }

    ################## Curriculum ##################

    def increase_push_intensity(
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor,
        intensities: list[tuple[float, float]],
    ) -> None:
        """Increase push intensity over time."""
        push_event_cfg = env.event_manager.get_term_cfg("push_robot")
        for step_threshold, intensity in intensities:
            if env.common_step_counter >= step_threshold:
                push_event_cfg.params["intensity"] = intensity

    curriculum = {
        "command_vel": CurriculumTermCfg(
            func=mdp_vel.commands_vel,
            params={
                "command_name": "twist",
                "velocity_stages": [
                    {"step": 0, "lin_vel_x": (-0.5, 0.5), "ang_vel_z": (-0.5, 0.5)},
                    {"step": 3001 * 24, "lin_vel_x": (-0.75, 0.75), "ang_vel_z": (-1.0, 1.0)},
                    {"step": 6001 * 24, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-1.5, 1.5)},
                ],
            },
        ),
        "push_intensity": CurriculumTermCfg(
            func=increase_push_intensity,
            params={
                "intensities": [
                    (0, 0.0),
                    (15001 * 24, 1.0),
                    (25001 * 24, 2.0),
                    (40001 * 24, 3.0),
                ]
            },
        ),
        # "terrain_levels": CurriculumTermCfg(
        #     func=mdp_vel.terrain_levels_vel,
        #     params={"command_name": "twist"},
        # ),
    }

    ################# Configuration ################

    cfg = ManagerBasedRlEnvCfg(
        scene=deepcopy(SCENE_CFG),
        viewer=deepcopy(VIEWER_CONFIG),
        sim=deepcopy(SIM_CFG),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum=curriculum if not play else None,
        decimation=4,
        episode_length_s=20.0 if not play else 10000.0,
    )

    cfg.scene.entities = {"robot": DEFAULT_UPKIE_CFG}

    feet_sensor_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_foot|right_foot)$",
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
        primary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
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
    cfg.scene.sensors = (nonfoot_ground_cfg,)

    return cfg


@dataclass
class UpkieVelRlCfg(RslRlOnPolicyRunnerCfg):
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
    save_interval: int = 5000
    num_steps_per_env: int = 24
    max_iterations: int = 60_000
