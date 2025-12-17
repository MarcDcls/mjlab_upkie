from mjlab.tasks.registry import register_mjlab_task

from .upkie_velocity_env_cfg import (
    upkie_velocity_env_cfg,
    UpkieRlCfg,
)


register_mjlab_task(
    task_id="Mjlab-Velocity-Upkie",
    env_cfg=upkie_velocity_env_cfg(reverse_knee=False),
    play_env_cfg=upkie_velocity_env_cfg(reverse_knee=False, play=True),
    rl_cfg=UpkieRlCfg(),
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Upkie-RK",
    env_cfg=upkie_velocity_env_cfg(reverse_knee=True),
    play_env_cfg=upkie_velocity_env_cfg(reverse_knee=True, play=True),
    rl_cfg=UpkieRlCfg(),
)


# import gymnasium as gym

# gym.register(
#     id="Mjlab-Velocity-Upkie",
#     entry_point="mjlab.envs:ManagerBasedRlEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.upkie_velocity_env_cfg:UpkieVelocityEnvCfg",
#         "rl_cfg_entry_point": f"{__name__}.upkie_velocity_env_cfg:UpkieCfg",
#     },
# )


# gym.register(
#     id="Mjlab-Velocity-Upkie-Play",
#     entry_point="mjlab.envs:ManagerBasedRlEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.upkie_velocity_env_cfg:UpkieVelocityEnvCfg_PLAY",
#         "rl_cfg_entry_point": f"{__name__}.upkie_velocity_env_cfg:UpkieCfg",
#     },
# )

# gym.register(
#     id="Mjlab-Velocity-Upkie-With-Push",
#     entry_point="mjlab.envs:ManagerBasedRlEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.upkie_velocity_env_cfg:UpkieVelocityEnvWithPushCfg",
#         "rl_cfg_entry_point": f"{__name__}.upkie_velocity_env_cfg:UpkieCfg",
#     },
# )


# gym.register(
#     id="Mjlab-Velocity-Upkie-Static-Push",
#     entry_point="mjlab.envs:ManagerBasedRlEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.upkie_velocity_env_cfg:UpkieVelocityEnvStaticPushCfg",
#         "rl_cfg_entry_point": f"{__name__}.upkie_velocity_env_cfg:UpkieCfg",
#     },
# )


# gym.register(
#     id="Mjlab-Velocity-Upkie-Static-Push-Play",
#     entry_point="mjlab.envs:ManagerBasedRlEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.upkie_velocity_env_cfg:UpkieVelocityEnvStaticPushCfg_PLAY",
#         "rl_cfg_entry_point": f"{__name__}.upkie_velocity_env_cfg:UpkieCfg",
#     },
# )


# gym.register(
#     id="Mjlab-Velocity-Upkie-Legs-Backward",
#     entry_point="mjlab.envs:ManagerBasedRlEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.upkie_velocity_env_cfg:UpkieVelocityEnvLegsBackwardCfg",
#         "rl_cfg_entry_point": f"{__name__}.upkie_velocity_env_cfg:UpkieCfg",
#     },
# )


# gym.register(
#     id="Mjlab-Velocity-Upkie-Legs-Backward-With-Push",
#     entry_point="mjlab.envs:ManagerBasedRlEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.upkie_velocity_env_cfg:UpkieVelocityEnvLegsBackwardWithPushCfg",
#         "rl_cfg_entry_point": f"{__name__}.upkie_velocity_env_cfg:UpkieCfg",
#     },
# )


# gym.register(
#     id="Mjlab-Velocity-Upkie-Legs-Backward-Play",
#     entry_point="mjlab.envs:ManagerBasedRlEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.upkie_velocity_env_cfg:UpkieVelocityEnvLegsBackwardCfg_PLAY",
#         "rl_cfg_entry_point": f"{__name__}.upkie_velocity_env_cfg:UpkieCfg",
#     },
# )
