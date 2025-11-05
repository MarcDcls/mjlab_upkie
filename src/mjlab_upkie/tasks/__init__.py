import gymnasium as gym

gym.register(
    id="Mjlab-Velocity-Upkie",
    entry_point="mjlab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.upkie_velocity_env_cfg:UpkieVelocityEnvCfg",
        "rl_cfg_entry_point": f"{__name__}.upkie_velocity_env_cfg:UpkieCfg",
    },
)

gym.register(
    id="Mjlab-Velocity-Upkie-Play",
    entry_point="mjlab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.upkie_velocity_env_cfg:UpkieVelocityEnvCfg_PLAY",
        "rl_cfg_entry_point": f"{__name__}.upkie_velocity_env_cfg:UpkieCfg",
    },
)