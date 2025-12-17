from mjlab.tasks.registry import register_mjlab_task

from .upkie_velocity_env_cfg import (
    upkie_velocity_env_cfg,
    UpkieRlCfg,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Upkie",
    env_cfg=upkie_velocity_env_cfg(reverse_knee=False),
    play_env_cfg=upkie_velocity_env_cfg(reverse_knee=False, play=True),
    rl_cfg=UpkieRlCfg(max_iterations=50_000),
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Upkie-Pushed",
    env_cfg=upkie_velocity_env_cfg(reverse_knee=False, pushed=True),
    play_env_cfg=upkie_velocity_env_cfg(reverse_knee=False, pushed=True, play=True),
    rl_cfg=UpkieRlCfg(max_iterations=80_000),
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Upkie-RK",
    env_cfg=upkie_velocity_env_cfg(reverse_knee=True),
    play_env_cfg=upkie_velocity_env_cfg(reverse_knee=True, play=True),
    rl_cfg=UpkieRlCfg(max_iterations=50_000),
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Upkie-RK-Pushed",
    env_cfg=upkie_velocity_env_cfg(reverse_knee=True, pushed=True),
    play_env_cfg=upkie_velocity_env_cfg(reverse_knee=True, pushed=True, play=True),
    rl_cfg=UpkieRlCfg(max_iterations=80_000),
)