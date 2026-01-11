from mjlab.tasks.registry import register_mjlab_task

from .upkie_velocity_env_cfg import (
    upkie_velocity_env_cfg,
    UpkieRlCfg
)

from mjlab.tasks.velocity.rl.runner import VelocityOnPolicyRunner

register_mjlab_task(
    task_id="Mjlab-Velocity-Upkie",
    env_cfg=upkie_velocity_env_cfg(reverse_knee=False),
    play_env_cfg=upkie_velocity_env_cfg(reverse_knee=False, play=True),
    rl_cfg=UpkieRlCfg(max_iterations=60_000),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Upkie-Static",
    env_cfg=upkie_velocity_env_cfg(reverse_knee=False, static=True),
    play_env_cfg=upkie_velocity_env_cfg(reverse_knee=False, static=True, play=True),
    rl_cfg=UpkieRlCfg(max_iterations=60_000),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Upkie-RK",
    env_cfg=upkie_velocity_env_cfg(reverse_knee=True),
    play_env_cfg=upkie_velocity_env_cfg(reverse_knee=True, play=True),
    rl_cfg=UpkieRlCfg(max_iterations=60_000),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Upkie-RK-Static",
    env_cfg=upkie_velocity_env_cfg(reverse_knee=True, static=True),
    play_env_cfg=upkie_velocity_env_cfg(reverse_knee=True, static=True, play=True),
    rl_cfg=UpkieRlCfg(max_iterations=60_000),
    runner_cls=VelocityOnPolicyRunner,
)