# Copyright 2025 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl.runner import VelocityOnPolicyRunner

from .upkie_velocity_env_cfg import (
    upkie_velocity_env_cfg,
    UpkieVelRlCfg
)

from .upkie_standup_env_cfg import (
    upkie_standup_env_cfg,
    UpkieStandupRlCfg
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Upkie",
    env_cfg=upkie_velocity_env_cfg(),
    play_env_cfg=upkie_velocity_env_cfg(play=True),
    rl_cfg=UpkieVelRlCfg(max_iterations=60_000),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Standup-Upkie",
    env_cfg=upkie_standup_env_cfg(),
    play_env_cfg=upkie_standup_env_cfg(play=True),
    rl_cfg=UpkieStandupRlCfg(),
    runner_cls=VelocityOnPolicyRunner,
)