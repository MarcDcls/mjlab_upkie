# Copyright 2025 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

from mjlab.tasks.registry import register_mjlab_task

from .upkie_velocity_env_cfg import (
    upkie_velocity_env_cfg,
    UpkieRlCfg
)

from mjlab.tasks.velocity.rl.runner import VelocityOnPolicyRunner

register_mjlab_task(
    task_id="Mjlab-Velocity-Upkie",
    env_cfg=upkie_velocity_env_cfg(),
    play_env_cfg=upkie_velocity_env_cfg(play=True),
    rl_cfg=UpkieRlCfg(max_iterations=60_000),
    runner_cls=VelocityOnPolicyRunner,
)
