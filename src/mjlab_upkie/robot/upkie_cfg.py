# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

import mujoco

from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.actuator import XmlPositionActuatorCfg, XmlVelocityActuatorCfg
from mjlab.utils.spec_config import CollisionCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

from mjlab_upkie.robot.upkie_constants import (
    UPKIE_XML,
    DEFAULT_HEIGHT,
    DEFAULT_POSE,
    POS_CTRL_JOINT_NAMES,
    VEL_CTRL_JOINT_NAMES,
)


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(UPKIE_XML))


FULL_COLLISION = CollisionCfg(
    geom_names_expr=tuple([".*_collision"]),
    condim={r"^(left|right)_foot_collision$": 3, ".*_collision*": 1},
    priority={r"^(left|right)_foot_collision$": 1},
    friction={r"^(left|right)_foot_collision$": (0.6,)},
)

ARTICULATION_CFG = EntityArticulationInfoCfg(
    actuators=(
        XmlPositionActuatorCfg(joint_names_expr=tuple(POS_CTRL_JOINT_NAMES)),
        XmlVelocityActuatorCfg(joint_names_expr=tuple(VEL_CTRL_JOINT_NAMES)),
    ),
)

DEFAULT_UPKIE_CFG = EntityCfg(
    spec_fn=get_spec,
    init_state=EntityCfg.InitialStateCfg(
        pos=(0, 0, DEFAULT_HEIGHT),
        joint_pos=DEFAULT_POSE,
        joint_vel={".*": 0.0},
    ),
    collisions=(FULL_COLLISION,),
    articulation=ARTICULATION_CFG,
)

VIEWER_CONFIG = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    asset_name="robot",
    body_name="trunk",
    distance=3.0,
    elevation=-15.0,
    azimuth=90.0,
)

SIM_CFG = SimulationCfg(
    mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
    ),
    nconmax=256,
    njmax=512,
)
