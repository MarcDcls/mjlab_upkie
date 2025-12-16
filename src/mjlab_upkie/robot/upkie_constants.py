from pathlib import Path
import mujoco
import numpy as np

import os
from dataclasses import dataclass
from mjlab.entity import Entity, EntityCfg, EntityArticulationInfoCfg
from mjlab.actuator import XmlPositionActuatorCfg, XmlVelocityActuatorCfg
from mjlab.utils.spec_config import CollisionCfg

UPKIE_XML: Path = Path(os.path.dirname(__file__)) / "upkie.xml"
assert UPKIE_XML.exists(), f"XML not found: {UPKIE_XML}"


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(UPKIE_XML))


POS_CTRL_JOINT_NAMES = ["left_hip", "left_knee", "right_hip", "right_knee"]
VEL_CTRL_JOINT_NAMES = ["left_wheel", "right_wheel"]

LEFT_HIP = 0
LEFT_KNEE = 1
LEFT_WHEEL = 2
RIGHT_HIP = 3
RIGHT_KNEE = 4
RIGHT_WHEEL = 5

POS_CTRL_JOINT_IDS = np.array([LEFT_HIP, LEFT_KNEE, RIGHT_HIP, RIGHT_KNEE])
VEL_CTRL_JOINT_IDS = np.array([LEFT_WHEEL, RIGHT_WHEEL])

DEFAULT_POSE = {
    "left_hip": 0.0,
    "left_knee": 0.0,
    "left_wheel": 0.0,
    "right_hip": 0.0,
    "right_knee": 0.0,
    "right_wheel": 0.0,
}

RK_POSE = {
    "left_hip": 0.3,
    "left_knee": -0.6,
    "left_wheel": 0.0,
    "right_hip": -0.3,
    "right_knee": 0.6,
    "right_wheel": 0.0,
}

DEFAULT_HEIGHT = 0.56
RK_HEIGHT = 0.545

FULL_COLLISION = CollisionCfg(
    geom_names_expr=tuple([".*_collision"]),
    condim={r"^(left|right)_foot_collision$": 3, ".*_collision*": 1},
    priority={r"^(left|right)_foot_collision$": 1},
    friction={r"^(left|right)_foot_collision$": (0.6,)},
)

ARTICULATION_CFG = EntityArticulationInfoCfg(
    actuators=(
        XmlPositionActuatorCfg(joint_names_expr=tuple(POS_CTRL_JOINT_NAMES)),
        XmlVelocityActuatorCfg(joint_names_expr=tuple(VEL_CTRL_JOINT_NAMES)), # XXX: should be velocity, but break the old agents
    ),
)

DEFAULT_UPKIE_CFG = EntityCfg(
    spec_fn=get_spec,
    init_state=EntityCfg.InitialStateCfg(
        joint_pos=DEFAULT_POSE,
        joint_vel={".*": 0.0},
    ),
    collisions=(FULL_COLLISION,),
    articulation=ARTICULATION_CFG,
)

RK_UPKIE_CFG = EntityCfg(
    spec_fn=get_spec,
    init_state=EntityCfg.InitialStateCfg(
        joint_pos=RK_POSE,
        joint_vel={".*": 0.0},
    ),
    collisions=(FULL_COLLISION,),
    articulation=ARTICULATION_CFG,
)

if __name__ == "__main__":
    import mujoco.viewer as viewer

    from mjlab.scene import SceneCfg, Scene
    from mjlab.terrains import TerrainImporterCfg
    from mjlab.terrains.config import ROUGH_TERRAINS_CFG

    SCENE_CFG = SceneCfg(
        terrain=TerrainImporterCfg(terrain_type="plane"),
        entities={"robot": DEFAULT_UPKIE_CFG},
    )

    scene = Scene(SCENE_CFG, device="cuda:0")

    viewer.launch(scene.compile())
