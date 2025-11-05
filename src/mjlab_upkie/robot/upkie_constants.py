from pathlib import Path
import mujoco

import os
from dataclasses import dataclass
from mjlab.entity import Entity, EntityCfg, EntityArticulationInfoCfg
from mjlab.utils.spec_config import ActuatorCfg, CollisionCfg

UPKIE_XML: Path = Path(os.path.dirname(__file__)) / "upkie.xml"
assert UPKIE_XML.exists(), f"XML not found: {UPKIE_XML}"


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(UPKIE_XML))


HOME_FRAME = EntityCfg.InitialStateCfg(
    joint_pos={
        "left_hip": 0.0,
        "left_knee": 0.0,
        "left_wheel": 0.0,
        "right_hip": 0.0,
        "right_knee": 0.0,
        "right_wheel": 0.0,
    },
    joint_vel={".*": 0.0},
)

UPKIE_CFG = EntityCfg(spec_fn=get_spec, init_state=HOME_FRAME)

if __name__ == "__main__":
    import mujoco.viewer as viewer

    from mjlab.scene import SceneCfg, Scene
    from mjlab.terrains import TerrainImporterCfg
    from mjlab.terrains.config import ROUGH_TERRAINS_CFG

    SCENE_CFG = SceneCfg(
        terrain=TerrainImporterCfg(terrain_type="plane"),
        entities={"robot": UPKIE_CFG},
    )

    scene = Scene(SCENE_CFG, device="cuda:0")

    viewer.launch(scene.compile())
