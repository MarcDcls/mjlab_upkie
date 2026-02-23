# Copyright 2025 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

from pathlib import Path
import numpy as np

import os

UPKIE_XML: Path = Path(os.path.dirname(__file__)) / "upkie/robot.xml"
assert UPKIE_XML.exists(), f"XML not found: {UPKIE_XML}"

UPKIE_MODEL_PATH = "src/mjlab_upkie/robot/upkie/scene.xml"

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

WHEEL_ACTION_SCALE = 100.0

DEFAULT_POSE = {
    "left_hip": 0.0,
    "left_knee": 0.0,
    "left_wheel": 0.0,
    "right_hip": 0.0,
    "right_knee": 0.0,
    "right_wheel": 0.0,
}

DEFAULT_HEIGHT = 0.343

if __name__ == "__main__":
    import mujoco.viewer as viewer

    from mjlab.scene import SceneCfg, Scene
    from mjlab.terrains import TerrainImporterCfg
    from mjlab.terrains.config import ROUGH_TERRAINS_CFG
    from mjlab_upkie.robot.upkie_cfg import DEFAULT_UPKIE_CFG

    SCENE_CFG = SceneCfg(
        terrain=TerrainImporterCfg(terrain_type="plane"),
        entities={"robot": DEFAULT_UPKIE_CFG},
    )

    scene = Scene(SCENE_CFG, device="cuda:0")

    viewer.launch(scene.compile())
