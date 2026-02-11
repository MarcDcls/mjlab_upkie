# Copyright 2025 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

import time
import mujoco
import argparse
import mujoco.viewer
import numpy as np

from mjlab_upkie.robot.upkie_constants import (
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_WHEEL,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_WHEEL,
    UPKIE_MODEL_PATH,
)

def fix_robot(model, data, height: float=1.0):
    """Set the robot in a fixed position for testing."""
    data.qpos[:7] = np.array([0.0] * 7)
    data.qvel[:6] = np.array([0.0] * 6)
    data.qpos[2] = height


def get_inputs(model, data, last_action, command):
    """Prepare observation dictionary for ONNX model inference."""
    obs = []

    # Joint positions
    joint_pos = [
        data.qpos[7 + LEFT_HIP],
        data.qpos[7 + LEFT_KNEE],
        data.qpos[7 + RIGHT_HIP],
        data.qpos[7 + RIGHT_KNEE],
    ]
    obs.extend(joint_pos)

    # Wheel velocities
    wheel_vel = [
        data.qvel[6 + LEFT_WHEEL],
        data.qvel[6 + RIGHT_WHEEL],
    ]
    obs.extend(wheel_vel)
    
    # IMU readings (quaternion)
    quat = data.qpos[3:7] 
    if quat[0] < 0: 
        quat = -quat
    obs.extend(quat)    
    
    # Gyro readings
    obs.extend(data.qvel[3:6])

    # Last action
    obs.extend(last_action)

    # Command
    obs.extend(command)

    return {
        "obs": [
            np.array(obs, dtype=np.float32),
        ]
    }


def log(data, t, observation, action, position: bool = False, velocity: bool = False):
    """Log read and target values depending on motor control type."""
    data["timestamp"].append(t) 
    
    if position:
        data["read"].append(float(observation[1]))
        data["target"].append(float(action[1]))
    elif velocity:
        data["read"].append(float(observation[5]))
        data["target"].append(float(action[5]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--position", action="store_true", help="Whether to log position observations and actions.")
    parser.add_argument("--velocity", action="store_true", help="Whether to log velocity observations and actions.")
    parser.add_argument("--log", action="store_true", help="Whether to log servo observations and actions for analysis.")
    parser.add_argument("--viewer", action="store_true", help="Run with MuJoCo viewer.")
    args = parser.parse_args()

    model: mujoco.MjModel = mujoco.MjModel.from_xml_path(UPKIE_MODEL_PATH)
    data: mujoco.MjData = mujoco.MjData(model)

    model.opt.timestep = 0.005  # 200 Hz simulation
    inf_period = 4  # 50 Hz inference
    step_counter = 0

    if args.log:
        servo_data = {"timestamp": [], "read": [], "target": []}

    viewer = None
    try:
        if args.viewer:
            viewer = mujoco.viewer.launch_passive(model, data)

        fix_robot(model, data)
        mujoco.mj_step(model, data)

        if viewer is not None:
            viewer.sync()

        t = 0
        t_start = time.perf_counter()
        while t < 12.0 and (viewer is None or viewer.is_running()):
            
            # 50 Hz control loop
            if step_counter % inf_period == 0:

                observation = get_inputs(model, data, last_action=np.zeros(6), command=np.zeros(3))["obs"][0]

                position = 0.0
                velocity = 0.0
                if args.position:
                    position = np.sin(t * np.pi) * 0.3
                elif args.velocity:
                    velocity = np.sin(t * np.pi / 2) * 6.0

                action = np.array([0.0, position, 0.0, 0.0, 0.0, velocity], dtype=np.float32)

                data.ctrl[LEFT_HIP] = action[0]
                data.ctrl[LEFT_KNEE] = action[1]
                data.ctrl[RIGHT_HIP] = action[2]
                data.ctrl[RIGHT_KNEE] = action[3]
                data.ctrl[LEFT_WHEEL] = action[4]
                data.ctrl[RIGHT_WHEEL] = action[5]

                # Log data
                if args.log:
                    log(servo_data, t, observation, action, position=args.position, velocity=args.velocity)

            fix_robot(model, data)
            mujoco.mj_step(model, data)

            if viewer is not None and viewer.is_running():
                viewer.sync()

            step_counter += 1
            t += model.opt.timestep
            time_until_next_step = (t_start + t) - time.perf_counter()
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    finally:
        if viewer is not None and viewer.is_running():
            viewer.close()

    # Save logged data
    if args.log and (args.position or args.velocity):
        import json

        filename = "logs/sim/sim_position_trajectory.json"
        if args.velocity:
            filename = "logs/sim/sim_velocity_trajectory.json"

        with open(filename, "w") as f:
            json.dump(servo_data, f, indent=2)
        print(f"Logged servo data saved to {filename}")

