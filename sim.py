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
import random

from mjlab_upkie.robot.upkie_constants import (
    DEFAULT_POSE,
    DEFAULT_HEIGHT,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_WHEEL,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_WHEEL,    
)

robot_path: str = "src/mjlab_upkie/robot/upkie/scene.xml"
wheel_action_scale: float = 100.0

# Global command variable
command: list[float] = [0.0, 0.0, 0.0]  

# Zero velocity management
world_target: list[float] = [0.0, 0.0, 0.0]

def reset_robot(model, data, yaw=0.0):
    """Reset robot to default pose with specified yaw orientation."""
    global world_target
    world_target = [0.0, 0.0, yaw]

    # Set initial joint positions
    data.qpos = np.array([0.0] * model.nq)
    data.qpos[7 + LEFT_HIP] = DEFAULT_POSE["left_hip"]
    data.qpos[7 + LEFT_KNEE] = DEFAULT_POSE["left_knee"]
    data.qpos[7 + RIGHT_HIP] = DEFAULT_POSE["right_hip"]
    data.qpos[7 + RIGHT_KNEE] = DEFAULT_POSE["right_knee"]
    data.qpos[7 + LEFT_WHEEL] = DEFAULT_POSE["left_wheel"]
    data.qpos[7 + RIGHT_WHEEL] = DEFAULT_POSE["right_wheel"]

    data.qvel = np.array([0.0] * model.nv)

    # Set robot initial position
    data.qpos[2] = DEFAULT_HEIGHT

    # Set robot initial orientation (quaternion)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    data.qpos[3] = cy   # w
    data.qpos[4] = 0.0  # x
    data.qpos[5] = 0.0  # y
    data.qpos[6] = sy   # z

    data.ctrl = np.array([0.0] * model.nu)

    mujoco.mj_forward(model, data)


def get_inputs(model, data, last_action, command):
    """Prepare observation dictionary for ONNX model inference."""
    obs = []

    # Joint positions
    obs.append(data.qpos[7 + LEFT_HIP])
    obs.append(data.qpos[7 + LEFT_KNEE])
    obs.append(data.qpos[7 + RIGHT_HIP])
    obs.append(data.qpos[7 + RIGHT_KNEE])

    # Wheel velocities
    obs.append(data.qvel[6 + LEFT_WHEEL])
    obs.append(data.qvel[6 + RIGHT_WHEEL])

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

    # Debug
    # print(f"joint positions: {np.array(obs[0:4]).round(2)}")
    # print(f"wheel velocities: {np.array(obs[4:6]).round(2)}")
    # print(f"IMU quaternion: {np.array(obs[6:10]).round(2)}")
    # print(f"gyro readings: {np.array(obs[10:13]).round(2)}")
    # print(f"last action: {np.array(obs[13:19]).round(2)}")
    # print(f"command: {np.array(obs[19:22]).round(2)}")
    # print("------------------------")

    return {
        "obs": [
            np.array(obs, dtype=np.float32),
        ]
    }


def keyboard_callback(keycode, data):
    """Handle keyboard input for command control."""
    global command
    global world_target
    
    previous_command = command.copy()
    if keycode == 265:  # Up arrow
        command[0] = min(command[0] + 0.25, 1.0)
    elif keycode == 264:  # Down arrow
        command[0] = max(command[0] - 0.25, -1.0)
    elif keycode == 263:  # Left arrow
        command[2] = min(command[2] + 0.5, 1.5)
    elif keycode == 262:  # Right arrow
        command[2] = max(command[2] - 0.5, -1.5)
    elif keycode == 32:  # Space bar - reset commands
        command = [0.0, 0.0, 0.0]

    if command != previous_command:
        print(f"Linear velocity: {command[0]:.2f} [m/s]  |  Angular velocity: {command[2]:.2f} [rad/s]")

    if command[0] == 0.0 and command[2] == 0.0:
        set_world_target(data)


def set_world_target(data):
    """Set the target position in the world frame when stopping."""
    global world_target

    yaw = np.arctan2(
        2.0 * (data.qpos[3] * data.qpos[6]),
        1.0 - 2.0 * (data.qpos[5] ** 2 + data.qpos[6] ** 2),
    )
    x = data.qpos[0]
    y = data.qpos[1]
    
    world_target[0] = x
    world_target[1] = y
    world_target[2] = yaw


def get_corrected_command(command, data, kp_pos=3.0, kv_pos=1.0, kp_yaw=1.0, kv_yaw=0.1):
    """Return command with a proportional controller on position error when stopped."""
    global world_target

    if command[0] != 0.0 or command[2] != 0.0:
        return command
        
    world_yaw = np.arctan2(
        2.0 * (data.qpos[3] * data.qpos[6]),
        1.0 - 2.0 * (data.qpos[5] ** 2 + data.qpos[6] ** 2),
    )

    # X error in robot frame
    ex_w = world_target[0] - data.qpos[0]
    ey_w = world_target[1] - data.qpos[1]
    ex_r = ex_w * np.cos(world_yaw) + ey_w * np.sin(world_yaw)

    # Yaw error (wrapped to [-pi, pi]) in robot frame
    eyaw = world_target[2] - world_yaw
    eyaw = (eyaw + np.pi) % (2 * np.pi) - np.pi

    # PD control
    vx_r = data.qvel[0] * np.cos(world_yaw) + data.qvel[1] * np.sin(world_yaw)
    wz_r = data.qvel[5]
    vx_cmd = np.clip(kp_pos * ex_r - kv_pos * vx_r, -1.0, 1.0)
    wz_cmd = np.clip(kp_yaw * eyaw - kv_yaw * wz_r, -1.5, 1.5)
    corrected_command = [vx_cmd, 0.0, wz_cmd]

    return corrected_command


if __name__ == "__main__":
    import onnxruntime as ort
    import onnx

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx-model-path", type=str, default="logs/rsl_rl/upkie_velocity/bests/default.onnx")
    args = parser.parse_args()

    onnx_model = onnx.load(args.onnx_model_path)
    onnx.checker.check_model(onnx_model)
    ort_sess = ort.InferenceSession(args.onnx_model_path)

    meta = ort_sess.get_modelmeta()
    print("ONNX model metadata:")
    for key, value in meta.custom_metadata_map.items():
        print(f"  {key}: {value}")

    model: mujoco.MjModel = mujoco.MjModel.from_xml_path(robot_path)
    data: mujoco.MjData = mujoco.MjData(model)

    model.opt.timestep = 0.005  # 200 Hz simulation
    inf_period = 4  # 50 Hz inference
    step_counter = 0

    reset_robot(model, data, yaw=random.uniform(0, 2*np.pi))
    last_action = [0.0] * 6

    print("\n=== Keyboard Controls ===")
    print("↑/↓ : Linear velocity ±0.25 m/s")
    print("←/→ : Angular velocity ±0.5 rad/s")
    print("SPACE : Reset commands")
    print("========================\n")

    with mujoco.viewer.launch_passive(model, data, key_callback=lambda k: keyboard_callback(k, data)) as viewer:
        while viewer.is_running():
            step_start = time.time()      

            # Infer at 50 Hz
            if step_counter % inf_period == 0:

                # Get observation and run inference
                inputs = get_inputs(model, data, last_action, get_corrected_command(command, data))
                outputs = ort_sess.run(None, inputs)
                action = outputs[0][0]
                last_action = action.tolist()

                # Apply action
                data.ctrl[LEFT_HIP] = action[0] + DEFAULT_POSE["left_hip"]
                data.ctrl[LEFT_KNEE] = action[1] + DEFAULT_POSE["left_knee"]
                data.ctrl[RIGHT_HIP] = action[2] + DEFAULT_POSE["right_hip"]
                data.ctrl[RIGHT_KNEE] = action[3] + DEFAULT_POSE["right_knee"]
                data.ctrl[LEFT_WHEEL] = action[4] * wheel_action_scale
                data.ctrl[RIGHT_WHEEL] = action[5] * wheel_action_scale

            # Step simulation
            mujoco.mj_step(model, data)
            viewer.sync()
            step_counter += 1

            # If the robot is falling, reset
            if data.qpos[2] < 0.1:
                print("Robot fell, resetting...")
                reset_robot(model, data, yaw=random.uniform(0, 2*np.pi))

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
