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

robot_path = "src/mjlab_upkie/robot/upkie/scene.xml"
wheel_action_scale = 100.0

command = [0.0, 0.0, 0.0]  # global command variable

def reset_robot(model, data, yaw=0.0):

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
    obs.extend(data.qpos[3:7])

    # Gyro readings
    obs.extend(data.qvel[3:6])

    # Last action
    obs.extend(last_action)

    # Command
    obs.extend(command)

    # Debug
    # print("joint positions:", obs[0:4])
    # print("wheel velocities:", obs[4:6])
    # print("IMU quaternion:", obs[6:10])
    # print("gyro readings:", obs[10:13])
    # print("last action:", obs[13:19])
    # print("command:", obs[19:22])
    # print("------------------------")

    return {
        "obs": [
            np.array(obs, dtype=np.float32),
        ]
    }


def keyboard_callback(keycode):
    """Handle keyboard input for command control."""
    global command
    
    if keycode == 265:  # Up arrow
        command[0] = min(command[0] + 0.25, 1.0)
        print(f"Linear velocity: {command[0]:.1f}")
    elif keycode == 264:  # Down arrow
        command[0] = max(command[0] - 0.25, -1.0)
        print(f"Linear velocity: {command[0]:.1f}")
    elif keycode == 263:  # Left arrow
        command[2] = min(command[2] + 0.5, 1.5)
        print(f"Angular velocity: {command[2]:.1f}")
    elif keycode == 262:  # Right arrow
        command[2] = max(command[2] - 0.5, -1.5)
        print(f"Angular velocity: {command[2]:.1f}")
    elif keycode == 32:  # Space bar - reset commands
        command = [0.0, 0.0, 0.0]
        print("Commands reset")


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

    model.opt.timestep = 0.001  # 1000 Hz simulation
    inf_period = 20 # 50 Hz inference
    step_counter = 0

    reset_robot(model, data, yaw=random.uniform(0, 2*np.pi))
    last_action = [0.0] * 6

    print("\n=== Keyboard Controls ===")
    print("↑/↓ : Linear velocity ±0.25 m/s")
    print("←/→ : Angular velocity ±0.5 rad/s")
    print("SPACE : Reset commands")
    print("========================\n")

    # Launch viewer with key callback
    with mujoco.viewer.launch_passive(model, data, key_callback=keyboard_callback) as viewer:
        while viewer.is_running():
            step_start = time.time()      

            # Infer at 50 Hz
            if step_counter % inf_period == 0:

                # Get observation and run inference
                inputs = get_inputs(model, data, last_action, command)
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
