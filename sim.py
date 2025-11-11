import time
import mujoco
import argparse
import mujoco.viewer
import numpy as np

from mjlab_upkie.tasks.upkie_velocity_env_cfg import (
    BACKWARD_LEGS_POS,
    LEFT_HIP,
    LEFT_KNEE,
    RIGHT_HIP,
    RIGHT_KNEE,
    LEFT_WHEEL,
    RIGHT_WHEEL,
    DEFAULT_ROBOT_HEIGHT,
    BACKWARD_LEG_ROBOT_HEIGHT,
)

robot_path = "src/mjlab_upkie/robot/scene.xml"

action_scale = 0.75

def reset_robot(model, data, backward_leg=False):
    default_joint_pos = np.array([0.0] * 6)
    if backward_leg:
        default_joint_pos = np.array(
            [
                BACKWARD_LEGS_POS["left_hip"],
                BACKWARD_LEGS_POS["left_knee"],
                BACKWARD_LEGS_POS["right_hip"],
                BACKWARD_LEGS_POS["right_knee"],
                0.0,
                0.0,
            ]
        )

    # Set initial joint positions
    data.qpos = np.array([0.0] * model.nq)
    data.qpos[7 + LEFT_HIP] = default_joint_pos[0]
    data.qpos[7 + LEFT_KNEE] = default_joint_pos[1]
    data.qpos[7 + RIGHT_HIP] = default_joint_pos[2]
    data.qpos[7 + RIGHT_KNEE] = default_joint_pos[3]
    data.qpos[7 + LEFT_WHEEL] = default_joint_pos[4]
    data.qpos[7 + RIGHT_WHEEL] = default_joint_pos[5]

    data.qvel = np.array([0.0] * model.nv)

    # Set robot initial position
    if args.backward_leg:
        data.qpos[2] = BACKWARD_LEG_ROBOT_HEIGHT
    else:
        data.qpos[2] = DEFAULT_ROBOT_HEIGHT

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


if __name__ == "__main__":
    import onnxruntime as ort
    import onnx

    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--backward-leg", action="store_true")
    parser.add_argument("-o", "--onnx-model-path", type=str, default="logs/rsl_rl/upkie_velocity/bests/default_no_push_tmp.onnx")
    args = parser.parse_args()

    onnx_model = onnx.load(args.onnx_model_path)
    onnx.checker.check_model(onnx_model)
    ort_sess = ort.InferenceSession(args.onnx_model_path)

    model: mujoco.MjModel = mujoco.MjModel.from_xml_path(robot_path)
    data: mujoco.MjData = mujoco.MjData(model)

    model.opt.timestep = 0.005  # 200 Hz control
    inf_period = 4  # 50 Hz inference
    step_counter = 0

    reset_robot(model, data, backward_leg=args.backward_leg)

    viewer = mujoco.viewer.launch_passive(model, data)

    last_action = [0.0] * 6
    command = [0.0, 0.0, 0.0]

    while viewer.is_running():

        # Infer at 50 Hz
        if step_counter % inf_period == 0:

            # Get observation and run inference
            inputs = get_inputs(model, data, last_action, command)

            # if step_counter > 0:
            #     exit()

            outputs = ort_sess.run(None, inputs)
            action = outputs[0][0]
            last_action = action.tolist()
            action = action * action_scale
            
            # Apply action
            data.ctrl[LEFT_HIP] = action[0]
            data.ctrl[LEFT_KNEE] = action[1]
            data.ctrl[LEFT_WHEEL] = action[2]
            data.ctrl[RIGHT_HIP] = action[3]
            data.ctrl[RIGHT_KNEE] = action[4]
            data.ctrl[RIGHT_WHEEL] = action[5]

            # data.ctrl[LEFT_HIP] = action[0]
            # data.ctrl[LEFT_KNEE] = action[1]
            # data.ctrl[RIGHT_HIP] = action[2]
            # data.ctrl[RIGHT_KNEE] = action[3]
            # data.ctrl[LEFT_WHEEL] = action[4]
            # data.ctrl[RIGHT_WHEEL] = action[5]

        # Step simulation
        step_start = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()
        step_counter += 1

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
