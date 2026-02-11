# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0


import mujoco


class MujocoModelWrapper:
    """
    MujocoModelWrapper is a class allowing to update friction parameters to do an identification.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        actuator_idx: list[int],
    ):
        self.model = model
        self.data = data
        self.actuator_idx = actuator_idx
        

    def set_parameters(self, frictionloss: float, damping: float, armature: float):
        """
        Set the friction parameters of the model.
        """
        self.model.dof_frictionloss[self.actuator_idx] = frictionloss
        self.model.dof_damping[self.actuator_idx] = damping
        self.model.dof_armature[self.actuator_idx] = armature
        mujoco.mj_setConst(self.model, self.data)


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    from sinus import fix_robot
    from sim import reset_robot
    from src.mjlab_upkie.robot.upkie_constants import (
        POS_CTRL_JOINT_IDS,
        LEFT_KNEE,
    )


    model = mujoco.MjModel.from_xml_path("src/mjlab_upkie/robot/upkie/scene.xml")
    data = mujoco.MjData(model)

    model.opt.timestep = 0.005

    wrapper = MujocoModelWrapper(model, data, actuator_idx=POS_CTRL_JOINT_IDS)

    params = [
        (0.001, 0.001, 0.001),
        (1.0, 0.001, 0.001),
        (0.001, 1.0, 0.001),
        (0.001, 0.001, 1.0),
    ]

    servo_data = {}
    for i, param in enumerate(params):
        wrapper.set_parameters(frictionloss=param[0], damping=param[1], armature=param[2])

        reset_robot(model, data)

        servo_data[i] = {"timestamp": [], "read": [], "target": []}

        t = 0
        while t < 4.0:
            fix_robot(model, data)
            data.ctrl[LEFT_KNEE] = 0.5 * np.sin(t**2 * np.pi)

            servo_data[i]["timestamp"].append(t)
            servo_data[i]["read"].append(float(data.qpos[LEFT_KNEE + 7]))
            servo_data[i]["target"].append(float(data.ctrl[LEFT_KNEE]))

            mujoco.mj_step(model, data)
            t += model.opt.timestep

    # Plotting the difference between read trajectories to check the effect of the different parameters
    fig, axes = plt.subplots(len(params) - 1, 1, sharex=True, figsize=(12, 8))
    for i, param in enumerate(params[1:]):
        ax = axes[i]
        diff = np.array(servo_data[i + 1]["read"]) - np.array(servo_data[0]["read"])
        ax.plot(servo_data[i + 1]["timestamp"], diff, label="Read Difference")
        ax.set_title(f"Friction Loss: {param[0]}, Damping: {param[1]}, Armature: {param[2]}")
        ax.set_ylabel("Error (rad)")
        ax.legend()
    axes[-1].set_xlabel("Time (s)")

    fig.tight_layout()
    plt.show()
        