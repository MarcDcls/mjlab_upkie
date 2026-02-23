# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

import time
import mujoco
import argparse
import mujoco.viewer
import numpy as np
import json
from pathlib import Path

from mjlab_upkie.sim import get_inputs
from mjlab_upkie.identification.plot import plot_leg_trajectories
from mjlab_upkie.robot.upkie_constants import (
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_WHEEL,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_WHEEL,
    UPKIE_MODEL_PATH,
)


JOINT_CTRL = {
    "left_hip":    LEFT_HIP,
    "left_knee":   LEFT_KNEE,
    "right_hip":   RIGHT_HIP,
    "right_knee":  RIGHT_KNEE,
    "left_wheel":  LEFT_WHEEL,
    "right_wheel": RIGHT_WHEEL,
}
JOINTS = list(JOINT_CTRL.keys())


def fix_robot(model, data, height: float=1.0):
    """Set the robot in a fixed upright position."""
    data.qpos[:7] = np.array([0.0] * 7)
    data.qvel[:6] = np.array([0.0] * 6)
    data.qpos[2] = height


def simulate(model, data, log: list[dict], use_viewer: bool = False) -> list[dict]:
    """Simulate a 50 Hz log and return the resulting trajectory.

    Args:
        model: MuJoCo model.
        data:  MuJoCo data.
        log:   List of entries at 50 Hz (output of resample.py).
        use_viewer: Whether to open the MuJoCo passive viewer.

    Returns:
        List of entries in the same format, with "read" filled from the
        simulation state and "target" copied from the log.
    """
    model.opt.timestep = 0.005  # 200 Hz simulation
    inf_period = 4              # 50 Hz inference
    step_counter = 0
    inf_step = 0

    results: list[dict] = []
    fix_robot(model, data)
    mujoco.mj_forward(model, data)

    viewer_ctx = None
    try:
        if use_viewer:
            viewer_ctx = mujoco.viewer.launch_passive(model, data)

        t = 0.0
        t_start = time.perf_counter()

        while inf_step < len(log):
            if step_counter % inf_period == 0:
                entry = log[inf_step]

                obs = get_inputs(model, data, last_action=np.zeros(6), command=np.zeros(3))["obs"][0]
                sim_reads = {
                    "left_hip":    float(obs[0]),
                    "left_knee":   float(obs[1]),
                    "right_hip":   float(obs[2]),
                    "right_knee":  float(obs[3]),
                    "left_wheel":  float(obs[4]),
                    "right_wheel": float(obs[5]),
                }

                # Send targets to actuators
                for joint, ctrl_idx in JOINT_CTRL.items():
                    data.ctrl[ctrl_idx] = entry[joint]["target"]

                # Record result
                result: dict = {"timestamp": entry["timestamp"]}
                for joint in JOINTS:
                    result[joint] = {
                        "read":   sim_reads[joint],
                        "target": entry[joint]["target"],
                    }
                results.append(result)
                inf_step += 1

            fix_robot(model, data)
            mujoco.mj_step(model, data)

            if viewer_ctx is not None and viewer_ctx.is_running():
                viewer_ctx.sync()

            step_counter += 1
            t += model.opt.timestep
            time_until_next_step = (t_start + t) - time.perf_counter()
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    finally:
        if viewer_ctx is not None and viewer_ctx.is_running():
            viewer_ctx.close()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path, help="Path to a source log JSON file.")
    parser.add_argument("--viewer", action="store_true", help="Run with MuJoCo viewer.")
    args = parser.parse_args()

    model: mujoco.MjModel = mujoco.MjModel.from_xml_path(UPKIE_MODEL_PATH)
    data: mujoco.MjData = mujoco.MjData(model)

    with open(args.log) as f:
        log = json.load(f)

        results = simulate(model, data, log, use_viewer=args.viewer)

        plot_leg_trajectories(results, title=f"Simulated - {args.log.stem}", show=False)
        plot_leg_trajectories(log, title=f"Original - {args.log.stem}", show=True)


if __name__ == "__main__":
    main()