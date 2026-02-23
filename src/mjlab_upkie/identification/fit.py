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
import json

from sinus import fix_robot
from sim import get_inputs, reset_robot
from model_wrapper import MujocoModelWrapper
from mjlab_upkie.robot.upkie_constants import (
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_WHEEL,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_WHEEL,
    UPKIE_MODEL_PATH,
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--position", action="store_true", help="Whether to log position observations and actions.")
    parser.add_argument("--velocity", action="store_true", help="Whether to log velocity observations and actions.")
    parser.add_argument("--viewer", action="store_true", help="Run with MuJoCo viewer.")
    args = parser.parse_args()

    model: mujoco.MjModel = mujoco.MjModel.from_xml_path(UPKIE_MODEL_PATH)
    data: mujoco.MjData = mujoco.MjData(model)

    motor_id = None
    if args.position:
        motor_id = [LEFT_KNEE]
    elif args.velocity:
        motor_id = [RIGHT_WHEEL]

    model_wrapper = MujocoModelWrapper(model, data, actuator_idx=motor_id)

    model.opt.timestep = 0.005  # 200 Hz simulation
    inf_period = 4  # 50 Hz inference
    step_counter = 0

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




if args.eval:
    model = load_model("params.json")
    print(f"Score: {compute_scores(model, logs)}")
else:
    study_name = f"study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Study URL (when multiple workers are used)
    study_url = f"sqlite:///study.db"
    # study_url = f"mysql://root:root@127.0.0.1:6033/optuna"

    if args.method == "cmaes":
        sampler = optuna.samplers.CmaEsSampler(
            # x0=model.get_parameter_values(),
            restart_strategy="bipop"
        )
    elif args.method == "random":
        sampler = optuna.samplers.RandomSampler()
    elif args.method == "nsgaii":
        sampler = optuna.samplers.NSGAIISampler()
    else:
        raise ValueError(f"Unknown method: {args.method}")

    def optuna_run(enable_monitoring=True):
        if args.workers > 1:
            study = optuna.load_study(study_name=study_name, storage=study_url)
        else:
            study = optuna.create_study(sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        callbacks = []
        if enable_monitoring:
            callbacks = [monitor]
        study.optimize(objective, n_trials=args.trials, n_jobs=1, callbacks=callbacks)

    if args.workers > 1:
        optuna.create_study(study_name=study_name, storage=study_url, sampler=sampler)

    # Running the other workers
    for k in range(args.workers - 1):
        p = Process(target=optuna_run, args=(False,))
        p.start()

    optuna_run(True)
