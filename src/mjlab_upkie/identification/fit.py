# Copyright 2025 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

from datetime import datetime
import time
import mujoco
import argparse
import numpy as np
import optuna
from multiprocessing import Process
import json
import sys
import os
from copy import deepcopy


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--trials", type=int, default=100_000)
    argparser.add_argument("--workers", type=int, default=1)
    argparser.add_argument("--reset_period", default=None, type=float)
    argparser.add_argument("--wandb", action="store_true")
    argparser.add_argument("--eval", action="store_true")
    args = argparser.parse_args()

    params_json_filename = "params.json"
    json.dump({}, open(params_json_filename, "w"))

    logs = []
    for file in os.listdir("logs/robot/24.8V/"):
        if file.endswith(".json"):
            with open(os.path.join("logs/robot/24.8V/", file), "r") as f:
                logs.append(json.load(f))
    print(f"{len(logs)} logs loaded.")

    validation_batch = []


    # def monitor(study, trial):
    #     global last_log, use_wandb, wandb_run

    #     elapsed = time.time() - last_log

    #     if use_wandb and wandb_run is None:
    #         wandb_run = wandb.init(
    #             name=f"upkie_identification_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    #             project=f"upkie_identification",
    #         )

    #     if elapsed > 0.2:
    #         last_log = time.time()
    #         data = deepcopy(study.best_params)
    #         trial_number = trial.number
    #         best_value = study.best_value
    #         wandb_log = {
    #             "optim/best_value": best_value,
    #             "optim/trial_number": trial_number,
    #         }

    #         model = make_model()
    #         model_parameters = model.get_parameters()
    #         for key in model_parameters:
    #             if key not in data:
    #                 data[key] = model_parameters[key].value
    #         data["model"] = args.model
    #         data["actuator"] = args.actuator

    #         json.dump(data, open(params_json_filename, "w"))

    #         if args.validation_kp > 0:
    #             val_model = load_model(params_json_filename)
    #             val_best_value = compute_score(val_model, validation_batch)
    #             wandb_log["optim/val_best_value"] = val_best_value

    #         print()
    #         message.bright(f"[Trial {trial_number}, Best score: {best_value}]")
    #         print(
    #             message.emphasis(f"Best params found (saved to {params_json_filename}): ")
    #         )
    #         for key in data:
    #             infos, warning = None, None

    #             if key in model_parameters:
    #                 if model_parameters[key].optimize:
    #                     infos = f"min: {model_parameters[key].min}, max: {model_parameters[key].max}"
    #                 else:
    #                     warning = "not optimized"

    #             message.print_parameter(key, data[key], infos, warning)

    #             if type(data[key]) == float:
    #                 wandb_log[f"params/{key}"] = data[key]

    #         if wandb_run is not None:
    #             wandb.log(wandb_log)
    #     sys.stdout.flush()


    # model: mujoco.MjModel = mujoco.MjModel.from_xml_path(UPKIE_MODEL_PATH)
    # data: mujoco.MjData = mujoco.MjData(model)

    # motor_id = None
    # if args.position:
    #     motor_id = [LEFT_KNEE]
    # elif args.velocity:
    #     motor_id = [RIGHT_WHEEL]

    # model_wrapper = MujocoModelWrapper(model, data, actuator_idx=motor_id)

    # model.opt.timestep = 0.005  # 200 Hz simulation
    # inf_period = 4  # 50 Hz inference
    # step_counter = 0



    # study_name = f"study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # # Use a SQLite database to store the study results, allowing multiple workers to share the same study.
    # study_url = f"sqlite:///study.db"

    # sampler = optuna.samplers.CmaEsSampler(restart_strategy="bipop")

    # def optuna_run(enable_monitoring=True):
    #     if args.workers > 1:
    #         study = optuna.load_study(study_name=study_name, storage=study_url)
    #     else:
    #         study = optuna.create_study(sampler=sampler)
    #     optuna.logging.set_verbosity(optuna.logging.WARNING)
    #     callbacks = []
    #     if enable_monitoring:
    #         callbacks = [monitor]
    #     study.optimize(objective, n_trials=args.trials, n_jobs=1, callbacks=callbacks)

    # if args.workers > 1:
    #     optuna.create_study(study_name=study_name, storage=study_url, sampler=sampler)

    # # Running the other workers
    # for k in range(args.workers - 1):
    #     p = Process(target=optuna_run, args=(False,))
    #     p.start()

    # optuna_run(True)


if __name__ == "__main__":
    main()
