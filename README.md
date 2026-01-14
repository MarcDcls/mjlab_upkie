# MjLab Upkie

<img src="https://github.com/user-attachments/assets/f6293fbc-5c59-4e56-bc7f-0ee930503f11" align="right" height="300px">

This repository contains simulation environments for the [Upkie](https://github.com/upkie/upkie) robot. 
They are built using the [MjLab](https://github.com/mujocolab/mjlab) framework.

I thank the MjLab team for their great work on this framework, as well as 
Stéphane Caron and the whole Upkie team for the robot original design.

Currently a velocity control task is implemented, allowing the robot to follow given 
target linear and angular velocities, while being able to resist external disturbances. 
This task is presented in the video on the right. 
The implementation of a standup task is also planned for the future. 

The robot model used in this repository is based on my own design of the Upkie robot. 
A presentation of this design is available [here](https://cad.onshape.com/documents/626c0feba56391e940274c5a/v/528b63837812ab4634b4ea70/e/f40baedc39efd40637593503?renderMode=0&uiState=69651b27fe1457b05503e479). 

## Install

To install the repository, you need the uv package manager. 
If you don't have it yet, you can install it by following the instructions [here](https://docs.astral.sh/uv/getting-started/installation/#installation-methods).

Then, clone this repository and run the following command in your terminal:
```
uv sync
```

## Using a velocity agent within its Mjlab environment (GPU required)

You can use a pre-trained agent directly in its environment, as shown in the video above, where random velocity commands are given to the robot at regular intervals. 
Linear velocity commands are represented by a blue arrow, while angular velocity commands are represented by a green vertical one.

To play with the velocity agent, use the following command:

```
uv run play Mjlab-Velocity-Upkie --checkpoint-file logs/rsl_rl/upkie_velocity/bests/default.pt
```

To push the robot while playing, you can double click on the trunk of the robot in the simulation window. 
Then, while holding the left-ctrl key, right-click and drag to apply a force to the robot.

## Using a velocity agent in a MuJoCo simulation (CPU supported)

You can also use the same agent in a MuJoCo simulation, avoiding the need for a GPU. 
To do so, run the following command:

```
uv run python sim.py
```

In this configuration, the commanded velocities can be set via keyboard inputs:
- Up/Down: increase/decrease linear velocity
- Left/Right: increase/decrease angular velocity
- Space: reset commanded velocities to zero

As before, you can push the robot by double clicking on its trunk, 
then right-clicking and dragging while holding the left-ctrl key.

![sim_cut_x1 5](https://github.com/user-attachments/assets/f2505df7-87fe-488f-9fbd-bc34e422ce9c)

## Training your own agent

You can also modify the environments to train your own agent. 
To do so, you can modify the environment configurations at `src/mjlab_upkie/tasks/*_env_cfg.py`.
In the following, we will suppose you want to train the velocity agent.

To test the environment before training, you can use the following commands to play with
a zero or random agent:

```
uv run play Mjlab-Velocity-Upkie --agent zero
uv run play Mjlab-Velocity-Upkie --agent random
```

You can then start the training process by using the following command:

```
uv run train Mjlab-Velocity-Upkie --env.scene.num-envs 2048
```

This command will start the training process using 2048 parallel environments.
Once you have trained your agent and obtained a checkpoint file, you can play it back 
as previously explained using the following commands:

```
uv run play Mjlab-Velocity-Upkie --checkpoint-file [path to your checkpoint]
```

Here, [path to your checkpoint] should be replaced with the actual path to the checkpoint file, 
which is typically located at `logs/rsl_rl/upkie_velocity/[date]/model_[number].pt`.

## Model

The Upkie model used in this repository is based on my own design of the Upkie robot. 
It has been made using Onshape CAD software and is available [here]
(https://cad.onshape.com/documents/626c0feba56391e940274c5a/v/528b63837812ab4634b4ea70/e/f40baedc39efd40637593503?renderMode=0&uiState=69651b27fe1457b05503e479).

It is mainly inspired by the original Upkie design, but features some differences, 
such as the use of a LiPo battery for power supply, a different IMU placement, and a slightly modified leg design. 
It also features openings on the front and back of the robot's body for better access to internal components.

The model files can be found at `src/mjlab_upkie/robot/upkie/`. They were generated using [onshape-to-robot]
(https://onshape-to-robot.readthedocs.io/en/latest/)

TODO: add pictures


