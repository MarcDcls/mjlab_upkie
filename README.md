# MjLab Upkie


<img src="https://github.com/user-attachments/assets/314c3cbd-bc42-4f79-831d-322b2074bf35" align="right" height="300px" margin="50px">

This repository is a work in progress to create a simulation environment for the [Upkie robot](https://github.com/upkie/upkie) using the [MjLab](https://github.com/mujocolab/mjlab) framework.
I thanks the MjLab team for their great work on this framework, as well as Stéphane Caron and the whole Upkie team for the robot design.

Currently a velocity control task is implemented, allowing the robot to follow given target linear and angular velocities, while being able to resist external disturbances (see video on the right).
Two other tasks are planned: the same velocity control task but with legs bent backward, and a standup task. 

I'm currently building my own version of the robot. The CAD  files are available [here](https://cad.onshape.com/documents/626c0feba56391e940274c5a/v/e55f6b42bf6a32302009cdcb/e/f40baedc39efd40637593503?renderMode=0&uiState=6911b15212b18eb69d61e2a6). The robot model currently used for simulation is quite simplified, I plan to use this CAD model to create a more accurate model in the future.

## Install

To install the repository, run the following command in your terminal:
```
uv sync
```

## Use an already trained agent

Some of my best trained agents are available in the repository in the `logs/rsl_rl/upkie_velocity/bests` folder. 
You can play them in an environment where random velocity commands are given to the robot at regular intervals. 
Linear velocity commands are represented by a blue arrow, while angular velocity commands are represented by a green vertical one.

To play with the velocity control agent with straight legs, use the following command:

```
uv run play Mjlab-Velocity-Upkie-Play --checkpoint-file logs/rsl_rl/upkie_velocity/bests/default.pt
```

To push the robot while playing, you can double click on the trunk of the robot in the simulation window. 
Then, while holding the left-ctrl key, right-click and drag to apply a force to the robot.

This agent aims to have straight legs and to reject moderate pushes, which tends to result in having alternated legs as it seems 
more effective for rejection of disturbances. To play with the velocity control agent with backward bent legs, use the following command:

```
uv run play Mjlab-Velocity-Upkie-Legs-Backward-Play --checkpoint-file logs/rsl_rl/upkie_velocity/bests/legs_backward.pt
```

This agent is expected to resist external pushes a bit less effectively, but hey, style matters! :sunglasses:.

## Training your own agent

You can modify the environments and train agents. To do so, modify the environment configurations in `src/mjlab_upkie/tasks/upkie_velocity_env_cfg.py`.

To train an agent for the default Upkie velocity control task or the backward knee Upkie velocity control task, 
you can use the following commands:

```
uv run train Mjlab-Velocity-Upkie --env.scene.num-envs 2048
uv run train Mjlab-Velocity-Upkie-Legs-Backward --env.scene.num-envs 2048
```

These commands will start the training process using 2048 parallel environments. The pose regulation reward encourages the robot to maintain straight or 
bent backward legs, depending of which command you use. 

## Playing with your agents

To play with a trained agent for the Upkie velocity control task, use the following command:

```
uv run play Mjlab-Velocity-Upkie-Play --checkpoint-file [path to your checkpoint]
uv run play Mjlab-Velocity-Upkie-Legs-Backward-Play --checkpoint-file [path to your checkpoint]
```

Here, [path to your checkpoint] should be replaced with the actual path to the checkpoint file, which is typically located at `logs/rsl_rl/upkie_velocity/[date]/model_[number].pt`.


## Debug

To debug an environment:

```
uv run play Mjlab-Velocity-Upkie --agent zero
uv run play Mjlab-Velocity-Upkie --agent random
```
