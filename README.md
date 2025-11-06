# MjLab Upkie (WIP)

## Install

To install the repository, run the following command in your terminal:
```
uv sync
```

## Training an agent

To train an agent for the Upkie velocity control task, use the following command:

```
uv run train Mjlab-Velocity-Upkie --env.scene.num-envs 2048
```

This command will start the training process using 2048 parallel environments. The pose regulation reward encourages the robot to maintain straight legs, whish can results in having alternated legs, which seems more effective for rejection of disturbances.

To have an agent that tends to walk with backward knees, you can use the following command:

```
uv run train Mjlab-Velocity-Upkie-Legs-Backward --env.scene.num-envs 2048
```

Such an agent is expected to resist external pressures a bit less effectively, but hey, style matters! :sunglasses:.

## Playing the agent

To play with a trained agent for the Upkie velocity control task, use the following command:

```
uv run play Mjlab-Velocity-Upkie-Play --checkpoint-file [path to your checkpoint]
```

Here, [path to your checkpoint] should be replaced with the actual path to the checkpoint file, which is typically located at `logs/rsl_rl/upkie_velocity/[date]/model_[number].pt`.

## Personnal notes

To debug an environment:

```
uv run play Mjlab-Velocity-Upkie --agent zero
```
