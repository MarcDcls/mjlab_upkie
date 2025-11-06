# MjLab Upkie (WIP)

## Install

```
uv sync
```

## Training an agent

```
uv run train Mjlab-Velocity-Upkie --env.scene.num-envs 2048
```


```
uv run train Mjlab-Velocity-Upkie-Legs-Backward --env.scene.num-envs 2048
```


## Playing the environnement

```
uv run play Mjlab-Velocity-Upkie-Play --checkpoint-file logs/rsl_rl/upkie_velocity/...
```
