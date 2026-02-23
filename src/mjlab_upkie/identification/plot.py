# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Plot the leg DOF trajectories (read vs target) from a log.

Log format:
[
    {
        "timestamp":   t,
        "left_hip":    { "read": ..., "target": ... },
        ...
        "right_wheel": { "read": ..., "target": ... },
    },
    ...
]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LEG_JOINTS = ["left_hip", "left_knee", "right_hip", "right_knee"]
JOINT_LABELS = {
    "left_hip":   "Left Hip",
    "left_knee":  "Left Knee",
    "right_hip":  "Right Hip",
    "right_knee": "Right Knee",
}


def plot_leg_trajectories(
    log: list[dict],
    title: str = "",
    show: bool = True,
) -> plt.Figure:
    """Plot read vs target trajectories for the 4 leg DOFs.

    Args:
        log: Log list.
        title: Optional figure title.
        show: Whether to call plt.show().

    Returns:
        The matplotlib figure.
    """
    timestamps = np.array([e["timestamp"] for e in log])

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    if title:
        fig.suptitle(title, fontsize=13)

    for ax, joint in zip(axes, LEG_JOINTS):
        read   = np.array([e[joint]["read"]   for e in log])
        target = np.array([e[joint]["target"] for e in log])

        ax.plot(timestamps, np.degrees(target), label="target", color="tab:blue",  linewidth=1.2, linestyle="--")
        ax.plot(timestamps, np.degrees(read),   label="read",   color="tab:orange", linewidth=1.2)
        ax.set_ylabel(f"{JOINT_LABELS[joint]}\n[deg]", fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, linewidth=0.5, alpha=0.7)

    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()

    if show:
        plt.show()

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot leg DOF trajectories from a log.")
    parser.add_argument("log", type=Path, help="Path to the log JSON file.")
    args = parser.parse_args()

    with open(args.log) as f:
        log = json.load(f)

    plot_leg_trajectories(log, title=args.log.stem, show=True)


if __name__ == "__main__":
    main()
