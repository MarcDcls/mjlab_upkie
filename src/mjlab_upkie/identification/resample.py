# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0


"""Resample a variable-framerate log to a fixed target framerate.

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


JOINTS = ["left_hip", "left_knee", "right_hip", "right_knee", "left_wheel", "right_wheel"]
FIELDS = ["read", "target"]


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b at position t in [0, 1]."""
    return a + (b - a) * t


def resample(entries: list[dict], target_fps: float) -> list[dict]:
    t_offset = entries[0]["timestamp"]
    t_end = entries[-1]["timestamp"] - t_offset
    dt = 1.0 / target_fps

    resampled = []
    t_cur = 0.0
    idx = 0

    while t_cur <= t_end:
        t_abs = t_cur + t_offset
        while idx < len(entries) - 2 and entries[idx + 1]["timestamp"] <= t_abs:
            idx += 1

        left  = entries[idx]
        right = entries[idx + 1]

        t_left  = left["timestamp"]
        t_right = right["timestamp"]
        span    = t_right - t_left

        alpha = max(0.0, min(1.0, (t_abs - t_left) / span)) if span > 0.0 else 0.0

        entry: dict = {"timestamp": t_cur}
        for joint in JOINTS:
            entry[joint] = {
                field: lerp(left[joint][field], right[joint][field], alpha)
                for field in FIELDS
            }
        resampled.append(entry)
        t_cur += dt

    return resampled


def resample_file(src_path: Path, dst_path: Path, fps: float) -> None:
    with open(src_path) as f:
        entries = json.load(f)
    resampled = resample(entries, fps)
    with open(dst_path, "w") as f:
        json.dump(resampled, f, indent=2)
    print(f"  {src_path.name}  →  {dst_path.name}  ({len(resampled)} entries @ {fps} Hz)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resample a variable-framerate log to a fixed target framerate."
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        help="Path to a source log JSON file.",
    )
    parser.add_argument(
        "--logdir",
        type=Path,
        default=None,
        help="Directory of JSON logs to resample (processes all *.json files).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=50.0,
        help="Target framerate in Hz (default: 50).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output file (single-file mode) or output directory (--logdir mode). "
            "Defaults to <input_stem>_<fps>hz.json beside the source, "
            "or a '<fps>hz/' sub-directory when using --logdir."
        ),
    )
    args = parser.parse_args()

    fps: float = args.fps

    if args.logdir is not None:
        src_dir: Path = args.logdir
        json_files = sorted(src_dir.glob("*.json"))
        if not json_files:
            print(f"No JSON files found in {src_dir}")
            return
        dst_dir: Path = args.output or src_dir / f"{int(fps)}hz"
        dst_dir.mkdir(parents=True, exist_ok=True)
        print(f"Resampling {len(json_files)} file(s) from {src_dir}  →  {dst_dir}")
        for json_file in json_files:
            dst_path = dst_dir / f"{json_file.stem}_{int(fps)}hz.json"
            resample_file(json_file, dst_path, fps)
    elif args.input is not None:
        src_path: Path = args.input
        dst_path = args.output or src_path.with_name(f"{src_path.stem}_{int(fps)}hz.json")
        print(f"Resampling {src_path}")
        resample_file(src_path, dst_path, fps)
    else:
        parser.error("Provide either a positional input file or --logdir.")

    print("Done.")


if __name__ == "__main__":
    main()
