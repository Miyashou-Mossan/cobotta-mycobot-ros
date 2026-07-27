#!/usr/bin/env python3
"""
Convert an external-PC RobotTrajectory YAML into the legacy COBOTTA Plan YAML format.

Input example:
  joint_trajectory:
    header: ...
    joint_names: [cobotta_joint_1, ..., cobotta_joint_6]
    points: [...]

Output example:
  plan_0:
    start_state_:
      joint_state: ...
    trajectory_:
      joint_trajectory: ...

This script only converts files. It does not connect to ROS or move the robot.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml


EXTERNAL_JOINT_NAMES = [f"cobotta_joint_{i}" for i in range(1, 7)]
LEGACY_JOINT_NAMES = [f"joint_{i}" for i in range(1, 7)]

# Legacy start_state_ includes gripper joints, while trajectory_ contains arm joints only.
LEGACY_START_STATE_NAMES = LEGACY_JOINT_NAMES


def fail(message: str) -> "NoReturn":
    raise ValueError(message)


def load_first_yaml_document(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        documents = [doc for doc in yaml.safe_load_all(f) if doc is not None]

    if not documents:
        fail(f"No YAML document found: {path}")

    root = documents[0]
    if not isinstance(root, dict):
        fail("The first YAML document must be a mapping.")

    return root


def ensure_number_list(value: Any, expected_len: int, field: str) -> List[float]:
    if not isinstance(value, list) or len(value) != expected_len:
        fail(f"{field} must be a list of length {expected_len}.")

    result: List[float] = []
    for i, item in enumerate(value):
        if not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            fail(f"{field}[{i}] is not a finite number.")
        result.append(float(item))
    return result


def validate_and_convert_joint_trajectory(root: Dict[str, Any]) -> Dict[str, Any]:
    jt = root.get("joint_trajectory")
    if not isinstance(jt, dict):
        fail("Top-level key 'joint_trajectory' was not found.")

    joint_names = jt.get("joint_names")
    if joint_names != EXTERNAL_JOINT_NAMES:
        fail(
            "Unexpected joint_names.\n"
            f"Expected: {EXTERNAL_JOINT_NAMES}\n"
            f"Actual:   {joint_names}"
        )

    points = jt.get("points")
    if not isinstance(points, list) or not points:
        fail("'joint_trajectory.points' must be a non-empty list.")

    converted_points: List[Dict[str, Any]] = []
    previous_time = -1.0

    for index, point in enumerate(points):
        if not isinstance(point, dict):
            fail(f"points[{index}] must be a mapping.")

        positions = ensure_number_list(point.get("positions"), 6, f"points[{index}].positions")

        velocities_raw = point.get("velocities", [])
        accelerations_raw = point.get("accelerations", [])
        effort_raw = point.get("effort", [])

        velocities = (
            ensure_number_list(velocities_raw, 6, f"points[{index}].velocities")
            if velocities_raw
            else [0.0] * 6
        )
        accelerations = (
            ensure_number_list(accelerations_raw, 6, f"points[{index}].accelerations")
            if accelerations_raw
            else [0.0] * 6
        )

        if not isinstance(effort_raw, list):
            fail(f"points[{index}].effort must be a list.")

        time_from_start = point.get("time_from_start")
        if not isinstance(time_from_start, dict):
            fail(f"points[{index}].time_from_start must be a mapping.")

        secs = int(time_from_start.get("secs", 0))
        nsecs = int(time_from_start.get("nsecs", 0))
        if secs < 0 or nsecs < 0 or nsecs >= 1_000_000_000:
            fail(f"points[{index}].time_from_start is invalid.")

        current_time = secs + nsecs * 1e-9
        if current_time <= previous_time and index > 0:
            fail(
                f"time_from_start is not strictly increasing at point {index}: "
                f"{current_time} <= {previous_time}"
            )
        previous_time = current_time

        converted_points.append(
            {
                "positions": positions,
                "velocities": velocities,
                "accelerations": accelerations,
                "effort": copy.deepcopy(effort_raw),
                "time_from_start": {"secs": secs, "nsecs": nsecs},
            }
        )

    header = copy.deepcopy(
        jt.get(
            "header",
            {
                "seq": 0,
                "stamp": {"secs": 0, "nsecs": 0},
                "frame_id": "",
            },
        )
    )

    first_positions = converted_points[0]["positions"]

    # The legacy loader later overwrites start_state_.joint_state.position
    # with the actual robot state immediately before execute().
    start_state_position = list(first_positions)

    legacy_plan = {
        "plan_0": {
            "start_state_": {
                "joint_state": {
                    "header": {
                        "seq": 0,
                        "stamp": {"secs": 0, "nsecs": 0},
                        "frame_id": "",
                    },
                    "name": list(LEGACY_START_STATE_NAMES),
                    "position": start_state_position,
                    "velocity": [0.0] * len(LEGACY_START_STATE_NAMES),
                    "effort": [0.0] * len(LEGACY_START_STATE_NAMES),
                }
            },
            "trajectory_": {
                "joint_trajectory": {
                    "header": header,
                    "joint_names": list(LEGACY_JOINT_NAMES),
                    "points": converted_points,
                }
            },
        }
    }

    return legacy_plan


def print_summary(output: Dict[str, Any], output_path: Path) -> None:
    jt = output["plan_0"]["trajectory_"]["joint_trajectory"]
    points = jt["points"]
    first = points[0]["positions"]
    last = points[-1]["positions"]
    duration = points[-1]["time_from_start"]
    total_seconds = duration["secs"] + duration["nsecs"] * 1e-9

    print("Conversion completed.")
    print(f"Output: {output_path}")
    print(f"Points: {len(points)}")
    print(f"Duration: {total_seconds:.9f} s")
    print(f"Joint names: {jt['joint_names']}")
    print(f"First positions: {first}")
    print(f"Last positions:  {last}")
    print()
    print("This file has only been converted. No robot command was sent.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert external COBOTTA RobotTrajectory YAML to legacy Plan YAML."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input RobotTrajectory YAML, e.g. ~/combined_robot_trajectory_0_19.yaml",
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path("combined_robot_trajectory_legacy_plan.yaml"),
        help="Output legacy Plan YAML",
    )
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not input_path.is_file():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        return 1

    try:
        root = load_first_yaml_document(input_path)
        converted = validate_and_convert_joint_trajectory(root)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                converted,
                f,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
                width=120,
            )

        # Re-read to verify the generated YAML is parseable.
        with output_path.open("r", encoding="utf-8") as f:
            verified = yaml.safe_load(f)
        if not isinstance(verified, dict) or "plan_0" not in verified:
            fail("Generated YAML verification failed.")

        print_summary(converted, output_path)
        return 0

    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
