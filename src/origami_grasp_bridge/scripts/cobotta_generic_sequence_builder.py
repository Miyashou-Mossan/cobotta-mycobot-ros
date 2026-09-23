#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
from pathlib import Path


DEFAULT_DUPLICATE_TOLERANCE_RAD = 1.0e-12


def max_joint_diff(q1, q2):
    if len(q1) != len(q2):
        raise ValueError("Joint vector length mismatch")

    return max(
        abs(a - b)
        for a, b in zip(q1, q2)
    )


def validate_input(data):
    if "joint_names" not in data:
        raise RuntimeError("Missing key: joint_names")

    if "combined_sequence" not in data:
        raise RuntimeError("Missing key: combined_sequence")

    joint_names = data["joint_names"]
    sequence = data["combined_sequence"]

    if not joint_names:
        raise RuntimeError("joint_names is empty")

    if not sequence:
        raise RuntimeError("combined_sequence is empty")

    joint_count = len(joint_names)

    for i, item in enumerate(sequence):
        if "joints_rad" not in item:
            raise RuntimeError(
                "Missing joints_rad at sequence index {}".format(i)
            )

        if "gripper_m" not in item:
            raise RuntimeError(
                "Missing gripper_m at sequence index {}".format(i)
            )

        q = item["joints_rad"]

        if len(q) != joint_count:
            raise RuntimeError(
                "Invalid joint count at sequence index {}: {} != {}".format(
                    i,
                    len(q),
                    joint_count,
                )
            )

        for j, value in enumerate(q):
            if not math.isfinite(float(value)):
                raise RuntimeError(
                    "NaN/Inf at sequence index {}, joint {}".format(
                        i,
                        joint_names[j],
                    )
                )

        if not math.isfinite(float(item["gripper_m"])):
            raise RuntimeError(
                "Invalid gripper_m at sequence index {}".format(i)
            )


def make_trace_point(item, sequence_index):
    return {
        "joints_rad": list(item["joints_rad"]),
        "source_sequence_index": sequence_index,
        "source_segment": item.get("segment"),
        "source_index": item.get("source_index"),
    }


def clean_motion_points(
    sequence,
    start_index,
    end_index,
    tolerance_rad,
):
    cleaned = []
    removed = []

    for i in range(start_index, end_index + 1):
        point = make_trace_point(sequence[i], i)

        if not cleaned:
            cleaned.append(point)
            continue

        diff = max_joint_diff(
            cleaned[-1]["joints_rad"],
            point["joints_rad"],
        )

        if diff <= tolerance_rad:
            removed.append({
                "source_sequence_index": i,
                "previous_kept_sequence_index":
                    cleaned[-1]["source_sequence_index"],
                "max_joint_diff_rad": diff,
            })
            continue

        cleaned.append(point)

    return cleaned, removed


def build_arm_motion(
    action_index,
    sequence,
    start_index,
    end_index,
    tolerance_rad,
):
    points, removed = clean_motion_points(
        sequence,
        start_index,
        end_index,
        tolerance_rad,
    )

    if not points:
        raise RuntimeError(
            "ARM_MOTION became empty: {} -> {}".format(
                start_index,
                end_index,
            )
        )

    return {
        "action_index": action_index,
        "type": "ARM_MOTION",
        "gripper_m": float(sequence[start_index]["gripper_m"]),
        "source_sequence_start": start_index,
        "source_sequence_end": end_index,
        "input_point_count": end_index - start_index + 1,
        "output_point_count": len(points),
        "removed_duplicate_count": len(removed),
        "removed_duplicates": removed,
        "points": points,
    }


def build_sequence(data, tolerance_rad):
    sequence = data["combined_sequence"]

    actions = []
    action_index = 0
    motion_start = 0

    for i in range(1, len(sequence)):
        previous = sequence[i - 1]
        current = sequence[i]

        previous_gripper = float(previous["gripper_m"])
        current_gripper = float(current["gripper_m"])

        # 現段階では gripper_m は離散指令値として扱い、
        # 厳密な値変化をイベントとして検出する。
        if previous_gripper == current_gripper:
            continue

        # まずイベント直前までを ARM_MOTION として確定する。
        actions.append(
            build_arm_motion(
                action_index,
                sequence,
                motion_start,
                i - 1,
                tolerance_rad,
            )
        )
        action_index += 1

        # GRIPPER_EVENT 中に腕が動いていないことを確認する。
        arm_diff = max_joint_diff(
            previous["joints_rad"],
            current["joints_rad"],
        )

        if arm_diff > tolerance_rad:
            raise RuntimeError(
                "Gripper changed while arm pose changed at "
                "{} -> {}: max_joint_diff={} rad > tolerance={} rad".format(
                    i - 1,
                    i,
                    arm_diff,
                    tolerance_rad,
                )
            )

        actions.append({
            "action_index": action_index,
            "type": "GRIPPER_EVENT",
            "from_m": previous_gripper,
            "to_m": current_gripper,
            "arm_joints_rad": list(current["joints_rad"]),
            "arm_pose_max_diff_rad": arm_diff,
            "source_transition_from": i - 1,
            "source_transition_to": i,
            "source_segment_from": previous.get("segment"),
            "source_segment_to": current.get("segment"),
        })
        action_index += 1

        # イベント後の同一姿勢を、次 ARM_MOTION の開始姿勢として残す。
        motion_start = i

    # 最後の ARM_MOTION
    actions.append(
        build_arm_motion(
            action_index,
            sequence,
            motion_start,
            len(sequence) - 1,
            tolerance_rad,
        )
    )

    return actions


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert a COBOTTA combined sequence into generic "
            "ARM_MOTION / GRIPPER_EVENT actions."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input complete-path JSON",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output generic-sequence JSON",
    )

    parser.add_argument(
        "--duplicate-joint-tolerance-rad",
        type=float,
        default=DEFAULT_DUPLICATE_TOLERANCE_RAD,
    )

    args = parser.parse_args()

    if args.duplicate_joint_tolerance_rad < 0.0:
        raise ValueError(
            "duplicate-joint-tolerance-rad must be >= 0"
        )

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    with input_path.open() as f:
        data = json.load(f)

    validate_input(data)

    actions = build_sequence(
        data,
        args.duplicate_joint_tolerance_rad,
    )

    arm_actions = [
        a for a in actions
        if a["type"] == "ARM_MOTION"
    ]

    gripper_actions = [
        a for a in actions
        if a["type"] == "GRIPPER_EVENT"
    ]

    removed_count = sum(
        a["removed_duplicate_count"]
        for a in arm_actions
    )

    source_metadata_keys = [
        "candidate_index",
        "finish_index",
        "approach_angle_deg",
        "normal_sign",
        "finish_branch_index",
    ]

    source_metadata = {
        key: data[key]
        for key in source_metadata_keys
        if key in data
    }

    output = {
        "schema_version": 1,
        "sequence_type": "COBOTTA_GENERIC_ACTION_SEQUENCE",
        "source_file": str(input_path),
        "joint_names": list(data["joint_names"]),
        "settings": {
            "duplicate_joint_tolerance_rad":
                args.duplicate_joint_tolerance_rad,
            "gripper_transition_requires_stationary_arm": True,
        },
        "source_metadata": source_metadata,
        "summary": {
            "source_point_count": len(data["combined_sequence"]),
            "action_count": len(actions),
            "arm_motion_count": len(arm_actions),
            "gripper_event_count": len(gripper_actions),
            "removed_duplicate_count": removed_count,
        },
        "actions": actions,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w") as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("===== GENERIC SEQUENCE BUILDER =====")
    print("input :", input_path)
    print("output:", output_path)
    print(
        "duplicate tolerance:",
        args.duplicate_joint_tolerance_rad,
        "rad",
    )

    print("\n===== SUMMARY =====")
    for key, value in output["summary"].items():
        print("{}: {}".format(key, value))

    print("\n===== ACTIONS =====")

    for action in actions:
        if action["type"] == "ARM_MOTION":
            print(
                "[{}] ARM_MOTION  source={}..{}  "
                "gripper_m={}  points={} -> {}  "
                "duplicates_removed={}".format(
                    action["action_index"],
                    action["source_sequence_start"],
                    action["source_sequence_end"],
                    action["gripper_m"],
                    action["input_point_count"],
                    action["output_point_count"],
                    action["removed_duplicate_count"],
                )
            )

        elif action["type"] == "GRIPPER_EVENT":
            print(
                "[{}] GRIPPER_EVENT  {} -> {} m  "
                "source={} -> {}  arm_diff={:.15e} rad".format(
                    action["action_index"],
                    action["from_m"],
                    action["to_m"],
                    action["source_transition_from"],
                    action["source_transition_to"],
                    action["arm_pose_max_diff_rad"],
                )
            )


if __name__ == "__main__":
    main()
