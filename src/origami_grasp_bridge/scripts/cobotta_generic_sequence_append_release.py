#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import json
import math
from pathlib import Path

import rospy


def load_json(path_text):
    path = Path(path_text).expanduser()

    if not path.is_absolute():
        path = Path.cwd() / path

    if not path.exists():
        raise RuntimeError(
            "JSON not found: {}".format(path)
        )

    with path.open() as f:
        data = json.load(f)

    return str(path), data


def max_joint_diff(a, b):
    if len(a) != len(b):
        raise RuntimeError(
            "joint vector length mismatch"
        )

    return max(
        abs(float(x) - float(y))
        for x, y in zip(a, b)
    )


def main():
    rospy.init_node(
        "cobotta_generic_sequence_append_release"
    )

    sequence_json = rospy.get_param(
        "~sequence_json"
    )

    retreat_json = rospy.get_param(
        "~retreat_json"
    )

    output_json = rospy.get_param(
        "~output_json"
    )

    seq_path, seq = load_json(
        sequence_json
    )

    retreat_path, retreat = load_json(
        retreat_json
    )

    if "actions" not in seq:
        raise RuntimeError(
            "sequence has no actions"
        )

    if not seq["actions"]:
        raise RuntimeError(
            "sequence actions are empty"
        )

    if "retreat_path" not in retreat:
        raise RuntimeError(
            "retreat JSON has no retreat_path"
        )

    retreat_points = retreat[
        "retreat_path"
    ]

    if not retreat_points:
        raise RuntimeError(
            "retreat_path is empty"
        )

    seq_joint_names = list(
        seq["joint_names"]
    )

    retreat_joint_names = list(
        retreat["arm_joint_names"]
    )

    if seq_joint_names != retreat_joint_names:
        raise RuntimeError(
            "joint_names mismatch"
        )

    last_action = seq["actions"][-1]

    if last_action.get("type") != "ARM_MOTION":
        raise RuntimeError(
            "last sequence action must be ARM_MOTION"
        )

    if not last_action.get("points"):
        raise RuntimeError(
            "last ARM_MOTION has no points"
        )

    current_gripper = float(
        last_action["gripper_m"]
    )

    retreat_gripper = float(
        retreat["gripper_closed_m"]
    )

    release_opening = float(
        retreat["release_opening_m"]
    )

    gripper_diff = abs(
        current_gripper
        - retreat_gripper
    )

    if gripper_diff > 1.0e-12:
        raise RuntimeError(
            "gripper mismatch: "
            "sequence={} retreat={}".format(
                current_gripper,
                retreat_gripper,
            )
        )

    sequence_last_joints = [
        float(q)
        for q in last_action[
            "points"
        ][-1]["joints_rad"]
    ]

    retreat_first_joints = [
        float(q)
        for q in retreat_points[
            0
        ]["joints_rad"]
    ]

    connection_diff = max_joint_diff(
        sequence_last_joints,
        retreat_first_joints,
    )

    duplicate_tolerance = float(
        seq.get(
            "settings",
            {}
        ).get(
            "duplicate_joint_tolerance_rad",
            1.0e-12,
        )
    )

    if connection_diff > duplicate_tolerance:
        raise RuntimeError(
            "sequence/retreat connection "
            "is not continuous: "
            "max diff = {:.16e} rad".format(
                connection_diff
            )
        )

    result = copy.deepcopy(seq)

    merged_action = result[
        "actions"
    ][-1]

    original_count = len(
        merged_action["points"]
    )

    generated_points = []

    # retreat index 0 はFold終端と完全重複するため除外。
    for p in retreat_points[1:]:
        generated_points.append({
            "joints_rad": [
                float(q)
                for q in p["joints_rad"]
            ],
            "source_sequence_index": None,
            "source_segment":
                "RELEASE_RETREAT",
            "source_index":
                int(p["index"]),
        })

    merged_action["points"].extend(
        generated_points
    )

    merged_action[
        "input_point_count"
    ] = (
        original_count
        + len(retreat_points)
    )

    merged_action[
        "output_point_count"
    ] = len(
        merged_action["points"]
    )

    merged_action[
        "removed_duplicate_count"
    ] = int(
        merged_action.get(
            "removed_duplicate_count",
            0,
        )
    ) + 1

    if "removed_duplicates" not in merged_action:
        merged_action[
            "removed_duplicates"
        ] = []

    merged_action[
        "removed_duplicates"
    ].append({
        "reason":
            "ARM_MOTION_CONNECTION_DUPLICATE",
        "source_segment":
            "RELEASE_RETREAT",
        "source_index":
            0,
        "max_joint_diff_rad":
            float(connection_diff),
    })

    merged_action[
        "release_retreat_extension"
    ] = {
        "source_file":
            retreat_path,
        "retreat_distance_m":
            float(
                retreat[
                    "retreat_distance_m"
                ]
            ),
        "path_resolution_m":
            float(
                retreat[
                    "path_resolution_m"
                ]
            ),
        "input_point_count":
            len(retreat_points),
        "appended_point_count":
            len(generated_points),
        "connection_duplicate_removed":
            True,
    }

    final_joints = [
        float(q)
        for q in retreat_points[
            -1
        ]["joints_rad"]
    ]

    gripper_event = {
        "action_index": None,
        "type": "GRIPPER_EVENT",
        "from_m":
            retreat_gripper,
        "to_m":
            release_opening,
        "arm_joints_rad":
            final_joints,
        "arm_pose_max_diff_rad":
            0.0,
        "source_transition_from":
            None,
        "source_transition_to":
            None,
        "source_segment_from":
            "RELEASE_READY",
        "source_segment_to":
            "RELEASED",
        "release_metadata": {
            "retreat_source_file":
                retreat_path,
            "relative_finger_opening_m":
                2.0 * release_opening,
            "collision_check":
                retreat[
                    "release_event"
                ][
                    "collision_check"
                ],
            "collision_sample_count":
                retreat[
                    "release_event"
                ][
                    "sample_count"
                ],
        },
    }

    result["actions"].append(
        gripper_event
    )

    # action_indexを全体で振り直す
    for i, action in enumerate(
        result["actions"]
    ):
        action["action_index"] = i

    summary = result.setdefault(
        "summary",
        {}
    )

    summary["action_count"] = len(
        result["actions"]
    )

    summary["arm_motion_count"] = sum(
        1
        for a in result["actions"]
        if a.get("type") == "ARM_MOTION"
    )

    summary["gripper_event_count"] = sum(
        1
        for a in result["actions"]
        if a.get("type") == "GRIPPER_EVENT"
    )

    summary[
        "removed_duplicate_count"
    ] = int(
        summary.get(
            "removed_duplicate_count",
            0,
        )
    ) + 1

    result[
        "release_extension"
    ] = {
        "retreat_file":
            retreat_path,
        "retreat_distance_m":
            float(
                retreat[
                    "retreat_distance_m"
                ]
            ),
        "release_opening_m":
            release_opening,
        "connection_max_joint_diff_rad":
            float(connection_diff),
        "merged_arm_motion":
            True,
    }

    output_path = Path(
        output_json
    ).expanduser()

    if not output_path.is_absolute():
        output_path = (
            Path.cwd()
            / output_path
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w") as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print()
    print(
        "===== Generic Sequence Release Append ====="
    )
    print(
        "sequence input       : {}".format(
            seq_path
        )
    )
    print(
        "retreat input        : {}".format(
            retreat_path
        )
    )
    print(
        "connection diff      : "
        "{:.16e} rad".format(
            connection_diff
        )
    )
    print(
        "original ARM points  : {}".format(
            original_count
        )
    )
    print(
        "retreat input points : {}".format(
            len(retreat_points)
        )
    )
    print(
        "duplicate removed    : 1"
    )
    print(
        "merged ARM points    : {}".format(
            len(
                merged_action["points"]
            )
        )
    )
    print(
        "release event        : "
        "{:.6f} -> {:.6f} mm/side".format(
            retreat_gripper * 1000.0,
            release_opening * 1000.0,
        )
    )
    print(
        "action count         : {}".format(
            len(result["actions"])
        )
    )
    print()
    print("===== ACTIONS =====")

    for action in result["actions"]:
        if action["type"] == "ARM_MOTION":
            print(
                "[{}] ARM_MOTION "
                "gripper={:.3f} mm "
                "points={}".format(
                    action["action_index"],
                    float(
                        action["gripper_m"]
                    ) * 1000.0,
                    len(action["points"]),
                )
            )

        elif action["type"] == "GRIPPER_EVENT":
            print(
                "[{}] GRIPPER_EVENT "
                "{:.3f} -> {:.3f} mm/side".format(
                    action["action_index"],
                    float(
                        action["from_m"]
                    ) * 1000.0,
                    float(
                        action["to_m"]
                    ) * 1000.0,
                )
            )

        else:
            print(
                "[{}] {}".format(
                    action["action_index"],
                    action["type"],
                )
            )

    print()
    print(
        "saved : {}".format(
            output_path
        )
    )


if __name__ == "__main__":
    main()
