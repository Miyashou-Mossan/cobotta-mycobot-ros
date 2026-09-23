#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import json
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
        "cobotta_generic_sequence_append_post_release"
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

    if not seq.get("actions"):
        raise RuntimeError(
            "sequence actions are empty"
        )

    if not retreat.get("retreat_path"):
        raise RuntimeError(
            "retreat_path is empty"
        )

    if (
        list(seq["joint_names"])
        != list(retreat["arm_joint_names"])
    ):
        raise RuntimeError(
            "joint_names mismatch"
        )

    last_action = seq["actions"][-1]

    if last_action.get("type") != "GRIPPER_EVENT":
        raise RuntimeError(
            "last action must be GRIPPER_EVENT"
        )

    release_opening = float(
        retreat["release_opening_m"]
    )

    full_open = float(
        retreat["full_open_m"]
    )

    event_to = float(
        last_action["to_m"]
    )

    if abs(event_to - release_opening) > 1.0e-12:
        raise RuntimeError(
            "release opening mismatch: "
            "sequence={} retreat={}".format(
                event_to,
                release_opening,
            )
        )

    retreat_points = retreat[
        "retreat_path"
    ]

    event_joints = [
        float(q)
        for q in last_action[
            "arm_joints_rad"
        ]
    ]

    first_joints = [
        float(q)
        for q in retreat_points[
            0
        ]["joints_rad"]
    ]

    connection_diff = max_joint_diff(
        event_joints,
        first_joints,
    )

    tolerance = float(
        seq.get(
            "settings",
            {}
        ).get(
            "duplicate_joint_tolerance_rad",
            1.0e-12,
        )
    )

    if connection_diff > tolerance:
        raise RuntimeError(
            "sequence/retreat discontinuity: "
            "{:.16e} rad".format(
                connection_diff
            )
        )

    result = copy.deepcopy(seq)

    arm_action = {
        "action_index": None,
        "type": "ARM_MOTION",
        "gripper_m": release_opening,
        "source_sequence_start": None,
        "source_sequence_end": None,
        "input_point_count":
            len(retreat_points),
        "output_point_count":
            len(retreat_points),
        "removed_duplicate_count": 0,
        "removed_duplicates": [],
        "points": [],
        "post_release_metadata": {
            "source_file":
                retreat_path,
            "search_pass":
                retreat["selected"][
                    "search_pass"
                ],
            "world_angle_deg":
                retreat["selected"][
                    "world_angle_deg"
                ],
            "retreat_distance_m":
                retreat["selected"][
                    "retreat_distance_m"
                ],
            "strategy":
                retreat["search_settings"][
                    "strategy"
                ],
        },
    }

    for p in retreat_points:
        arm_action["points"].append({
            "joints_rad": [
                float(q)
                for q in p["joints_rad"]
            ],
            "source_sequence_index": None,
            "source_segment":
                "POST_RELEASE_RETREAT",
            "source_index":
                int(p["index"]),
        })

    result["actions"].append(
        arm_action
    )

    final_joints = [
        float(q)
        for q in retreat_points[
            -1
        ]["joints_rad"]
    ]

    full_open_event = {
        "action_index": None,
        "type": "GRIPPER_EVENT",
        "from_m":
            release_opening,
        "to_m":
            full_open,
        "arm_joints_rad":
            final_joints,
        "arm_pose_max_diff_rad":
            0.0,
        "source_transition_from":
            None,
        "source_transition_to":
            None,
        "source_segment_from":
            "POST_RELEASE_RETREAT",
        "source_segment_to":
            "FULL_OPEN",
        "post_release_metadata": {
            "source_file":
                retreat_path,
            "collision_check":
                retreat[
                    "full_open_event"
                ][
                    "collision_check"
                ],
            "collision_sample_count":
                retreat[
                    "full_open_event"
                ][
                    "sample_count"
                ],
        },
    }

    result["actions"].append(
        full_open_event
    )

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

    result[
        "post_release_extension"
    ] = {
        "retreat_file":
            retreat_path,
        "connection_max_joint_diff_rad":
            float(connection_diff),
        "retreat_distance_m":
            float(
                retreat["selected"][
                    "retreat_distance_m"
                ]
            ),
        "world_angle_deg":
            float(
                retreat["selected"][
                    "world_angle_deg"
                ]
            ),
        "release_opening_m":
            release_opening,
        "full_open_m":
            full_open,
        "strategy":
            "FEASIBLE_FIRST",
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
        "===== Generic Sequence "
        "Post-Release Append ====="
    )
    print(
        "sequence input      : {}".format(
            seq_path
        )
    )
    print(
        "retreat input       : {}".format(
            retreat_path
        )
    )
    print(
        "connection diff     : "
        "{:.16e} rad".format(
            connection_diff
        )
    )
    print(
        "retreat points      : {}".format(
            len(retreat_points)
        )
    )
    print(
        "retreat distance    : {:.3f} mm".format(
            retreat["selected"][
                "retreat_distance_m"
            ] * 1000.0
        )
    )
    print(
        "retreat world angle : {:.3f} deg".format(
            retreat["selected"][
                "world_angle_deg"
            ]
        )
    )
    print(
        "FULL OPEN event     : "
        "{:.3f} -> {:.3f} mm/side".format(
            release_opening * 1000.0,
            full_open * 1000.0,
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
