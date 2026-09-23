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
    return max(
        abs(float(x) - float(y))
        for x, y in zip(a, b)
    )


def main():
    rospy.init_node(
        "cobotta_generic_sequence_append_return_safezero"
    )

    sequence_json = rospy.get_param(
        "~sequence_json"
    )

    return_json = rospy.get_param(
        "~return_json"
    )

    output_json = rospy.get_param(
        "~output_json"
    )

    seq_path, seq = load_json(
        sequence_json
    )

    return_path_name, ret = load_json(
        return_json
    )

    if not seq.get("actions"):
        raise RuntimeError(
            "sequence actions are empty"
        )

    if not ret.get("return_path"):
        raise RuntimeError(
            "return_path is empty"
        )

    if (
        list(seq["joint_names"])
        != list(ret["arm_joint_names"])
    ):
        raise RuntimeError(
            "joint_names mismatch"
        )

    last_action = seq["actions"][-1]

    if last_action.get("type") != "GRIPPER_EVENT":
        raise RuntimeError(
            "sequence must end with GRIPPER_EVENT"
        )

    gripper_opening = float(
        ret["gripper_opening_m"]
    )

    if abs(
        float(last_action["to_m"])
        - gripper_opening
    ) > 1.0e-12:
        raise RuntimeError(
            "gripper opening mismatch"
        )

    path = ret["return_path"]

    sequence_end_joints = [
        float(q)
        for q in last_action[
            "arm_joints_rad"
        ]
    ]

    return_start_joints = [
        float(q)
        for q in path[0][
            "joints_rad"
        ]
    ]

    connection_diff = max_joint_diff(
        sequence_end_joints,
        return_start_joints,
    )

    if connection_diff > 1.0e-12:
        raise RuntimeError(
            "sequence/return discontinuity: "
            "{:.16e} rad".format(
                connection_diff
            )
        )

    result = copy.deepcopy(seq)

    arm_action = {
        "action_index": None,
        "type": "ARM_MOTION",
        "gripper_m": gripper_opening,
        "source_sequence_start": None,
        "source_sequence_end": None,
        "input_point_count": len(path),
        "output_point_count": len(path),
        "removed_duplicate_count": 0,
        "removed_duplicates": [],
        "points": [],
        "return_safezero_metadata": {
            "source_file":
                return_path_name,
            "dense_state_count":
                ret["validation"][
                    "dense_state_count"
                ],
            "safezero_end_error_rad":
                ret["validation"][
                    "safezero_end_error_rad"
                ],
            "max_joint_step_deg":
                ret["validation"][
                    "max_joint_step_deg"
                ],
        },
    }

    for p in path:
        arm_action["points"].append({
            "joints_rad": [
                float(q)
                for q in p[
                    "joints_rad"
                ]
            ],
            "source_sequence_index": None,
            "source_segment":
                "RETURN_TO_SAFEZERO",
            "source_index":
                int(p["index"]),
        })

    result["actions"].append(
        arm_action
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

    result["return_safezero_extension"] = {
        "source_file":
            return_path_name,
        "connection_max_joint_diff_rad":
            float(connection_diff),
        "gripper_opening_m":
            gripper_opening,
        "safezero_end_error_rad":
            float(
                ret["validation"][
                    "safezero_end_error_rad"
                ]
            ),
        "dense_state_count":
            int(
                ret["validation"][
                    "dense_state_count"
                ]
            ),
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
        "Append Return-to-SafeZero ====="
    )
    print(
        "sequence input     : {}".format(
            seq_path
        )
    )
    print(
        "return input       : {}".format(
            return_path_name
        )
    )
    print(
        "connection diff    : "
        "{:.16e} rad".format(
            connection_diff
        )
    )
    print(
        "return points      : {}".format(
            len(path)
        )
    )
    print(
        "gripper            : "
        "{:.3f} mm/side".format(
            gripper_opening * 1000.0
        )
    )
    print(
        "SafeZero end error : "
        "{:.12e} rad".format(
            ret["validation"][
                "safezero_end_error_rad"
            ]
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

    print()
    print(
        "saved : {}".format(
            output_path
        )
    )


if __name__ == "__main__":
    main()
