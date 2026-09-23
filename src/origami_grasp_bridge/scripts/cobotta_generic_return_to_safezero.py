#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import json
import math
import sys
from pathlib import Path

import rospy
import moveit_commander

from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import (
    GetStateValidity,
    GetStateValidityRequest,
)


GROUP = "cobotta_arm"

COBOTTA_JOINTS = [
    "cobotta_joint_1",
    "cobotta_joint_2",
    "cobotta_joint_3",
    "cobotta_joint_4",
    "cobotta_joint_5",
    "cobotta_joint_6",
]

SAFE_ZERO = [
    0.0,
    0.0,
    0.35,
    0.0,
    0.0,
    0.0,
]

FULL_OPEN = 0.015

PLANNING_TIME = 10.0
PLANNING_ATTEMPTS = 20

MAX_INTERPOLATION_STEP = 0.01
MAX_END_ERROR = 0.01


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


def make_full_state(
    base_joint_state,
    cobotta_positions,
    gripper_opening,
):
    state = RobotState()
    state.joint_state = copy.deepcopy(
        base_joint_state
    )

    names = list(
        state.joint_state.name
    )

    positions = list(
        state.joint_state.position
    )

    lookup = {
        name: i
        for i, name in enumerate(names)
    }

    for name, value in zip(
        COBOTTA_JOINTS,
        cobotta_positions,
    ):
        if name not in lookup:
            raise RuntimeError(
                "Joint not found in /joint_states: "
                + name
            )

        positions[lookup[name]] = float(value)

    if "cobotta_joint_gripper" in lookup:
        positions[
            lookup["cobotta_joint_gripper"]
        ] = float(gripper_opening)

    if (
        "cobotta_joint_gripper_mimic"
        in lookup
    ):
        positions[
            lookup[
                "cobotta_joint_gripper_mimic"
            ]
        ] = -float(gripper_opening)

    state.joint_state.position = positions

    return state


def trajectory_point_to_cobotta(
    joint_names,
    positions,
):
    lookup = {
        name: float(value)
        for name, value in zip(
            joint_names,
            positions,
        )
    }

    result = []

    for name in COBOTTA_JOINTS:
        if name not in lookup:
            raise RuntimeError(
                "Trajectory joint missing: "
                + name
            )

        result.append(
            lookup[name]
        )

    return result


def contact_to_dict(contact):
    body_1 = getattr(
        contact,
        "contact_body_1",
        getattr(
            contact,
            "body_name_1",
            "?",
        ),
    )

    body_2 = getattr(
        contact,
        "contact_body_2",
        getattr(
            contact,
            "body_name_2",
            "?",
        ),
    )

    return {
        "body_1": str(body_1),
        "body_2": str(body_2),
        "depth": float(
            getattr(
                contact,
                "depth",
                0.0,
            )
        ),
    }


def check_state(
    proxy,
    base_joint_state,
    joints,
    gripper_opening,
):
    req = GetStateValidityRequest()

    req.robot_state = make_full_state(
        base_joint_state,
        joints,
        gripper_opening,
    )

    req.group_name = GROUP

    result = proxy(req)

    contacts = [
        contact_to_dict(c)
        for c in result.contacts
    ]

    return bool(result.valid), contacts


def main():
    rospy.init_node(
        "cobotta_generic_return_to_safezero"
    )

    moveit_commander.roscpp_initialize(
        sys.argv
    )

    sequence_json = rospy.get_param(
        "~sequence_json"
    )

    output_json = rospy.get_param(
        "~output_json"
    )

    (
        resolved_sequence,
        sequence,
    ) = load_json(
        sequence_json
    )

    if (
        list(sequence["joint_names"])
        != COBOTTA_JOINTS
    ):
        raise RuntimeError(
            "sequence joint_names mismatch"
        )

    if not sequence.get("actions"):
        raise RuntimeError(
            "sequence actions are empty"
        )

    last_action = sequence["actions"][-1]

    if (
        last_action.get("type")
        != "GRIPPER_EVENT"
    ):
        raise RuntimeError(
            "sequence must end with "
            "GRIPPER_EVENT"
        )

    start_joints = [
        float(q)
        for q in last_action[
            "arm_joints_rad"
        ]
    ]

    start_gripper = float(
        last_action["to_m"]
    )

    if abs(
        start_gripper - FULL_OPEN
    ) > 1.0e-12:
        raise RuntimeError(
            "sequence does not end "
            "FULL OPEN: {:.9f} m".format(
                start_gripper
            )
        )

    print()
    print(
        "===== Generic Return-to-SafeZero ====="
    )
    print(
        "sequence input      : {}".format(
            resolved_sequence
        )
    )
    print(
        "group               : {}".format(
            GROUP
        )
    )
    print(
        "gripper             : "
        "{:.3f} mm/side".format(
            start_gripper * 1000.0
        )
    )
    print(
        "planning time       : {:.1f} s".format(
            PLANNING_TIME
        )
    )
    print(
        "planning attempts   : {}".format(
            PLANNING_ATTEMPTS
        )
    )
    print(
        "dense max step      : {:.6f} rad".format(
            MAX_INTERPOLATION_STEP
        )
    )

    current = rospy.wait_for_message(
        "/joint_states",
        JointState,
        timeout=15.0,
    )

    rospy.wait_for_service(
        "/check_state_validity",
        timeout=30.0,
    )

    check = rospy.ServiceProxy(
        "/check_state_validity",
        GetStateValidity,
        persistent=True,
    )

    # ------------------------------------
    # Start state validation
    # ------------------------------------
    start_valid, start_contacts = (
        check_state(
            check,
            current,
            start_joints,
            start_gripper,
        )
    )

    print()
    print(
        "start state valid   : {}".format(
            start_valid
        )
    )

    if not start_valid:
        for c in start_contacts:
            print(
                "  CONTACT: {} <-> {} "
                "depth={}".format(
                    c["body_1"],
                    c["body_2"],
                    c["depth"],
                )
            )

        raise RuntimeError(
            "FULL OPEN start state "
            "is in collision"
        )

    # ------------------------------------
    # MoveIt planning
    # FULL OPEN position -> SafeZero
    # ------------------------------------
    group = (
        moveit_commander.MoveGroupCommander(
            GROUP
        )
    )

    group.set_planning_time(
        PLANNING_TIME
    )

    group.set_num_planning_attempts(
        PLANNING_ATTEMPTS
    )

    start_state = make_full_state(
        current,
        start_joints,
        start_gripper,
    )

    group.stop()
    group.clear_pose_targets()

    group.set_start_state(
        start_state
    )

    target_map = {
        name: value
        for name, value in zip(
            COBOTTA_JOINTS,
            SAFE_ZERO,
        )
    }

    group.set_joint_value_target(
        target_map
    )

    result = group.plan()

    if isinstance(result, tuple):
        plan_success = bool(
            result[0]
        )
        plan = result[1]
    else:
        plan = result
        plan_success = (
            len(
                plan.joint_trajectory.points
            ) > 0
        )

    if not plan_success:
        raise RuntimeError(
            "FULL OPEN -> SafeZero "
            "planning failed"
        )

    points = (
        plan.joint_trajectory.points
    )

    trajectory_joint_names = list(
        plan.joint_trajectory.joint_names
    )

    if not points:
        raise RuntimeError(
            "planned trajectory is empty"
        )

    final_joints = (
        trajectory_point_to_cobotta(
            trajectory_joint_names,
            points[-1].positions,
        )
    )

    end_error = max_joint_diff(
        final_joints,
        SAFE_ZERO,
    )

    print()
    print(
        "MoveIt plan points  : {}".format(
            len(points)
        )
    )
    print(
        "SafeZero end error  : "
        "{:.12e} rad".format(
            end_error
        )
    )

    if end_error >= MAX_END_ERROR:
        raise RuntimeError(
            "SafeZero end error too large: "
            "{} rad".format(
                end_error
            )
        )

    # ------------------------------------
    # Dense Collision Validation
    #
    # SafeZero -> PRE と同じく、
    # 各関節差が0.01 rad以下になるよう
    # 補間した全状態を再確認する。
    # ------------------------------------
    previous = list(
        start_joints
    )

    dense_state_count = 1
    invalid_found = False
    invalid_dense_index = None
    invalid_contacts = []

    path = [{
        "index": 0,
        "joints_rad": list(
            start_joints
        ),
        "source":
            "POST_RELEASE_FULL_OPEN",
        "time_from_start_sec": 0.0,
    }]

    duplicate_tol = float(
        sequence.get(
            "settings",
            {}
        ).get(
            "duplicate_joint_tolerance_rad",
            1.0e-12,
        )
    )

    for point in points:
        current_q = (
            trajectory_point_to_cobotta(
                trajectory_joint_names,
                point.positions,
            )
        )

        max_delta = max_joint_diff(
            previous,
            current_q,
        )

        subdivisions = max(
            1,
            int(
                math.ceil(
                    max_delta
                    / MAX_INTERPOLATION_STEP
                )
            ),
        )

        for k in range(
            1,
            subdivisions + 1,
        ):
            t = (
                float(k)
                / float(subdivisions)
            )

            q = [
                a + t * (b - a)
                for a, b in zip(
                    previous,
                    current_q,
                )
            ]

            valid, contacts = (
                check_state(
                    check,
                    current,
                    q,
                    start_gripper,
                )
            )

            if not valid:
                invalid_found = True
                invalid_dense_index = (
                    dense_state_count
                )
                invalid_contacts = contacts
                break

            dense_state_count += 1

        if invalid_found:
            break

        if (
            max_joint_diff(
                path[-1]["joints_rad"],
                current_q,
            )
            > duplicate_tol
        ):
            path.append({
                "index": len(path),
                "joints_rad":
                    list(current_q),
                "source":
                    "MOVEIT_PLAN",
                "time_from_start_sec":
                    float(
                        point
                        .time_from_start
                        .to_sec()
                    ),
            })

        previous = current_q

    print(
        "dense states        : {}".format(
            dense_state_count
        )
    )

    if invalid_found:
        print(
            "RESULT              : "
            "DENSE_COLLISION"
        )
        print(
            "invalid dense index : {}".format(
                invalid_dense_index
            )
        )

        for c in invalid_contacts:
            print(
                "  CONTACT: {} <-> {} "
                "depth={}".format(
                    c["body_1"],
                    c["body_2"],
                    c["depth"],
                )
            )

        raise RuntimeError(
            "dense collision validation failed"
        )

    # ------------------------------------
    # Final path statistics
    # ------------------------------------
    max_step_rad = 0.0
    max_step_from = None
    max_step_to = None
    max_step_joint = None

    for i in range(
        1,
        len(path),
    ):
        a = path[i - 1][
            "joints_rad"
        ]
        b = path[i][
            "joints_rad"
        ]

        for j, (
            qa,
            qb,
        ) in enumerate(
            zip(a, b)
        ):
            d = abs(qb - qa)

            if d > max_step_rad:
                max_step_rad = d
                max_step_from = i - 1
                max_step_to = i
                max_step_joint = (
                    COBOTTA_JOINTS[j]
                )

    output = {
        "schema":
            "cobotta_generic_return_to_safezero_v1",

        "sequence_source_json":
            resolved_sequence,

        "arm_joint_names":
            COBOTTA_JOINTS,

        "gripper_opening_m":
            float(start_gripper),

        "start_joints_rad":
            list(start_joints),

        "safezero_joints_rad":
            list(SAFE_ZERO),

        "planning_settings": {
            "group":
                GROUP,
            "planning_time_sec":
                PLANNING_TIME,
            "planning_attempts":
                PLANNING_ATTEMPTS,
            "max_interpolation_step_rad":
                MAX_INTERPOLATION_STEP,
            "max_end_error_rad":
                MAX_END_ERROR,
        },

        "validation": {
            "start_state_valid":
                True,
            "safezero_end_error_rad":
                float(end_error),
            "dense_state_count":
                int(dense_state_count),
            "dense_collision_valid":
                True,
            "max_joint_step_rad":
                float(max_step_rad),
            "max_joint_step_deg":
                float(
                    math.degrees(
                        max_step_rad
                    )
                ),
            "max_joint_step_from":
                max_step_from,
            "max_joint_step_to":
                max_step_to,
            "max_joint_step_joint":
                max_step_joint,
        },

        "return_path":
            path,
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
            output,
            f,
            indent=2,
        )

    print()
    print(
        "===== RETURN RESULT ====="
    )
    print(
        "RESULT              : VALID"
    )
    print(
        "return path points  : {}".format(
            len(path)
        )
    )
    print(
        "dense states        : {}".format(
            dense_state_count
        )
    )
    print(
        "end error           : "
        "{:.12e} rad".format(
            end_error
        )
    )
    print(
        "max joint step      : "
        "{:.6f} deg".format(
            math.degrees(
                max_step_rad
            )
        )
    )

    if max_step_joint is not None:
        print(
            "location            : "
            "{} -> {} ({})".format(
                max_step_from,
                max_step_to,
                max_step_joint,
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
