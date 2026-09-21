#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import json
import math
import sys

import rospy
import moveit_commander

from sensor_msgs.msg import JointState
from std_msgs.msg import String
from moveit_msgs.msg import Constraints, RobotState
from moveit_msgs.srv import GetStateValidity


GROUP = "cobotta_arm"

INPUT_TOPIC = (
    "/origami/debug/"
    "cobotta_pregrasp_feasibility_results"
)

OUTPUT_TOPIC = (
    "/origami/debug/"
    "cobotta_safezero_to_pregrasp_results"
)

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

GRIPPER_OPEN = 0.015

PLANNING_TIME = 10.0
PLANNING_ATTEMPTS = 20

# 既存Safe-Zero経路検証と同じ条件。
# MoveIt/OMPLのcollision checkingとは別に、
# 計画後の軌道を各関節0.01 rad以下に細分化して再確認する。
MAX_INTERPOLATION_STEP = 0.01

MAX_END_ERROR = 0.01


def make_full_state(base_joint_state, cobotta_positions):
    state = RobotState()
    state.joint_state = copy.deepcopy(base_joint_state)

    names = list(state.joint_state.name)
    positions = list(state.joint_state.position)

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

    # PRE-GRASPまで紙をまだ把持していないため
    # グリッパはOPEN 15 mm。
    if "cobotta_joint_gripper" in lookup:
        positions[
            lookup["cobotta_joint_gripper"]
        ] = GRIPPER_OPEN

    if "cobotta_joint_gripper_mimic" in lookup:
        positions[
            lookup[
                "cobotta_joint_gripper_mimic"
            ]
        ] = -GRIPPER_OPEN

    state.joint_state.position = positions
    state.joint_state.header.stamp = rospy.Time.now()

    return state


def trajectory_point_to_cobotta(
    trajectory_joint_names,
    positions,
):
    q_map = dict(zip(
        trajectory_joint_names,
        positions,
    ))

    missing = [
        name
        for name in COBOTTA_JOINTS
        if name not in q_map
    ]

    if missing:
        raise RuntimeError(
            "Missing COBOTTA joints in trajectory: "
            + ", ".join(missing)
        )

    return [
        float(q_map[name])
        for name in COBOTTA_JOINTS
    ]


def check_state(
    check_service,
    base_joint_state,
    q,
):
    state = make_full_state(
        base_joint_state,
        q,
    )

    res = check_service(
        state,
        GROUP,
        Constraints(),
    )

    contacts = []

    for c in res.contacts:
        contacts.append({
            "body_1": str(c.contact_body_1),
            "body_2": str(c.contact_body_2),
            "depth": float(c.depth),
        })

    return bool(res.valid), contacts


def main():
    rospy.init_node(
        "cobotta_safezero_to_pregrasp_"
        "feasibility_diagnostic"
    )

    moveit_commander.roscpp_initialize(
        sys.argv
    )

    print(
        "===== SAFE-ZERO -> PRE-GRASP "
        "FEASIBILITY DIAGNOSTIC ====="
    )

    print("safe-zero :", SAFE_ZERO)
    print(
        "gripper   : OPEN {:.1f} mm".format(
            GRIPPER_OPEN * 1000.0
        )
    )
    print(
        "planning  : {:.1f} s x {} attempts".format(
            PLANNING_TIME,
            PLANNING_ATTEMPTS,
        )
    )
    print(
        "dense step: <= {:.4f} rad".format(
            MAX_INTERPOLATION_STEP
        )
    )

    current = rospy.wait_for_message(
        "/joint_states",
        JointState,
        timeout=5.0,
    )

    print()
    print(
        "Waiting for structured "
        "PRE-GRASP feasibility results..."
    )

    input_msg = rospy.wait_for_message(
        INPUT_TOPIC,
        String,
        timeout=10.0,
    )

    input_data = json.loads(
        input_msg.data
    )

    candidates = []

    for p0 in input_data["p0_results"]:
        for candidate in p0["candidates"]:

            if (
                candidate.get("result")
                != "FEASIBLE_FOUND"
            ):
                continue

            joints = candidate.get(
                "pregrasp_joints_rad"
            )

            if (
                joints is None
                or len(joints) != 6
            ):
                raise RuntimeError(
                    "FEASIBLE candidate does not "
                    "contain 6 PRE-GRASP joints."
                )

            candidates.append({
                "p0_index":
                    int(p0["p0_index"]),
                "candidate_index":
                    int(
                        candidate[
                            "candidate_index"
                        ]
                    ),
                "edge":
                    int(candidate["edge"]),
                "normal_sign":
                    str(
                        candidate[
                            "normal_sign"
                        ]
                    ),
                "pregrasp_joints_rad": [
                    float(v)
                    for v in joints
                ],
            })

    print()
    print(
        "input FEASIBLE candidates : {}".format(
            len(candidates)
        )
    )

    rospy.wait_for_service(
        "/check_state_validity",
        timeout=10.0,
    )

    check = rospy.ServiceProxy(
        "/check_state_validity",
        GetStateValidity,
    )

    group = moveit_commander.MoveGroupCommander(
        GROUP
    )

    group.set_planning_time(
        PLANNING_TIME
    )

    group.set_num_planning_attempts(
        PLANNING_ATTEMPTS
    )

    safezero_valid, safezero_contacts = (
        check_state(
            check,
            current,
            SAFE_ZERO,
        )
    )

    print(
        "Safe-Zero state valid : {}".format(
            safezero_valid
        )
    )

    if not safezero_valid:
        print(
            "Safe-Zero contacts : {}".format(
                safezero_contacts
            )
        )

    records = []

    for number, candidate in enumerate(
        candidates,
        start=1,
    ):
        p0_index = candidate["p0_index"]
        candidate_index = (
            candidate["candidate_index"]
        )
        edge = candidate["edge"]
        normal_sign = (
            candidate["normal_sign"]
        )
        target = (
            candidate[
                "pregrasp_joints_rad"
            ]
        )

        print()
        print(
            "========================================"
        )
        print(
            "[{}/{}] P0[{}] candidate {} "
            "edge{} {}".format(
                number,
                len(candidates),
                p0_index,
                candidate_index,
                edge,
                normal_sign,
            )
        )
        print(
            "========================================"
        )

        record = copy.deepcopy(
            candidate
        )

        if not safezero_valid:
            print(
                "RESULT: START_STATE_INVALID"
            )

            record.update({
                "result":
                    "START_STATE_INVALID",
                "plan_points": 0,
                "duration_sec": None,
                "dense_state_count": 0,
                "max_end_error_rad": None,
                "contacts":
                    safezero_contacts,
            })

            records.append(record)
            continue

        target_valid, target_contacts = (
            check_state(
                check,
                current,
                target,
            )
        )

        if not target_valid:
            print(
                "RESULT: TARGET_STATE_INVALID"
            )

            print(
                "contacts : {}".format(
                    target_contacts
                )
            )

            record.update({
                "result":
                    "TARGET_STATE_INVALID",
                "plan_points": 0,
                "duration_sec": None,
                "dense_state_count": 0,
                "max_end_error_rad": None,
                "contacts":
                    target_contacts,
            })

            records.append(record)
            continue

        start_state = make_full_state(
            current,
            SAFE_ZERO,
        )

        group.stop()
        group.clear_pose_targets()

        group.set_start_state(
            start_state
        )

        # PRE-GRASP PoseからIKを解き直さない。
        # PRE→P0で安全確認した同じ関節姿勢を
        # そのままMoveItの目標にする。
        target_map = {
            name: value
            for name, value in zip(
                COBOTTA_JOINTS,
                target,
            )
        }

        group.set_joint_value_target(
            target_map
        )

        result = group.plan()

        if isinstance(result, tuple):
            success = bool(result[0])
            plan = result[1]
        else:
            plan = result
            success = (
                len(
                    plan.joint_trajectory.points
                ) > 0
            )

        points = (
            plan.joint_trajectory.points
        )

        trajectory_joint_names = list(
            plan.joint_trajectory.joint_names
        )

        print(
            "plan success : {}".format(
                success
            )
        )
        print(
            "plan points  : {}".format(
                len(points)
            )
        )

        if not success or not points:
            print("RESULT: PLAN_FAILED")

            record.update({
                "result":
                    "PLAN_FAILED",
                "plan_points":
                    int(len(points)),
                "duration_sec": None,
                "dense_state_count": 0,
                "max_end_error_rad": None,
                "contacts": [],
            })

            records.append(record)
            continue

        final_q = (
            trajectory_point_to_cobotta(
                trajectory_joint_names,
                points[-1].positions,
            )
        )

        max_end_error = max(
            abs(a - b)
            for a, b in zip(
                final_q,
                target,
            )
        )

        duration_sec = (
            points[-1]
            .time_from_start
            .to_sec()
        )

        print(
            "duration     : {:.6f} s".format(
                duration_sec
            )
        )
        print(
            "max end error: {:.9f} rad".format(
                max_end_error
            )
        )

        if max_end_error >= MAX_END_ERROR:
            print("RESULT: END_ERROR")

            record.update({
                "result":
                    "END_ERROR",
                "plan_points":
                    int(len(points)),
                "duration_sec":
                    float(duration_sec),
                "dense_state_count": 0,
                "max_end_error_rad":
                    float(max_end_error),
                "contacts": [],
            })

            records.append(record)
            continue

        # ----------------------------------------
        # Dense Collision Validation
        # ----------------------------------------
        previous = list(SAFE_ZERO)

        dense_state_count = 1
        invalid_found = False
        invalid_dense_index = None
        invalid_contacts = []

        for point in points:

            current_q = (
                trajectory_point_to_cobotta(
                    trajectory_joint_names,
                    point.positions,
                )
            )

            max_delta = max(
                abs(b - a)
                for a, b in zip(
                    previous,
                    current_q,
                )
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
                    )
                )

                if not valid:
                    invalid_found = True
                    invalid_dense_index = (
                        dense_state_count
                    )
                    invalid_contacts = (
                        contacts
                    )
                    break

                dense_state_count += 1

            if invalid_found:
                break

            previous = current_q

        print(
            "dense states : {}".format(
                dense_state_count
            )
        )

        if invalid_found:
            print(
                "RESULT: DENSE_COLLISION"
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

            record.update({
                "result":
                    "DENSE_COLLISION",
                "plan_points":
                    int(len(points)),
                "duration_sec":
                    float(duration_sec),
                "dense_state_count":
                    int(dense_state_count),
                "invalid_dense_index":
                    int(
                        invalid_dense_index
                    ),
                "max_end_error_rad":
                    float(max_end_error),
                "contacts":
                    invalid_contacts,
            })

            records.append(record)
            continue

        print(
            "RESULT: "
            "SAFEZERO_TO_PREGRASP_FEASIBLE"
        )

        record.update({
            "result":
                "SAFEZERO_TO_PREGRASP_FEASIBLE",
            "plan_points":
                int(len(points)),
            "duration_sec":
                float(duration_sec),
            "dense_state_count":
                int(dense_state_count),
            "max_end_error_rad":
                float(max_end_error),
            "contacts": [],
        })

        records.append(record)

    feasible = [
        r
        for r in records
        if r["result"]
        == "SAFEZERO_TO_PREGRASP_FEASIBLE"
    ]

    print()
    print("===== Summary =====")
    print(
        "input candidates : {}".format(
            len(candidates)
        )
    )
    print(
        "Safe-Zero -> PRE-GRASP feasible : "
        "{}/{}".format(
            len(feasible),
            len(candidates),
        )
    )

    for r in records:
        print(
            "P0[{}] edge{} {} : {}".format(
                r["p0_index"],
                r["edge"],
                r["normal_sign"],
                r["result"],
            )
        )

    output = {
        "schema_version": 1,
        "safe_zero_rad":
            list(SAFE_ZERO),
        "gripper_open_m":
            float(GRIPPER_OPEN),
        "planning_time_sec":
            float(PLANNING_TIME),
        "planning_attempts":
            int(PLANNING_ATTEMPTS),
        "max_interpolation_step_rad":
            float(
                MAX_INTERPOLATION_STEP
            ),
        "input_candidate_count":
            int(len(candidates)),
        "feasible_count":
            int(len(feasible)),
        "candidates":
            records,
    }

    pub = rospy.Publisher(
        OUTPUT_TOPIC,
        String,
        queue_size=1,
        latch=True,
    )

    rospy.sleep(0.5)

    msg = String()
    msg.data = json.dumps(
        output,
        ensure_ascii=False,
        sort_keys=True,
    )

    pub.publish(msg)

    print()
    print(
        "Published structured result:"
    )
    print("  " + OUTPUT_TOPIC)

    rospy.spin()


if __name__ == "__main__":
    main()
