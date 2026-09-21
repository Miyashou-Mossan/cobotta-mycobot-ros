#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import rospy
import moveit_commander

from geometry_msgs.msg import PoseArray
from std_msgs.msg import String

from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import (
    GetPositionIK,
    GetPositionIKRequest,
    GetStateValidity,
)


TARGET_CANDIDATE = 522
TARGET_INDEX = 73

GRIPPER_OPEN = 0.015
PRE_STEPS = 30

BASE_FILE = Path(
    "/home/maeda/catkin_ws/src/"
    "origami_grasp_bridge/scripts/"
    "cobotta_phase1d_deepest_first_reverse_search.py"
)

RESULT_FILE = Path(
    "/home/maeda/catkin_ws/results/phase1d/"
    "candidate522_index73_open_pre_connection.json"
)


def load_base_module():
    spec = importlib.util.spec_from_file_location(
        "deepest_base",
        str(BASE_FILE),
    )

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module


def set_gripper_open(state):
    state = copy.deepcopy(state)

    names = list(state.joint_state.name)
    positions = list(state.joint_state.position)

    lookup = {
        name: i
        for i, name in enumerate(names)
    }

    if "cobotta_joint_gripper" in lookup:
        positions[
            lookup["cobotta_joint_gripper"]
        ] = GRIPPER_OPEN

    if (
        "cobotta_joint_gripper_mimic"
        in lookup
    ):
        positions[
            lookup[
                "cobotta_joint_gripper_mimic"
            ]
        ] = -GRIPPER_OPEN

    state.joint_state.position = positions

    return state


def solve_ik_open(
    base,
    compute_ik,
    target,
    seed_state,
):
    req = GetPositionIKRequest()

    req.ik_request.group_name = base.GROUP
    req.ik_request.ik_link_name = base.TIP
    req.ik_request.pose_stamped = target

    req.ik_request.robot_state = copy.deepcopy(
        seed_state
    )

    req.ik_request.avoid_collisions = False

    req.ik_request.timeout = rospy.Duration(
        base.IK_TIMEOUT
    )

    res = compute_ik(req)

    if (
        res.error_code.val
        != MoveItErrorCodes.SUCCESS
    ):
        return None

    return set_gripper_open(
        res.solution
    )


def reverse_p0_to_pre_open(
    base,
    p0_state_closed,
    pre_record,
    compute_ik,
    check_validity,
):
    # J1-J6はreverse foldで得たP0枝のまま、
    # グリッパだけOPEN 15 mmにする。
    p0_state_open = set_gripper_open(
        p0_state_closed
    )

    valid, pairs = base.collision_check(
        check_validity,
        p0_state_open,
    )

    if not valid:
        return (
            False,
            None,
            {
                "step": 0,
                "reason": "P0_OPEN_COLLISION",
                "pairs": [
                    "{}<->{}".format(a, b)
                    for a, b in pairs
                ],
            },
        )

    grasp_pose = base.dict_pose_to_pose(
        pre_record[
            "grasp_tool_pose"
        ]
    )

    pre_pose = base.reconstruct_pre_pose(
        pre_record
    )

    frame_id = str(
        pre_record[
            "grasp_tool_pose"
        ].get(
            "frame_id",
            "paper_center",
        )
    )

    seed = copy.deepcopy(
        p0_state_open
    )

    previous_joints = (
        base.extract_arm_joints(
            seed
        )
    )

    path_joints = [
        list(previous_joints)
    ]

    for step in range(
        1,
        PRE_STEPS + 1,
    ):
        alpha = (
            float(step)
            / float(PRE_STEPS)
        )

        target = base.interpolate_pose(
            grasp_pose,
            pre_pose,
            alpha,
            frame_id,
        )

        solution = solve_ik_open(
            base,
            compute_ik,
            target,
            seed,
        )

        if solution is None:
            return (
                False,
                None,
                {
                    "step": step,
                    "reason": "IK_FAILED",
                },
            )

        current_joints = (
            base.extract_arm_joints(
                solution
            )
        )

        jump = base.joint_distance_max(
            previous_joints,
            current_joints,
        )

        if jump > base.MAX_JUMP_RAD:
            return (
                False,
                None,
                {
                    "step": step,
                    "reason": "JOINT_JUMP",
                    "jump_deg":
                        math.degrees(jump),
                },
            )

        valid, pairs = base.collision_check(
            check_validity,
            solution,
        )

        if not valid:
            return (
                False,
                None,
                {
                    "step": step,
                    "reason": "COLLISION",
                    "pairs": [
                        "{}<->{}".format(a, b)
                        for a, b in pairs
                    ],
                },
            )

        path_joints.append(
            list(current_joints)
        )

        seed = copy.deepcopy(
            solution
        )

        previous_joints = (
            current_joints
        )

    return (
        True,
        seed,
        {
            "path_joints_rad":
                path_joints,
        },
    )


def main():
    rospy.init_node(
        "cobotta_candidate522_"
        "open_pre_connection"
    )

    moveit_commander.roscpp_initialize(
        sys.argv
    )

    base = load_base_module()

    print(
        "===== CANDIDATE 522 OPEN PRE CONNECTION ====="
    )

    print(
        "finish index     : {}".format(
            TARGET_INDEX
        )
    )

    print(
        "gripper fold     : CLOSED 0 mm"
    )

    print(
        "gripper PRE path : OPEN 15 mm"
    )

    robot = (
        moveit_commander.RobotCommander()
    )

    rospy.wait_for_service(
        "/compute_ik",
        timeout=30.0,
    )

    rospy.wait_for_service(
        "/check_state_validity",
        timeout=30.0,
    )

    compute_ik = rospy.ServiceProxy(
        "/compute_ik",
        GetPositionIK,
        persistent=True,
    )

    check_validity = rospy.ServiceProxy(
        "/check_state_validity",
        GetStateValidity,
        persistent=True,
    )

    fold_meta_msg = rospy.wait_for_message(
        base.FOLD_META_TOPIC,
        String,
        timeout=15.0,
    )

    pre_msg = rospy.wait_for_message(
        base.PRE_RESULT_TOPIC,
        String,
        timeout=15.0,
    )

    fold_meta = json.loads(
        fold_meta_msg.data
    )

    pre_data = json.loads(
        pre_msg.data
    )

    candidate = None

    for c in fold_meta["candidates"]:
        if (
            int(c["candidate_index"])
            == TARGET_CANDIDATE
        ):
            candidate = c
            break

    if candidate is None:
        raise RuntimeError(
            "candidate 522 not found"
        )

    pre_record = None

    for c in pre_data["candidates"]:
        if (
            int(c["candidate_index"])
            == TARGET_CANDIDATE
        ):
            pre_record = c
            break

    if pre_record is None:
        raise RuntimeError(
            "candidate 522 PRE record not found"
        )

    trajectory = rospy.wait_for_message(
        candidate[
            "trajectory_topic"
        ],
        PoseArray,
        timeout=15.0,
    )

    q_original_grasp = [
        float(v)
        for v in candidate[
            "grasp_joints_rad"
        ]
    ]

    joint_limits = (
        base.get_joint_limits()
    )

    midpoint_seed = [
        0.5 * (lo + hi)
        for lo, hi in joint_limits
    ]

    global_seeds = (
        base.make_global_seeds(
            joint_limits,
            base.GLOBAL_SEED_COUNT,
        )
    )

    seed_bank = []

    base.append_unique_seed(
        seed_bank,
        q_original_grasp,
    )

    base.append_unique_seed(
        seed_bank,
        midpoint_seed,
    )

    for q in global_seeds:
        base.append_unique_seed(
            seed_bank,
            q,
        )

    robot_state = (
        robot.get_current_state()
    )

    finish_target = base.make_target(
        trajectory,
        TARGET_INDEX,
    )

    finish_solutions = []

    start_time = (
        time.perf_counter()
    )

    # -----------------------------------
    # index73 のFINISH IK枝を再生成
    # -----------------------------------
    for seed_joints in seed_bank:
        seed_state = base.set_arm_joints(
            robot_state,
            seed_joints,
        )

        seed_state = (
            base.set_gripper_closed(
                seed_state
            )
        )

        solution = base.solve_ik(
            compute_ik,
            finish_target,
            seed_state,
        )

        if solution is None:
            continue

        valid, pairs = base.collision_check(
            check_validity,
            solution,
        )

        if not valid:
            continue

        base.add_distinct_solution(
            finish_solutions,
            solution,
        )

    print()
    print(
        "distinct FINISH branches = {}"
        .format(
            len(finish_solutions)
        )
    )

    results = []

    # -----------------------------------
    # FINISH→P0 CLOSED
    # P0→PRE OPEN
    # -----------------------------------
    for branch_index, branch in enumerate(
        finish_solutions
    ):
        print()
        print(
            "----- branch {} -----"
            .format(
                branch_index
            )
        )

        (
            fold_ok,
            p0_state,
            fold_failure,
        ) = base.reverse_fold_track(
            TARGET_INDEX,
            branch["state"],
            trajectory,
            compute_ik,
            check_validity,
        )

        if not fold_ok:
            print(
                "FINISH -> P0 CLOSED : FAIL"
            )

            print(
                "failure             : {}"
                .format(
                    fold_failure
                )
            )

            results.append({
                "branch_index":
                    branch_index,
                "fold_reverse":
                    False,
                "fold_failure":
                    fold_failure,
            })

            continue

        print(
            "FINISH -> P0 CLOSED : SUCCESS"
        )

        q_p0 = base.extract_arm_joints(
            p0_state
        )

        (
            pre_ok,
            pre_state,
            pre_info,
        ) = reverse_p0_to_pre_open(
            base,
            p0_state,
            pre_record,
            compute_ik,
            check_validity,
        )

        if not pre_ok:
            print(
                "P0 -> PRE OPEN      : FAIL"
            )

            print(
                "failure             : {}"
                .format(
                    pre_info
                )
            )

            results.append({
                "branch_index":
                    branch_index,
                "fold_reverse":
                    True,
                "p0_joints_rad":
                    q_p0,
                "open_pre_connection":
                    False,
                "pre_failure":
                    pre_info,
            })

            continue

        print(
            "P0 -> PRE OPEN      : SUCCESS"
        )

        q_pre = base.extract_arm_joints(
            pre_state
        )

        print()
        print(
            "reverse P0 joints [deg]:"
        )

        print(
            "  "
            + " ".join(
                "{:+.3f}".format(
                    math.degrees(v)
                )
                for v in q_p0
            )
        )

        print(
            "new PRE joints [deg]:"
        )

        print(
            "  "
            + " ".join(
                "{:+.3f}".format(
                    math.degrees(v)
                )
                for v in q_pre
            )
        )

        results.append({
            "branch_index":
                branch_index,

            "fold_reverse":
                True,

            "p0_joints_rad":
                q_p0,

            "open_pre_connection":
                True,

            "new_pre_joints_rad":
                q_pre,

            "open_path_joints_rad":
                pre_info[
                    "path_joints_rad"
                ],
        })

    elapsed = (
        time.perf_counter()
        - start_time
    )

    complete = [
        r
        for r in results
        if (
            r.get("fold_reverse")
            and
            r.get(
                "open_pre_connection"
            )
        )
    ]

    output = {
        "candidate_index":
            TARGET_CANDIDATE,

        "finish_index":
            TARGET_INDEX,

        "approach_angle_deg":
            float(
                candidate[
                    "approach_angle_deg"
                ]
            ),

        "normal_sign":
            str(
                candidate[
                    "normal_sign"
                ]
            ),

        "gripper_fold_m":
            0.0,

        "gripper_pre_m":
            GRIPPER_OPEN,

        "pre_steps":
            PRE_STEPS,

        "finish_branch_count":
            len(
                finish_solutions
            ),

        "complete_branch_count":
            len(
                complete
            ),

        "results":
            results,

        "elapsed_sec":
            elapsed,
    }

    RESULT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_FILE.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print(
        "========================================"
    )

    print(
        " FINAL OPEN PRE CONNECTION RESULT"
    )

    print(
        "========================================"
    )

    print(
        "FINISH branches          : {}"
        .format(
            len(
                finish_solutions
            )
        )
    )

    print(
        "complete OPEN branches   : {}"
        .format(
            len(
                complete
            )
        )
    )

    print(
        "elapsed                  : "
        "{:.3f} s"
        .format(
            elapsed
        )
    )

    if complete:
        print(
            "RESULT                   : SUCCESS"
        )

        print(
            "usable branch            : {}"
            .format(
                complete[0][
                    "branch_index"
                ]
            )
        )
    else:
        print(
            "RESULT                   : FAIL"
        )

    print()
    print(
        "saved JSON:"
    )

    print(
        "  {}".format(
            RESULT_FILE
        )
    )


if __name__ == "__main__":
    main()
