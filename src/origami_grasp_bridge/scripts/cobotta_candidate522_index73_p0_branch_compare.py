#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
from moveit_msgs.srv import (
    GetPositionIK,
    GetStateValidity,
)


TARGET_CANDIDATE = 522
TARGET_INDEX = 73

BASE_FILE = Path(
    "/home/maeda/catkin_ws/src/"
    "origami_grasp_bridge/scripts/"
    "cobotta_phase1d_deepest_first_reverse_search.py"
)

RESULT_FILE = Path(
    "/home/maeda/catkin_ws/results/phase1d/"
    "candidate522_index73_p0_branch_compare.json"
)


def load_base_module():
    spec = importlib.util.spec_from_file_location(
        "deepest_base",
        str(BASE_FILE),
    )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def main():
    rospy.init_node(
        "cobotta_candidate522_index73_p0_branch_compare"
    )

    moveit_commander.roscpp_initialize(
        sys.argv
    )

    base = load_base_module()

    print(
        "===== CANDIDATE 522 / INDEX 73 "
        "P0 BRANCH COMPARISON ====="
    )

    print(
        "This is NOT a full deepest-first rerun."
    )

    print(
        "candidate = {}".format(
            TARGET_CANDIDATE
        )
    )

    print(
        "index     = {}".format(
            TARGET_INDEX
        )
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

    # ------------------------------------------
    # fold metadata
    # ------------------------------------------
    fold_meta_msg = rospy.wait_for_message(
        base.FOLD_META_TOPIC,
        String,
        timeout=15.0,
    )

    fold_meta = json.loads(
        fold_meta_msg.data
    )

    fold_candidate = None

    for c in fold_meta["candidates"]:
        if (
            int(c["candidate_index"])
            == TARGET_CANDIDATE
        ):
            fold_candidate = c
            break

    if fold_candidate is None:
        raise RuntimeError(
            "candidate 522 not found "
            "in fold metadata"
        )

    # ------------------------------------------
    # PRE->P0 result
    # ------------------------------------------
    pre_msg = rospy.wait_for_message(
        base.PRE_RESULT_TOPIC,
        String,
        timeout=15.0,
    )

    pre_data = json.loads(
        pre_msg.data
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
            "candidate 522 not found "
            "in PRE result"
        )

    if (
        pre_record.get("result")
        != "FEASIBLE_FOUND"
    ):
        raise RuntimeError(
            "candidate 522 was not "
            "PRE->P0 feasible"
        )

    q_pre_p0 = [
        float(v)
        for v in pre_record[
            "grasp_joints_rad"
        ]
    ]

    # ------------------------------------------
    # trajectory
    # ------------------------------------------
    trajectory = rospy.wait_for_message(
        fold_candidate[
            "trajectory_topic"
        ],
        PoseArray,
        timeout=15.0,
    )

    if TARGET_INDEX >= len(
        trajectory.poses
    ):
        raise RuntimeError(
            "target index outside trajectory"
        )

    print()
    print(
        "trajectory poses = {}".format(
            len(trajectory.poses)
        )
    )

    print(
        "approach angle   = {:.1f} deg"
        .format(
            float(
                fold_candidate[
                    "approach_angle_deg"
                ]
            )
        )
    )

    print(
        "normal sign      = {}".format(
            fold_candidate[
                "normal_sign"
            ]
        )
    )

    # ------------------------------------------
    # 34 FINISH seeds
    # grasp + midpoint + 32 Halton
    # ------------------------------------------
    joint_limits = (
        base.get_joint_limits()
    )

    midpoint_seed = [
        0.5 * (lo + hi)
        for lo, hi
        in joint_limits
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
        q_pre_p0,
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

    print(
        "FINISH seed count = {}".format(
            len(seed_bank)
        )
    )

    target = base.make_target(
        trajectory,
        TARGET_INDEX,
    )

    robot_state = (
        robot.get_current_state()
    )

    finish_solutions = []

    ik_success_count = 0
    collision_free_raw_count = 0

    start_time = (
        time.perf_counter()
    )

    for seed_no, seed_joints in enumerate(
        seed_bank
    ):
        seed_state = (
            base.set_arm_joints(
                robot_state,
                seed_joints,
            )
        )

        seed_state = (
            base.set_gripper_closed(
                seed_state
            )
        )

        solution = base.solve_ik(
            compute_ik,
            target,
            seed_state,
        )

        if solution is None:
            continue

        ik_success_count += 1

        valid, pairs = (
            base.collision_check(
                check_validity,
                solution,
            )
        )

        if not valid:
            continue

        collision_free_raw_count += 1

        base.add_distinct_solution(
            finish_solutions,
            solution,
        )

    print()
    print(
        "FINISH IK success         = {}/{}"
        .format(
            ik_success_count,
            len(seed_bank),
        )
    )

    print(
        "collision-free raw        = {}"
        .format(
            collision_free_raw_count
        )
    )

    print(
        "distinct free branches    = {}"
        .format(
            len(
                finish_solutions
            )
        )
    )

    # ------------------------------------------
    # 各FINISH枝をindex0まで逆追跡
    # ------------------------------------------
    successful_branches = []

    for branch_index, branch in enumerate(
        finish_solutions
    ):
        (
            reverse_ok,
            p0_state,
            failure,
        ) = base.reverse_fold_track(
            TARGET_INDEX,
            branch["state"],
            trajectory,
            compute_ik,
            check_validity,
        )

        print()
        print(
            "----- FINISH branch {} -----"
            .format(
                branch_index
            )
        )

        if not reverse_ok:
            print(
                "reverse to P0 : FAIL"
            )

            print(
                "failure       : {}"
                .format(
                    failure
                )
            )

            continue

        print(
            "reverse to P0 : SUCCESS"
        )

        q_reverse_p0 = (
            base.extract_arm_joints(
                p0_state
            )
        )

        deltas_deg = [
            math.degrees(
                b - a
            )
            for a, b in zip(
                q_pre_p0,
                q_reverse_p0,
            )
        ]

        abs_deltas = [
            abs(v)
            for v in deltas_deg
        ]

        max_delta_deg = max(
            abs_deltas
        )

        max_joint_index = (
            abs_deltas.index(
                max_delta_deg
            )
        )

        joint_names = [
            "J1",
            "J2",
            "J3",
            "J4",
            "J5",
            "J6",
        ]

        print()
        print(
            "{:<3s} | {:>11s} | {:>11s} | {:>11s}"
            .format(
                "",
                "PRE->P0",
                "REVERSE P0",
                "delta",
            )
        )

        print(
            "----+-------------+"
            "-------------+-------------"
        )

        for name, q_a, q_b, d in zip(
            joint_names,
            q_pre_p0,
            q_reverse_p0,
            deltas_deg,
        ):
            print(
                "{:<3s} | {:+11.3f} | "
                "{:+11.3f} | {:+11.3f}"
                .format(
                    name,
                    math.degrees(q_a),
                    math.degrees(q_b),
                    d,
                )
            )

        print()
        print(
            "max abs joint difference = "
            "{:.3f} deg ({})"
            .format(
                max_delta_deg,
                joint_names[
                    max_joint_index
                ],
            )
        )

        successful_branches.append({
            "finish_branch_index":
                int(branch_index),

            "finish_joints_rad": [
                float(v)
                for v in branch[
                    "joints"
                ]
            ],

            "reverse_p0_joints_rad": [
                float(v)
                for v in q_reverse_p0
            ],

            "pregrasp_p0_joints_rad": [
                float(v)
                for v in q_pre_p0
            ],

            "delta_deg": [
                float(v)
                for v in deltas_deg
            ],

            "max_abs_delta_deg":
                float(max_delta_deg),

            "max_delta_joint":
                joint_names[
                    max_joint_index
                ],
        })

    elapsed = (
        time.perf_counter()
        - start_time
    )

    # ------------------------------------------
    # 最もPRE側P0に近いreverse branch
    # ------------------------------------------
    best_branch = None

    if successful_branches:
        best_branch = min(
            successful_branches,
            key=lambda x:
                x[
                    "max_abs_delta_deg"
                ],
        )

    result = {
        "candidate_index":
            TARGET_CANDIDATE,

        "finish_index":
            TARGET_INDEX,

        "approach_angle_deg":
            float(
                fold_candidate[
                    "approach_angle_deg"
                ]
            ),

        "normal_sign":
            str(
                fold_candidate[
                    "normal_sign"
                ]
            ),

        "finish_seed_count":
            int(
                len(seed_bank)
            ),

        "finish_ik_success_count":
            int(
                ik_success_count
            ),

        "finish_collision_free_raw_count":
            int(
                collision_free_raw_count
            ),

        "finish_distinct_free_branch_count":
            int(
                len(
                    finish_solutions
                )
            ),

        "reverse_p0_success_count":
            int(
                len(
                    successful_branches
                )
            ),

        "successful_branches":
            successful_branches,

        "best_branch":
            best_branch,

        "elapsed_sec":
            float(
                elapsed
            ),
    }

    RESULT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_FILE.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print(
        "========================================"
    )

    print(
        " FINAL COMPARISON SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        "reverse P0 successes = {}"
        .format(
            len(
                successful_branches
            )
        )
    )

    if best_branch is None:
        print(
            "No branch reached P0."
        )
    else:
        print(
            "closest branch        = {}"
            .format(
                best_branch[
                    "finish_branch_index"
                ]
            )
        )

        print(
            "max P0 joint diff     = "
            "{:.3f} deg ({})"
            .format(
                best_branch[
                    "max_abs_delta_deg"
                ],
                best_branch[
                    "max_delta_joint"
                ],
            )
        )

    print(
        "elapsed               = "
        "{:.3f} s"
        .format(
            elapsed
        )
    )

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
