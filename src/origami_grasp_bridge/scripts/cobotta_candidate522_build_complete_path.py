#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import importlib.util
import json
import math
import sys
from pathlib import Path

import rospy
import moveit_commander

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseArray
from std_msgs.msg import String

from moveit_msgs.msg import CollisionObject
from moveit_msgs.srv import (
    GetPositionIK,
    GetStateValidity,
)


TARGET_CANDIDATE = 522
TARGET_INDEX = 73

DEEP_FILE = Path(
    "/home/maeda/catkin_ws/src/"
    "origami_grasp_bridge/scripts/"
    "cobotta_phase1d_deepest_first_reverse_search.py"
)

SAFE_FILE = Path(
    "/home/maeda/catkin_ws/src/"
    "origami_grasp_bridge/scripts/"
    "cobotta_safezero_to_pregrasp_feasibility_diagnostic.py"
)

OPEN_RESULT_FILE = Path(
    "/home/maeda/catkin_ws/results/phase1d/"
    "candidate522_index73_open_pre_connection.json"
)

OUTPUT_FILE = Path(
    "/home/maeda/catkin_ws/results/phase1d/"
    "candidate522_complete_path_to_index73.json"
)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(
        name,
        str(path),
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def max_joint_diff(q1, q2):
    return max(
        abs(a - b)
        for a, b in zip(q1, q2)
    )


def collect_reverse_fold_path(
    deep,
    start_index,
    finish_state,
    trajectory,
    compute_ik,
    check_validity,
):
    seed = copy.deepcopy(
        finish_state
    )

    previous_joints = (
        deep.extract_arm_joints(seed)
    )

    # [index73, index72, ..., index0]
    path = [{
        "index": int(start_index),
        "joints_rad":
            list(previous_joints),
    }]

    for index in range(
        start_index - 1,
        -1,
        -1,
    ):
        target = deep.make_target(
            trajectory,
            index,
        )

        solution = deep.solve_ik(
            compute_ik,
            target,
            seed,
        )

        if solution is None:
            return False, None, {
                "index": index,
                "reason": "IK_FAILED",
            }

        current_joints = (
            deep.extract_arm_joints(
                solution
            )
        )

        jump = deep.joint_distance_max(
            previous_joints,
            current_joints,
        )

        if jump > deep.MAX_JUMP_RAD:
            return False, None, {
                "index": index,
                "reason": "JOINT_JUMP",
                "jump_deg":
                    math.degrees(jump),
            }

        valid, pairs = (
            deep.collision_check(
                check_validity,
                solution,
            )
        )

        if not valid:
            return False, None, {
                "index": index,
                "reason": "COLLISION",
                "pairs": [
                    "{}<->{}".format(a, b)
                    for a, b in pairs
                ],
            }

        path.append({
            "index": int(index),
            "joints_rad":
                list(current_joints),
        })

        seed = copy.deepcopy(
            solution
        )

        previous_joints = (
            current_joints
        )

    return True, path, None


def main():
    rospy.init_node(
        "cobotta_candidate522_"
        "build_complete_path"
    )

    moveit_commander.roscpp_initialize(
        sys.argv
    )

    deep = load_module(
        "deepest_base",
        DEEP_FILE,
    )

    safe = load_module(
        "safezero_base",
        SAFE_FILE,
    )

    print(
        "===== BUILD COMPLETE PATH ====="
    )

    print(
        "candidate : {}".format(
            TARGET_CANDIDATE
        )
    )

    print(
        "finish    : index {}".format(
            TARGET_INDEX
        )
    )

    # ------------------------------------
    # 前回保存したOPEN PRE経路
    # ------------------------------------
    if not OPEN_RESULT_FILE.exists():
        raise RuntimeError(
            "Missing: "
            + str(OPEN_RESULT_FILE)
        )

    open_data = json.loads(
        OPEN_RESULT_FILE.read_text()
    )

    successful = [
        r
        for r in open_data["results"]
        if (
            r.get("fold_reverse")
            and
            r.get("open_pre_connection")
        )
    ]

    if not successful:
        raise RuntimeError(
            "No successful OPEN PRE branch."
        )

    selected_open = successful[0]

    q_saved_p0 = [
        float(v)
        for v in selected_open[
            "p0_joints_rad"
        ]
    ]

    q_new_pre = [
        float(v)
        for v in selected_open[
            "new_pre_joints_rad"
        ]
    ]

    # 保存済みpathは P0 -> PRE
    p0_to_pre = [
        [float(v) for v in q]
        for q in selected_open[
            "open_path_joints_rad"
        ]
    ]

    # 前進実行では PRE -> P0
    pre_to_p0 = list(
        reversed(p0_to_pre)
    )

    print(
        "PRE -> P0 points : {}"
        .format(
            len(pre_to_p0)
        )
    )

    # ------------------------------------
    # ROS / MoveIt準備
    # ------------------------------------
    current = rospy.wait_for_message(
        "/joint_states",
        JointState,
        timeout=5.0,
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

    group = (
        moveit_commander.MoveGroupCommander(
            safe.GROUP
        )
    )

    group.set_planning_time(
        safe.PLANNING_TIME
    )

    group.set_num_planning_attempts(
        safe.PLANNING_ATTEMPTS
    )

    # ====================================
    # 1. SafeZero -> new PRE
    # ====================================
    start_state = safe.make_full_state(
        current,
        safe.SAFE_ZERO,
    )

    group.stop()
    group.clear_pose_targets()

    group.set_start_state(
        start_state
    )

    target_map = {
        name: value
        for name, value in zip(
            safe.COBOTTA_JOINTS,
            q_new_pre,
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
            "SafeZero -> new PRE "
            "planning failed."
        )

    plan_points = (
        plan.joint_trajectory.points
    )

    plan_joint_names = list(
        plan.joint_trajectory.joint_names
    )

    safezero_to_pre = []

    # SafeZeroを明示的に先頭へ
    safezero_to_pre.append({
        "joints_rad":
            list(safe.SAFE_ZERO),
        "time_from_start_sec":
            0.0,
    })

    for p in plan_points:
        q = (
            safe.trajectory_point_to_cobotta(
                plan_joint_names,
                p.positions,
            )
        )

        safezero_to_pre.append({
            "joints_rad":
                list(q),
            "time_from_start_sec":
                float(
                    p.time_from_start.to_sec()
                ),
        })

    final_pre = (
        safezero_to_pre[-1][
            "joints_rad"
        ]
    )

    pre_error = max_joint_diff(
        final_pre,
        q_new_pre,
    )

    if pre_error >= safe.MAX_END_ERROR:
        raise RuntimeError(
            "SafeZero plan PRE error too large: "
            "{} rad".format(pre_error)
        )

    print(
        "SafeZero -> PRE points : {}"
        .format(
            len(safezero_to_pre)
        )
    )

    print(
        "PRE end error          : "
        "{:.9f} rad"
        .format(
            pre_error
        )
    )

    # ====================================
    # SafeZero -> PRE ではT0紙Collisionを使用した。
    #
    # PRE以降は紙へ意図的に接近・把持するため、
    # 静止T0紙CollisionはここでPlanning Sceneから外す。
    # ====================================
    paper_collision_pub = rospy.Publisher(
        "/collision_object",
        CollisionObject,
        queue_size=1,
        latch=True,
    )

    rospy.sleep(0.5)

    remove_paper = CollisionObject()
    remove_paper.header.frame_id = "paper_center"
    remove_paper.header.stamp = rospy.Time.now()
    remove_paper.id = "cobotta_t0_full_paper_collision"
    remove_paper.operation = CollisionObject.REMOVE

    paper_collision_pub.publish(
        remove_paper
    )

    rospy.sleep(1.0)

    print()
    print(
        "T0 full-paper collision : REMOVED "
        "after SafeZero -> PRE"
    )

    # ====================================
    # 2. fold metadata / trajectory
    # ====================================
    fold_meta_msg = (
        rospy.wait_for_message(
            deep.FOLD_META_TOPIC,
            String,
            timeout=15.0,
        )
    )

    fold_meta = json.loads(
        fold_meta_msg.data
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
            "candidate 522 not found."
        )

    trajectory = (
        rospy.wait_for_message(
            candidate[
                "trajectory_topic"
            ],
            PoseArray,
            timeout=15.0,
        )
    )

    # ====================================
    # 3. index73で34-seed FINISH IK
    # ====================================
    joint_limits = (
        deep.get_joint_limits()
    )

    midpoint_seed = [
        0.5 * (lo + hi)
        for lo, hi in joint_limits
    ]

    global_seeds = (
        deep.make_global_seeds(
            joint_limits,
            deep.GLOBAL_SEED_COUNT,
        )
    )

    seed_bank = []

    deep.append_unique_seed(
        seed_bank,
        candidate[
            "grasp_joints_rad"
        ],
    )

    deep.append_unique_seed(
        seed_bank,
        midpoint_seed,
    )

    for q in global_seeds:
        deep.append_unique_seed(
            seed_bank,
            q,
        )

    robot = (
        moveit_commander.RobotCommander()
    )

    robot_state = (
        robot.get_current_state()
    )

    finish_target = deep.make_target(
        trajectory,
        TARGET_INDEX,
    )

    finish_solutions = []

    for seed_joints in seed_bank:
        seed_state = (
            deep.set_arm_joints(
                robot_state,
                seed_joints,
            )
        )

        seed_state = (
            deep.set_gripper_closed(
                seed_state
            )
        )

        solution = deep.solve_ik(
            compute_ik,
            finish_target,
            seed_state,
        )

        if solution is None:
            continue

        valid, pairs = (
            deep.collision_check(
                check_validity,
                solution,
            )
        )

        if not valid:
            continue

        deep.add_distinct_solution(
            finish_solutions,
            solution,
        )

    print(
        "FINISH branches        : {}"
        .format(
            len(finish_solutions)
        )
    )

    # ====================================
    # 4. 各枝をindex0まで戻し、
    #    保存済みP0に最も近い枝を選択
    # ====================================
    reverse_candidates = []

    for branch_index, branch in enumerate(
        finish_solutions
    ):
        (
            ok,
            reverse_path,
            failure,
        ) = collect_reverse_fold_path(
            deep,
            TARGET_INDEX,
            branch["state"],
            trajectory,
            compute_ik,
            check_validity,
        )

        if not ok:
            print(
                "branch {} : reverse FAIL {}"
                .format(
                    branch_index,
                    failure,
                )
            )
            continue

        q0 = reverse_path[-1][
            "joints_rad"
        ]

        diff = max_joint_diff(
            q0,
            q_saved_p0,
        )

        print(
            "branch {} : reverse SUCCESS "
            "P0 diff={:.9f} rad"
            .format(
                branch_index,
                diff,
            )
        )

        reverse_candidates.append({
            "branch_index":
                int(branch_index),
            "p0_diff_rad":
                float(diff),
            "reverse_path":
                reverse_path,
        })

    if not reverse_candidates:
        raise RuntimeError(
            "No FINISH branch reached P0."
        )

    selected_fold = min(
        reverse_candidates,
        key=lambda x:
            x["p0_diff_rad"],
    )

    # 保存済みP0との一致確認
    if (
        selected_fold["p0_diff_rad"]
        > 1.0e-3
    ):
        raise RuntimeError(
            "Reconstructed fold branch "
            "does not match saved P0. "
            "diff={} rad"
            .format(
                selected_fold[
                    "p0_diff_rad"
                ]
            )
        )

    # reverse:
    # index73 -> ... -> index0
    #
    # forward:
    # index0 -> ... -> index73
    fold_forward = list(
        reversed(
            selected_fold[
                "reverse_path"
            ]
        )
    )

    print(
        "selected FINISH branch : {}"
        .format(
            selected_fold[
                "branch_index"
            ]
        )
    )

    print(
        "fold forward points     : {}"
        .format(
            len(fold_forward)
        )
    )

    # ====================================
    # 5. 一本のsequenceへ統合
    # ====================================
    sequence = []

    # SafeZero -> PRE
    for i, p in enumerate(
        safezero_to_pre
    ):
        sequence.append({
            "segment":
                "SAFEZERO_TO_PRE",
            "source_index":
                int(i),
            "joints_rad":
                list(
                    p["joints_rad"]
                ),
            "gripper_m":
                float(
                    safe.GRIPPER_OPEN
                ),
        })

    # PRE -> P0
    # 最初のPREは重複するためskip
    for i, q in enumerate(
        pre_to_p0[1:],
        start=1,
    ):
        sequence.append({
            "segment":
                "PRE_TO_P0_OPEN",
            "source_index":
                int(i),
            "joints_rad":
                list(q),
            "gripper_m":
                float(
                    safe.GRIPPER_OPEN
                ),
        })

    # P0で把持
    sequence.append({
        "segment":
            "GRIPPER_CLOSE_AT_P0",
        "source_index":
            0,
        "joints_rad":
            list(q_saved_p0),
        "gripper_m":
            0.0,
    })

    # P0 -> index73
    # index0はP0として重複するためskip
    for item in fold_forward[1:]:
        sequence.append({
            "segment":
                "FOLD_CLOSED",
            "source_index":
                int(item["index"]),
            "joints_rad":
                list(
                    item["joints_rad"]
                ),
            "gripper_m":
                0.0,
        })

    # ====================================
    # 6. 接続点確認
    # ====================================
    safe_pre_diff = max_joint_diff(
        safezero_to_pre[-1][
            "joints_rad"
        ],
        pre_to_p0[0],
    )

    prep0_fold_diff = max_joint_diff(
        pre_to_p0[-1],
        fold_forward[0][
            "joints_rad"
        ],
    )

    print()
    print(
        "===== CONNECTION CHECK ====="
    )

    print(
        "SafeZero path PRE <-> "
        "PRE->P0 start : "
        "{:.9f} rad"
        .format(
            safe_pre_diff
        )
    )

    print(
        "PRE->P0 end <-> "
        "fold index0   : "
        "{:.9f} rad"
        .format(
            prep0_fold_diff
        )
    )

    if safe_pre_diff >= 1.0e-3:
        raise RuntimeError(
            "PRE connection mismatch."
        )

    if prep0_fold_diff >= 1.0e-3:
        raise RuntimeError(
            "P0 connection mismatch."
        )

    output = {
        "schema_version": 1,

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

        "finish_branch_index":
            int(
                selected_fold[
                    "branch_index"
                ]
            ),

        "joint_names":
            list(
                deep.COBOTTA_JOINTS
            ),

        "safezero_joints_rad":
            list(
                safe.SAFE_ZERO
            ),

        "new_pre_joints_rad":
            list(q_new_pre),

        "new_p0_joints_rad":
            list(q_saved_p0),

        "connection_checks": {
            "pre_connection_max_diff_rad":
                float(
                    safe_pre_diff
                ),
            "p0_connection_max_diff_rad":
                float(
                    prep0_fold_diff
                ),
        },

        "segment_counts": {
            "safezero_to_pre":
                len(
                    safezero_to_pre
                ),
            "pre_to_p0_open":
                len(
                    pre_to_p0
                ),
            "fold_p0_to_finish":
                len(
                    fold_forward
                ),
            "combined_sequence":
                len(sequence),
        },

        "safezero_to_pre":
            safezero_to_pre,

        "pre_to_p0_open_joints_rad":
            pre_to_p0,

        "fold_p0_to_finish":
            fold_forward,

        "combined_sequence":
            sequence,
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FILE.write_text(
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
        " COMPLETE PATH SAVED"
    )

    print(
        "========================================"
    )

    print(
        "combined points : {}"
        .format(
            len(sequence)
        )
    )

    print(
        "finish index    : {}"
        .format(
            TARGET_INDEX
        )
    )

    print(
        "saved:"
    )

    print(
        "  {}".format(
            OUTPUT_FILE
        )
    )


if __name__ == "__main__":
    main()
