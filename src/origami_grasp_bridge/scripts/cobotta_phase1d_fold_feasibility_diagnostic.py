#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import json
import math
import sys

import moveit_commander
import rospy

from geometry_msgs.msg import PoseArray, PoseStamped
from std_msgs.msg import String

from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import (
    GetPositionIK,
    GetPositionIKRequest,
    GetStateValidity,
    GetStateValidityRequest,
)


GROUP = "cobotta_arm"
TIP = "cobotta_tool_link"

METADATA_TOPIC = (
    "/origami/debug/"
    "cobotta_phase1d_fold_candidate_metadata"
)

OUTPUT_TOPIC = (
    "/origami/debug/"
    "cobotta_phase1d_fold_feasibility_results"
)

COBOTTA_JOINTS = [
    "cobotta_joint_1",
    "cobotta_joint_2",
    "cobotta_joint_3",
    "cobotta_joint_4",
    "cobotta_joint_5",
    "cobotta_joint_6",
]

IK_TIMEOUT = 0.10

# 既存Phase1-Cと同じ関節ジャンプ判定
MAX_JUMP_DEG = 10.0
MAX_JUMP_RAD = math.radians(MAX_JUMP_DEG)

# 折り動作中は把持済みなので全閉
GRIPPER_CLOSED = 0.0


def set_cobotta_arm_joints(
    state,
    joints,
):
    state = copy.deepcopy(state)

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
        joints,
    ):
        if name not in lookup:
            raise RuntimeError(
                "Missing joint in RobotState: "
                + name
            )

        positions[
            lookup[name]
        ] = float(value)

    state.joint_state.position = positions

    return state


def set_gripper_closed(state):
    state = copy.deepcopy(state)

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

    if "cobotta_joint_gripper" in lookup:
        positions[
            lookup[
                "cobotta_joint_gripper"
            ]
        ] = GRIPPER_CLOSED

    if (
        "cobotta_joint_gripper_mimic"
        in lookup
    ):
        positions[
            lookup[
                "cobotta_joint_gripper_mimic"
            ]
        ] = -GRIPPER_CLOSED

    state.joint_state.position = positions

    return state


def extract_arm_joints(state):
    lookup = dict(zip(
        state.joint_state.name,
        state.joint_state.position,
    ))

    missing = [
        name
        for name in COBOTTA_JOINTS
        if name not in lookup
    ]

    if missing:
        raise RuntimeError(
            "RobotState missing joints: "
            + ", ".join(missing)
        )

    return [
        float(lookup[name])
        for name in COBOTTA_JOINTS
    ]


def solve_ik(
    compute_ik,
    target,
    seed_state,
):
    req = GetPositionIKRequest()

    req.ik_request.group_name = GROUP
    req.ik_request.ik_link_name = TIP

    req.ik_request.pose_stamped = target

    req.ik_request.robot_state = (
        copy.deepcopy(seed_state)
    )

    # IKとCollisionは分離して評価
    req.ik_request.avoid_collisions = False

    req.ik_request.timeout = (
        rospy.Duration(IK_TIMEOUT)
    )

    res = compute_ik(req)

    if (
        res.error_code.val
        != MoveItErrorCodes.SUCCESS
    ):
        return (
            None,
            res.error_code.val,
        )

    solution = set_gripper_closed(
        res.solution
    )

    return (
        solution,
        res.error_code.val,
    )


def check_collision(
    check_validity,
    state,
):
    req = GetStateValidityRequest()

    req.robot_state = state
    req.group_name = GROUP

    res = check_validity(req)

    pairs = []

    if not res.valid:
        for contact in res.contacts:
            pairs.append(
                tuple(sorted([
                    contact.contact_body_1,
                    contact.contact_body_2,
                ]))
            )

    return bool(res.valid), pairs


def max_joint_delta(
    previous,
    current,
):
    return max(
        abs(a - b)
        for a, b in zip(
            previous,
            current,
        )
    )


def diagnose_candidate(
    candidate,
    trajectory_msg,
    robot_state,
    compute_ik,
    check_validity,
):
    p0_index = candidate[
        "p0_index"
    ]

    candidate_index = candidate[
        "candidate_index"
    ]

    edge = candidate["edge"]
    normal_sign = candidate[
        "normal_sign"
    ]

    grasp_joints = [
        float(v)
        for v in candidate[
            "grasp_joints_rad"
        ]
    ]

    print()
    print(
        "========================================"
    )
    print(
        "P0[{}] candidate {} edge{} {}"
        .format(
            p0_index,
            candidate_index,
            edge,
            normal_sign,
        )
    )
    print(
        "========================================"
    )

    # --------------------------------------
    # P0到着状態を、この折り軌道専用の
    # 最初のIK seedとして使う。
    # グリッパはここでCLOSED 0 mm。
    # --------------------------------------
    seed_state = set_cobotta_arm_joints(
        robot_state,
        grasp_joints,
    )

    seed_state = set_gripper_closed(
        seed_state
    )

    previous_joints = list(
        grasp_joints
    )

    valid_count = 0
    prefix_count = 0
    prefix_broken = False

    ik_fail_count = 0
    collision_count = 0
    joint_jump_count = 0

    first_invalid_index = None
    first_invalid_reason = None

    max_joint_step_rad = 0.0

    collision_pairs_count = {}

    final_joints = None

    for index, pose in enumerate(
        trajectory_msg.poses
    ):
        target = PoseStamped()

        target.header.stamp = rospy.Time(0)

        target.header.frame_id = (
            trajectory_msg.header.frame_id
            if trajectory_msg.header.frame_id
            else "paper_center"
        )

        target.pose = pose

        solution, error_code = solve_ik(
            compute_ik,
            target,
            seed_state,
        )

        if solution is None:
            ik_fail_count += 1

            reason = (
                "IK_FAIL({})"
                .format(error_code)
            )

            print(
                "index {:3d}: {}"
                .format(
                    index,
                    reason,
                )
            )

            if first_invalid_index is None:
                first_invalid_index = index
                first_invalid_reason = reason

            prefix_broken = True

            # IK失敗時は直前の成功seedを保持
            continue

        current_joints = extract_arm_joints(
            solution
        )

        collision_free, pairs = (
            check_collision(
                check_validity,
                solution,
            )
        )

        if not collision_free:
            collision_count += 1

            for pair in pairs:
                key = (
                    "{}<->{}"
                    .format(
                        pair[0],
                        pair[1],
                    )
                )

                collision_pairs_count[key] = (
                    collision_pairs_count.get(
                        key,
                        0,
                    )
                    + 1
                )

        # P0到着関節角から
        # 折り軌道1点目への変化も含めて評価
        jump_rad = max_joint_delta(
            previous_joints,
            current_joints,
        )

        max_joint_step_rad = max(
            max_joint_step_rad,
            jump_rad,
        )

        jump_invalid = (
            jump_rad > MAX_JUMP_RAD
        )

        if jump_invalid:
            joint_jump_count += 1

        reasons = []

        if not collision_free:
            reasons.append(
                "COLLISION"
            )

        if jump_invalid:
            reasons.append(
                "JOINT_JUMP({:.3f}deg)"
                .format(
                    math.degrees(
                        jump_rad
                    )
                )
            )

        point_valid = (
            collision_free
            and not jump_invalid
        )

        if point_valid:
            valid_count += 1

            if not prefix_broken:
                prefix_count += 1

        else:
            reason = "+".join(
                reasons
            )

            if first_invalid_index is None:
                first_invalid_index = index
                first_invalid_reason = reason

            prefix_broken = True

            print(
                "index {:3d}: {}"
                .format(
                    index,
                    reason,
                )
            )

        # Collisionしていても、
        # 既存Phase1-Cと同様にIK枝の診断を継続。
        seed_state = copy.deepcopy(
            solution
        )

        previous_joints = list(
            current_joints
        )

        final_joints = list(
            current_joints
        )

    pose_count = len(
        trajectory_msg.poses
    )

    complete = (
        prefix_count == pose_count
    )

    if first_invalid_index is None:
        first_invalid_index_value = None
        first_invalid_reason = "PASS"
    else:
        first_invalid_index_value = int(
            first_invalid_index
        )

    print()
    print(
        "  valid                   = {}/{}"
        .format(
            valid_count,
            pose_count,
        )
    )

    print(
        "  continuous valid prefix = {}"
        .format(
            prefix_count
        )
    )

    print(
        "  first invalid index     = {}"
        .format(
            "NONE"
            if first_invalid_index_value is None
            else first_invalid_index_value
        )
    )

    print(
        "  first invalid reason    = {}"
        .format(
            first_invalid_reason
        )
    )

    print(
        "  IK_FAIL                 = {}"
        .format(
            ik_fail_count
        )
    )

    print(
        "  COLLISION               = {}"
        .format(
            collision_count
        )
    )

    print(
        "  JOINT_JUMP              = {}"
        .format(
            joint_jump_count
        )
    )

    print(
        "  max joint step          = {:.3f} deg"
        .format(
            math.degrees(
                max_joint_step_rad
            )
        )
    )

    print(
        "  FINISH reached safely   = {}"
        .format(
            complete
        )
    )

    return {
        "p0_index":
            int(p0_index),
        "candidate_index":
            int(candidate_index),
        "edge":
            int(edge),
        "normal_sign":
            str(normal_sign),
        "pose_count":
            int(pose_count),
        "valid_count":
            int(valid_count),
        "continuous_valid_prefix":
            int(prefix_count),
        "first_invalid_index":
            first_invalid_index_value,
        "first_invalid_reason":
            str(first_invalid_reason),
        "ik_fail_count":
            int(ik_fail_count),
        "collision_count":
            int(collision_count),
        "joint_jump_count":
            int(joint_jump_count),
        "max_joint_step_deg":
            float(
                math.degrees(
                    max_joint_step_rad
                )
            ),
        "finish_reached_safely":
            bool(complete),
        "gripper_closed_m":
            float(
                GRIPPER_CLOSED
            ),
        "grasp_joints_rad":
            grasp_joints,
        "final_joints_rad":
            final_joints,
        "collision_pairs":
            collision_pairs_count,
    }


def main():
    rospy.init_node(
        "cobotta_phase1d_fold_feasibility_diagnostic"
    )

    moveit_commander.roscpp_initialize(
        sys.argv
    )

    print(
        "===== PHASE1-D FOLD "
        "FEASIBILITY DIAGNOSTIC ====="
    )

    print(
        "IK strategy : previous solution -> next seed"
    )

    print(
        "gripper     : CLOSED 0 mm"
    )

    print(
        "IK timeout  : {:.3f} s".format(
            IK_TIMEOUT
        )
    )

    print(
        "max jump    : {:.1f} deg".format(
            MAX_JUMP_DEG
        )
    )

    robot = (
        moveit_commander
        .RobotCommander()
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

    print()
    print(
        "Waiting for fold candidate metadata..."
    )

    meta_msg = rospy.wait_for_message(
        METADATA_TOPIC,
        String,
        timeout=15.0,
    )

    metadata = json.loads(
        meta_msg.data
    )

    candidates = metadata[
        "candidates"
    ]

    print(
        "candidate count = {}".format(
            len(candidates)
        )
    )

    trajectories = []

    for candidate in candidates:
        topic = candidate[
            "trajectory_topic"
        ]

        print(
            "Waiting for {}".format(
                topic
            )
        )

        msg = rospy.wait_for_message(
            topic,
            PoseArray,
            timeout=15.0,
        )

        trajectories.append(
            msg
        )

    robot_state = (
        robot.get_current_state()
    )

    results = []

    for candidate, trajectory in zip(
        candidates,
        trajectories,
    ):
        result = diagnose_candidate(
            candidate,
            trajectory,
            robot_state,
            compute_ik,
            check_validity,
        )

        results.append(
            result
        )

    successful = [
        r
        for r in results
        if r[
            "finish_reached_safely"
        ]
    ]

    print()
    print(
        "========================================"
    )
    print(
        " PHASE1-D FOLD FINAL SUMMARY"
    )
    print(
        "========================================"
    )

    print(
        "input candidates : {}".format(
            len(results)
        )
    )

    print(
        "FINISH reached safely : {}/{}"
        .format(
            len(successful),
            len(results),
        )
    )

    print()

    for r in results:
        print(
            "P0[{p0_index}] "
            "edge{edge} {normal_sign} : "
            "prefix={continuous_valid_prefix}/{pose_count} "
            "first={first_invalid_index} "
            "{first_invalid_reason} "
            "IK={ik_fail_count} "
            "COL={collision_count} "
            "JUMP={joint_jump_count} "
            "maxStep={max_joint_step_deg:.3f}deg "
            "FINISH={finish_reached_safely}"
            .format(**r)
        )

    output = {
        "schema_version": 1,
        "gripper_closed_m":
            float(GRIPPER_CLOSED),
        "ik_timeout_sec":
            float(IK_TIMEOUT),
        "max_jump_deg":
            float(MAX_JUMP_DEG),
        "candidate_count":
            int(len(results)),
        "finish_reached_safely_count":
            int(len(successful)),
        "candidates":
            results,
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

    pub.publish(
        msg
    )

    print()
    print(
        "Published structured result:"
    )
    print(
        "  {}".format(
            OUTPUT_TOPIC
        )
    )

    rospy.spin()


if __name__ == "__main__":
    main()
