#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import json
import sys

import rospy
import moveit_commander

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

META_TOPIC = (
    "/origami/debug/"
    "cobotta_phase1d_exhaustive_fold_candidate_metadata"
)

COBOTTA_JOINTS = [
    "cobotta_joint_1",
    "cobotta_joint_2",
    "cobotta_joint_3",
    "cobotta_joint_4",
    "cobotta_joint_5",
    "cobotta_joint_6",
]


def set_arm_joints(state, joints):
    state = copy.deepcopy(state)

    names = list(state.joint_state.name)
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
        positions[
            lookup[name]
        ] = float(value)

    state.joint_state.position = (
        positions
    )

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
        ] = 0.0

    if (
        "cobotta_joint_gripper_mimic"
        in lookup
    ):
        positions[
            lookup[
                "cobotta_joint_gripper_mimic"
            ]
        ] = 0.0

    state.joint_state.position = (
        positions
    )

    return state


def solve_ik(
    compute_ik,
    pose,
    frame_id,
    seed,
):
    target = PoseStamped()

    target.header.frame_id = frame_id
    target.header.stamp = rospy.Time(0)
    target.pose = pose

    req = GetPositionIKRequest()

    req.ik_request.group_name = GROUP
    req.ik_request.ik_link_name = TIP
    req.ik_request.pose_stamped = target
    req.ik_request.robot_state = (
        copy.deepcopy(seed)
    )
    req.ik_request.avoid_collisions = False
    req.ik_request.timeout = (
        rospy.Duration(0.10)
    )

    res = compute_ik(req)

    if (
        res.error_code.val
        != MoveItErrorCodes.SUCCESS
    ):
        return None

    return set_gripper_closed(
        res.solution
    )


def collision_pairs(
    check_validity,
    state,
):
    req = GetStateValidityRequest()

    req.robot_state = state
    req.group_name = GROUP

    res = check_validity(req)

    pairs = sorted(set(
        tuple(sorted([
            c.contact_body_1,
            c.contact_body_2,
        ]))
        for c in res.contacts
    ))

    return bool(res.valid), pairs


def main():
    rospy.init_node(
        "cobotta_fold_end_collision_detail"
    )

    moveit_commander.roscpp_initialize(
        sys.argv
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

    meta_msg = rospy.wait_for_message(
        META_TOPIC,
        String,
        timeout=15.0,
    )

    data = json.loads(
        meta_msg.data
    )

    target = None

    for c in data["candidates"]:
        if (
            int(c["direction_index"]) == 248
            and c["normal_sign"] == "N-"
        ):
            target = c
            break

    if target is None:
        raise RuntimeError(
            "direction 248 N- not found"
        )

    topic = target[
        "trajectory_topic"
    ]

    trajectory = rospy.wait_for_message(
        topic,
        PoseArray,
        timeout=15.0,
    )

    seed = set_arm_joints(
        robot.get_current_state(),
        target["grasp_joints_rad"],
    )

    seed = set_gripper_closed(
        seed
    )

    print(
        "===== FOLD END COLLISION DETAIL ====="
    )
    print(
        "trajectory : {}".format(
            topic
        )
    )
    print(
        "poses      : {}".format(
            len(trajectory.poses)
        )
    )

    for index, pose in enumerate(
        trajectory.poses
    ):
        solution = solve_ik(
            compute_ik,
            pose,
            trajectory.header.frame_id,
            seed,
        )

        if solution is None:
            if index >= 69:
                print(
                    "index {:2d} : IK_FAIL"
                    .format(index)
                )
            continue

        valid, pairs = collision_pairs(
            check_validity,
            solution,
        )

        if index >= 69:
            print()
            print(
                "index {:2d} : {}"
                .format(
                    index,
                    "VALID"
                    if valid
                    else "COLLISION",
                )
            )

            for a, b in pairs:
                print(
                    "  {} <-> {}"
                    .format(a, b)
                )

        seed = solution


if __name__ == "__main__":
    main()
