#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import json
import sys

import moveit_commander
import rospy

from moveit_msgs.msg import DisplayRobotState
from std_msgs.msg import String


INPUT_TOPIC = (
    "/origami/debug/"
    "cobotta_pregrasp_feasibility_results"
)

OUTPUT_TOPIC = (
    "/origami/debug/"
    "cobotta_grasp_posture"
)

COBOTTA_JOINTS = [
    "cobotta_joint_1",
    "cobotta_joint_2",
    "cobotta_joint_3",
    "cobotta_joint_4",
    "cobotta_joint_5",
    "cobotta_joint_6",
]


def collect_candidate(
    obj,
    target_p0,
    target_sign,
    inherited_p0=None,
    found=None,
):
    if found is None:
        found = []

    if isinstance(obj, dict):
        current_p0 = obj.get(
            "p0_index",
            inherited_p0
        )

        if (
            current_p0 == target_p0
            and obj.get("normal_sign") == target_sign
            and obj.get("result") == "FEASIBLE_FOUND"
            and "grasp_joints_rad" in obj
        ):
            found.append(obj)

        for value in obj.values():
            collect_candidate(
                value,
                target_p0,
                target_sign,
                current_p0,
                found,
            )

    elif isinstance(obj, list):
        for value in obj:
            collect_candidate(
                value,
                target_p0,
                target_sign,
                inherited_p0,
                found,
            )

    return found


def set_joint(state, name, value):
    names = list(
        state.joint_state.name
    )

    positions = list(
        state.joint_state.position
    )

    if name not in names:
        rospy.logwarn(
            "Joint not found: %s",
            name
        )
        return

    index = names.index(name)
    positions[index] = float(value)

    state.joint_state.position = positions


def main():
    rospy.init_node(
        "cobotta_grasp_posture_visualizer"
    )

    moveit_commander.roscpp_initialize(
        sys.argv
    )

    target_p0 = int(
        rospy.get_param(
            "~p0_index",
            0,
        )
    )

    target_sign = str(
        rospy.get_param(
            "~normal_sign",
            "N+",
        )
    )

    if target_sign not in ["N+", "N-"]:
        raise RuntimeError(
            "normal_sign must be N+ or N-"
        )

    print(
        "===== COBOTTA GRASP POSTURE VISUALIZER ====="
    )

    print(
        "target = P0[{}] {}".format(
            target_p0,
            target_sign
        )
    )

    print(
        "gripper = CLOSED 0 mm"
    )

    msg = rospy.wait_for_message(
        INPUT_TOPIC,
        String,
        timeout=15.0,
    )

    data = json.loads(msg.data)

    matches = collect_candidate(
        data,
        target_p0,
        target_sign,
    )

    if len(matches) != 1:
        raise RuntimeError(
            "Expected one candidate, found {}"
            .format(len(matches))
        )

    candidate = matches[0]

    grasp_joints = [
        float(v)
        for v in candidate[
            "grasp_joints_rad"
        ]
    ]

    robot = (
        moveit_commander
        .RobotCommander()
    )

    state = copy.deepcopy(
        robot.get_current_state()
    )

    for name, value in zip(
        COBOTTA_JOINTS,
        grasp_joints,
    ):
        set_joint(
            state,
            name,
            value,
        )

    set_joint(
        state,
        "cobotta_joint_gripper",
        0.0,
    )

    set_joint(
        state,
        "cobotta_joint_gripper_mimic",
        0.0,
    )

    display = DisplayRobotState()
    display.state = state

    pub = rospy.Publisher(
        OUTPUT_TOPIC,
        DisplayRobotState,
        queue_size=1,
        latch=True,
    )

    print()
    print(
        "candidate index = {}".format(
            candidate.get(
                "candidate_index",
                "?"
            )
        )
    )

    print(
        "edge = {}".format(
            candidate.get(
                "edge",
                "?"
            )
        )
    )

    print()

    print(
        "grasp_joints_rad:"
    )

    for name, value in zip(
        COBOTTA_JOINTS,
        grasp_joints,
    ):
        print(
            "  {} = {:.9f} rad"
            .format(
                name,
                value
            )
        )

    print()

    print(
        "Publishing:"
    )
    print(
        "  {}".format(
            OUTPUT_TOPIC
        )
    )

    print(
        "P0[{}] {} grasp posture"
        .format(
            target_p0,
            target_sign
        )
    )

    print(
        "continuous publish = 1 Hz"
    )

    rate = rospy.Rate(1.0)

    while not rospy.is_shutdown():
        display.state.joint_state.header.stamp = (
            rospy.Time.now()
        )

        pub.publish(display)

        rate.sleep()


if __name__ == "__main__":
    main()
