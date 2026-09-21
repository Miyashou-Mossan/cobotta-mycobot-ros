#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import json
import sys
import time
from pathlib import Path

import moveit_commander
import rospy

from moveit_msgs.msg import DisplayRobotState


INPUT_FILE = Path(
    "/home/maeda/catkin_ws/results/phase1d/"
    "candidate522_complete_path_to_index73.json"
)

OUTPUT_TOPIC = (
    "/origami/debug/"
    "cobotta_candidate522_complete_path_state"
)

COBOTTA_JOINTS = [
    "cobotta_joint_1",
    "cobotta_joint_2",
    "cobotta_joint_3",
    "cobotta_joint_4",
    "cobotta_joint_5",
    "cobotta_joint_6",
]


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

    positions[index] = float(
        value
    )

    state.joint_state.position = (
        positions
    )


def apply_sequence_point(
    base_state,
    point,
):
    state = copy.deepcopy(
        base_state
    )

    joints = point[
        "joints_rad"
    ]

    if len(joints) != 6:
        raise RuntimeError(
            "Expected 6 COBOTTA joints."
        )

    for name, value in zip(
        COBOTTA_JOINTS,
        joints,
    ):
        set_joint(
            state,
            name,
            value,
        )

    gripper = float(
        point[
            "gripper_m"
        ]
    )

    set_joint(
        state,
        "cobotta_joint_gripper",
        gripper,
    )

    set_joint(
        state,
        "cobotta_joint_gripper_mimic",
        -gripper,
    )

    state.joint_state.header.stamp = (
        rospy.Time.now()
    )

    return state


def main():
    rospy.init_node(
        "cobotta_candidate522_"
        "complete_path_visualizer"
    )

    moveit_commander.roscpp_initialize(
        sys.argv
    )

    if not INPUT_FILE.exists():
        raise RuntimeError(
            "Input file not found: "
            + str(INPUT_FILE)
        )

    data = json.loads(
        INPUT_FILE.read_text()
    )

    sequence = data[
        "combined_sequence"
    ]

    if not sequence:
        raise RuntimeError(
            "combined_sequence is empty."
        )

    playback_hz = float(
        rospy.get_param(
            "~playback_hz",
            10.0,
        )
    )

    loop = bool(
        rospy.get_param(
            "~loop",
            False,
        )
    )

    hold_sec = float(
        rospy.get_param(
            "~hold_sec",
            2.0,
        )
    )

    print(
        "===== COBOTTA COMPLETE PATH VISUALIZER ====="
    )

    print(
        "candidate      : {}".format(
            data.get(
                "candidate_index",
                "?"
            )
        )
    )

    print(
        "finish index   : {}".format(
            data.get(
                "finish_index",
                "?"
            )
        )
    )

    print(
        "sequence points: {}".format(
            len(sequence)
        )
    )

    print(
        "playback rate  : {:.2f} Hz".format(
            playback_hz
        )
    )

    print(
        "loop           : {}".format(
            loop
        )
    )

    robot = (
        moveit_commander
        .RobotCommander()
    )

    base_state = copy.deepcopy(
        robot.get_current_state()
    )

    pub = rospy.Publisher(
        OUTPUT_TOPIC,
        DisplayRobotState,
        queue_size=1,
        latch=True,
    )

    rospy.sleep(0.5)

    rate = rospy.Rate(
        playback_hz
    )

    previous_segment = None

    while not rospy.is_shutdown():

        for i, point in enumerate(
            sequence
        ):
            segment = point[
                "segment"
            ]

            if segment != previous_segment:
                print()
                print(
                    "===== {} ====="
                    .format(
                        segment
                    )
                )

                previous_segment = (
                    segment
                )

            display = DisplayRobotState()

            display.state = (
                apply_sequence_point(
                    base_state,
                    point,
                )
            )

            pub.publish(
                display
            )

            if (
                i == 0
                or i == len(sequence) - 1
                or i % 20 == 0
            ):
                print(
                    "[{}/{}] segment={} "
                    "source_index={} "
                    "gripper={:.1f} mm"
                    .format(
                        i + 1,
                        len(sequence),
                        segment,
                        point.get(
                            "source_index",
                            -1
                        ),
                        float(
                            point[
                                "gripper_m"
                            ]
                        ) * 1000.0,
                    )
                )

            rate.sleep()

            if rospy.is_shutdown():
                return

        print()
        print(
            "===== PLAYBACK FINISHED ====="
        )

        print(
            "final segment : {}".format(
                sequence[-1][
                    "segment"
                ]
            )
        )

        print(
            "final source index : {}"
            .format(
                sequence[-1].get(
                    "source_index",
                    "?"
                )
            )
        )

        if not loop:
            print(
                "Holding final posture..."
            )

            rospy.sleep(
                hold_sec
            )

            rospy.spin()
            return

        print(
            "Restarting playback..."
        )


if __name__ == "__main__":
    main()
