#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import os
import sys

import moveit_commander
import rospy

from moveit_msgs.msg import DisplayRobotState


CSV_PATH = os.path.expanduser(
    "~/cobotta_branch_preserving_best_path.csv"
)

TOPIC = "/origami/debug/cobotta_branch_search_state"

GROUP = "cobotta_arm"


class Visualizer:

    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)

        self.index = int(
            rospy.get_param("~index", 340)
        )

        self.group = moveit_commander.MoveGroupCommander(
            GROUP,
            wait_for_servers=20.0,
        )

        self.active_joints = list(
            self.group.get_active_joints()
        )

        self.pub = rospy.Publisher(
            TOPIC,
            DisplayRobotState,
            queue_size=1,
            latch=True,
        )

    def load_row(self):
        with open(
            CSV_PATH,
            newline="",
            encoding="utf-8-sig",
        ) as f:
            rows = list(csv.DictReader(f))

        matches = [
            r for r in rows
            if int(r["index"]) == self.index
        ]

        if not matches:
            available = [
                int(r["index"])
                for r in rows
            ]

            raise RuntimeError(
                "index {} not found. "
                "available range = {}..{}"
                .format(
                    self.index,
                    min(available),
                    max(available),
                )
            )

        return matches[0]

    def make_state(self, row):
        state = self.group.get_current_state()

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

        for name in self.active_joints:
            positions[
                lookup[name]
            ] = float(row[name])

        state.joint_state.position = positions
        state.joint_state.header.stamp = rospy.Time(0)
        state.is_diff = False

        return state

    def run(self):
        row = self.load_row()
        state = self.make_state(row)

        msg = DisplayRobotState()
        msg.state = state

        self.pub.publish(msg)

        print("===== Branch search RViz state =====")
        print("index   :", self.index)
        print(
            "local X :",
            row["local_x_deg"],
            "deg",
        )
        print(
            "local Y :",
            row["local_y_deg"],
            "deg",
        )
        print("topic   :", TOPIC)

        print()
        print("joints:")
        for name in self.active_joints:
            print(
                "  {:16s} {:9.3f} deg"
                .format(
                    name,
                    float(row[name])
                    * 180.0 / 3.141592653589793,
                )
            )

        rospy.spin()


if __name__ == "__main__":
    rospy.init_node(
        "cobotta_branch_search_state_visualizer"
    )

    Visualizer().run()
