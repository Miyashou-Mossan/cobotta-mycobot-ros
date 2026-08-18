#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import csv
import math
import os
import sys

import moveit_commander
import rospy

from moveit_msgs.msg import (
    DisplayTrajectory,
    RobotTrajectory,
)
from trajectory_msgs.msg import JointTrajectoryPoint

from trajectory_yaml_io import save_robot_trajectory


DEFAULT_INPUT = os.path.expanduser(
    "~/cobotta_branch_preserving_full_path_0_340_edge_fixed.csv"
)

DEFAULT_OUTPUT = os.path.expanduser(
    "~/cobotta_branch_preserving_finish340_robot_trajectory.yaml"
)

GROUP_NAME = "cobotta_arm"


class JointPathToRobotTrajectory:

    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)

        self.input_file = os.path.expanduser(
            rospy.get_param(
                "~input_file",
                DEFAULT_INPUT,
            )
        )

        self.output_file = os.path.expanduser(
            rospy.get_param(
                "~output_file",
                DEFAULT_OUTPUT,
            )
        )

        self.group_name = rospy.get_param(
            "~group_name",
            GROUP_NAME,
        )

        self.velocity_scaling = float(
            rospy.get_param(
                "~velocity_scaling",
                0.10,
            )
        )

        self.acceleration_scaling = float(
            rospy.get_param(
                "~acceleration_scaling",
                0.10,
            )
        )

        self.display_wait = float(
            rospy.get_param(
                "~display_wait",
                5.0,
            )
        )

        if not (
            0.0 < self.velocity_scaling <= 1.0
        ):
            raise ValueError(
                "velocity_scaling must be in (0, 1]"
            )

        if not (
            0.0 < self.acceleration_scaling <= 1.0
        ):
            raise ValueError(
                "acceleration_scaling must be in (0, 1]"
            )

        self.group = moveit_commander.MoveGroupCommander(
            self.group_name,
            wait_for_servers=20.0,
        )

        self.active_joints = list(
            self.group.get_active_joints()
        )

        if not self.active_joints:
            raise RuntimeError(
                "No active joints found for group {}".format(
                    self.group_name
                )
            )

        self.display_publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            DisplayTrajectory,
            queue_size=1,
            latch=True,
        )

    def load_path(self):
        if not os.path.exists(
            self.input_file
        ):
            raise RuntimeError(
                "Input file does not exist: {}".format(
                    self.input_file
                )
            )

        with open(
            self.input_file,
            newline="",
            encoding="utf-8-sig",
        ) as f:
            rows = list(
                csv.DictReader(f)
            )

        if len(rows) < 2:
            raise RuntimeError(
                "Path must contain at least 2 points"
            )

        missing_columns = [
            name
            for name in self.active_joints
            if name not in rows[0]
        ]

        if missing_columns:
            raise RuntimeError(
                "CSV is missing joint columns: {}".format(
                    missing_columns
                )
            )

        path = []

        for row_index, row in enumerate(rows):

            joints = []

            for name in self.active_joints:
                value = float(row[name])

                if not math.isfinite(value):
                    raise RuntimeError(
                        "NaN/Inf at row {} joint {}".format(
                            row_index,
                            name,
                        )
                    )

                joints.append(value)

            path.append({
                "positions": joints,
                "row": row,
            })

        return path

    def build_initial_state(
        self,
        first_positions,
    ):
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

        missing = [
            name
            for name in self.active_joints
            if name not in lookup
        ]

        if missing:
            raise RuntimeError(
                "Current RobotState is missing joints: {}".format(
                    missing
                )
            )

        for name, value in zip(
            self.active_joints,
            first_positions,
        ):
            positions[
                lookup[name]
            ] = value

        state.joint_state.position = positions
        state.joint_state.header.stamp = rospy.Time(0)
        state.is_diff = False

        return state

    def build_untimed_trajectory(
        self,
        path,
    ):
        trajectory = RobotTrajectory()

        trajectory.joint_trajectory.header.frame_id = ""

        trajectory.joint_trajectory.joint_names = list(
            self.active_joints
        )

        for item in path:
            point = JointTrajectoryPoint()

            point.positions = list(
                item["positions"]
            )

            point.velocities = []
            point.accelerations = []
            point.effort = []
            point.time_from_start = rospy.Duration(0)

            trajectory.joint_trajectory.points.append(
                point
            )

        return trajectory

    def retime(
        self,
        initial_state,
        trajectory,
    ):
        retimed = self.group.retime_trajectory(
            initial_state,
            trajectory,
            velocity_scaling_factor=(
                self.velocity_scaling
            ),
            acceleration_scaling_factor=(
                self.acceleration_scaling
            ),
            algorithm=(
                "iterative_time_parameterization"
            ),
        )

        if (
            retimed is None
            or not retimed.joint_trajectory.points
        ):
            raise RuntimeError(
                "retime_trajectory failed"
            )

        return retimed

    def validate(
        self,
        trajectory,
        original_path,
    ):
        points = (
            trajectory.joint_trajectory.points
        )

        joint_names = list(
            trajectory.joint_trajectory.joint_names
        )

        joint_count = len(joint_names)

        if len(points) != len(original_path):
            raise RuntimeError(
                "Point count changed during retiming: "
                "{} -> {}".format(
                    len(original_path),
                    len(points),
                )
            )

        times = []

        max_adjacent_delta = -1.0
        max_adjacent_from = None
        max_adjacent_joint = None

        for i, point in enumerate(points):

            if len(point.positions) != joint_count:
                raise RuntimeError(
                    "Invalid positions length at point {}".format(
                        i
                    )
                )

            if not all(
                math.isfinite(v)
                for v in point.positions
            ):
                raise RuntimeError(
                    "Non-finite position at point {}".format(
                        i
                    )
                )

            if point.velocities and not all(
                math.isfinite(v)
                for v in point.velocities
            ):
                raise RuntimeError(
                    "Non-finite velocity at point {}".format(
                        i
                    )
                )

            if point.accelerations and not all(
                math.isfinite(v)
                for v in point.accelerations
            ):
                raise RuntimeError(
                    "Non-finite acceleration at point {}".format(
                        i
                    )
                )

            # retimeでpositions自体が変化していないことを確認
            source_positions = (
                original_path[i]["positions"]
            )

            max_position_error = max(
                abs(a-b)
                for a, b in zip(
                    point.positions,
                    source_positions,
                )
            )

            if max_position_error > 1.0e-10:
                raise RuntimeError(
                    "Retiming changed joint positions at point "
                    "{}: max error={:.12e}".format(
                        i,
                        max_position_error,
                    )
                )

            times.append(
                point.time_from_start.to_sec()
            )

            if i > 0:
                deltas = [
                    abs(a-b)
                    for a, b in zip(
                        points[i-1].positions,
                        point.positions,
                    )
                ]

                max_delta = max(deltas)

                if max_delta > max_adjacent_delta:
                    max_adjacent_delta = max_delta
                    max_adjacent_from = i - 1
                    max_adjacent_joint = joint_names[
                        deltas.index(max_delta)
                    ]

        non_increasing = [
            i
            for i in range(1, len(times))
            if times[i] <= times[i-1]
        ]

        if non_increasing:
            raise RuntimeError(
                "time_from_start is not strictly increasing: "
                "{}".format(
                    non_increasing
                )
            )

        minimum_interval = min(
            times[i] - times[i-1]
            for i in range(1, len(times))
        )

        return {
            "point_count": len(points),
            "total_duration": times[-1],
            "minimum_interval": minimum_interval,
            "max_adjacent_delta_rad":
                max_adjacent_delta,
            "max_adjacent_delta_deg":
                math.degrees(
                    max_adjacent_delta
                ),
            "max_adjacent_from":
                max_adjacent_from,
            "max_adjacent_to":
                max_adjacent_from + 1,
            "max_adjacent_joint":
                max_adjacent_joint,
        }

    def publish(
        self,
        initial_state,
        trajectory,
    ):
        display = DisplayTrajectory()

        display.trajectory_start = copy.deepcopy(
            initial_state
        )

        display.trajectory.append(
            trajectory
        )

        self.display_publisher.publish(
            display
        )

    def run(self):
        print("=" * 90)
        print(
            "COBOTTA joint path -> timed RobotTrajectory"
        )
        print("=" * 90)

        print("input :", self.input_file)
        print("output:", self.output_file)
        print("group :", self.group_name)
        print(
            "velocity scaling    :",
            self.velocity_scaling,
        )
        print(
            "acceleration scaling:",
            self.acceleration_scaling,
        )

        path = self.load_path()

        print(
            "input path points    :",
            len(path),
        )

        initial_state = (
            self.build_initial_state(
                path[0]["positions"]
            )
        )

        untimed = (
            self.build_untimed_trajectory(
                path
            )
        )

        timed = self.retime(
            initial_state,
            untimed,
        )

        result = self.validate(
            timed,
            path,
        )

        output_dir = os.path.dirname(
            self.output_file
        )

        if output_dir:
            os.makedirs(
                output_dir,
                exist_ok=True,
            )

        metadata = {
            "generator":
                "cobotta_joint_path_to_robot_trajectory.py",
            "group_name":
                self.group_name,
            "input_file":
                self.input_file,
            "velocity_scaling":
                self.velocity_scaling,
            "acceleration_scaling":
                self.acceleration_scaling,
            "time_parameterization":
                "iterative_time_parameterization",
            "robot_finish_paper_index":
                340,
            "actual_grasp_correction":
                True,
            "edge_collision_validation":
                True,
            "source_path_points":
                len(path),
        }

        save_robot_trajectory(
            self.output_file,
            timed,
            initial_state,
            metadata=metadata,
        )

        self.publish(
            initial_state,
            timed,
        )

        print()
        print("=" * 90)
        print("RESULT")
        print("=" * 90)

        print(
            "trajectory points :",
            result["point_count"],
        )

        print(
            "total duration    : "
            "{:.6f} s".format(
                result["total_duration"]
            )
        )

        print(
            "minimum interval  : "
            "{:.9f} s".format(
                result["minimum_interval"]
            )
        )

        print(
            "max adjacent delta: "
            "{:.6f} deg".format(
                result[
                    "max_adjacent_delta_deg"
                ]
            )
        )

        print(
            "  path_point      : "
            "{} -> {}".format(
                result[
                    "max_adjacent_from"
                ],
                result[
                    "max_adjacent_to"
                ],
            )
        )

        print(
            "  joint           :",
            result[
                "max_adjacent_joint"
            ],
        )

        print()
        print(
            "time_from_start   : strictly increasing"
        )

        print(
            "joint positions   : unchanged by retiming"
        )

        print(
            "RobotTrajectory YAML saved:"
        )
        print(
            " ",
            self.output_file,
        )

        print()
        print(
            "RViz publish      : DONE"
        )

        print(
            "Real robot Execute: NOT PERFORMED"
        )

        print()
        print(
            "ROBOT TRAJECTORY BUILD: SUCCESS"
        )

        rospy.sleep(
            self.display_wait
        )


def main():
    rospy.init_node(
        "cobotta_joint_path_to_robot_trajectory"
    )

    JointPathToRobotTrajectory().run()


if __name__ == "__main__":
    main()
