#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import moveit_commander
import rospy

from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

from trajectory_yaml_io import save_robot_trajectory


POSITION_TOLERANCE_RAD = 1.0e-10


class GenericArmMotionTimeParameterizer:

    def __init__(
        self,
        input_file,
        output_dir,
        group_name,
        velocity_scaling,
        acceleration_scaling,
    ):
        moveit_commander.roscpp_initialize(sys.argv)

        self.input_file = Path(input_file).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser().resolve()

        self.group_name = group_name
        self.velocity_scaling = float(velocity_scaling)
        self.acceleration_scaling = float(acceleration_scaling)

        if not (0.0 < self.velocity_scaling <= 1.0):
            raise ValueError(
                "velocity_scaling must be in (0, 1]"
            )

        if not (0.0 < self.acceleration_scaling <= 1.0):
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

    def load_sequence(self):
        if not self.input_file.exists():
            raise RuntimeError(
                "Input file does not exist: {}".format(
                    self.input_file
                )
            )

        with self.input_file.open() as f:
            data = json.load(f)

        if "joint_names" not in data:
            raise RuntimeError("Missing joint_names")

        if "actions" not in data:
            raise RuntimeError("Missing actions")

        source_joint_names = list(data["joint_names"])

        missing = [
            name
            for name in self.active_joints
            if name not in source_joint_names
        ]

        extra = [
            name
            for name in source_joint_names
            if name not in self.active_joints
        ]

        if missing or extra:
            raise RuntimeError(
                "Joint-name mismatch. missing={} extra={}".format(
                    missing,
                    extra,
                )
            )

        return data

    def reorder_positions(
        self,
        source_joint_names,
        joints_rad,
    ):
        if len(source_joint_names) != len(joints_rad):
            raise RuntimeError(
                "Joint-name / position length mismatch"
            )

        lookup = {
            name: i
            for i, name in enumerate(source_joint_names)
        }

        positions = [
            float(joints_rad[lookup[name]])
            for name in self.active_joints
        ]

        if not all(math.isfinite(v) for v in positions):
            raise RuntimeError(
                "Non-finite joint position detected"
            )

        return positions

    def build_initial_state(
        self,
        first_positions,
    ):
        state = self.group.get_current_state()

        names = list(state.joint_state.name)
        positions = list(state.joint_state.position)

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
                "Current RobotState missing joints: {}".format(
                    missing
                )
            )

        for name, value in zip(
            self.active_joints,
            first_positions,
        ):
            positions[lookup[name]] = value

        state.joint_state.position = positions
        state.joint_state.header.stamp = rospy.Time(0)
        state.is_diff = False

        return state

    def build_untimed_trajectory(
        self,
        path_positions,
    ):
        trajectory = RobotTrajectory()

        trajectory.joint_trajectory.joint_names = list(
            self.active_joints
        )

        for positions in path_positions:
            point = JointTrajectoryPoint()

            point.positions = list(positions)
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
        result = self.group.retime_trajectory(
            initial_state,
            trajectory,
            velocity_scaling_factor=self.velocity_scaling,
            acceleration_scaling_factor=self.acceleration_scaling,
            algorithm="iterative_time_parameterization",
        )

        if (
            result is None
            or not result.joint_trajectory.points
        ):
            raise RuntimeError(
                "retime_trajectory failed"
            )

        return result

    def validate(
        self,
        trajectory,
        original_positions,
    ):
        points = trajectory.joint_trajectory.points

        if len(points) != len(original_positions):
            raise RuntimeError(
                "Point count changed during retiming: {} -> {}".format(
                    len(original_positions),
                    len(points),
                )
            )

        if len(points) < 2:
            raise RuntimeError(
                "ARM_MOTION must contain at least 2 points"
            )

        times = []

        max_adjacent_delta = -1.0
        max_adjacent_from = None
        max_adjacent_joint = None

        for i, point in enumerate(points):

            if not all(
                math.isfinite(v)
                for v in point.positions
            ):
                raise RuntimeError(
                    "Non-finite position at point {}".format(i)
                )

            if point.velocities and not all(
                math.isfinite(v)
                for v in point.velocities
            ):
                raise RuntimeError(
                    "Non-finite velocity at point {}".format(i)
                )

            if point.accelerations and not all(
                math.isfinite(v)
                for v in point.accelerations
            ):
                raise RuntimeError(
                    "Non-finite acceleration at point {}".format(i)
                )

            max_position_error = max(
                abs(a - b)
                for a, b in zip(
                    point.positions,
                    original_positions[i],
                )
            )

            if max_position_error > POSITION_TOLERANCE_RAD:
                raise RuntimeError(
                    "Retiming changed joint positions at point {}: "
                    "max error={:.12e}".format(
                        i,
                        max_position_error,
                    )
                )

            times.append(
                point.time_from_start.to_sec()
            )

            if i > 0:
                deltas = [
                    abs(a - b)
                    for a, b in zip(
                        points[i - 1].positions,
                        point.positions,
                    )
                ]

                max_delta = max(deltas)

                if max_delta > max_adjacent_delta:
                    max_adjacent_delta = max_delta
                    max_adjacent_from = i - 1
                    max_adjacent_joint = self.active_joints[
                        deltas.index(max_delta)
                    ]

        non_increasing = [
            i
            for i in range(1, len(times))
            if times[i] <= times[i - 1]
        ]

        if non_increasing:
            raise RuntimeError(
                "time_from_start is not strictly increasing: {}".format(
                    non_increasing
                )
            )

        minimum_interval = min(
            times[i] - times[i - 1]
            for i in range(1, len(times))
        )

        return {
            "point_count": len(points),
            "total_duration_sec": times[-1],
            "minimum_interval_sec": minimum_interval,
            "max_adjacent_delta_rad": max_adjacent_delta,
            "max_adjacent_delta_deg":
                math.degrees(max_adjacent_delta),
            "max_adjacent_from": max_adjacent_from,
            "max_adjacent_to": max_adjacent_from + 1,
            "max_adjacent_joint": max_adjacent_joint,
        }

    def run(self):
        data = self.load_sequence()

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        source_joint_names = list(
            data["joint_names"]
        )

        output_actions = []
        timed_count = 0

        print("=" * 80)
        print("COBOTTA Generic ARM_MOTION Time Parameterizer")
        print("=" * 80)
        print("input :", self.input_file)
        print("output:", self.output_dir)
        print("group :", self.group_name)
        print("velocity scaling    :", self.velocity_scaling)
        print("acceleration scaling:", self.acceleration_scaling)

        for action in data["actions"]:

            if action["type"] != "ARM_MOTION":
                output_actions.append(
                    copy.deepcopy(action)
                )
                continue

            action_index = action["action_index"]
            points = action["points"]

            if len(points) < 2:
                raise RuntimeError(
                    "ARM_MOTION {} has fewer than 2 points".format(
                        action_index
                    )
                )

            path_positions = [
                self.reorder_positions(
                    source_joint_names,
                    point["joints_rad"],
                )
                for point in points
            ]

            initial_state = self.build_initial_state(
                path_positions[0]
            )

            untimed = self.build_untimed_trajectory(
                path_positions
            )

            timed = self.retime(
                initial_state,
                untimed,
            )

            validation = self.validate(
                timed,
                path_positions,
            )

            yaml_name = (
                "arm_motion_{:02d}_timed.yaml".format(
                    action_index
                )
            )

            yaml_path = self.output_dir / yaml_name

            metadata = {
                "generator":
                    "cobotta_generic_arm_motion_time_parameterizer.py",
                "action_index":
                    action_index,
                "action_type":
                    "ARM_MOTION",
                "group_name":
                    self.group_name,
                "gripper_m":
                    action.get("gripper_m"),
                "source_sequence_start":
                    action.get("source_sequence_start"),
                "source_sequence_end":
                    action.get("source_sequence_end"),
                "source_point_count":
                    len(points),
                "velocity_scaling":
                    self.velocity_scaling,
                "acceleration_scaling":
                    self.acceleration_scaling,
                "time_parameterization":
                    "iterative_time_parameterization",
            }

            save_robot_trajectory(
                str(yaml_path),
                timed,
                initial_state,
                metadata=metadata,
            )

            timed_action = copy.deepcopy(action)

            timed_action["timing"] = {
                "trajectory_file": str(yaml_path),
                "velocity_scaling":
                    self.velocity_scaling,
                "acceleration_scaling":
                    self.acceleration_scaling,
                "time_parameterization":
                    "iterative_time_parameterization",
                "validation": validation,
            }

            output_actions.append(
                timed_action
            )

            timed_count += 1

            print()
            print(
                "[{}] ARM_MOTION".format(
                    action_index
                )
            )
            print(
                "  points       :",
                validation["point_count"],
            )
            print(
                "  duration [s] :",
                validation["total_duration_sec"],
            )
            print(
                "  min dt [s]   :",
                validation["minimum_interval_sec"],
            )
            print(
                "  max delta    : {:.9f} rad = {:.6f} deg".format(
                    validation["max_adjacent_delta_rad"],
                    validation["max_adjacent_delta_deg"],
                )
            )
            print(
                "  location     : {} -> {} ({})".format(
                    validation["max_adjacent_from"],
                    validation["max_adjacent_to"],
                    validation["max_adjacent_joint"],
                )
            )
            print(
                "  saved        :",
                yaml_path,
            )

        output_data = copy.deepcopy(data)

        output_data["timing"] = {
            "generator":
                "cobotta_generic_arm_motion_time_parameterizer.py",
            "group_name":
                self.group_name,
            "velocity_scaling":
                self.velocity_scaling,
            "acceleration_scaling":
                self.acceleration_scaling,
            "time_parameterization":
                "iterative_time_parameterization",
            "timed_arm_motion_count":
                timed_count,
        }

        output_data["actions"] = output_actions

        sequence_output = (
            self.output_dir
            / "generic_sequence_timed.json"
        )

        with sequence_output.open("w") as f:
            json.dump(
                output_data,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print()
        print("===== COMPLETE =====")
        print("timed ARM_MOTION count:", timed_count)
        print("sequence output:", sequence_output)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Generic sequence JSON",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--group-name",
        default="cobotta_arm",
    )

    parser.add_argument(
        "--velocity-scaling",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--acceleration-scaling",
        type=float,
        default=0.10,
    )

    args = parser.parse_args()

    rospy.init_node(
        "cobotta_generic_arm_motion_time_parameterizer",
        anonymous=True,
    )

    runner = GenericArmMotionTimeParameterizer(
        input_file=args.input,
        output_dir=args.output_dir,
        group_name=args.group_name,
        velocity_scaling=args.velocity_scaling,
        acceleration_scaling=args.acceleration_scaling,
    )

    runner.run()


if __name__ == "__main__":
    main()
