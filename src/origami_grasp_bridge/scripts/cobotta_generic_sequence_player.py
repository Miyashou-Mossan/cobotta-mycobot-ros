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
import yaml

from moveit_msgs.msg import DisplayRobotState

from cobotta_generic_gripper import (
    build_gripper_frames,
    set_gripper,
)

from cobotta_generic_sync import (
    wait_for_bool_topic,
)


DISPLAY_TOPIC = (
    "/origami/debug/"
    "cobotta_generic_sequence_state"
)


def duration_to_sec(time_dict):
    return (
        float(time_dict.get("secs", 0))
        + float(time_dict.get("nsecs", 0)) * 1.0e-9
    )


def load_json(path):
    path = Path(path).expanduser().resolve()

    with path.open() as f:
        return json.load(f), path


def load_timed_trajectory(path):
    path = Path(path).expanduser().resolve()

    with path.open() as f:
        data = yaml.safe_load(f)

    if "joint_trajectory" not in data:
        raise RuntimeError(
            "Missing joint_trajectory in {}".format(path)
        )

    jt = data["joint_trajectory"]

    joint_names = list(
        jt.get("joint_names", [])
    )

    points = list(
        jt.get("points", [])
    )

    if not joint_names:
        raise RuntimeError(
            "Empty joint_names in {}".format(path)
        )

    if not points:
        raise RuntimeError(
            "Empty trajectory points in {}".format(path)
        )

    return {
        "path": path,
        "joint_names": joint_names,
        "points": points,
    }


def set_arm_joints(
    state,
    joint_names,
    positions,
):
    result = copy.deepcopy(state)

    state_names = list(
        result.joint_state.name
    )

    state_positions = list(
        result.joint_state.position
    )

    lookup = {
        name: i
        for i, name in enumerate(state_names)
    }

    if len(joint_names) != len(positions):
        raise RuntimeError(
            "Joint-name / position length mismatch"
        )

    missing = [
        name
        for name in joint_names
        if name not in lookup
    ]

    if missing:
        raise RuntimeError(
            "RobotState missing joints: {}".format(
                missing
            )
        )

    for name, value in zip(
        joint_names,
        positions,
    ):
        value = float(value)

        if not math.isfinite(value):
            raise RuntimeError(
                "Non-finite joint value for {}".format(
                    name
                )
            )

        state_positions[
            lookup[name]
        ] = value

    result.joint_state.position = (
        state_positions
    )

    result.joint_state.header.stamp = (
        rospy.Time(0)
    )

    result.is_diff = False

    return result


class GenericSequencePlayer:

    def __init__(
        self,
        input_file,
        gripper_steps,
        gripper_frame_period_sec,
        final_hold_sec,
    ):
        self.data, self.input_path = (
            load_json(input_file)
        )

        self.gripper_steps = int(
            gripper_steps
        )

        self.gripper_frame_period_sec = float(
            gripper_frame_period_sec
        )

        self.final_hold_sec = float(
            final_hold_sec
        )

        if self.gripper_steps < 1:
            raise ValueError(
                "gripper_steps must be >= 1"
            )

        if self.gripper_frame_period_sec <= 0.0:
            raise ValueError(
                "gripper_frame_period_sec must be > 0"
            )

        if self.final_hold_sec < 0.0:
            raise ValueError(
                "final_hold_sec must be >= 0"
            )

        if "actions" not in self.data:
            raise RuntimeError(
                "Missing actions"
            )

        if "joint_names" not in self.data:
            raise RuntimeError(
                "Missing joint_names"
            )

        self.source_joint_names = list(
            self.data["joint_names"]
        )

        moveit_commander.roscpp_initialize(
            sys.argv
        )

        self.robot = (
            moveit_commander.RobotCommander()
        )

        self.template_state = (
            self.robot.get_current_state()
        )

        self.pub = rospy.Publisher(
            DISPLAY_TOPIC,
            DisplayRobotState,
            queue_size=1,
            latch=True,
        )

        self.current_state = copy.deepcopy(
            self.template_state
        )

    def publish_state(self, state):
        msg = DisplayRobotState()

        msg.state = copy.deepcopy(
            state
        )

        self.pub.publish(msg)

    def play_arm_motion(
        self,
        action,
    ):
        timing = action.get(
            "timing",
            {}
        )

        trajectory_file = timing.get(
            "trajectory_file"
        )

        if not trajectory_file:
            raise RuntimeError(
                "ARM_MOTION {} has no timed trajectory".format(
                    action["action_index"]
                )
            )

        trajectory = load_timed_trajectory(
            trajectory_file
        )

        gripper_m = float(
            action["gripper_m"]
        )

        points = trajectory["points"]

        print(
            "[{}] ARM_MOTION".format(
                action["action_index"]
            )
        )
        print(
            "  trajectory:",
            trajectory["path"],
        )
        print(
            "  points    :",
            len(points),
        )
        print(
            "  gripper_m :",
            gripper_m,
        )

        previous_time = 0.0

        for i, point in enumerate(points):
            positions = point.get(
                "positions",
                []
            )

            current_time = duration_to_sec(
                point.get(
                    "time_from_start",
                    {}
                )
            )

            if i > 0:
                dt = (
                    current_time
                    - previous_time
                )

                if dt <= 0.0:
                    raise RuntimeError(
                        "Non-increasing trajectory time "
                        "at ARM_MOTION {} point {}".format(
                            action["action_index"],
                            i,
                        )
                    )

                rospy.sleep(dt)

            state = set_arm_joints(
                self.current_state,
                trajectory["joint_names"],
                positions,
            )

            state = set_gripper(
                state,
                gripper_m,
            )

            self.current_state = (
                copy.deepcopy(state)
            )

            self.publish_state(
                self.current_state
            )

            previous_time = (
                current_time
            )

        print(
            "  COMPLETE duration={:.6f} sec".format(
                previous_time
            )
        )

    def play_gripper_event(
        self,
        action,
    ):
        print(
            "[{}] GRIPPER_EVENT".format(
                action["action_index"]
            )
        )

        from_m = float(
            action["from_m"]
        )

        to_m = float(
            action["to_m"]
        )

        arm_joints_rad = action.get(
            "arm_joints_rad"
        )

        if arm_joints_rad is None:
            raise RuntimeError(
                "GRIPPER_EVENT {} missing "
                "arm_joints_rad".format(
                    action["action_index"]
                )
            )

        # Eventに記録された腕姿勢へ固定する
        state = set_arm_joints(
            self.current_state,
            self.source_joint_names,
            arm_joints_rad,
        )

        state = set_gripper(
            state,
            from_m,
        )

        frames = build_gripper_frames(
            state,
            from_m,
            to_m,
            self.gripper_steps,
        )

        print(
            "  from_m :",
            from_m,
        )
        print(
            "  to_m   :",
            to_m,
        )
        print(
            "  frames :",
            len(frames),
        )

        for i, frame in enumerate(frames):
            self.current_state = (
                copy.deepcopy(frame)
            )

            self.publish_state(
                self.current_state
            )

            if i < len(frames) - 1:
                rospy.sleep(
                    self.gripper_frame_period_sec
                )

        print("  COMPLETE")

    def play_wait(
        self,
        action,
    ):
        duration_sec = float(
            action["duration_sec"]
        )

        if (
            not math.isfinite(duration_sec)
            or duration_sec <= 0.0
        ):
            raise RuntimeError(
                "Invalid WAIT duration: {}".format(
                    duration_sec
                )
            )

        print(
            "[{}] WAIT".format(
                action["action_index"]
            )
        )
        print(
            "  duration_sec:",
            duration_sec,
        )

        # 現在の腕姿勢・グリッパ状態をそのまま維持
        self.publish_state(
            self.current_state
        )

        rospy.sleep(
            duration_sec
        )

        print("  COMPLETE")

    def play_sync_event(
        self,
        action,
    ):
        topic = str(
            action["topic"]
        )

        message_type = action.get(
            "message_type",
            "std_msgs/Bool",
        )

        expected_value = action[
            "expected_value"
        ]

        timeout_sec = float(
            action["timeout_sec"]
        )

        if message_type != "std_msgs/Bool":
            raise RuntimeError(
                "Unsupported SYNC_EVENT message_type: {}".format(
                    message_type
                )
            )

        if not isinstance(
            expected_value,
            bool,
        ):
            raise RuntimeError(
                "SYNC_EVENT expected_value must be bool"
            )

        print(
            "[{}] SYNC_EVENT".format(
                action["action_index"]
            )
        )
        print(
            "  topic         :",
            topic,
        )
        print(
            "  expected_value:",
            expected_value,
        )
        print(
            "  timeout_sec   :",
            timeout_sec,
        )

        # 現在姿勢・現在gripper状態を維持して待つ
        self.publish_state(
            self.current_state
        )

        result = wait_for_bool_topic(
            topic=topic,
            expected_value=expected_value,
            timeout_sec=timeout_sec,
        )

        print(
            "  received_count:",
            result["received_count"],
        )
        print(
            "  elapsed_sec   :",
            result["elapsed_sec"],
        )
        print(
            "  COMPLETE"
        )

    def run(self):
        print("=" * 80)
        print(
            "COBOTTA Generic Sequence Player v1"
        )
        print("=" * 80)
        print(
            "input :",
            self.input_path,
        )
        print(
            "topic :",
            DISPLAY_TOPIC,
        )
        print(
            "actions:",
            len(self.data["actions"]),
        )

        rospy.sleep(1.0)

        for action in self.data["actions"]:

            action_type = action.get(
                "type"
            )

            print()

            if action_type == "ARM_MOTION":
                self.play_arm_motion(
                    action
                )

            elif action_type == "GRIPPER_EVENT":
                self.play_gripper_event(
                    action
                )

            elif action_type == "WAIT":
                self.play_wait(
                    action
                )

            elif action_type == "SYNC_EVENT":
                self.play_sync_event(
                    action
                )

            else:
                raise RuntimeError(
                    "Unsupported action type: {}".format(
                        action_type
                    )
                )

        print()
        print(
            "===== SEQUENCE COMPLETE ====="
        )

        if self.final_hold_sec > 0.0:
            print(
                "holding final state for",
                self.final_hold_sec,
                "sec",
            )

            rospy.sleep(
                self.final_hold_sec
            )


def main():
    argv = rospy.myargv(
        argv=sys.argv
    )

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--gripper-steps",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--gripper-frame-period-sec",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--final-hold-sec",
        type=float,
        default=10.0,
    )

    args = parser.parse_args(
        argv[1:]
    )

    rospy.init_node(
        "cobotta_generic_sequence_player"
    )

    player = GenericSequencePlayer(
        input_file=args.input,
        gripper_steps=args.gripper_steps,
        gripper_frame_period_sec=
            args.gripper_frame_period_sec,
        final_hold_sec=args.final_hold_sec,
    )

    player.run()


if __name__ == "__main__":
    main()
