#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import csv
import math
import os
import sys

import moveit_commander
import rospy
import yaml

from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import (
    GetPositionIK,
    GetPositionIKRequest,
    GetStateValidity,
    GetStateValidityRequest,
)


ERROR_NAMES = {
    1: "SUCCESS",
    99999: "FAILURE",
    -1: "FAILURE",
    -2: "PLANNING_FAILED",
    -3: "INVALID_MOTION_PLAN",
    -4: "MOTION_PLAN_INVALIDATED",
    -5: "CONTROL_FAILED",
    -6: "UNABLE_TO_ACQUIRE_SENSOR_DATA",
    -7: "TIMED_OUT",
    -10: "PREEMPTED",
    -11: "START_STATE_IN_COLLISION",
    -12: "START_STATE_VIOLATES_PATH_CONSTRAINTS",
    -13: "GOAL_IN_COLLISION",
    -14: "GOAL_VIOLATES_PATH_CONSTRAINTS",
    -15: "GOAL_CONSTRAINTS_VIOLATED",
    -16: "INVALID_GROUP_NAME",
    -17: "INVALID_GOAL_CONSTRAINTS",
    -18: "INVALID_ROBOT_STATE",
    -19: "INVALID_LINK_NAME",
    -21: "FRAME_TRANSFORM_FAILURE",
    -22: "COLLISION_CHECKING_UNAVAILABLE",
    -23: "ROBOT_STATE_STALE",
    -24: "SENSOR_INFO_STALE",
    -31: "NO_IK_SOLUTION",
}


def error_name(code):
    return ERROR_NAMES.get(code, "ERROR_{}".format(code))


class Test9PaperFollowingIKCollisionScan:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)

        self.trajectory_file = os.path.expanduser(
            rospy.get_param(
                "~trajectory_file",
                "~/test9_pf_current_pose_128points_multidof.yaml",
            )
        )

        self.output_csv = os.path.expanduser(
            rospy.get_param(
                "~output_csv",
                "~/test9_pf_ik_collision_scan.csv",
            )
        )

        self.group_name = rospy.get_param(
            "~group_name",
            "cobotta_arm",
        )

        self.end_effector_link = rospy.get_param(
            "~end_effector_link",
            "cobotta_tool_link",
        )

        self.reference_frame = rospy.get_param(
            "~reference_frame",
            "paper_center",
        )

        self.min_z = float(
            rospy.get_param("~min_z", 0.030)
        )

        self.ik_timeout = float(
            rospy.get_param("~ik_timeout", 0.20)
        )

        self.ik_attempts = int(
            rospy.get_param("~ik_attempts", 5)
        )

        self.retry_collision_aware = bool(
            rospy.get_param(
                "~retry_collision_aware",
                True,
            )
        )

        self.compute_ik_service = rospy.get_param(
            "~compute_ik_service",
            "/compute_ik",
        )

        self.state_validity_service = rospy.get_param(
            "~state_validity_service",
            "/check_state_validity",
        )

        start_joint_positions = rospy.get_param(
            "~start_joint_positions",
            [],
        )

        if isinstance(start_joint_positions, str):
            start_joint_positions = yaml.safe_load(
                start_joint_positions
            )

        self.start_joint_positions = [
            float(value)
            for value in start_joint_positions
        ]

        self.move_group = moveit_commander.MoveGroupCommander(
            self.group_name,
            wait_for_servers=20.0,
        )

        self.move_group.set_end_effector_link(
            self.end_effector_link
        )

        self.active_joints = list(
            self.move_group.get_active_joints()
        )

        rospy.loginfo(
            "Waiting for service: %s",
            self.compute_ik_service,
        )
        rospy.wait_for_service(
            self.compute_ik_service,
            timeout=30.0,
        )

        rospy.loginfo(
            "Waiting for service: %s",
            self.state_validity_service,
        )
        rospy.wait_for_service(
            self.state_validity_service,
            timeout=30.0,
        )

        self.compute_ik = rospy.ServiceProxy(
            self.compute_ik_service,
            GetPositionIK,
            persistent=True,
        )

        self.check_state_validity = rospy.ServiceProxy(
            self.state_validity_service,
            GetStateValidity,
            persistent=True,
        )

    def load_points(self):
        if not os.path.isfile(self.trajectory_file):
            raise FileNotFoundError(
                "入力軌道がありません: {}".format(
                    self.trajectory_file
                )
            )

        with open(
            self.trajectory_file,
            "r",
            encoding="utf-8",
        ) as yaml_file:
            document = yaml.safe_load(yaml_file)

        if not isinstance(document, dict):
            raise ValueError(
                "入力YAMLのルートがmappingではありません。"
            )

        points = document.get("points")

        if not isinstance(points, list) or not points:
            raise ValueError(
                "有効なpoints配列がありません。"
            )

        return points

    def make_initial_state(self):
        state = self.move_group.get_current_state()

        if not self.start_joint_positions:
            rospy.loginfo(
                "開始シードにMoveIt現在状態を使用します。"
            )
            return state

        if (
            len(self.start_joint_positions)
            != len(self.active_joints)
        ):
            raise ValueError(
                "start_joint_positionsは{}要素必要ですが、"
                "{}要素です。".format(
                    len(self.active_joints),
                    len(self.start_joint_positions),
                )
            )

        state_names = list(state.joint_state.name)
        state_positions = list(
            state.joint_state.position
        )

        name_to_index = {
            name: index
            for index, name in enumerate(state_names)
        }

        for joint_name, joint_position in zip(
            self.active_joints,
            self.start_joint_positions,
        ):
            if joint_name not in name_to_index:
                raise RuntimeError(
                    "RobotStateに関節がありません: {}".format(
                        joint_name
                    )
                )

            state_positions[
                name_to_index[joint_name]
            ] = joint_position

        state.joint_state.position = state_positions
        state.joint_state.header.stamp = rospy.Time(0)
        state.is_diff = False

        rospy.loginfo(
            "開始シードにTest9実機関節角を使用: %s",
            self.start_joint_positions,
        )

        return state

    def parse_point(self, point, point_index):
        try:
            transform = point["transforms"][0]
            translation = transform["translation"]
            rotation = transform["rotation"]

            position = [
                float(translation["x"]),
                float(translation["y"]),
                float(translation["z"]),
            ]

            quaternion = [
                float(rotation["x"]),
                float(rotation["y"]),
                float(rotation["z"]),
                float(rotation["w"]),
            ]

            duration = point.get(
                "time_from_start",
                {},
            )

            time_from_start = (
                float(duration.get("secs", 0))
                + float(duration.get("nsecs", 0))
                * 1.0e-9
            )

        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
        ) as error:
            raise ValueError(
                "index {} のPose取得に失敗: {}".format(
                    point_index,
                    error,
                )
            )

        values = (
            position
            + quaternion
            + [time_from_start]
        )

        if not all(
            math.isfinite(value)
            for value in values
        ):
            raise ValueError(
                "index {} にNaN/Infがあります。".format(
                    point_index
                )
            )

        quaternion_norm = math.sqrt(
            sum(value * value for value in quaternion)
        )

        if quaternion_norm <= 1.0e-12:
            raise ValueError(
                "index {} のQuaternionノルムが0です。".format(
                    point_index
                )
            )

        quaternion = [
            value / quaternion_norm
            for value in quaternion
        ]

        pose = PoseStamped()
        pose.header.frame_id = self.reference_frame
        pose.header.stamp = rospy.Time(0)

        pose.pose.position.x = position[0]
        pose.pose.position.y = position[1]
        pose.pose.position.z = position[2]

        pose.pose.orientation.x = quaternion[0]
        pose.pose.orientation.y = quaternion[1]
        pose.pose.orientation.z = quaternion[2]
        pose.pose.orientation.w = quaternion[3]

        return pose, time_from_start

    def request_ik(
        self,
        pose,
        seed_state,
        avoid_collisions,
    ):
        request = GetPositionIKRequest()

        request.ik_request.group_name = (
            self.group_name
        )
        request.ik_request.robot_state = copy.deepcopy(
            seed_state
        )
        request.ik_request.avoid_collisions = (
            avoid_collisions
        )
        request.ik_request.ik_link_name = (
            self.end_effector_link
        )
        request.ik_request.pose_stamped = pose
        request.ik_request.timeout = rospy.Duration(
            self.ik_timeout
        )
        # ROS NoeticのPositionIKRequestには
        # attemptsフィールドがないため、timeoutのみ指定する。
        return self.compute_ik(request)

    def request_state_validity(self, robot_state):
        request = GetStateValidityRequest()
        request.robot_state = robot_state
        request.group_name = self.group_name

        return self.check_state_validity(request)

    def active_positions(self, robot_state):
        names = list(robot_state.joint_state.name)
        positions = list(
            robot_state.joint_state.position
        )

        name_to_position = {
            name: position
            for name, position in zip(
                names,
                positions,
            )
        }

        missing = [
            name
            for name in self.active_joints
            if name not in name_to_position
        ]

        if missing:
            raise RuntimeError(
                "IK結果に関節がありません: {}".format(
                    missing
                )
            )

        return [
            float(name_to_position[name])
            for name in self.active_joints
        ]

    @staticmethod
    def contact_summary(contacts):
        if not contacts:
            return ""

        summaries = []
        used = set()

        for contact in contacts:
            body_1 = getattr(
                contact,
                "contact_body_1",
                "unknown",
            )
            body_2 = getattr(
                contact,
                "contact_body_2",
                "unknown",
            )
            depth = float(
                getattr(contact, "depth", 0.0)
            )

            pair = tuple(sorted([body_1, body_2]))

            if pair in used:
                continue

            used.add(pair)

            summaries.append(
                "{}<->{} depth={:.6f}".format(
                    body_1,
                    body_2,
                    depth,
                )
            )

            if len(summaries) >= 10:
                break

        return "; ".join(summaries)

    @staticmethod
    def find_valid_segments(rows):
        segments = []
        current_start = None

        for row in rows:
            index = int(row["index"])
            valid = bool(row["valid"])

            if valid and current_start is None:
                current_start = index

            if not valid and current_start is not None:
                segments.append(
                    (current_start, index - 1)
                )
                current_start = None

        if current_start is not None:
            segments.append(
                (
                    current_start,
                    int(rows[-1]["index"]),
                )
            )

        return segments

    def run(self):
        points = self.load_points()
        seed_state = self.make_initial_state()

        rows = []

        previous_valid_index = None
        previous_valid_positions = None

        global_max_delta = 0.0
        global_max_delta_joint = ""
        global_max_delta_segment = ""

        for point_index, point in enumerate(points):
            if rospy.is_shutdown():
                raise rospy.ROSInterruptException()

            pose, time_from_start = self.parse_point(
                point,
                point_index,
            )

            z_value = pose.pose.position.z

            if z_value < self.min_z:
                raise RuntimeError(
                    "index {} が30 mm未満です: {:.9f}".format(
                        point_index,
                        z_value,
                    )
                )

            free_response = self.request_ik(
                pose,
                seed_state,
                avoid_collisions=False,
            )

            free_code = int(
                free_response.error_code.val
            )

            result = "IK_FAILED"
            valid = False
            contacts_text = ""
            joint_positions = []
            used_state = None
            collision_aware_code = 0

            if free_code == MoveItErrorCodes.SUCCESS:
                used_state = free_response.solution

                validity_response = (
                    self.request_state_validity(
                        used_state
                    )
                )

                valid = bool(
                    validity_response.valid
                )

                contacts_text = self.contact_summary(
                    validity_response.contacts
                )

                if valid:
                    result = "VALID"
                else:
                    result = "COLLISION"

                    if self.retry_collision_aware:
                        collision_response = (
                            self.request_ik(
                                pose,
                                seed_state,
                                avoid_collisions=True,
                            )
                        )

                        collision_aware_code = int(
                            collision_response.error_code.val
                        )

                        if (
                            collision_aware_code
                            == MoveItErrorCodes.SUCCESS
                        ):
                            alternative_validity = (
                                self.request_state_validity(
                                    collision_response.solution
                                )
                            )

                            if alternative_validity.valid:
                                used_state = (
                                    collision_response.solution
                                )
                                valid = True
                                result = "VALID_ALTERNATIVE"
                                contacts_text = ""
                            else:
                                contacts_text = (
                                    self.contact_summary(
                                        alternative_validity.contacts
                                    )
                                )

                # IK解が衝突状態でも次点のシードとして使い、
                # 同じIK分岐をできるだけ連続追跡する。
                seed_state = copy.deepcopy(used_state)

                joint_positions = self.active_positions(
                    used_state
                )

            max_adjacent_delta = ""
            max_adjacent_joint = ""

            if valid and joint_positions:
                if (
                    previous_valid_index is not None
                    and previous_valid_positions is not None
                    and point_index
                    == previous_valid_index + 1
                ):
                    deltas = [
                        abs(current - previous)
                        for current, previous in zip(
                            joint_positions,
                            previous_valid_positions,
                        )
                    ]

                    max_delta_index = max(
                        range(len(deltas)),
                        key=lambda index: deltas[index],
                    )

                    max_adjacent_delta = deltas[
                        max_delta_index
                    ]
                    max_adjacent_joint = self.active_joints[
                        max_delta_index
                    ]

                    if (
                        max_adjacent_delta
                        > global_max_delta
                    ):
                        global_max_delta = (
                            max_adjacent_delta
                        )
                        global_max_delta_joint = (
                            max_adjacent_joint
                        )
                        global_max_delta_segment = (
                            "{}->{}".format(
                                previous_valid_index,
                                point_index,
                            )
                        )

                previous_valid_index = point_index
                previous_valid_positions = list(
                    joint_positions
                )
            else:
                previous_valid_index = None
                previous_valid_positions = None

            row = {
                "index": point_index,
                "time_from_start": time_from_start,
                "x": pose.pose.position.x,
                "y": pose.pose.position.y,
                "z": pose.pose.position.z,
                "qx": pose.pose.orientation.x,
                "qy": pose.pose.orientation.y,
                "qz": pose.pose.orientation.z,
                "qw": pose.pose.orientation.w,
                "result": result,
                "valid": valid,
                "free_ik_code": free_code,
                "free_ik_name": error_name(free_code),
                "collision_aware_ik_code": (
                    collision_aware_code
                ),
                "collision_aware_ik_name": (
                    error_name(collision_aware_code)
                    if collision_aware_code != 0
                    else ""
                ),
                "contacts": contacts_text,
                "max_adjacent_delta_rad": (
                    max_adjacent_delta
                ),
                "max_adjacent_joint": (
                    max_adjacent_joint
                ),
            }

            for joint_index, joint_name in enumerate(
                self.active_joints
            ):
                row[joint_name] = (
                    joint_positions[joint_index]
                    if joint_positions
                    else ""
                )

            rows.append(row)

            rospy.loginfo(
                "[%03d/%03d] %-17s "
                "z=%.6f freeIK=%s contacts=%s",
                point_index,
                len(points) - 1,
                result,
                z_value,
                error_name(free_code),
                contacts_text if contacts_text else "-",
            )

        output_directory = os.path.dirname(
            self.output_csv
        )

        if output_directory:
            os.makedirs(
                output_directory,
                exist_ok=True,
            )

        fieldnames = [
            "index",
            "time_from_start",
            "x",
            "y",
            "z",
            "qx",
            "qy",
            "qz",
            "qw",
            "result",
            "valid",
            "free_ik_code",
            "free_ik_name",
            "collision_aware_ik_code",
            "collision_aware_ik_name",
            "contacts",
            "max_adjacent_delta_rad",
            "max_adjacent_joint",
        ] + self.active_joints

        with open(
            self.output_csv,
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=fieldnames,
            )
            writer.writeheader()
            writer.writerows(rows)

        valid_count = sum(
            1 for row in rows if row["valid"]
        )
        collision_count = sum(
            1
            for row in rows
            if row["result"] == "COLLISION"
        )
        ik_failure_count = sum(
            1
            for row in rows
            if row["result"] == "IK_FAILED"
        )
        alternative_count = sum(
            1
            for row in rows
            if row["result"] == "VALID_ALTERNATIVE"
        )

        segments = self.find_valid_segments(rows)

        longest_segment = None

        if segments:
            longest_segment = max(
                segments,
                key=lambda segment: (
                    segment[1] - segment[0] + 1
                ),
            )

        print("\n===== Test9 paper-following IK / Collision scan =====")
        print("input points          :", len(rows))
        print("valid points          :", valid_count)
        print("IK failed             :", ik_failure_count)
        print("collision             :", collision_count)
        print("valid alternatives    :", alternative_count)
        print("valid segments        :", segments)

        if longest_segment is not None:
            print(
                "longest valid segment : {} - {} ({} points)".format(
                    longest_segment[0],
                    longest_segment[1],
                    longest_segment[1]
                    - longest_segment[0]
                    + 1,
                )
            )
        else:
            print("longest valid segment : none")

        print(
            "max adjacent delta    : {:.9f} rad, {} at {}".format(
                global_max_delta,
                global_max_delta_joint
                if global_max_delta_joint
                else "-",
                global_max_delta_segment
                if global_max_delta_segment
                else "-",
            )
        )
        print("output CSV            :", self.output_csv)
        print("RESULT: IK / Collision scan completed")


def main():
    rospy.init_node(
        "test9_paper_following_ik_collision_scan"
    )

    scanner = Test9PaperFollowingIKCollisionScan()
    scanner.run()


if __name__ == "__main__":
    main()
