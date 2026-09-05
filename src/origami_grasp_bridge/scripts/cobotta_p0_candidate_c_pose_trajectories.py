#!/usr/bin/env python3

import math

import numpy as np
import rospy

from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray

from tf.transformations import (
    quaternion_inverse,
    quaternion_matrix,
    quaternion_multiply,
)


class CobottaP0CandidateCPoseTrajectories:
    def __init__(self):
        self.paper_history_msg = None
        self.p0_upper_msg = None
        self.p0_lower_msg = None
        self.candidate_c_msg = None

        self.last_processed_stamp = None

        # cobotta_tool_link -> actual_grasp_point
        # FINISH340で使用していた実測オフセット [m]
        self.r_grasp = np.array([
            +0.000401,
            -0.001507,
            -0.004894,
        ], dtype=float)

        self.trajectory_publishers = {}

        self.marker_pub = rospy.Publisher(
            "/origami/cobotta_p0_candidate_c_pose_markers",
            MarkerArray,
            queue_size=1,
            latch=True,
        )

        rospy.Subscriber(
            "/origami/active_paper_pose_history_ros",
            PoseArray,
            self.paper_history_callback,
            queue_size=1,
        )

        rospy.Subscriber(
            "/origami/cobotta_p0_candidates_upper",
            PoseArray,
            self.p0_upper_callback,
            queue_size=1,
        )

        rospy.Subscriber(
            "/origami/cobotta_p0_candidates_lower",
            PoseArray,
            self.p0_lower_callback,
            queue_size=1,
        )

        rospy.Subscriber(
            "/origami/debug/cobotta_tool_candidate_c_pose",
            PoseStamped,
            self.candidate_c_callback,
            queue_size=1,
        )

        rospy.loginfo(
            "COBOTTA P0 Candidate-C Pose trajectory generator started."
        )
        rospy.loginfo(
            "Candidate C only: local-Y/local-Z fixed offsets are NOT applied."
        )
        rospy.loginfo(
            "actual_grasp offset = "
            "(%.3f, %.3f, %.3f) mm",
            self.r_grasp[0] * 1000.0,
            self.r_grasp[1] * 1000.0,
            self.r_grasp[2] * 1000.0,
        )

    @staticmethod
    def q_array(q):
        return np.array([
            q.x,
            q.y,
            q.z,
            q.w,
        ], dtype=float)

    @staticmethod
    def normalize_q(q):
        norm = np.linalg.norm(q)

        if norm < 1.0e-12:
            raise ValueError(
                "Quaternion norm is zero."
            )

        return q / norm

    @staticmethod
    def rotate_vector(q, v):
        rotation = quaternion_matrix(q)[:3, :3]
        return rotation.dot(v)

    @staticmethod
    def quaternion_angle_deg(q0, q1):
        q0 = q0 / np.linalg.norm(q0)
        q1 = q1 / np.linalg.norm(q1)

        dot = abs(float(np.dot(q0, q1)))
        dot = max(-1.0, min(1.0, dot))

        return math.degrees(
            2.0 * math.acos(dot)
        )

    @staticmethod
    def stamp_nsec(msg):
        return msg.header.stamp.to_nsec()

    def paper_history_callback(self, msg):
        self.paper_history_msg = msg
        self.update()

    def p0_upper_callback(self, msg):
        self.p0_upper_msg = msg
        self.update()

    def p0_lower_callback(self, msg):
        self.p0_lower_msg = msg
        self.update()

    def candidate_c_callback(self, msg):
        self.candidate_c_msg = msg
        self.update()

    def get_publisher(self, side, index):
        key = (side, index)

        if key not in self.trajectory_publishers:
            topic = (
                "/origami/"
                "cobotta_p0_candidate_c_trajectory/"
                "{}_{}".format(side, index)
            )

            self.trajectory_publishers[key] = (
                rospy.Publisher(
                    topic,
                    PoseArray,
                    queue_size=1,
                    latch=True,
                )
            )

            rospy.loginfo(
                "Created trajectory topic: %s",
                topic,
            )

        return self.trajectory_publishers[key]

    def make_tool_quaternions(self, paper_poses):
        """
        現在のCandidate CをPose履歴終端の紙Poseに対する
        相対姿勢へ変換し、その相対姿勢を全紙Poseへ適用する。
        """
        q_paper_last = self.normalize_q(
            self.q_array(
                paper_poses[-1].orientation
            )
        )

        q_candidate_last = self.normalize_q(
            self.q_array(
                self.candidate_c_msg.pose.orientation
            )
        )

        q_relative = self.normalize_q(
            quaternion_multiply(
                quaternion_inverse(
                    q_paper_last
                ),
                q_candidate_last,
            )
        )

        tool_quaternions = []

        for paper_pose in paper_poses:
            q_paper = self.normalize_q(
                self.q_array(
                    paper_pose.orientation
                )
            )

            q_tool = self.normalize_q(
                quaternion_multiply(
                    q_paper,
                    q_relative,
                )
            )

            # q / -q の符号を連続化
            if tool_quaternions:
                if np.dot(
                    tool_quaternions[-1],
                    q_tool,
                ) < 0.0:
                    q_tool = -q_tool

            tool_quaternions.append(q_tool)

        return q_relative, tool_quaternions

    def make_pose_trajectory(
        self,
        p0_pose,
        paper_poses,
        tool_quaternions,
    ):
        """
        P0をT0紙Poseのローカル点へ変換し、
        全紙Poseへ適用する。

        その点をactual_grasp_pointとして扱い、
        tool姿勢とr_graspからcobotta_tool_link位置を逆算する。
        """
        pose0 = paper_poses[0]

        q0 = self.normalize_q(
            self.q_array(
                pose0.orientation
            )
        )

        delta0 = np.array([
            p0_pose.position.x
            - pose0.position.x,

            p0_pose.position.y
            - pose0.position.y,

            p0_pose.position.z
            - pose0.position.z,
        ], dtype=float)

        local_p0 = self.rotate_vector(
            quaternion_inverse(q0),
            delta0,
        )

        output_poses = []

        max_grasp_reconstruction_error = 0.0

        for paper_pose, q_tool in zip(
            paper_poses,
            tool_quaternions,
        ):
            q_paper = self.normalize_q(
                self.q_array(
                    paper_pose.orientation
                )
            )

            paper_origin = np.array([
                paper_pose.position.x,
                paper_pose.position.y,
                paper_pose.position.z,
            ], dtype=float)

            grasp_point = (
                paper_origin
                + self.rotate_vector(
                    q_paper,
                    local_p0,
                )
            )

            offset_world = self.rotate_vector(
                q_tool,
                self.r_grasp,
            )

            tool_link_position = (
                grasp_point
                - offset_world
            )

            # 逆算したtool_link位置から
            # actual_grasp_pointを再構成して検証
            reconstructed_grasp = (
                tool_link_position
                + offset_world
            )

            reconstruction_error = np.linalg.norm(
                reconstructed_grasp
                - grasp_point
            )

            max_grasp_reconstruction_error = max(
                max_grasp_reconstruction_error,
                reconstruction_error,
            )

            pose = Pose()

            pose.position.x = (
                tool_link_position[0]
            )
            pose.position.y = (
                tool_link_position[1]
            )
            pose.position.z = (
                tool_link_position[2]
            )

            pose.orientation.x = q_tool[0]
            pose.orientation.y = q_tool[1]
            pose.orientation.z = q_tool[2]
            pose.orientation.w = q_tool[3]

            output_poses.append(pose)

        return (
            local_p0,
            output_poses,
            max_grasp_reconstruction_error,
        )

    def add_trajectory_markers(
        self,
        marker_array,
        side,
        index,
        poses,
    ):
        line = Marker()

        line.header.frame_id = "paper_center"
        line.header.stamp = rospy.Time(0)

        line.ns = (
            "candidate_c_tool_path_" + side
        )
        line.id = index

        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD

        line.pose.orientation.w = 1.0

        line.scale.x = 0.0015

        if side == "lower":
            line.color.r = 0.0
            line.color.g = 0.8
            line.color.b = 1.0
        else:
            line.color.r = 1.0
            line.color.g = 0.5
            line.color.b = 0.0

        line.color.a = 0.9

        line.points = [
            pose.position
            for pose in poses
        ]

        marker_array.markers.append(line)

        # 姿勢確認用。
        # 10点ごと＋最後の点に工具X軸方向の矢印を表示する。
        arrow_indices = list(
            range(0, len(poses), 10)
        )

        if (
            len(poses) > 0
            and (len(poses) - 1)
            not in arrow_indices
        ):
            arrow_indices.append(
                len(poses) - 1
            )

        for arrow_number, pose_index in enumerate(
            arrow_indices
        ):
            arrow = Marker()

            arrow.header.frame_id = "paper_center"
            arrow.header.stamp = rospy.Time(0)

            arrow.ns = (
                "candidate_c_tool_orientation_"
                + side
                + "_{}"
                .format(index)
            )

            arrow.id = arrow_number

            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD

            arrow.pose = poses[pose_index]

            arrow.scale.x = 0.012
            arrow.scale.y = 0.002
            arrow.scale.z = 0.002

            if side == "lower":
                arrow.color.r = 0.1
                arrow.color.g = 0.9
                arrow.color.b = 1.0
            else:
                arrow.color.r = 1.0
                arrow.color.g = 0.6
                arrow.color.b = 0.1

            arrow.color.a = 0.8

            marker_array.markers.append(
                arrow
            )

    def process_side(
        self,
        side,
        p0_msg,
        paper_poses,
        tool_quaternions,
        marker_array,
    ):
        for index, p0_pose in enumerate(
            p0_msg.poses
        ):
            (
                local_p0,
                trajectory,
                reconstruction_error,
            ) = self.make_pose_trajectory(
                p0_pose,
                paper_poses,
                tool_quaternions,
            )

            output_msg = PoseArray()

            output_msg.header.stamp = (
                self.paper_history_msg.header.stamp
            )
            output_msg.header.frame_id = (
                "paper_center"
            )

            output_msg.poses = trajectory

            publisher = self.get_publisher(
                side,
                index,
            )

            publisher.publish(output_msg)

            self.add_trajectory_markers(
                marker_array,
                side,
                index,
                trajectory,
            )

            start = trajectory[0].position
            end = trajectory[-1].position

            rospy.loginfo(
                "%s[%d]: poses=%d | "
                "localP0=(%.3f, %.3f, %.3f) mm | "
                "graspReconstructionMax=%.9f mm | "
                "tool START=(%.3f, %.3f, %.3f) mm | "
                "END=(%.3f, %.3f, %.3f) mm",
                side,
                index,
                len(trajectory),
                local_p0[0] * 1000.0,
                local_p0[1] * 1000.0,
                local_p0[2] * 1000.0,
                reconstruction_error * 1000.0,
                start.x * 1000.0,
                start.y * 1000.0,
                start.z * 1000.0,
                end.x * 1000.0,
                end.y * 1000.0,
                end.z * 1000.0,
            )

    def update(self):
        if (
            self.paper_history_msg is None
            or self.p0_upper_msg is None
            or self.p0_lower_msg is None
            or self.candidate_c_msg is None
        ):
            return

        pose_stamp = self.stamp_nsec(
            self.paper_history_msg
        )

        upper_stamp = self.stamp_nsec(
            self.p0_upper_msg
        )

        lower_stamp = self.stamp_nsec(
            self.p0_lower_msg
        )

        # Phase1-Cで既に合わせている
        # T0由来stampが一致している場合のみ使用する。
        if not (
            pose_stamp
            == upper_stamp
            == lower_stamp
        ):
            return

        if self.last_processed_stamp == pose_stamp:
            return

        paper_poses = (
            self.paper_history_msg.poses
        )

        if len(paper_poses) == 0:
            return

        try:
            (
                q_relative,
                tool_quaternions,
            ) = self.make_tool_quaternions(
                paper_poses
            )
        except ValueError as exc:
            rospy.logerr(
                "Candidate C trajectory generation "
                "failed: %s",
                str(exc),
            )
            return

        max_orientation_step = 0.0

        for q_prev, q_curr in zip(
            tool_quaternions[:-1],
            tool_quaternions[1:],
        ):
            max_orientation_step = max(
                max_orientation_step,
                self.quaternion_angle_deg(
                    q_prev,
                    q_curr,
                ),
            )

        marker_array = MarkerArray()

        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(
            delete_all
        )

        self.process_side(
            "upper",
            self.p0_upper_msg,
            paper_poses,
            tool_quaternions,
            marker_array,
        )

        self.process_side(
            "lower",
            self.p0_lower_msg,
            paper_poses,
            tool_quaternions,
            marker_array,
        )

        self.marker_pub.publish(
            marker_array
        )

        self.last_processed_stamp = pose_stamp

        rospy.loginfo(
            "Candidate C Pose trajectories published: "
            "paperPoses=%d upper=%d lower=%d "
            "maxOrientationStep=%.3f deg",
            len(paper_poses),
            len(self.p0_upper_msg.poses),
            len(self.p0_lower_msg.poses),
            max_orientation_step,
        )

        rospy.loginfo(
            "Candidate C relative quaternion = "
            "[%.6f, %.6f, %.6f, %.6f]",
            q_relative[0],
            q_relative[1],
            q_relative[2],
            q_relative[3],
        )


if __name__ == "__main__":
    rospy.init_node(
        "cobotta_p0_candidate_c_pose_trajectories"
    )

    CobottaP0CandidateCPoseTrajectories()

    rospy.spin()
