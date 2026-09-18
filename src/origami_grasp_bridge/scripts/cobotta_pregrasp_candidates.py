#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy

import numpy as np
import rospy

from geometry_msgs.msg import (
    Point,
    PolygonStamped,
    Pose,
    PoseArray,
)
from tf.transformations import quaternion_from_matrix
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import UInt64MultiArray


PAPER_TOPIC = "/origami/active_folding_paper_t0_ros"

PRE_POSE_TOPIC = (
    "/origami/debug/cobotta_pregrasp_candidate_poses"
)

GRASP_POSE_TOPIC = (
    "/origami/debug/cobotta_grasp_candidate_poses"
)

MARKER_TOPIC = (
    "/origami/debug/cobotta_pregrasp_candidates"
)

BATCH_TOPIC = (
    "/origami/debug/"
    "cobotta_pregrasp_batch"
)

DONE_TOPIC = (
    "/origami/debug/"
    "cobotta_pregrasp_feasibility_done"
)

# cobotta_tool_link -> actual_grasp_point [m]
R_GRASP = np.array(
    [0.002000, 0.000000, -0.004894],
    dtype=float,
)

# OPENした実Collision形状から求めた
# actual_grasp_pointより+Z_tool側の最大張り出し
L_FRONT = 0.009894


class PreGraspCandidates:

    def __init__(self):
        self.side = rospy.get_param(
            "~side",
            "lower",
        )

        self.p0_index = int(
            rospy.get_param(
                "~p0_index",
                0,
            )
        )

        self.all_p0 = bool(
            rospy.get_param(
                "~all_p0",
                False,
            )
        )

        # all_p0では必ずP0[0]から開始する
        if self.all_p0:
            self.p0_index = 0

        self.paper_msg = None
        self.p0_msg = None
        self.done = False

        # 現在Feasibility結果を待っている
        # P0 batchのtimestamp
        self.current_batch_stamp = None

        self.pre_pub = rospy.Publisher(
            PRE_POSE_TOPIC,
            PoseArray,
            queue_size=1,
            latch=True,
        )

        self.grasp_pub = rospy.Publisher(
            GRASP_POSE_TOPIC,
            PoseArray,
            queue_size=1,
            latch=True,
        )

        self.marker_pub = rospy.Publisher(
            MARKER_TOPIC,
            MarkerArray,
            queue_size=1,
            latch=True,
        )

        self.batch_pub = rospy.Publisher(
            BATCH_TOPIC,
            UInt64MultiArray,
            queue_size=1,
            latch=True,
        )

        rospy.Subscriber(
            PAPER_TOPIC,
            PolygonStamped,
            self.paper_cb,
            queue_size=1,
        )

        rospy.Subscriber(
            "/origami/cobotta_p0_candidates_{}".format(
                self.side
            ),
            PoseArray,
            self.p0_cb,
            queue_size=1,
        )

        rospy.Subscriber(
            DONE_TOPIC,
            UInt64MultiArray,
            self.done_cb,
            queue_size=1,
        )

        rospy.loginfo(
            "Waiting for paper and P0..."
        )

        rospy.loginfo(
            "all_p0 mode: %s",
            self.all_p0,
        )

    def paper_cb(self, msg):
        self.paper_msg = msg
        self.try_run()

    def p0_cb(self, msg):
        self.p0_msg = msg
        self.try_run()

    def done_cb(self, msg):
        if not self.all_p0:
            return

        if self.p0_msg is None:
            return

        if self.current_batch_stamp is None:
            return

        if len(msg.data) != 3:
            rospy.logwarn(
                "Ignoring malformed feasibility ACK."
            )
            return

        ack_index = int(msg.data[0])
        ack_secs = int(msg.data[1])
        ack_nsecs = int(msg.data[2])

        same_index = (
            ack_index
            == self.p0_index
        )

        same_stamp = (
            ack_secs
            == self.current_batch_stamp.secs
            and ack_nsecs
            == self.current_batch_stamp.nsecs
        )

        if not (
            same_index
            and same_stamp
        ):
            rospy.logwarn(
                "Ignoring feasibility ACK: "
                "received=(P0[%d], %d.%09d) "
                "current=(P0[%d], %d.%09d)",
                ack_index,
                ack_secs,
                ack_nsecs,
                self.p0_index,
                self.current_batch_stamp.secs,
                self.current_batch_stamp.nsecs,
            )
            return

        total = len(
            self.p0_msg.poses
        )

        rospy.loginfo(
            "P0[%d] feasibility completed.",
            self.p0_index,
        )

        if self.p0_index + 1 >= total:
            rospy.loginfo(
                "ALL P0 COMPLETE: %d/%d",
                total,
                total,
            )
            return

        self.p0_index += 1
        self.done = False

        rospy.loginfo(
            "Proceeding to P0[%d].",
            self.p0_index,
        )

        self.try_run()

    @staticmethod
    def point(v):
        p = Point()
        p.x = float(v[0])
        p.y = float(v[1])
        p.z = float(v[2])
        return p

    @staticmethod
    def closest_point_on_segment(
        p,
        a,
        b,
    ):
        ab = b - a
        denom = np.dot(ab, ab)

        if denom < 1.0e-12:
            return a.copy(), 0.0

        t = np.dot(
            p - a,
            ab,
        ) / denom

        t = max(
            0.0,
            min(1.0, float(t))
        )

        return a + t * ab, t

    @staticmethod
    def paper_normal(points):
        # 最初に見つかる非退化な3点を使う
        p0 = points[0]

        for i in range(
            1,
            len(points) - 1,
        ):
            a = points[i] - p0
            b = points[i + 1] - p0

            n = np.cross(a, b)
            norm = np.linalg.norm(n)

            if norm > 1.0e-9:
                return n / norm

        raise RuntimeError(
            "Could not calculate paper normal."
        )

    @staticmethod
    def make_rotation(
        y_tool,
        z_direction,
    ):
        y = (
            y_tool
            / np.linalg.norm(y_tool)
        )

        # Zを紙面法線Yと直交させる
        z = (
            z_direction
            - np.dot(
                z_direction,
                y,
            ) * y
        )

        z = z / np.linalg.norm(z)

        # 右手座標系
        x = np.cross(y, z)
        x = x / np.linalg.norm(x)

        z = np.cross(x, y)
        z = z / np.linalg.norm(z)

        R = np.eye(3)

        R[:, 0] = x
        R[:, 1] = y
        R[:, 2] = z

        return R

    @staticmethod
    def pose_from_tool(
        position,
        R,
    ):
        M = np.eye(4)
        M[:3, :3] = R

        q = quaternion_from_matrix(M)

        pose = Pose()

        pose.position.x = float(
            position[0]
        )
        pose.position.y = float(
            position[1]
        )
        pose.position.z = float(
            position[2]
        )

        pose.orientation.x = float(q[0])
        pose.orientation.y = float(q[1])
        pose.orientation.z = float(q[2])
        pose.orientation.w = float(q[3])

        return pose

    @staticmethod
    def tool_position_from_grasp(
        grasp_point,
        R,
    ):
        return (
            grasp_point
            - R.dot(R_GRASP)
        )

    def add_sphere(
        self,
        markers,
        frame_id,
        marker_id,
        ns,
        position,
        r,
        g,
        b,
        scale=0.006,
    ):
        m = Marker()

        m.header.frame_id = frame_id
        m.header.stamp = rospy.Time(0)

        m.ns = ns
        m.id = marker_id

        m.type = Marker.SPHERE
        m.action = Marker.ADD

        m.pose.position = self.point(
            position
        )
        m.pose.orientation.w = 1.0

        m.scale.x = scale
        m.scale.y = scale
        m.scale.z = scale

        m.color.r = r
        m.color.g = g
        m.color.b = b
        m.color.a = 1.0

        markers.markers.append(m)

    def add_line(
        self,
        markers,
        frame_id,
        marker_id,
        start,
        end,
    ):
        m = Marker()

        m.header.frame_id = frame_id
        m.header.stamp = rospy.Time(0)

        m.ns = "approach"
        m.id = marker_id

        m.type = Marker.ARROW
        m.action = Marker.ADD

        m.pose.orientation.w = 1.0

        m.scale.x = 0.002
        m.scale.y = 0.005
        m.scale.z = 0.007

        m.color.r = 1.0
        m.color.g = 1.0
        m.color.a = 1.0

        m.points = [
            self.point(start),
            self.point(end),
        ]

        markers.markers.append(m)

    def add_label(
        self,
        markers,
        frame_id,
        marker_id,
        position,
        text,
    ):
        m = Marker()

        m.header.frame_id = frame_id
        m.header.stamp = rospy.Time(0)

        m.ns = "labels"
        m.id = marker_id

        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD

        m.pose.position = self.point(
            position
        )
        m.pose.orientation.w = 1.0

        m.scale.z = 0.008

        m.color.r = 1.0
        m.color.g = 1.0
        m.color.b = 1.0
        m.color.a = 1.0

        m.text = text

        markers.markers.append(m)

    def try_run(self):
        if self.done:
            return

        if (
            self.paper_msg is None
            or self.p0_msg is None
        ):
            return

        if self.p0_index >= len(
            self.p0_msg.poses
        ):
            rospy.logerr(
                "P0 index out of range."
            )
            return

        self.done = True

        frame_id = (
            self.paper_msg.header.frame_id
        )

        paper = np.array(
            [
                [p.x, p.y, p.z]
                for p
                in self.paper_msg.polygon.points
            ],
            dtype=float,
        )

        if len(paper) < 3:
            rospy.logerr(
                "Paper polygon needs >= 3 vertices."
            )
            return

        pp = self.p0_msg.poses[
            self.p0_index
        ].position

        p0 = np.array(
            [
                pp.x,
                pp.y,
                pp.z,
            ],
            dtype=float,
        )

        normal = self.paper_normal(
            paper
        )

        pre_array = PoseArray()
        pre_array.header.frame_id = frame_id
        pre_array.header.stamp = rospy.Time.now()

        # このstampと一致するACKだけ受理する
        self.current_batch_stamp = copy.deepcopy(
            pre_array.header.stamp
        )

        grasp_array = PoseArray()
        grasp_array.header = copy.deepcopy(
            pre_array.header
        )

        markers = MarkerArray()

        print()
        print(
            "===== PRE-GRASP Candidates ====="
        )
        print(
            "P0 index : {}".format(
                self.p0_index
            )
        )
        print(
            "L_front  : {:.3f} mm".format(
                L_FRONT * 1000.0
            )
        )
        print()

        candidate_id = 0

        for edge_id in range(
            len(paper)
        ):
            a = paper[edge_id]
            b = paper[
                (edge_id + 1)
                % len(paper)
            ]

            e, t = (
                self.closest_point_on_segment(
                    p0,
                    a,
                    b,
                )
            )

            entry = p0 - e
            d_edge = np.linalg.norm(
                entry
            )

            if d_edge < 1.0e-9:
                continue

            z_direction = (
                entry / d_edge
            )

            for sign, sign_name in [
                (+1.0, "N+"),
                (-1.0, "N-"),
            ]:
                R = self.make_rotation(
                    sign * normal,
                    z_direction,
                )

                # make_rotation後の厳密な+Z_tool
                z_tool = R[:, 2]

                d_pre = (
                    d_edge
                    + L_FRONT
                )

                # PRE-GRASPにおける
                # actual_grasp_point位置
                pre_grasp_point = (
                    p0
                    - d_pre * z_tool
                )

                # 実際にIKへ渡すtool_link位置
                pre_tool = (
                    self.tool_position_from_grasp(
                        pre_grasp_point,
                        R,
                    )
                )

                grasp_tool = (
                    self.tool_position_from_grasp(
                        p0,
                        R,
                    )
                )

                pre_array.poses.append(
                    self.pose_from_tool(
                        pre_tool,
                        R,
                    )
                )

                grasp_array.poses.append(
                    self.pose_from_tool(
                        grasp_tool,
                        R,
                    )
                )

                print(
                    "candidate {} : "
                    "edge {} {}".format(
                        candidate_id,
                        edge_id,
                        sign_name,
                    )
                )

                print(
                    "  |E-P0|     = "
                    "{:.3f} mm".format(
                        d_edge * 1000.0
                    )
                )

                print(
                    "  d_pre      = "
                    "{:.3f} mm".format(
                        d_pre * 1000.0
                    )
                )

                print(
                    "  E          = "
                    "({:.3f}, {:.3f}, {:.3f}) mm".format(
                        *(e * 1000.0)
                    )
                )

                print(
                    "  PRE grasp  = "
                    "({:.3f}, {:.3f}, {:.3f}) mm".format(
                        *(
                            pre_grasp_point
                            * 1000.0
                        )
                    )
                )

                print(
                    "  PRE tool   = "
                    "({:.3f}, {:.3f}, {:.3f}) mm".format(
                        *(pre_tool * 1000.0)
                    )
                )

                print()

                base = (
                    candidate_id * 10
                )

                # PRE actual_grasp_point
                self.add_sphere(
                    markers,
                    frame_id,
                    base + 0,
                    "pre",
                    pre_grasp_point,
                    1.0,
                    0.0,
                    1.0,
                )

                # 紙端E
                self.add_sphere(
                    markers,
                    frame_id,
                    base + 1,
                    "edge",
                    e,
                    1.0,
                    0.5,
                    0.0,
                )

                # P0
                self.add_sphere(
                    markers,
                    frame_id,
                    base + 2,
                    "p0",
                    p0,
                    0.0,
                    1.0,
                    0.0,
                )

                # PRE → P0
                self.add_line(
                    markers,
                    frame_id,
                    base + 3,
                    pre_grasp_point,
                    p0,
                )

                self.add_label(
                    markers,
                    frame_id,
                    base + 4,
                    pre_grasp_point
                    + np.array([
                        0.0,
                        0.0,
                        0.012,
                    ]),
                    "edge {} {} PRE".format(
                        edge_id,
                        sign_name,
                    ),
                )

                candidate_id += 1

        batch_msg = UInt64MultiArray()
        batch_msg.data = [
            int(self.p0_index),
            int(pre_array.header.stamp.secs),
            int(pre_array.header.stamp.nsecs),
        ]

        # Batch情報を先にpublishし、
        # PRE/GRASPとtimestampで対応付ける
        self.batch_pub.publish(
            batch_msg
        )

        self.pre_pub.publish(
            pre_array
        )

        self.grasp_pub.publish(
            grasp_array
        )

        self.marker_pub.publish(
            markers
        )

        print(
            "total candidates : {}".format(
                candidate_id
            )
        )

        print(
            "Published:"
        )
        print(
            "  {}".format(
                PRE_POSE_TOPIC
            )
        )
        print(
            "  {}".format(
                GRASP_POSE_TOPIC
            )
        )
        print(
            "  {}".format(
                MARKER_TOPIC
            )
        )


def main():
    rospy.init_node(
        "cobotta_pregrasp_candidates"
    )

    PreGraspCandidates()

    rospy.spin()


if __name__ == "__main__":
    main()
