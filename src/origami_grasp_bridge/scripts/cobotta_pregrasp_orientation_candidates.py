#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import rospy

from geometry_msgs.msg import (
    Point,
    PolygonStamped,
    Pose,
    PoseArray,
)
from visualization_msgs.msg import Marker, MarkerArray

from tf.transformations import quaternion_from_matrix


PAPER_TOPIC = "/origami/active_folding_paper_t0_ros"
P0_TOPIC = "/origami/cobotta_p0_candidates_lower"

POSE_TOPIC = "/origami/debug/cobotta_pregrasp_orientation_candidates"
MARKER_TOPIC = "/origami/debug/cobotta_pregrasp_orientation_markers"


class OrientationCandidates:

    def __init__(self):
        self.p0_index = int(
            rospy.get_param("~p0_index", 0)
        )

        self.paper_msg = None
        self.p0_msg = None
        self.done = False

        self.pose_pub = rospy.Publisher(
            POSE_TOPIC,
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

        rospy.Subscriber(
            PAPER_TOPIC,
            PolygonStamped,
            self.paper_cb,
            queue_size=1,
        )

        rospy.Subscriber(
            P0_TOPIC,
            PoseArray,
            self.p0_cb,
            queue_size=1,
        )

        rospy.loginfo(
            "Waiting for T0 paper and P0..."
        )

    def paper_cb(self, msg):
        self.paper_msg = msg
        self.try_run()

    def p0_cb(self, msg):
        self.p0_msg = msg
        self.try_run()

    @staticmethod
    def point(v):
        p = Point()
        p.x = float(v[0])
        p.y = float(v[1])
        p.z = float(v[2])
        return p

    @staticmethod
    def closest_point_on_segment(p, a, b):
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

        e = a + t * ab

        return e, t

    @staticmethod
    def make_rotation(
        y_tool,
        z_tool,
    ):
        y = y_tool / np.linalg.norm(
            y_tool
        )

        z = z_tool - (
            np.dot(z_tool, y) * y
        )

        z_norm = np.linalg.norm(z)

        if z_norm < 1.0e-9:
            raise RuntimeError(
                "Z_tool is parallel to Y_tool."
            )

        z = z / z_norm

        # X × Y = Z となる右手座標系
        x = np.cross(
            y,
            z,
        )

        x = x / np.linalg.norm(x)

        # 数値誤差を除く
        z = np.cross(
            x,
            y,
        )

        z = z / np.linalg.norm(z)

        R = np.eye(4)

        R[:3, 0] = x
        R[:3, 1] = y
        R[:3, 2] = z

        return R

    @staticmethod
    def make_pose(
        p0,
        R,
    ):
        q = quaternion_from_matrix(R)

        pose = Pose()

        pose.position.x = float(p0[0])
        pose.position.y = float(p0[1])
        pose.position.z = float(p0[2])

        pose.orientation.x = float(q[0])
        pose.orientation.y = float(q[1])
        pose.orientation.z = float(q[2])
        pose.orientation.w = float(q[3])

        return pose

    def add_axis(
        self,
        arr,
        frame_id,
        start,
        direction,
        marker_id,
        axis_name,
    ):
        length = 0.030

        end = (
            start
            + length * direction
        )

        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = rospy.Time(0)

        m.ns = "tool_axes"
        m.id = marker_id

        m.type = Marker.ARROW
        m.action = Marker.ADD

        m.pose.orientation.w = 1.0

        m.scale.x = 0.002
        m.scale.y = 0.004
        m.scale.z = 0.006

        m.points = [
            self.point(start),
            self.point(end),
        ]

        # RViz座標軸と同じ
        if axis_name == "X":
            m.color.r = 1.0

        elif axis_name == "Y":
            m.color.g = 1.0

        elif axis_name == "Z":
            m.color.b = 1.0

        m.color.a = 1.0

        arr.markers.append(m)

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

        paper = np.array([
            [p.x, p.y, p.z]
            for p
            in self.paper_msg.polygon.points
        ], dtype=float)

        if len(paper) < 3:
            rospy.logerr(
                "Paper needs >= 3 vertices."
            )
            return

        pp = self.p0_msg.poses[
            self.p0_index
        ].position

        p0 = np.array([
            pp.x,
            pp.y,
            pp.z,
        ], dtype=float)

        # T0紙面法線
        v1 = paper[1] - paper[0]
        v2 = paper[2] - paper[0]

        normal = np.cross(
            v1,
            v2,
        )

        n_norm = np.linalg.norm(
            normal
        )

        if n_norm < 1.0e-9:
            rospy.logerr(
                "Paper normal cannot be calculated."
            )
            return

        normal = normal / n_norm

        poses = PoseArray()
        poses.header.frame_id = frame_id
        poses.header.stamp = rospy.Time.now()

        markers = MarkerArray()

        print()
        print(
            "===== PRE-GRASP Orientation Candidates ====="
        )

        print(
            "P0 = ({:.3f}, {:.3f}, {:.3f}) mm".format(
                *(p0 * 1000.0)
            )
        )

        print(
            "paper normal = "
            "[{:+.6f}, {:+.6f}, {:+.6f}]".format(
                *normal
            )
        )

        print()

        n_edges = len(paper)

        candidate_id = 0

        for edge_id in range(n_edges):
            a = paper[edge_id]

            b = paper[
                (edge_id + 1) % n_edges
            ]

            e, t = (
                self.closest_point_on_segment(
                    p0,
                    a,
                    b,
                )
            )

            z_tool = p0 - e

            distance = np.linalg.norm(
                z_tool
            )

            if distance < 1.0e-9:
                continue

            z_tool = (
                z_tool / distance
            )

            for normal_sign in [
                +1.0,
                -1.0,
            ]:
                y_tool = (
                    normal_sign
                    * normal
                )

                R = self.make_rotation(
                    y_tool,
                    z_tool,
                )

                x = R[:3, 0]
                y = R[:3, 1]
                z = R[:3, 2]

                pose = self.make_pose(
                    p0,
                    R,
                )

                poses.poses.append(
                    pose
                )

                sign_name = (
                    "N+"
                    if normal_sign > 0
                    else "N-"
                )

                print(
                    "candidate {:d}: "
                    "edge {} {}".format(
                        candidate_id,
                        edge_id,
                        sign_name,
                    )
                )

                print(
                    "  E      = "
                    "({:.3f}, {:.3f}, {:.3f}) mm".format(
                        *(e * 1000.0)
                    )
                )

                print(
                    "  dist   = {:.3f} mm".format(
                        distance * 1000.0
                    )
                )

                print(
                    "  X_tool = "
                    "[{:+.5f}, {:+.5f}, {:+.5f}]".format(
                        *x
                    )
                )

                print(
                    "  Y_tool = "
                    "[{:+.5f}, {:+.5f}, {:+.5f}]".format(
                        *y
                    )
                )

                print(
                    "  Z_tool = "
                    "[{:+.5f}, {:+.5f}, {:+.5f}]".format(
                        *z
                    )
                )

                print()

                # 6候補が重ならないよう
                # 表示だけ少しZ方向へずらす
                display_offset = (
                    0.008
                    * candidate_id
                    * normal
                )

                origin = (
                    p0
                    + display_offset
                )

                base_id = (
                    candidate_id * 10
                )

                self.add_axis(
                    markers,
                    frame_id,
                    origin,
                    x,
                    base_id + 0,
                    "X",
                )

                self.add_axis(
                    markers,
                    frame_id,
                    origin,
                    y,
                    base_id + 1,
                    "Y",
                )

                self.add_axis(
                    markers,
                    frame_id,
                    origin,
                    z,
                    base_id + 2,
                    "Z",
                )

                label = Marker()

                label.header.frame_id = (
                    frame_id
                )
                label.header.stamp = (
                    rospy.Time(0)
                )

                label.ns = "labels"
                label.id = base_id + 3

                label.type = (
                    Marker.TEXT_VIEW_FACING
                )
                label.action = Marker.ADD

                label.pose.position = (
                    self.point(
                        origin
                        + np.array([
                            0.0,
                            0.0,
                            0.015,
                        ])
                    )
                )

                label.pose.orientation.w = 1.0

                label.scale.z = 0.008

                label.color.r = 1.0
                label.color.g = 1.0
                label.color.b = 1.0
                label.color.a = 1.0

                label.text = (
                    "edge {} {}".format(
                        edge_id,
                        sign_name,
                    )
                )

                markers.markers.append(
                    label
                )

                candidate_id += 1

        self.pose_pub.publish(
            poses
        )

        self.marker_pub.publish(
            markers
        )

        print(
            "total candidates : {}".format(
                len(poses.poses)
            )
        )

        print(
            "Published:"
        )
        print(
            "  {}".format(
                POSE_TOPIC
            )
        )
        print(
            "  {}".format(
                MARKER_TOPIC
            )
        )


def main():
    rospy.init_node(
        "cobotta_pregrasp_orientation_candidates"
    )

    OrientationCandidates()

    rospy.spin()


if __name__ == "__main__":
    main()
