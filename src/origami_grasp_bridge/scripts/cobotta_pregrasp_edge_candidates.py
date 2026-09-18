#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import rospy

from geometry_msgs.msg import Point, PolygonStamped, PoseArray
from visualization_msgs.msg import Marker, MarkerArray


PAPER_TOPIC = "/origami/active_folding_paper_t0_ros"
P0_TOPIC = "/origami/cobotta_p0_candidates_lower"
MARKER_TOPIC = "/origami/debug/cobotta_pregrasp_edge_candidates"


class EdgeCandidateVisualizer:

    def __init__(self):
        self.p0_index = int(
            rospy.get_param("~p0_index", 0)
        )

        self.paper_msg = None
        self.p0_msg = None
        self.done = False

        self.pub = rospy.Publisher(
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
            "Waiting for T0 paper and P0 candidates..."
        )

    def paper_cb(self, msg):
        self.paper_msg = msg
        self.try_run()

    def p0_cb(self, msg):
        self.p0_msg = msg
        self.try_run()

    @staticmethod
    def to_point(v):
        p = Point()
        p.x = float(v[0])
        p.y = float(v[1])
        p.z = float(v[2])
        return p

    @staticmethod
    def closest_point_on_segment(p, a, b):
        v = b - a
        vv = np.dot(v, v)

        if vv < 1.0e-12:
            return a.copy(), 0.0

        t = np.dot(p - a, v) / vv

        t_clamped = max(
            0.0,
            min(1.0, float(t))
        )

        e = a + t_clamped * v

        return e, t_clamped

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
                "P0 index out of range: %d / %d",
                self.p0_index,
                len(self.p0_msg.poses),
            )
            return

        self.done = True

        frame_id = self.paper_msg.header.frame_id

        paper = np.array([
            [p.x, p.y, p.z]
            for p in self.paper_msg.polygon.points
        ], dtype=float)

        pp = self.p0_msg.poses[
            self.p0_index
        ].position

        p0 = np.array([
            pp.x,
            pp.y,
            pp.z,
        ], dtype=float)

        n = len(paper)

        if n < 3:
            rospy.logerr(
                "Paper polygon needs >= 3 vertices."
            )
            return

        print()
        print(
            "===== PRE-GRASP Edge Candidates ====="
        )
        print(
            "P0 index : {}".format(
                self.p0_index
            )
        )
        print(
            "P0       : "
            "({:.3f}, {:.3f}, {:.3f}) mm".format(
                *(p0 * 1000.0)
            )
        )
        print(
            "paper vertices : {}".format(n)
        )
        print()

        markers = MarkerArray()

        # 紙外形
        outline = Marker()
        outline.header.frame_id = frame_id
        outline.header.stamp = rospy.Time(0)
        outline.ns = "paper"
        outline.id = 0
        outline.type = Marker.LINE_STRIP
        outline.action = Marker.ADD
        outline.pose.orientation.w = 1.0
        outline.scale.x = 0.0015
        outline.color.r = 1.0
        outline.color.g = 1.0
        outline.color.b = 1.0
        outline.color.a = 1.0

        for p in paper:
            outline.points.append(
                self.to_point(p)
            )

        outline.points.append(
            self.to_point(paper[0])
        )

        markers.markers.append(outline)

        # P0
        p0_marker = Marker()
        p0_marker.header.frame_id = frame_id
        p0_marker.header.stamp = rospy.Time(0)
        p0_marker.ns = "p0"
        p0_marker.id = 1
        p0_marker.type = Marker.SPHERE
        p0_marker.action = Marker.ADD
        p0_marker.pose.position = self.to_point(p0)
        p0_marker.pose.orientation.w = 1.0
        p0_marker.scale.x = 0.008
        p0_marker.scale.y = 0.008
        p0_marker.scale.z = 0.008
        p0_marker.color.g = 1.0
        p0_marker.color.a = 1.0

        markers.markers.append(p0_marker)

        candidates = []

        for i in range(n):
            a = paper[i]
            b = paper[(i + 1) % n]

            e, t = self.closest_point_on_segment(
                p0,
                a,
                b,
            )

            vec = p0 - e
            dist = np.linalg.norm(vec)

            if dist > 1.0e-9:
                direction = vec / dist
            else:
                direction = np.zeros(3)

            candidates.append(
                (
                    i,
                    e,
                    t,
                    dist,
                    direction,
                )
            )

        # 近い紙端から順に表示
        candidates_sorted = sorted(
            candidates,
            key=lambda x: x[3]
        )

        for rank, (
            edge_id,
            e,
            t,
            dist,
            direction,
        ) in enumerate(candidates_sorted):

            print(
                "rank {:d} | edge {:d} | "
                "distance {:7.3f} mm | "
                "t={:.3f}".format(
                    rank,
                    edge_id,
                    dist * 1000.0,
                    t,
                )
            )

            print(
                "         E = "
                "({:.3f}, {:.3f}, {:.3f}) mm".format(
                    *(e * 1000.0)
                )
            )

            print(
                "         E->P0 = "
                "[{:+.5f}, {:+.5f}, {:+.5f}]".format(
                    *direction
                )
            )

            # E点
            sphere = Marker()
            sphere.header.frame_id = frame_id
            sphere.header.stamp = rospy.Time(0)
            sphere.ns = "entry_points"
            sphere.id = 100 + edge_id
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position = self.to_point(e)
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = 0.006
            sphere.scale.y = 0.006
            sphere.scale.z = 0.006
            sphere.color.r = 1.0
            sphere.color.g = 0.5
            sphere.color.a = 1.0

            markers.markers.append(sphere)

            # E -> P0
            line = Marker()
            line.header.frame_id = frame_id
            line.header.stamp = rospy.Time(0)
            line.ns = "approach_lines"
            line.id = 200 + edge_id
            line.type = Marker.ARROW
            line.action = Marker.ADD
            line.pose.orientation.w = 1.0
            line.scale.x = 0.002
            line.scale.y = 0.005
            line.scale.z = 0.007
            line.points = [
                self.to_point(e),
                self.to_point(p0),
            ]

            # 色は候補順位を見やすくするだけ
            if rank == 0:
                line.color.g = 1.0
            else:
                line.color.r = 1.0
                line.color.g = 0.5

            line.color.a = 1.0

            markers.markers.append(line)

            # ラベル
            label = Marker()
            label.header.frame_id = frame_id
            label.header.stamp = rospy.Time(0)
            label.ns = "edge_labels"
            label.id = 300 + edge_id
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position = self.to_point(
                e + np.array([0.0, 0.0, 0.010])
            )
            label.pose.orientation.w = 1.0
            label.scale.z = 0.008
            label.color.r = 1.0
            label.color.g = 1.0
            label.color.b = 1.0
            label.color.a = 1.0
            label.text = "edge {}".format(
                edge_id
            )

            markers.markers.append(label)

        self.pub.publish(markers)

        print()
        print(
            "Published:",
            MARKER_TOPIC,
        )


def main():
    rospy.init_node(
        "cobotta_pregrasp_edge_candidates"
    )

    EdgeCandidateVisualizer()

    rospy.spin()


if __name__ == "__main__":
    main()
