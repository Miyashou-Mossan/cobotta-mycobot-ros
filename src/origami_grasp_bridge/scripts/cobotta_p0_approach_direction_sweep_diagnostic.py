#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import rospy

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PolygonStamped, PoseArray


PAPER_TOPIC = "/origami/active_folding_paper_t0_ros"
P0_TOPIC = "/origami/cobotta_p0_candidates_lower"

MARKER_TOPIC = (
    "/origami/debug/"
    "cobotta_p0_approach_direction_sweep"
)

L_FRONT = 0.009894

ANGLE_STEP_DEG = 1.0


def point_msg(v):
    p = Point()
    p.x = float(v[0])
    p.y = float(v[1])
    p.z = float(v[2])
    return p


def paper_normal(points):
    p0 = points[0]

    for i in range(1, len(points) - 1):
        a = points[i] - p0
        b = points[i + 1] - p0

        n = np.cross(a, b)
        norm = np.linalg.norm(n)

        if norm > 1.0e-9:
            return n / norm

    raise RuntimeError(
        "Could not calculate paper normal."
    )


def rotate_about_axis(v, axis, angle_rad):
    """
    Rodrigues rotation formula.
    vをaxisまわりにangle_rad回転。
    """
    axis = axis / np.linalg.norm(axis)

    return (
        v * math.cos(angle_rad)
        + np.cross(axis, v)
        * math.sin(angle_rad)
        + axis
        * np.dot(axis, v)
        * (1.0 - math.cos(angle_rad))
    )


def closest_point_on_segment(p, a, b):
    ab = b - a
    denom = np.dot(ab, ab)

    if denom < 1.0e-12:
        return a.copy(), 0.0

    t = np.dot(p - a, ab) / denom

    t = max(
        0.0,
        min(1.0, float(t))
    )

    return a + t * ab, t


def ray_segment_intersection(
    ray_origin,
    ray_direction,
    seg_a,
    seg_b,
    normal,
):
    """
    紙面上で、

        ray_origin + s * ray_direction
        seg_a      + t * (seg_b-seg_a)

    の交点を求める。

    s >= 0
    0 <= t <= 1

    の場合のみ有効。

    3Dベクトルだが、紙面法線normalを使って
    紙面内2D問題として解く。
    """

    edge = seg_b - seg_a

    denom = np.dot(
        normal,
        np.cross(
            ray_direction,
            edge
        )
    )

    if abs(denom) < 1.0e-12:
        return None

    delta = seg_a - ray_origin

    s = (
        np.dot(
            normal,
            np.cross(
                delta,
                edge
            )
        )
        / denom
    )

    t = (
        np.dot(
            normal,
            np.cross(
                delta,
                ray_direction
            )
        )
        / denom
    )

    eps = 1.0e-9

    if s < eps:
        return None

    if t < -eps or t > 1.0 + eps:
        return None

    point = (
        ray_origin
        + s * ray_direction
    )

    return {
        "point": point,
        "ray_distance": float(s),
        "edge_t": float(t),
    }


def add_line(
    array,
    marker_id,
    frame_id,
    start,
    end,
):
    m = Marker()

    m.header.frame_id = frame_id
    m.header.stamp = rospy.Time(0)

    m.ns = "approach_directions"
    m.id = marker_id

    m.type = Marker.LINE_LIST
    m.action = Marker.ADD

    m.pose.orientation.w = 1.0

    m.scale.x = 0.0008

    # 色はRVizで識別しやすい白
    m.color.r = 1.0
    m.color.g = 1.0
    m.color.b = 1.0
    m.color.a = 0.45

    m.points = [
        point_msg(start),
        point_msg(end),
    ]

    array.markers.append(m)


def add_sphere(
    array,
    marker_id,
    frame_id,
    position,
    scale,
):
    m = Marker()

    m.header.frame_id = frame_id
    m.header.stamp = rospy.Time(0)

    m.ns = "p0"
    m.id = marker_id

    m.type = Marker.SPHERE
    m.action = Marker.ADD

    m.pose.position = point_msg(
        position
    )
    m.pose.orientation.w = 1.0

    m.scale.x = scale
    m.scale.y = scale
    m.scale.z = scale

    m.color.r = 1.0
    m.color.g = 0.0
    m.color.b = 0.0
    m.color.a = 1.0

    array.markers.append(m)


def main():
    rospy.init_node(
        "cobotta_p0_approach_direction_sweep_diagnostic"
    )

    p0_index = int(
        rospy.get_param(
            "~p0_index",
            0
        )
    )

    angle_step_deg = float(
        rospy.get_param(
            "~angle_step_deg",
            ANGLE_STEP_DEG
        )
    )

    print(
        "===== P0 APPROACH DIRECTION SWEEP ====="
    )

    print(
        "P0 index       : {}".format(
            p0_index
        )
    )

    print(
        "angle range    : 0 ... <360 deg"
    )

    print(
        "angle step     : {:.3f} deg"
        .format(
            angle_step_deg
        )
    )

    print(
        "IK/Collision   : NOT evaluated"
    )

    print(
        "geometry only  : YES"
    )

    paper_msg = rospy.wait_for_message(
        PAPER_TOPIC,
        PolygonStamped,
        timeout=15.0,
    )

    p0_msg = rospy.wait_for_message(
        P0_TOPIC,
        PoseArray,
        timeout=15.0,
    )

    if p0_index >= len(
        p0_msg.poses
    ):
        raise RuntimeError(
            "P0 index out of range."
        )

    frame_id = (
        paper_msg.header.frame_id
    )

    paper = np.array(
        [
            [p.x, p.y, p.z]
            for p in
            paper_msg.polygon.points
        ],
        dtype=float,
    )

    if len(paper) < 3:
        raise RuntimeError(
            "Paper polygon needs >=3 vertices."
        )

    pp = p0_msg.poses[
        p0_index
    ].position

    p0 = np.array(
        [pp.x, pp.y, pp.z],
        dtype=float,
    )

    normal = paper_normal(
        paper
    )

    # -----------------------------------------
    # theta=0 の基準方向
    #
    # 既存方式と対応させるため、
    # edge0への最近点方向を基準にする。
    #
    # 360°全域を走査するため、
    # 基準軸そのものは候補漏れには影響しない。
    # -----------------------------------------
    e0, _ = closest_point_on_segment(
        p0,
        paper[0],
        paper[1],
    )

    base = p0 - e0

    # 念のため紙面へ投影
    base = (
        base
        - np.dot(
            base,
            normal
        ) * normal
    )

    base_norm = np.linalg.norm(
        base
    )

    if base_norm < 1.0e-9:
        raise RuntimeError(
            "Could not define base direction."
        )

    base = base / base_norm

    candidates = []

    angle_deg = 0.0

    while angle_deg < (
        360.0 - 1.0e-9
    ):
        # +Z_tool = 境界 -> P0
        z_tool = rotate_about_axis(
            base,
            normal,
            math.radians(
                angle_deg
            )
        )

        z_tool = (
            z_tool
            / np.linalg.norm(z_tool)
        )

        # P0から境界側へ逆向きにrayを飛ばす
        ray_direction = -z_tool

        hits = []

        for edge_id in range(
            len(paper)
        ):
            a = paper[edge_id]
            b = paper[
                (edge_id + 1)
                % len(paper)
            ]

            hit = ray_segment_intersection(
                p0,
                ray_direction,
                a,
                b,
                normal,
            )

            if hit is None:
                continue

            hit["edge_id"] = edge_id
            hits.append(hit)

        if hits:
            # P0から最初に出会う境界
            hit = min(
                hits,
                key=lambda x:
                    x["ray_distance"]
            )

            edge_point = hit[
                "point"
            ]

            d_edge = hit[
                "ray_distance"
            ]

            # 紙端からさらにL_FRONTだけ外側
            pre_grasp_point = (
                p0
                - (
                    d_edge
                    + L_FRONT
                ) * z_tool
            )

            candidates.append({
                "angle_deg":
                    float(angle_deg),
                "edge_id":
                    int(
                        hit["edge_id"]
                    ),
                "edge_point":
                    edge_point,
                "pre_grasp_point":
                    pre_grasp_point,
                "z_tool":
                    z_tool,
                "d_edge":
                    float(d_edge),
            })

        angle_deg += angle_step_deg

    print()
    print(
        "generated directions = {}"
        .format(
            len(candidates)
        )
    )

    edge_counts = {}

    for c in candidates:
        edge_id = c[
            "edge_id"
        ]

        edge_counts[edge_id] = (
            edge_counts.get(
                edge_id,
                0
            )
            + 1
        )

    print(
        "entry edge distribution:"
    )

    for edge_id in sorted(
        edge_counts
    ):
        print(
            "  edge {} : {} directions"
            .format(
                edge_id,
                edge_counts[
                    edge_id
                ]
            )
        )

    print()
    print(
        "P0 [mm] = "
        "[{:+.3f}, {:+.3f}, {:+.3f}]"
        .format(
            p0[0] * 1000.0,
            p0[1] * 1000.0,
            p0[2] * 1000.0,
        )
    )

    # -----------------------------------------
    # Marker生成
    #
    # PRE -> P0 の全候補を線で表示
    # -----------------------------------------
    markers = MarkerArray()

    marker_id = 0

    add_sphere(
        markers,
        marker_id,
        frame_id,
        p0,
        0.006,
    )

    marker_id += 1

    for c in candidates:
        add_line(
            markers,
            marker_id,
            frame_id,
            c["pre_grasp_point"],
            p0,
        )

        marker_id += 1

    pub = rospy.Publisher(
        MARKER_TOPIC,
        MarkerArray,
        queue_size=1,
        latch=True,
    )

    print()
    print(
        "Publishing markers:"
    )

    print(
        "  {}".format(
            MARKER_TOPIC
        )
    )

    print(
        "Each line = PRE -> P0"
    )

    print(
        "continuous publish = 1 Hz"
    )

    rate = rospy.Rate(1.0)

    while not rospy.is_shutdown():
        now = rospy.Time.now()

        for m in markers.markers:
            m.header.stamp = now

        pub.publish(
            markers
        )

        rate.sleep()


if __name__ == "__main__":
    main()
