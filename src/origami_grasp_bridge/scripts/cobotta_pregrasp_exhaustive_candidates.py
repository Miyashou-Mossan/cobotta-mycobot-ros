#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import numpy as np
import rospy

from geometry_msgs.msg import (
    Point,
    Pose,
    PoseArray,
    PolygonStamped,
)
from std_msgs.msg import String
from visualization_msgs.msg import (
    Marker,
    MarkerArray,
)

from tf.transformations import quaternion_from_matrix


PAPER_TOPIC = (
    "/origami/active_folding_paper_t0_ros"
)

P0_TOPIC = (
    "/origami/cobotta_p0_candidates_lower"
)

PRE_TOPIC = (
    "/origami/debug/"
    "cobotta_pregrasp_exhaustive_candidate_poses"
)

GRASP_TOPIC = (
    "/origami/debug/"
    "cobotta_grasp_exhaustive_candidate_poses"
)

METADATA_TOPIC = (
    "/origami/debug/"
    "cobotta_pregrasp_exhaustive_metadata"
)

MARKER_TOPIC = (
    "/origami/debug/"
    "cobotta_pregrasp_exhaustive_markers"
)


# actual_grasp_point relative to tool_link
R_GRASP = np.array(
    [
        0.002000,
        0.000000,
        -0.004894,
    ],
    dtype=float,
)

# actual_grasp_pointより
# +Z_tool側の最大張り出し
L_FRONT = 0.009894

DEFAULT_ANGLE_STEP_DEG = 1.0


def point_msg(v):
    p = Point()
    p.x = float(v[0])
    p.y = float(v[1])
    p.z = float(v[2])
    return p


def paper_normal(points):
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


def closest_point_on_segment(
    p,
    a,
    b,
):
    ab = b - a
    denom = np.dot(ab, ab)

    if denom < 1.0e-12:
        return a.copy(), 0.0

    t = (
        np.dot(
            p - a,
            ab,
        )
        / denom
    )

    t = max(
        0.0,
        min(1.0, float(t))
    )

    return a + t * ab, t


def rotate_about_axis(
    v,
    axis,
    angle_rad,
):
    axis = (
        axis
        / np.linalg.norm(axis)
    )

    return (
        v * math.cos(angle_rad)
        + np.cross(
            axis,
            v,
        ) * math.sin(angle_rad)
        + axis
        * np.dot(axis, v)
        * (
            1.0
            - math.cos(angle_rad)
        )
    )


def ray_segment_intersection(
    ray_origin,
    ray_direction,
    seg_a,
    seg_b,
    normal,
):
    edge = seg_b - seg_a

    denom = np.dot(
        normal,
        np.cross(
            ray_direction,
            edge,
        ),
    )

    if abs(denom) < 1.0e-12:
        return None

    delta = seg_a - ray_origin

    s = (
        np.dot(
            normal,
            np.cross(
                delta,
                edge,
            ),
        )
        / denom
    )

    t = (
        np.dot(
            normal,
            np.cross(
                delta,
                ray_direction,
            ),
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
        "point":
            point,
        "ray_distance":
            float(s),
        "edge_t":
            float(t),
    }


def make_rotation(
    y_tool,
    z_direction,
):
    y = (
        y_tool
        / np.linalg.norm(y_tool)
    )

    # Z_toolを紙面へ投影
    z = (
        z_direction
        - np.dot(
            z_direction,
            y,
        ) * y
    )

    z = (
        z
        / np.linalg.norm(z)
    )

    # right-handed
    x = np.cross(
        y,
        z,
    )

    x = (
        x
        / np.linalg.norm(x)
    )

    z = np.cross(
        x,
        y,
    )

    z = (
        z
        / np.linalg.norm(z)
    )

    R = np.eye(3)

    R[:, 0] = x
    R[:, 1] = y
    R[:, 2] = z

    return R


def tool_position_from_grasp(
    grasp_point,
    R,
):
    return (
        grasp_point
        - R.dot(R_GRASP)
    )


def pose_from_tool(
    position,
    R,
):
    M = np.eye(4)
    M[:3, :3] = R

    q = quaternion_from_matrix(
        M
    )

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

    pose.orientation.x = float(
        q[0]
    )
    pose.orientation.y = float(
        q[1]
    )
    pose.orientation.z = float(
        q[2]
    )
    pose.orientation.w = float(
        q[3]
    )

    return pose


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

    m.ns = "exhaustive_approach"
    m.id = marker_id

    m.type = Marker.LINE_LIST
    m.action = Marker.ADD

    m.pose.orientation.w = 1.0

    m.scale.x = 0.0007

    m.color.r = 1.0
    m.color.g = 1.0
    m.color.b = 1.0
    m.color.a = 0.25

    m.points = [
        point_msg(start),
        point_msg(end),
    ]

    array.markers.append(m)


def main():
    rospy.init_node(
        "cobotta_pregrasp_exhaustive_candidates"
    )

    p0_index = int(
        rospy.get_param(
            "~p0_index",
            0,
        )
    )

    angle_step_deg = float(
        rospy.get_param(
            "~angle_step_deg",
            DEFAULT_ANGLE_STEP_DEG,
        )
    )

    print(
        "===== EXHAUSTIVE PRE-GRASP CANDIDATES ====="
    )

    print(
        "P0 index        : {}".format(
            p0_index
        )
    )

    print(
        "angle range     : 0 <= theta < 360 deg"
    )

    print(
        "angle step      : {:.3f} deg"
        .format(
            angle_step_deg
        )
    )

    print(
        "N+/N-           : BOTH"
    )

    print(
        "IK/Collision    : NOT evaluated here"
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

    if (
        p0_index < 0
        or p0_index
        >= len(p0_msg.poses)
    ):
        raise RuntimeError(
            "P0 index out of range."
        )

    frame_id = (
        paper_msg.header.frame_id
    )

    paper = np.array(
        [
            [
                p.x,
                p.y,
                p.z,
            ]
            for p in
            paper_msg.polygon.points
        ],
        dtype=float,
    )

    if len(paper) < 3:
        raise RuntimeError(
            "Paper polygon needs >=3 vertices."
        )

    pp = (
        p0_msg.poses[
            p0_index
        ].position
    )

    p0 = np.array(
        [
            pp.x,
            pp.y,
            pp.z,
        ],
        dtype=float,
    )

    normal = paper_normal(
        paper
    )

    # -------------------------------------
    # theta=0 基準方向
    #
    # edge0の最近点 -> P0
    #
    # 360°全探索なので、
    # 基準方向は単に角度表示の基準。
    # -------------------------------------
    e0, _ = closest_point_on_segment(
        p0,
        paper[0],
        paper[1],
    )

    base_direction = (
        p0 - e0
    )

    # 紙面内へ投影
    base_direction = (
        base_direction
        - np.dot(
            base_direction,
            normal,
        ) * normal
    )

    base_norm = np.linalg.norm(
        base_direction
    )

    if base_norm < 1.0e-9:
        raise RuntimeError(
            "Could not define base direction."
        )

    base_direction = (
        base_direction
        / base_norm
    )

    pre_array = PoseArray()
    pre_array.header.frame_id = (
        frame_id
    )
    pre_array.header.stamp = (
        rospy.Time.now()
    )

    grasp_array = PoseArray()
    grasp_array.header = (
        pre_array.header
    )

    markers = MarkerArray()

    metadata = []

    direction_id = 0
    candidate_id = 0

    angle_deg = 0.0

    while (
        angle_deg
        < 360.0 - 1.0e-9
    ):
        z_nominal = rotate_about_axis(
            base_direction,
            normal,
            math.radians(
                angle_deg
            ),
        )

        z_nominal = (
            z_nominal
            / np.linalg.norm(
                z_nominal
            )
        )

        # P0から紙外側へ逆向きray
        ray_direction = (
            -z_nominal
        )

        hits = []

        for edge_id in range(
            len(paper)
        ):
            a = paper[
                edge_id
            ]

            b = paper[
                (edge_id + 1)
                % len(paper)
            ]

            hit = (
                ray_segment_intersection(
                    p0,
                    ray_direction,
                    a,
                    b,
                    normal,
                )
            )

            if hit is None:
                continue

            hit["edge_id"] = (
                edge_id
            )

            hits.append(
                hit
            )

        if hits:
            # P0から見て最初の紙境界
            hit = min(
                hits,
                key=lambda x:
                    x["ray_distance"]
            )

            edge_id = int(
                hit["edge_id"]
            )

            edge_point = (
                hit["point"]
            )

            d_edge = float(
                hit["ray_distance"]
            )

            # N+ / N-を必ず隣接ペアにする
            for sign, sign_name in [
                (+1.0, "N+"),
                (-1.0, "N-"),
            ]:
                R = make_rotation(
                    sign * normal,
                    z_nominal,
                )

                # make_rotation後の厳密なZ
                z_tool = (
                    R[:, 2]
                )

                d_pre = (
                    d_edge
                    + L_FRONT
                )

                pre_grasp_point = (
                    p0
                    - d_pre
                    * z_tool
                )

                pre_tool = (
                    tool_position_from_grasp(
                        pre_grasp_point,
                        R,
                    )
                )

                grasp_tool = (
                    tool_position_from_grasp(
                        p0,
                        R,
                    )
                )

                pre_array.poses.append(
                    pose_from_tool(
                        pre_tool,
                        R,
                    )
                )

                grasp_array.poses.append(
                    pose_from_tool(
                        grasp_tool,
                        R,
                    )
                )

                metadata.append({
                    "candidate_index":
                        int(candidate_id),
                    "direction_index":
                        int(direction_id),
                    "p0_index":
                        int(p0_index),
                    "approach_angle_deg":
                        float(angle_deg),
                    "entry_edge":
                        int(edge_id),
                    "edge_t":
                        float(
                            hit["edge_t"]
                        ),
                    "normal_sign":
                        str(sign_name),
                    "distance_edge_to_p0_m":
                        float(d_edge),
                    "pre_distance_m":
                        float(d_pre),
                    "edge_point_m": [
                        float(v)
                        for v
                        in edge_point
                    ],
                    "pre_grasp_point_m": [
                        float(v)
                        for v
                        in pre_grasp_point
                    ],
                    "z_tool": [
                        float(v)
                        for v
                        in z_tool
                    ],
                })

                candidate_id += 1

            # 進入線はN±で共通なので1本だけ描く
            add_line(
                markers,
                direction_id,
                frame_id,
                (
                    p0
                    - (
                        d_edge
                        + L_FRONT
                    ) * z_nominal
                ),
                p0,
            )

            direction_id += 1

        angle_deg += (
            angle_step_deg
        )

    metadata_msg = String()

    metadata_msg.data = json.dumps(
        {
            "schema_version": 1,
            "p0_index":
                int(p0_index),
            "frame_id":
                str(frame_id),
            "angle_step_deg":
                float(angle_step_deg),
            "direction_count":
                int(direction_id),
            "candidate_count":
                int(candidate_id),
            "pair_rule":
                "adjacent N+/N-",
            "candidates":
                metadata,
        },
        ensure_ascii=False,
        sort_keys=True,
    )

    pre_pub = rospy.Publisher(
        PRE_TOPIC,
        PoseArray,
        queue_size=1,
        latch=True,
    )

    grasp_pub = rospy.Publisher(
        GRASP_TOPIC,
        PoseArray,
        queue_size=1,
        latch=True,
    )

    metadata_pub = rospy.Publisher(
        METADATA_TOPIC,
        String,
        queue_size=1,
        latch=True,
    )

    marker_pub = rospy.Publisher(
        MARKER_TOPIC,
        MarkerArray,
        queue_size=1,
        latch=True,
    )

    print()
    print(
        "direction count = {}".format(
            direction_id
        )
    )

    print(
        "candidate count = {}".format(
            candidate_id
        )
    )

    edge_counts = {}

    for c in metadata[::2]:
        edge = c["entry_edge"]

        edge_counts[edge] = (
            edge_counts.get(
                edge,
                0,
            )
            + 1
        )

    print()
    print(
        "entry edge distribution:"
    )

    for edge in sorted(
        edge_counts
    ):
        print(
            "  edge {} : {} directions"
            .format(
                edge,
                edge_counts[edge],
            )
        )

    print()
    print(
        "Publishing:"
    )
    print(
        "  {}".format(
            PRE_TOPIC
        )
    )
    print(
        "  {}".format(
            GRASP_TOPIC
        )
    )
    print(
        "  {}".format(
            METADATA_TOPIC
        )
    )
    print(
        "  {}".format(
            MARKER_TOPIC
        )
    )

    rate = rospy.Rate(1.0)

    while not rospy.is_shutdown():
        now = rospy.Time.now()

        pre_array.header.stamp = now
        grasp_array.header.stamp = now

        for marker in (
            markers.markers
        ):
            marker.header.stamp = now

        pre_pub.publish(
            pre_array
        )

        grasp_pub.publish(
            grasp_array
        )

        metadata_pub.publish(
            metadata_msg
        )

        marker_pub.publish(
            markers
        )

        rate.sleep()


if __name__ == "__main__":
    main()
