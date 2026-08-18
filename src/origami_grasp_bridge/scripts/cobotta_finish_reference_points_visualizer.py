#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
import yaml
import rospy

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray


DEFAULT_YAML = os.path.expanduser(
    "~/directionA_reverse_local_z_m25_multidof.yaml"
)

FRAME = "paper_center"
TOPIC = "/origami/debug/cobotta_finish_reference_points"

PAPER_HALF = 0.075  # 150 mm paper


def load_position(points, index):
    tf = points[index]["transforms"][0]
    t = tf["translation"]

    return [
        float(t["x"]),
        float(t["y"]),
        float(t["z"]),
    ]


def distance_mm(a, b):
    return 1000.0 * math.sqrt(
        sum((x - y) ** 2 for x, y in zip(a, b))
    )


def make_sphere(marker_id, position, scale, r, g, b):
    m = Marker()
    m.header.frame_id = FRAME
    m.header.stamp = rospy.Time(0)

    m.ns = "finish_reference_points"
    m.id = marker_id
    m.type = Marker.SPHERE
    m.action = Marker.ADD

    m.pose.position.x = position[0]
    m.pose.position.y = position[1]
    m.pose.position.z = position[2]
    m.pose.orientation.w = 1.0

    m.scale.x = scale
    m.scale.y = scale
    m.scale.z = scale

    m.color.r = r
    m.color.g = g
    m.color.b = b
    m.color.a = 1.0

    return m


def make_text(marker_id, position, text, r, g, b):
    m = Marker()
    m.header.frame_id = FRAME
    m.header.stamp = rospy.Time(0)

    m.ns = "finish_reference_text"
    m.id = marker_id
    m.type = Marker.TEXT_VIEW_FACING
    m.action = Marker.ADD

    m.pose.position.x = position[0]
    m.pose.position.y = position[1]
    m.pose.position.z = position[2] + 0.012
    m.pose.orientation.w = 1.0

    m.scale.z = 0.009

    m.color.r = r
    m.color.g = g
    m.color.b = b
    m.color.a = 1.0

    m.text = text

    return m


def make_line(marker_id, points):
    m = Marker()
    m.header.frame_id = FRAME
    m.header.stamp = rospy.Time(0)

    m.ns = "finish_reference_line"
    m.id = marker_id
    m.type = Marker.LINE_STRIP
    m.action = Marker.ADD

    m.scale.x = 0.0015

    m.color.r = 1.0
    m.color.g = 1.0
    m.color.b = 1.0
    m.color.a = 0.8

    for xyz in points:
        p = Point()
        p.x = xyz[0]
        p.y = xyz[1]
        p.z = xyz[2]
        m.points.append(p)

    return m


def main():
    rospy.init_node(
        "cobotta_finish_reference_points_visualizer"
    )

    yaml_path = rospy.get_param(
        "~yaml",
        DEFAULT_YAML
    )

    current_index = int(
        rospy.get_param(
            "~index",
            340
        )
    )

    with open(
        yaml_path,
        "r",
        encoding="utf-8"
    ) as f:
        doc = yaml.safe_load(f)

    points = doc["points"]

    if current_index >= len(points):
        raise RuntimeError(
            "index {} exceeds trajectory length {}"
            .format(current_index, len(points))
        )

    # Reverse trajectory:
    # index 399 = Unity geometric full-fold GOAL
    unity_goal_index = len(points) - 1

    unity_goal = load_position(
        points,
        unity_goal_index
    )

    current_finish = load_position(
        points,
        current_index
    )

    # paper_center frameでの4隅
    paper_corners = [
        [-PAPER_HALF, -PAPER_HALF, 0.0],
        [-PAPER_HALF, +PAPER_HALF, 0.0],
        [+PAPER_HALF, -PAPER_HALF, 0.0],
        [+PAPER_HALF, +PAPER_HALF, 0.0],
    ]

    # Unity GOALに最も近い紙角を自動選択
    paper_corner = min(
        paper_corners,
        key=lambda c:
            (c[0] - unity_goal[0]) ** 2
            + (c[1] - unity_goal[1]) ** 2
    )

    pub = rospy.Publisher(
        TOPIC,
        MarkerArray,
        queue_size=1,
        latch=True
    )

    rospy.sleep(1.0)

    msg = MarkerArray()

    # 赤: 紙角
    msg.markers.append(
        make_sphere(
            0,
            paper_corner,
            0.010,
            1.0, 0.1, 0.1
        )
    )
    msg.markers.append(
        make_text(
            10,
            paper_corner,
            "PAPER CORNER",
            1.0, 0.1, 0.1
        )
    )

    # 青: Unity完全GOAL
    msg.markers.append(
        make_sphere(
            1,
            unity_goal,
            0.009,
            0.1, 0.3, 1.0
        )
    )
    msg.markers.append(
        make_text(
            11,
            unity_goal,
            "UNITY GOAL (index {})".format(
                unity_goal_index
            ),
            0.1, 0.3, 1.0
        )
    )

    # 黄: 現在のindex 340
    msg.markers.append(
        make_sphere(
            2,
            current_finish,
            0.008,
            1.0, 0.8, 0.1
        )
    )
    msg.markers.append(
        make_text(
            12,
            current_finish,
            "CURRENT index {}".format(
                current_index
            ),
            1.0, 0.8, 0.1
        )
    )

    # 位置関係を白線で接続
    msg.markers.append(
        make_line(
            20,
            [
                paper_corner,
                unity_goal,
                current_finish
            ]
        )
    )

    pub.publish(msg)

    print("===== FINISH reference points =====")
    print("frame:", FRAME)
    print()

    print(
        "PAPER CORNER : "
        "({:+.3f}, {:+.3f}, {:+.3f}) mm"
        .format(
            paper_corner[0] * 1000.0,
            paper_corner[1] * 1000.0,
            paper_corner[2] * 1000.0
        )
    )

    print(
        "UNITY GOAL   : "
        "({:+.3f}, {:+.3f}, {:+.3f}) mm "
        "[index {}]"
        .format(
            unity_goal[0] * 1000.0,
            unity_goal[1] * 1000.0,
            unity_goal[2] * 1000.0,
            unity_goal_index
        )
    )

    print(
        "CURRENT      : "
        "({:+.3f}, {:+.3f}, {:+.3f}) mm "
        "[index {}]"
        .format(
            current_finish[0] * 1000.0,
            current_finish[1] * 1000.0,
            current_finish[2] * 1000.0,
            current_index
        )
    )

    print()
    print(
        "corner -> Unity GOAL : {:.3f} mm"
        .format(
            distance_mm(
                paper_corner,
                unity_goal
            )
        )
    )

    print(
        "Unity GOAL -> current: {:.3f} mm"
        .format(
            distance_mm(
                unity_goal,
                current_finish
            )
        )
    )

    print(
        "corner -> current    : {:.3f} mm"
        .format(
            distance_mm(
                paper_corner,
                current_finish
            )
        )
    )

    print()
    print("topic:", TOPIC)

    rospy.spin()


if __name__ == "__main__":
    main()
