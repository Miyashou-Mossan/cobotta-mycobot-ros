#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import rospy
import tf
import numpy as np

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker


CSV_PATH = "/home/maeda/20230214OrigamiSim/Assets/folding_face_90_base_triangle.csv"


def unity_to_ros(x_u, y_u, z_u):
    """
    Unity -> ROS (paper_center)
    x_ros =  z_unity
    y_ros = -x_unity
    z_ros =  y_unity
    """
    return np.array([
        z_u,
        -x_u,
        y_u
    ], dtype=float)


def load_face_points():
    rows = []

    with open(CSV_PATH, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            rows.append(line)

    reader = csv.DictReader(rows)

    points = []

    for row in reader:
        points.append(
            unity_to_ros(
                float(row["x_m"]),
                float(row["y_m"]),
                float(row["z_m"])
            )
        )

    if len(points) != 3:
        raise RuntimeError(
            "Expected 3 points, got {}".format(
                len(points)
            )
        )

    return points


def ros_point(p):
    q = Point()
    q.x = float(p[0])
    q.y = float(p[1])
    q.z = float(p[2])
    return q


def main():
    rospy.init_node(
        "fourfold_virtual_face_visualizer"
    )

    listener = tf.TransformListener()

    pub = rospy.Publisher(
        "/origami/fourfold_half_face_marker",
        Marker,
        queue_size=1,
        latch=True
    )

    rospy.sleep(1.0)

    # Unity由来90°三角形
    p0, p1, p2 = load_face_points()

    # 折り筋P0-P1の中央
    midpoint = (p0 + p1) * 0.5

    # COBOTTA位置をpaper_center基準で取得
    listener.waitForTransform(
        "paper_center",
        "cobotta_base_link",
        rospy.Time(0),
        rospy.Duration(3.0)
    )

    cobotta, _ = listener.lookupTransform(
        "paper_center",
        "cobotta_base_link",
        rospy.Time(0)
    )

    cobotta = np.array(cobotta, dtype=float)

    # P0/P1のうちCOBOTTAに近い側を残す
    d0 = np.linalg.norm(
        p0[:2] - cobotta[:2]
    )

    d1 = np.linalg.norm(
        p1[:2] - cobotta[:2]
    )

    if d0 <= d1:
        keep = p0
        keep_name = "P0"
    else:
        keep = p1
        keep_name = "P1"

    marker = Marker()

    marker.header.frame_id = "paper_center"
    marker.header.stamp = rospy.Time.now()

    marker.ns = "fourfold_half_face"
    marker.id = 0

    marker.type = Marker.TRIANGLE_LIST
    marker.action = Marker.ADD

    marker.scale.x = 1.0
    marker.scale.y = 1.0
    marker.scale.z = 1.0

    # 元三角形を半分にして
    # COBOTTA側だけ残した三角形
    marker.points = [
        ros_point(keep),
        ros_point(midpoint),
        ros_point(p2)
    ]

    marker.color.r = 0.0
    marker.color.g = 0.8
    marker.color.b = 1.0
    marker.color.a = 0.55

    pub.publish(marker)

    rospy.loginfo(
        "Published half of Unity 90deg triangle"
    )

    rospy.loginfo(
        "COBOTTA-side endpoint = %s",
        keep_name
    )

    rospy.loginfo(
        "distance P0 -> COBOTTA = %.1f mm",
        d0 * 1000.0
    )

    rospy.loginfo(
        "distance P1 -> COBOTTA = %.1f mm",
        d1 * 1000.0
    )

    rospy.loginfo(
        "kept triangle:"
    )

    for name, p in [
        (keep_name, keep),
        ("MID", midpoint),
        ("P2", p2)
    ]:
        rospy.loginfo(
            "%s = (%.6f, %.6f, %.6f)",
            name,
            p[0],
            p[1],
            p[2]
        )

    rospy.spin()


if __name__ == "__main__":
    main()
