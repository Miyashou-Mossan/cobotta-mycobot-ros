#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import rospy

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker


CSV_PATH = "/home/maeda/20230214OrigamiSim/Assets/folding_face_90.csv"


def unity_to_ros(x_u, y_u, z_u):
    """
    Unity -> ROS (paper_center basis)

    x_ros =  z_unity
    y_ros = -x_unity
    z_ros =  y_unity
    """
    return z_u, -x_u, y_u


def load_points_from_csv(path):
    points = []

    with open(path, "r") as f:
        rows = [
            line for line in f
            if not line.startswith("#")
        ]

    reader = csv.DictReader(rows)

    for row in reader:
        x_u = float(row["x_m"])
        y_u = float(row["y_m"])
        z_u = float(row["z_m"])

        x_r, y_r, z_r = unity_to_ros(
            x_u,
            y_u,
            z_u
        )

        points.append(
            (x_r, y_r, z_r)
        )

    return points


def main():
    rospy.init_node(
        "folding_face_90_csv_visualizer"
    )

    pub = rospy.Publisher(
        "/origami/folding_face_90_marker",
        Marker,
        queue_size=1,
        latch=True
    )

    points = load_points_from_csv(
        CSV_PATH
    )

    if len(points) != 3:
        rospy.logerr(
            "Expected 3 points, got %d",
            len(points)
        )
        return

    rospy.loginfo(
        "Loaded 90deg folding face:"
    )

    for i, p in enumerate(points):
        rospy.loginfo(
            "P%d = (%.6f, %.6f, %.6f)",
            i,
            p[0],
            p[1],
            p[2]
        )

    marker = Marker()
    marker.header.frame_id = "paper_center"
    marker.header.stamp = rospy.Time.now()

    marker.ns = "folding_face_90"
    marker.id = 0
    marker.type = Marker.TRIANGLE_LIST
    marker.action = Marker.ADD

    marker.scale.x = 1.0
    marker.scale.y = 1.0
    marker.scale.z = 1.0

    marker.color.r = 1.0
    marker.color.g = 0.2
    marker.color.b = 0.2
    marker.color.a = 0.45

    p0 = Point()
    p0.x, p0.y, p0.z = points[0]

    p1 = Point()
    p1.x, p1.y, p1.z = points[1]

    p2 = Point()
    p2.x, p2.y, p2.z = points[2]

    marker.points = [
        p0,
        p1,
        p2
    ]

    pub.publish(marker)

    rospy.loginfo(
        "Published marker: "
        "/origami/folding_face_90_marker"
    )

    rospy.spin()


if __name__ == "__main__":
    main()
