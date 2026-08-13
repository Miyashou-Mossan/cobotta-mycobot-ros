#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rospy
import numpy as np

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker


# STL -> mycobot_tool_link
T = np.array([
    -0.018,
    -0.018,
    +0.043
])

ROLL = -math.pi / 2.0

RX = np.array([
    [1.0, 0.0, 0.0],
    [0.0, math.cos(ROLL), -math.sin(ROLL)],
    [0.0, math.sin(ROLL),  math.cos(ROLL)]
])


def stl_to_tool(x_mm, y_mm, z_mm):
    p_stl = np.array([
        x_mm,
        y_mm,
        z_mm
    ]) / 1000.0

    return T + RX.dot(p_stl)


def ros_point(p):
    q = Point()
    q.x = float(p[0])
    q.y = float(p[1])
    q.z = float(p[2])
    return q


def main():
    rospy.init_node(
        "mycobot_allowed_contact_region_visualizer"
    )

    region_pub = rospy.Publisher(
        "/origami/mycobot_allowed_contact_region",
        Marker,
        queue_size=1,
        latch=True
    )

    edge_pub = rospy.Publisher(
        "/origami/mycobot_contact_edge",
        Marker,
        queue_size=1,
        latch=True
    )

    rospy.sleep(1.0)

    # ----------------------------------------
    # 接触辺
    # STL:
    # A = (18, 0, 4)
    # B = (18, 0, 32)
    # ----------------------------------------

    A = stl_to_tool(
        18.0, 0.0, 4.0
    )

    B = stl_to_tool(
        18.0, 0.0, 32.0
    )

    # ----------------------------------------
    # 斜面先端側50%
    #
    # 全斜面：
    # y = 0 -> 11.18 mm
    #
    # 50%：
    # y = 0 -> 5.59 mm
    #
    # +X側 midpoint x=23
    # -X側 midpoint x=13
    # ----------------------------------------

    plus_bottom = stl_to_tool(
        23.0, 5.59, 4.0
    )

    plus_top = stl_to_tool(
        23.0, 5.59, 32.0
    )

    minus_bottom = stl_to_tool(
        13.0, 5.59, 4.0
    )

    minus_top = stl_to_tool(
        13.0, 5.59, 32.0
    )

    # ----------------------------------------
    # 許容接触領域
    # 左右2つの四角形を三角形4枚で表示
    # ----------------------------------------

    marker = Marker()

    marker.header.frame_id = "mycobot_tool_link"
    marker.header.stamp = rospy.Time.now()

    marker.ns = "allowed_contact_region"
    marker.id = 0

    marker.type = Marker.TRIANGLE_LIST
    marker.action = Marker.ADD

    marker.scale.x = 1.0
    marker.scale.y = 1.0
    marker.scale.z = 1.0

    marker.color.r = 0.1
    marker.color.g = 1.0
    marker.color.b = 0.2
    marker.color.a = 0.65

    marker.points = [
        # +X側
        ros_point(A),
        ros_point(B),
        ros_point(plus_top),

        ros_point(A),
        ros_point(plus_top),
        ros_point(plus_bottom),

        # -X側
        ros_point(A),
        ros_point(minus_top),
        ros_point(B),

        ros_point(A),
        ros_point(minus_bottom),
        ros_point(minus_top),
    ]

    region_pub.publish(marker)

    # ----------------------------------------
    # 接触辺A-B
    # ----------------------------------------

    edge = Marker()

    edge.header.frame_id = "mycobot_tool_link"
    edge.header.stamp = rospy.Time.now()

    edge.ns = "contact_edge"
    edge.id = 0

    edge.type = Marker.LINE_LIST
    edge.action = Marker.ADD

    edge.scale.x = 0.0015

    edge.color.r = 1.0
    edge.color.g = 0.1
    edge.color.b = 0.1
    edge.color.a = 1.0

    edge.points = [
        ros_point(A),
        ros_point(B)
    ]

    edge_pub.publish(edge)

    print("")
    print("===== Allowed contact region =====")
    print("sloped surface depth : 15.0 mm")
    print("allowed fraction     : 50 %")
    print("allowed depth        : 7.5 mm")

    print("")
    print("A =", A)
    print("B =", B)

    print("")
    print("+X half boundary:")
    print(" bottom =", plus_bottom)
    print(" top    =", plus_top)

    print("")
    print("-X half boundary:")
    print(" bottom =", minus_bottom)
    print(" top    =", minus_top)

    rospy.spin()


if __name__ == "__main__":
    main()
