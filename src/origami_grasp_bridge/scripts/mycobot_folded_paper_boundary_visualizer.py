#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


def main():
    rospy.init_node("mycobot_folded_paper_boundary_visualizer")

    pub = rospy.Publisher(
        "/mycobot/folded_paper_boundary",
        Marker,
        queue_size=1,
        latch=True
    )

    rospy.sleep(1.0)

    marker = Marker()
    marker.header.frame_id = "paper_center"
    marker.header.stamp = rospy.Time.now()

    marker.ns = "folded_paper_boundary"
    marker.id = 0
    marker.type = Marker.TRIANGLE_LIST
    marker.action = Marker.ADD

    marker.scale.x = 1.0
    marker.scale.y = 1.0
    marker.scale.z = 1.0

    marker.color.r = 1.0
    marker.color.g = 0.3
    marker.color.b = 0.3
    marker.color.a = 0.45

    # paper_center基準
    # 150x150 mm紙の対角頂点を結ぶ折り筋
    sx, sy = -0.075, -0.075
    ex, ey =  0.075,  0.075

    # 境界面の高さ
    h = 0.120

    p1 = Point()
    p1.x = sx
    p1.y = sy
    p1.z = 0.0

    p2 = Point()
    p2.x = ex
    p2.y = ey
    p2.z = 0.0

    p3 = Point()
    p3.x = ex
    p3.y = ey
    p3.z = h

    p4 = Point()
    p4.x = sx
    p4.y = sy
    p4.z = h

    # 折り筋から鉛直に立つ境界面
    marker.points = [
        p1, p2, p3,
        p1, p3, p4
    ]

    pub.publish(marker)

    rospy.loginfo("Published vertical crease boundary plane")
    rospy.loginfo(
        "crease start = ({:.3f}, {:.3f}, 0.000)".format(sx, sy)
    )
    rospy.loginfo(
        "crease end   = ({:.3f}, {:.3f}, 0.000)".format(ex, ey)
    )

    rospy.spin()


if __name__ == "__main__":
    main()
