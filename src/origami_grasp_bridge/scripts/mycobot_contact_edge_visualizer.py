#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


def main():
    rospy.init_node("mycobot_contact_edge_visualizer")

    pub = rospy.Publisher(
        "/mycobot/contact_edge_marker",
        Marker,
        queue_size=1,
        latch=True
    )

    rospy.sleep(1.0)

    marker = Marker()
    marker.header.frame_id = "mycobot_tool_link"
    marker.header.stamp = rospy.Time.now()

    marker.ns = "mycobot_contact_edge"
    marker.id = 0
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD

    marker.scale.x = 0.002  # 2 mmの太さ

    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 1.0

    p1 = Point()
    p1.x = 0.000
    p1.y = -0.014
    p1.z = 0.043

    p2 = Point()
    p2.x = 0.000
    p2.y = 0.014
    p2.z = 0.043

    marker.points = [p1, p2]

    pub.publish(marker)

    rospy.loginfo("Published MyCobot contact edge marker")
    rospy.loginfo("start = (0.000, -0.014, 0.043)")
    rospy.loginfo("end   = (0.000, +0.014, 0.043)")

    rospy.spin()


if __name__ == "__main__":
    main()
