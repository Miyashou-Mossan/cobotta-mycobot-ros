#!/usr/bin/env python3

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


def main():
    rospy.init_node("mycobot_foldline_visualizer")

    pub = rospy.Publisher(
        "/origami/fold_line_marker",
        Marker,
        queue_size=1,
        latch=True
    )

    rospy.sleep(1.0)

    marker = Marker()
    marker.header.frame_id = "paper_center"
    marker.header.stamp = rospy.Time.now()

    marker.ns = "fold_line"
    marker.id = 0
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD

    marker.scale.x = 0.003

    marker.color.r = 1.0
    marker.color.g = 0.2
    marker.color.b = 0.2
    marker.color.a = 1.0

    # 仮の折り筋
    # paper_center を中心に X方向へ100 mm
    p1 = Point()
    p1.x = -0.075
    p1.y = 0.0
    p1.z = 0.003

    p2 = Point()
    p2.x = 0.075
    p2.y = 0.0
    p2.z = 0.003

    marker.points = [p1, p2]

    pub.publish(marker)

    rospy.loginfo("Published temporary fold line.")
    rospy.loginfo(
        "start = (%.3f, %.3f, %.3f)",
        p1.x, p1.y, p1.z
    )
    rospy.loginfo(
        "end   = (%.3f, %.3f, %.3f)",
        p2.x, p2.y, p2.z
    )

    rospy.spin()


if __name__ == "__main__":
    main()
