#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Point, PoseStamped


class GraspPointToPose:
    def __init__(self):
        self.publisher = rospy.Publisher(
            "/origami/grasp_pose",
            PoseStamped,
            queue_size=10
        )

        rospy.Subscriber(
            "/origami/grasp_point_ros",
            Point,
            self.point_callback,
            queue_size=1
        )

        rospy.loginfo("grasp_point_to_pose node started")
        rospy.loginfo("Input : /origami/grasp_point_ros")
        rospy.loginfo("Output: /origami/grasp_pose")

    def point_callback(self, point):
        pose = PoseStamped()

        pose.header.stamp = rospy.Time.now()

        # grasp_point_rosはmycobot_base基準として扱う
        pose.header.frame_id = "mycobot_base"

        pose.pose.position.x = point.x
        pose.pose.position.y = point.y
        pose.pose.position.z = point.z

        # 仮の姿勢：回転なし
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 1.0

        self.publisher.publish(pose)

        rospy.loginfo_throttle(
            1.0,
            "Pose: x=%.4f, y=%.4f, z=%.4f",
            point.x,
            point.y,
            point.z
        )


if __name__ == "__main__":
    rospy.init_node("grasp_point_to_pose")

    GraspPointToPose()

    rospy.spin()
