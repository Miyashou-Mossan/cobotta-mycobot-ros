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

        # 仮の工具姿勢。修正版テスト6で紙保持姿勢を検証する
        self.orientation_x = rospy.get_param("~orientation_x", -0.636)
        self.orientation_y = rospy.get_param("~orientation_y", 0.0)
        self.orientation_z = rospy.get_param("~orientation_z", 0.772)
        self.orientation_w = rospy.get_param("~orientation_w", 0.0)

        rospy.loginfo("grasp_point_to_pose node started")
        rospy.loginfo("Input : /origami/grasp_point_ros")
        rospy.loginfo("Output: /origami/grasp_pose")
        rospy.logwarn(
            "Tool orientation is provisional: "
            "(%.4f, %.4f, %.4f, %.4f)",
            self.orientation_x,
            self.orientation_y,
            self.orientation_z,
            self.orientation_w,
        )

    def point_callback(self, point):
        pose = PoseStamped()

        pose.header.stamp = rospy.Time.now()

        # grasp_point_rosはmycobot_base基準として扱う
        pose.header.frame_id = "mycobot_base"

        pose.pose.position.x = point.x
        pose.pose.position.y = point.y
        pose.pose.position.z = point.z

        # 仮姿勢。position_only計画では使用されない
        # pose計画へ進む前に修正版テスト6で検証する
        pose.pose.orientation.x = self.orientation_x
        pose.pose.orientation.y = self.orientation_y
        pose.pose.orientation.z = self.orientation_z
        pose.pose.orientation.w = self.orientation_w

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
