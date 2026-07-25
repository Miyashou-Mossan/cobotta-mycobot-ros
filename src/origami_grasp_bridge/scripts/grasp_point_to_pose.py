#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PointStamped, PoseStamped


class GraspPointToPose:
    def __init__(self):
        self.publisher = rospy.Publisher(
            "/origami/grasp_pose",
            PoseStamped,
            queue_size=10
        )

        rospy.Subscriber(
            "/origami/grasp_point_ros",
            PointStamped,
            self.point_callback,
            queue_size=1
        )

        # 仮の工具姿勢。
        # position_only計画では使用しない。
        self.orientation_x = rospy.get_param(
            "~orientation_x",
            -0.636
        )
        self.orientation_y = rospy.get_param(
            "~orientation_y",
            0.0
        )
        self.orientation_z = rospy.get_param(
            "~orientation_z",
            0.772
        )
        self.orientation_w = rospy.get_param(
            "~orientation_w",
            0.0
        )

        rospy.loginfo("grasp_point_to_pose node started")
        rospy.loginfo(
            "Input : /origami/grasp_point_ros [PointStamped]"
        )
        rospy.loginfo(
            "Output: /origami/grasp_pose [PoseStamped]"
        )
        rospy.logwarn(
            "Tool orientation is provisional. "
            "Use position_only planning until orientation is verified."
        )

    def point_callback(self, point_message):
        pose = PoseStamped()

        # 入力された基準座標をそのまま引き継ぐ
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = point_message.header.frame_id

        pose.pose.position.x = point_message.point.x
        pose.pose.position.y = point_message.point.y
        pose.pose.position.z = point_message.point.z

        # 仮姿勢。position_only計画では使用されない
        pose.pose.orientation.x = self.orientation_x
        pose.pose.orientation.y = self.orientation_y
        pose.pose.orientation.z = self.orientation_z
        pose.pose.orientation.w = self.orientation_w

        self.publisher.publish(pose)

        rospy.loginfo_throttle(
            1.0,
            "Pose: frame=%s, x=%.4f, y=%.4f, z=%.4f",
            pose.header.frame_id,
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z
        )


if __name__ == "__main__":
    rospy.init_node("grasp_point_to_pose")

    GraspPointToPose()
    rospy.spin()
