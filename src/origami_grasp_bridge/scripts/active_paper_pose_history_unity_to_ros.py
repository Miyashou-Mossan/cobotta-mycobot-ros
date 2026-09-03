#!/usr/bin/env python3

import math

import rospy
from geometry_msgs.msg import Pose, PoseArray


class ActivePaperPoseHistoryUnityToRos:
    def __init__(self):
        self.input_topic = rospy.get_param(
            "~input_topic",
            "/origami/active_paper_pose_history_unity"
        )

        self.output_topic = rospy.get_param(
            "~output_topic",
            "/origami/active_paper_pose_history_ros"
        )

        self.output_frame = rospy.get_param(
            "~output_frame",
            "paper_center"
        )

        self.publisher = rospy.Publisher(
            self.output_topic,
            PoseArray,
            queue_size=1
        )

        self.subscriber = rospy.Subscriber(
            self.input_topic,
            PoseArray,
            self.callback,
            queue_size=1
        )

        rospy.loginfo("Active paper Pose履歴 Unity->ROS 変換ノードを開始しました。")
        rospy.loginfo("入力 : %s", self.input_topic)
        rospy.loginfo("出力 : %s", self.output_topic)
        rospy.loginfo("frame: %s", self.output_frame)

    @staticmethod
    def convert_pose(unity_pose):
        """
        Unity RUF:
            x = Right
            y = Up
            z = Forward

        ROS FLU / paper_center:
            x = Forward
            y = Left
            z = Up

        Unity World -> ROS paper_center は、
        追加Yaw補正を行わずRUF -> FLUのみを適用する。
        """

        ros_pose = Pose()

        # Position: Unity RUF -> ROS FLU
        ros_pose.position.x = unity_pose.position.z
        ros_pose.position.y = -unity_pose.position.x
        ros_pose.position.z = unity_pose.position.y

        # Quaternion: Unity RUF -> ROS FLU
        qx = unity_pose.orientation.z
        qy = -unity_pose.orientation.x
        qz = unity_pose.orientation.y
        qw = -unity_pose.orientation.w

        # 数値誤差に備えて正規化
        norm = math.sqrt(
            qx * qx +
            qy * qy +
            qz * qz +
            qw * qw
        )

        if norm < 1.0e-12:
            raise ValueError("Quaternion norm is zero.")

        ros_pose.orientation.x = qx / norm
        ros_pose.orientation.y = qy / norm
        ros_pose.orientation.z = qz / norm
        ros_pose.orientation.w = qw / norm

        return ros_pose

    def callback(self, unity_msg):
        ros_msg = PoseArray()

        ros_msg.header.stamp = rospy.Time.now()
        ros_msg.header.frame_id = self.output_frame

        try:
            converted = [
                self.convert_pose(pose)
                for pose in unity_msg.poses
            ]
        except ValueError as exc:
            rospy.logerr(
                "Pose履歴の変換に失敗しました: %s",
                str(exc)
            )
            return

        # Quaternionの符号を時系列で連続化する。
        # q と -q は同じ姿勢なので、物理姿勢は変化しない。
        for i in range(1, len(converted)):
            q_prev = converted[i - 1].orientation
            q_curr = converted[i].orientation

            dot = (
                q_prev.x * q_curr.x +
                q_prev.y * q_curr.y +
                q_prev.z * q_curr.z +
                q_prev.w * q_curr.w
            )

            if dot < 0.0:
                q_curr.x *= -1.0
                q_curr.y *= -1.0
                q_curr.z *= -1.0
                q_curr.w *= -1.0

        ros_msg.poses = converted

        self.publisher.publish(ros_msg)

        rospy.loginfo(
            "Pose履歴を変換してPublishしました: poseCount=%d",
            len(ros_msg.poses)
        )


def main():
    rospy.init_node(
        "active_paper_pose_history_unity_to_ros"
    )

    ActivePaperPoseHistoryUnityToRos()

    rospy.spin()


if __name__ == "__main__":
    main()
