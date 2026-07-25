#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Point, PointStamped


publisher = None
output_frame = "paper_center"


def grasp_point_callback(unity_point):
    """
    Unity上の紙中心基準座標を、
    paper_center基準のROS座標へ変換する。

    Unity:
        x = 右
        y = 上
        z = 前

    ROS / paper_center:
        x = 前
        y = 左
        z = 上
    """

    ros_point = PointStamped()

    ros_point.header.stamp = rospy.Time.now()
    ros_point.header.frame_id = output_frame

    # 軸変換のみを行う。
    # 紙中心の位置・向きはpaper_centerのTFで表現する。
    ros_point.point.x = unity_point.z
    ros_point.point.y = -unity_point.x
    ros_point.point.z = unity_point.y

    publisher.publish(ros_point)


def main():
    global publisher
    global output_frame

    rospy.init_node("unity_to_ros_point")

    input_topic = rospy.get_param(
        "~input_topic",
        "/origami/grasp_point_unity"
    )

    output_topic = rospy.get_param(
        "~output_topic",
        "/origami/grasp_point_ros"
    )

    output_frame = rospy.get_param(
        "~output_frame",
        "paper_center"
    )

    publisher = rospy.Publisher(
        output_topic,
        PointStamped,
        queue_size=10
    )

    rospy.Subscriber(
        input_topic,
        Point,
        grasp_point_callback,
        queue_size=10
    )

    rospy.loginfo("把持点座標変換ノードを開始しました。")
    rospy.loginfo("入力: %s", input_topic)
    rospy.loginfo("出力: %s", output_topic)
    rospy.loginfo("出力基準座標: %s", output_frame)
    rospy.loginfo(
        "Unity原点は%sの原点として出力されます。",
        output_frame
    )

    rospy.spin()


if __name__ == "__main__":
    main()
