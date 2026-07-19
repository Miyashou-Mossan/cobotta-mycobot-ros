#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Point


publisher = None

paper_center_x = 0.0
paper_center_y = 0.0
paper_center_z = 0.0


def grasp_point_callback(unity_point):
    """
    Unity上の紙中心基準の座標を、
    mycobot_base基準のROS座標へ変換する。

    Unity:
        x = 右
        y = 上
        z = 前

    ROS / mycobot_base:
        x = 前
        y = 左
        z = 上
    """

    ros_point = Point()

    # 軸変換 + 紙中心位置のオフセット
    ros_point.x = paper_center_x + unity_point.z
    ros_point.y = paper_center_y - unity_point.x
    ros_point.z = paper_center_z + unity_point.y

    publisher.publish(ros_point)


def main():
    global publisher
    global paper_center_x
    global paper_center_y
    global paper_center_z

    rospy.init_node("unity_to_ros_point")

    input_topic = rospy.get_param(
        "~input_topic",
        "/origami/grasp_point_unity"
    )

    output_topic = rospy.get_param(
        "~output_topic",
        "/origami/grasp_point_ros"
    )

    # 紙中心のmycobot_base基準座標
    paper_center_x = rospy.get_param("~paper_center_x", 0.25)
    paper_center_y = rospy.get_param("~paper_center_y", 0.00)

    # 高さはまだ未確定なので、現段階では仮に0.0
    paper_center_z = rospy.get_param("~paper_center_z", 0.00)

    publisher = rospy.Publisher(
        output_topic,
        Point,
        queue_size=10
    )

    rospy.Subscriber(
        input_topic,
        Point,
        grasp_point_callback,
        queue_size=10
    )

    rospy.loginfo("把持点座標変換ノードを開始しました。")
    rospy.loginfo("入力: %s  出力: %s", input_topic, output_topic)
    rospy.loginfo(
        "紙中心: x=%.3f, y=%.3f, z=%.3f [m]",
        paper_center_x,
        paper_center_y,
        paper_center_z
    )

    rospy.spin()


if __name__ == "__main__":
    main()
