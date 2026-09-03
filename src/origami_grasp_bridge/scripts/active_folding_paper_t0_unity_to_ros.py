#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PolygonStamped, Point32


class ActiveFoldingPaperT0UnityToRos:
    def __init__(self):
        self.input_topic = rospy.get_param(
            "~input_topic",
            "/origami/active_folding_paper_t0_unity"
        )

        self.output_topic = rospy.get_param(
            "~output_topic",
            "/origami/active_folding_paper_t0_ros"
        )

        self.output_frame = rospy.get_param(
            "~output_frame",
            "paper_center"
        )

        self.publisher = rospy.Publisher(
            self.output_topic,
            PolygonStamped,
            queue_size=1
        )

        self.subscriber = rospy.Subscriber(
            self.input_topic,
            PolygonStamped,
            self.callback,
            queue_size=1
        )

        rospy.loginfo(
            "T0 ActiveFoldingPaper Unity->ROS変換ノードを開始しました。"
        )
        rospy.loginfo("入力 : %s", self.input_topic)
        rospy.loginfo("出力 : %s", self.output_topic)
        rospy.loginfo("frame: %s", self.output_frame)

    def callback(self, unity_msg):
        ros_msg = PolygonStamped()

        ros_msg.header.stamp = rospy.Time.now()
        ros_msg.header.frame_id = self.output_frame

        for p in unity_msg.polygon.points:
            ros_point = Point32()

            # Unity RUF -> ROS FLU
            #
            # ROS paper_center +X = Unity +Z
            # ROS paper_center +Y = Unity -X
            # ROS paper_center +Z = Unity +Y
            ros_point.x = p.z
            ros_point.y = -p.x
            ros_point.z = p.y

            ros_msg.polygon.points.append(
                ros_point
            )

        self.publisher.publish(ros_msg)

        rospy.loginfo(
            "T0紙面を変換してPublishしました: vertexCount=%d",
            len(ros_msg.polygon.points)
        )


def main():
    rospy.init_node(
        "active_folding_paper_t0_unity_to_ros"
    )

    ActiveFoldingPaperT0UnityToRos()

    rospy.spin()


if __name__ == "__main__":
    main()
