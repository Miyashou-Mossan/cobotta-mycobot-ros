#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import rospy

from interactive_markers.interactive_marker_server import (
    InteractiveMarkerServer,
)
from visualization_msgs.msg import (
    InteractiveMarker,
    InteractiveMarkerControl,
    InteractiveMarkerFeedback,
    Marker,
)


FRAME_ID = "cobotta_gripper_base"

# 現在の仮TCPを初期位置にする
INITIAL_X = -0.002
INITIAL_Y = 0.000
INITIAL_Z = 0.080


def make_axis_control(name, qx, qy, qz, qw):
    control = InteractiveMarkerControl()
    control.name = name

    control.orientation.x = qx
    control.orientation.y = qy
    control.orientation.z = qz
    control.orientation.w = qw

    control.interaction_mode = (
        InteractiveMarkerControl.MOVE_AXIS
    )

    return control


def feedback_cb(feedback):
    p = feedback.pose.position

    # ドラッグを離した瞬間だけ最終値を表示
    if (
        feedback.event_type
        == InteractiveMarkerFeedback.MOUSE_UP
    ):
        rospy.loginfo(
            "\n"
            "===== actual_grasp_point candidate =====\n"
            "frame : %s\n"
            "xyz   : (%.6f, %.6f, %.6f) m\n"
            "xyz   : (%.3f, %.3f, %.3f) mm",
            feedback.header.frame_id,
            p.x,
            p.y,
            p.z,
            p.x * 1000.0,
            p.y * 1000.0,
            p.z * 1000.0,
        )

        rospy.set_param(
            "/cobotta_actual_grasp_point/x",
            p.x,
        )
        rospy.set_param(
            "/cobotta_actual_grasp_point/y",
            p.y,
        )
        rospy.set_param(
            "/cobotta_actual_grasp_point/z",
            p.z,
        )


def main():
    rospy.init_node(
        "cobotta_actual_grasp_point_visualizer"
    )

    server = InteractiveMarkerServer(
        "origami/debug/cobotta_actual_grasp_point"
    )

    marker = InteractiveMarker()
    marker.header.frame_id = FRAME_ID
    marker.header.stamp = rospy.Time(0)

    marker.name = "actual_grasp_point"
    marker.description = (
        "actual_grasp_point\n"
        "drag to paper pinch position"
    )

    marker.scale = 0.060

    marker.pose.position.x = INITIAL_X
    marker.pose.position.y = INITIAL_Y
    marker.pose.position.z = INITIAL_Z
    marker.pose.orientation.w = 1.0

    # 実把持点本体：オレンジ球
    sphere = Marker()
    sphere.type = Marker.SPHERE

    sphere.scale.x = 0.008
    sphere.scale.y = 0.008
    sphere.scale.z = 0.008

    sphere.color.r = 1.0
    sphere.color.g = 0.45
    sphere.color.b = 0.0
    sphere.color.a = 0.95

    visual = InteractiveMarkerControl()
    visual.name = "actual_grasp_point_visual"
    visual.always_visible = True
    visual.markers.append(sphere)

    marker.controls.append(visual)

    # X軸：赤
    marker.controls.append(
        make_axis_control(
            "move_x",
            0.0,
            0.0,
            0.0,
            1.0,
        )
    )

    # Y軸：緑
    s = math.sqrt(0.5)
    marker.controls.append(
        make_axis_control(
            "move_y",
            0.0,
            0.0,
            s,
            s,
        )
    )

    # Z軸：青
    marker.controls.append(
        make_axis_control(
            "move_z",
            0.0,
            -s,
            0.0,
            s,
        )
    )

    server.insert(
        marker,
        feedback_cb,
    )
    server.applyChanges()

    rospy.loginfo(
        "actual_grasp_point marker started."
    )
    rospy.loginfo(
        "Initial xyz in %s = "
        "(%.3f, %.3f, %.3f) mm",
        FRAME_ID,
        INITIAL_X * 1000.0,
        INITIAL_Y * 1000.0,
        INITIAL_Z * 1000.0,
    )

    rospy.spin()


if __name__ == "__main__":
    main()
