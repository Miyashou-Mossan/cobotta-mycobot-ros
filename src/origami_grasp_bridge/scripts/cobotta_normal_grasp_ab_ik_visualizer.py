#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import sys

import moveit_commander
import numpy as np
import rospy

from geometry_msgs.msg import PoseArray, PoseStamped, Point
from moveit_msgs.msg import DisplayRobotState, MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from std_msgs.msg import ColorRGBA
from tf.transformations import quaternion_matrix
from visualization_msgs.msg import Marker, MarkerArray


GROUP = "cobotta_arm"
TIP = "cobotta_tool_link"

P0_TOPIC = "/origami/cobotta_p0_candidates_lower"

POSE_A_TOPIC = "/origami/debug/cobotta_tool_target_pose_a"
POSE_B_TOPIC = "/origami/debug/cobotta_tool_target_pose_b"

STATE_A_TOPIC = "/origami/debug/cobotta_normal_grasp_a_state"
STATE_B_TOPIC = "/origami/debug/cobotta_normal_grasp_b_state"

AXES_A_TOPIC = "/origami/debug/cobotta_normal_grasp_a_axes"
AXES_B_TOPIC = "/origami/debug/cobotta_normal_grasp_b_axes"

# cobotta_tool_link -> actual_grasp_point [m]
R_GRASP = np.array(
    [0.002000, 0.000000, -0.004894],
    dtype=float,
)


class Visualizer:

    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)

        self.group = moveit_commander.MoveGroupCommander(
            GROUP,
            wait_for_servers=20.0,
        )

        rospy.wait_for_service("/compute_ik")
        self.compute_ik = rospy.ServiceProxy(
            "/compute_ik",
            GetPositionIK,
        )

        self.state_a_pub = rospy.Publisher(
            STATE_A_TOPIC,
            DisplayRobotState,
            queue_size=1,
            latch=True,
        )

        self.state_b_pub = rospy.Publisher(
            STATE_B_TOPIC,
            DisplayRobotState,
            queue_size=1,
            latch=True,
        )

        self.axes_a_pub = rospy.Publisher(
            AXES_A_TOPIC,
            MarkerArray,
            queue_size=1,
            latch=True,
        )

        self.axes_b_pub = rospy.Publisher(
            AXES_B_TOPIC,
            MarkerArray,
            queue_size=1,
            latch=True,
        )

        self.p0 = None
        self.orientation_a = None
        self.orientation_b = None
        self.done = False

        rospy.Subscriber(
            P0_TOPIC,
            PoseArray,
            self.p0_callback,
            queue_size=1,
        )

        rospy.Subscriber(
            POSE_A_TOPIC,
            PoseStamped,
            self.pose_a_callback,
            queue_size=1,
        )

        rospy.Subscriber(
            POSE_B_TOPIC,
            PoseStamped,
            self.pose_b_callback,
            queue_size=1,
        )

        rospy.loginfo(
            "Waiting for P0 + Candidate A/B orientations..."
        )

    def p0_callback(self, msg):
        if not msg.poses:
            return

        p = msg.poses[0].position

        self.p0 = np.array(
            [p.x, p.y, p.z],
            dtype=float,
        )

        self.frame_id = msg.header.frame_id

        self.try_generate()

    def pose_a_callback(self, msg):
        self.orientation_a = copy.deepcopy(
            msg.pose.orientation
        )
        self.try_generate()

    def pose_b_callback(self, msg):
        self.orientation_b = copy.deepcopy(
            msg.pose.orientation
        )
        self.try_generate()

    @staticmethod
    def quat_array(q):
        return np.array(
            [q.x, q.y, q.z, q.w],
            dtype=float,
        )

    @staticmethod
    def point(v):
        p = Point()
        p.x = float(v[0])
        p.y = float(v[1])
        p.z = float(v[2])
        return p

    @staticmethod
    def color(r, g, b):
        c = ColorRGBA()
        c.r = r
        c.g = g
        c.b = b
        c.a = 1.0
        return c

    def make_tool_pose(self, orientation):
        q = self.quat_array(orientation)
        rotation = quaternion_matrix(q)[:3, :3]

        # actual_grasp_point=P0 を固定して
        # tool_link位置を逆算する。
        tool_position = self.p0 - rotation.dot(
            R_GRASP
        )

        target = PoseStamped()
        target.header.frame_id = self.frame_id
        target.header.stamp = rospy.Time.now()

        target.pose.position.x = float(
            tool_position[0]
        )
        target.pose.position.y = float(
            tool_position[1]
        )
        target.pose.position.z = float(
            tool_position[2]
        )

        target.pose.orientation = copy.deepcopy(
            orientation
        )

        return target, rotation, tool_position

    def solve_ik(self, target):
        req = GetPositionIKRequest()

        req.ik_request.group_name = GROUP
        req.ik_request.ik_link_name = TIP
        req.ik_request.pose_stamped = target
        req.ik_request.robot_state = (
            self.group.get_current_state()
        )

        req.ik_request.avoid_collisions = False
        req.ik_request.timeout = rospy.Duration(
            0.1
        )

        return self.compute_ik(req)

    def axis_marker(
        self,
        ns,
        marker_id,
        origin,
        direction,
        color,
    ):
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = rospy.Time.now()

        m.ns = ns
        m.id = marker_id
        m.type = Marker.ARROW
        m.action = Marker.ADD

        length = 0.050

        m.points = [
            self.point(origin),
            self.point(
                origin + length * direction
            ),
        ]

        m.scale.x = 0.003
        m.scale.y = 0.006
        m.scale.z = 0.009

        m.color = color

        return m

    def make_axes(
        self,
        ns,
        rotation,
        origin,
    ):
        markers = MarkerArray()

        markers.markers.append(
            self.axis_marker(
                ns,
                0,
                origin,
                rotation[:, 0],
                self.color(1.0, 0.0, 0.0),
            )
        )

        markers.markers.append(
            self.axis_marker(
                ns,
                1,
                origin,
                rotation[:, 1],
                self.color(0.0, 1.0, 0.0),
            )
        )

        markers.markers.append(
            self.axis_marker(
                ns,
                2,
                origin,
                rotation[:, 2],
                self.color(0.0, 0.0, 1.0),
            )
        )

        return markers

    def try_generate(self):
        if self.done:
            return

        if (
            self.p0 is None
            or self.orientation_a is None
            or self.orientation_b is None
        ):
            return

        target_a, rot_a, pos_a = (
            self.make_tool_pose(
                self.orientation_a
            )
        )

        target_b, rot_b, pos_b = (
            self.make_tool_pose(
                self.orientation_b
            )
        )

        res_a = self.solve_ik(target_a)
        res_b = self.solve_ik(target_b)

        print("===== Normal grasp A/B check =====")

        if (
            res_a.error_code.val
            == MoveItErrorCodes.SUCCESS
        ):
            msg = DisplayRobotState()
            msg.state = res_a.solution
            self.state_a_pub.publish(msg)

            self.axes_a_pub.publish(
                self.make_axes(
                    "normal_grasp_a",
                    rot_a,
                    pos_a,
                )
            )

            print("Candidate A : IK SUCCESS")
        else:
            print(
                "Candidate A : IK FAIL",
                res_a.error_code.val,
            )

        if (
            res_b.error_code.val
            == MoveItErrorCodes.SUCCESS
        ):
            msg = DisplayRobotState()
            msg.state = res_b.solution
            self.state_b_pub.publish(msg)

            self.axes_b_pub.publish(
                self.make_axes(
                    "normal_grasp_b",
                    rot_b,
                    pos_b,
                )
            )

            print("Candidate B : IK SUCCESS")
        else:
            print(
                "Candidate B : IK FAIL",
                res_b.error_code.val,
            )

        print()
        print("red   = X_tool")
        print("green = Y_tool  (gripper opening)")
        print("blue  = Z_tool  (tool longitudinal)")
        print()
        print("A state:", STATE_A_TOPIC)
        print("B state:", STATE_B_TOPIC)

        self.done = True


def main():
    rospy.init_node(
        "cobotta_normal_grasp_ab_ik_visualizer"
    )

    Visualizer()
    rospy.spin()


if __name__ == "__main__":
    main()
