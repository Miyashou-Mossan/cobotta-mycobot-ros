#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import sys

import numpy as np
import rospy
import moveit_commander

from geometry_msgs.msg import Point, PolygonStamped, PoseArray, PoseStamped
from moveit_msgs.msg import DisplayRobotState, MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from tf.transformations import quaternion_matrix
from visualization_msgs.msg import Marker, MarkerArray


GROUP = "cobotta_arm"
TIP = "cobotta_tool_link"

PAPER_TOPIC = "/origami/active_folding_paper_t0_ros"
POSE_A_TOPIC = "/origami/debug/cobotta_tool_target_pose_a"

STATE_TOPIC = "/origami/debug/cobotta_pregrasp_animation_state"
MARKER_TOPIC = "/origami/debug/cobotta_pregrasp_animation_markers"

R_GRASP = np.array(
    [0.002000, 0.000000, -0.004894],
    dtype=float,
)


class PreGraspAnimation:

    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)

        self.side = rospy.get_param(
            "~side",
            "lower",
        )

        self.p0_index = int(
            rospy.get_param(
                "~p0_index",
                0,
            )
        )

        # 今回実測したfingerの
        # P0より進行方向側への張り出し
        self.finger_front = (
            float(
                rospy.get_param(
                    "~finger_front_mm",
                    9.894,
                )
            )
            / 1000.0
        )

        # 今回は0 mmで形だけ確認する
        self.clearance = (
            float(
                rospy.get_param(
                    "~clearance_mm",
                    0.0,
                )
            )
            / 1000.0
        )

        self.approach_steps = int(
            rospy.get_param(
                "~approach_steps",
                30,
            )
        )

        self.close_steps = int(
            rospy.get_param(
                "~close_steps",
                15,
            )
        )

        self.paper_msg = None
        self.p0_msg = None
        self.pose_a_msg = None

        self.frames = []
        self.frame_index = 0
        self.built = False

        self.robot = moveit_commander.RobotCommander()

        rospy.wait_for_service(
            "/compute_ik",
            timeout=30.0,
        )

        self.compute_ik = rospy.ServiceProxy(
            "/compute_ik",
            GetPositionIK,
            persistent=True,
        )

        self.state_pub = rospy.Publisher(
            STATE_TOPIC,
            DisplayRobotState,
            queue_size=1,
            latch=True,
        )

        # 確認用の3つの静止状態
        self.pregrasp_state_pub = rospy.Publisher(
            "/origami/debug/cobotta_pregrasp_state",
            DisplayRobotState,
            queue_size=1,
            latch=True,
        )

        self.fold_start_open_state_pub = rospy.Publisher(
            "/origami/debug/cobotta_fold_start_open_state",
            DisplayRobotState,
            queue_size=1,
            latch=True,
        )

        self.fold_start_closed_state_pub = rospy.Publisher(
            "/origami/debug/cobotta_fold_start_closed_state",
            DisplayRobotState,
            queue_size=1,
            latch=True,
        )

        self.marker_pub = rospy.Publisher(
            MARKER_TOPIC,
            MarkerArray,
            queue_size=1,
            latch=True,
        )

        rospy.Subscriber(
            PAPER_TOPIC,
            PolygonStamped,
            self.paper_cb,
            queue_size=1,
        )

        rospy.Subscriber(
            "/origami/cobotta_p0_candidates_{}".format(
                self.side
            ),
            PoseArray,
            self.p0_cb,
            queue_size=1,
        )

        rospy.Subscriber(
            POSE_A_TOPIC,
            PoseStamped,
            self.pose_a_cb,
            queue_size=1,
        )

        self.timer = rospy.Timer(
            rospy.Duration(0.10),
            self.timer_cb,
        )

        rospy.loginfo(
            "PRE-GRASP animation started: "
            "side=%s P0=%d",
            self.side,
            self.p0_index,
        )

    def paper_cb(self, msg):
        self.paper_msg = msg
        self.try_build()

    def p0_cb(self, msg):
        self.p0_msg = msg
        self.try_build()

    def pose_a_cb(self, msg):
        self.pose_a_msg = msg
        self.try_build()

    @staticmethod
    def cross2(a, b):
        return (
            a[0] * b[1]
            - a[1] * b[0]
        )

    def ray_polygon_intersection(
        self,
        p0,
        ray3,
    ):
        poly = self.paper_msg.polygon.points

        ray_xy_norm = np.linalg.norm(
            ray3[:2]
        )

        if ray_xy_norm < 1.0e-9:
            raise RuntimeError(
                "Z_tool has almost no paper-plane component."
            )

        ray2 = ray3[:2] / ray_xy_norm
        origin2 = p0[:2]

        best_t = None

        for i in range(len(poly)):
            a = np.array([
                poly[i].x,
                poly[i].y,
            ])

            b = np.array([
                poly[(i + 1) % len(poly)].x,
                poly[(i + 1) % len(poly)].y,
            ])

            edge = b - a

            denom = self.cross2(
                ray2,
                edge,
            )

            if abs(denom) < 1.0e-12:
                continue

            delta = a - origin2

            t = self.cross2(
                delta,
                edge,
            ) / denom

            u = self.cross2(
                delta,
                ray2,
            ) / denom

            if (
                t > 1.0e-6
                and -1.0e-9 <= u <= 1.0 + 1.0e-9
            ):
                if (
                    best_t is None
                    or t < best_t
                ):
                    best_t = t

        if best_t is None:
            raise RuntimeError(
                "No paper-edge intersection found."
            )

        # 2D距離を実際の3D Z_tool距離へ戻す
        distance_3d = (
            best_t / ray_xy_norm
        )

        edge_point = (
            p0
            + distance_3d * ray3
        )

        return distance_3d, edge_point

    @staticmethod
    def make_point(v):
        p = Point()
        p.x = float(v[0])
        p.y = float(v[1])
        p.z = float(v[2])
        return p

    def make_tool_pose(
        self,
        grasp_point,
        rotation,
        orientation,
        frame_id,
    ):
        tool_position = (
            grasp_point
            - rotation.dot(R_GRASP)
        )

        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.header.stamp = rospy.Time(0)

        pose.pose.position.x = float(
            tool_position[0]
        )
        pose.pose.position.y = float(
            tool_position[1]
        )
        pose.pose.position.z = float(
            tool_position[2]
        )

        pose.pose.orientation = copy.deepcopy(
            orientation
        )

        return pose

    def solve_ik(
        self,
        pose,
        seed,
    ):
        req = GetPositionIKRequest()

        req.ik_request.group_name = GROUP
        req.ik_request.ik_link_name = TIP
        req.ik_request.pose_stamped = pose

        req.ik_request.robot_state = (
            copy.deepcopy(seed)
        )

        # 今回は進入形状の可視化なので
        # Paper Collision判定はまだ入れない。
        req.ik_request.avoid_collisions = False
        req.ik_request.timeout = rospy.Duration(
            0.10
        )

        res = self.compute_ik(req)

        if (
            res.error_code.val
            != MoveItErrorCodes.SUCCESS
        ):
            return None

        return res.solution

    @staticmethod
    def set_gripper(
        state,
        opening,
    ):
        state = copy.deepcopy(state)

        names = list(
            state.joint_state.name
        )

        positions = list(
            state.joint_state.position
        )

        lookup = {
            name: i
            for i, name in enumerate(names)
        }

        if "cobotta_joint_gripper" in lookup:
            positions[
                lookup["cobotta_joint_gripper"]
            ] = opening

        # mimic jointがRobotStateに含まれている場合
        if (
            "cobotta_joint_gripper_mimic"
            in lookup
        ):
            positions[
                lookup[
                    "cobotta_joint_gripper_mimic"
                ]
            ] = -opening

        state.joint_state.position = positions
        state.joint_state.header.stamp = rospy.Time(0)

        return state

    def make_markers(
        self,
        p0,
        edge,
        pre,
        frame_id,
    ):
        arr = MarkerArray()

        # T0紙面を半透明で表示
        paper_points = self.paper_msg.polygon.points

        if len(paper_points) >= 3:
            fill = Marker()
            fill.header.frame_id = frame_id
            fill.header.stamp = rospy.Time(0)
            fill.ns = "pregrasp_paper"
            fill.id = 10
            fill.type = Marker.TRIANGLE_LIST
            fill.action = Marker.ADD
            fill.pose.orientation.w = 1.0
            fill.scale.x = 1.0
            fill.scale.y = 1.0
            fill.scale.z = 1.0

            fill.color.r = 0.8
            fill.color.g = 0.8
            fill.color.b = 0.8
            fill.color.a = 0.35

            p_first = paper_points[0]

            for i in range(1, len(paper_points) - 1):
                for p in (
                    p_first,
                    paper_points[i],
                    paper_points[i + 1],
                ):
                    fill.points.append(
                        self.make_point(
                            np.array([
                                p.x,
                                p.y,
                                p.z + 0.0003,
                            ])
                        )
                    )

            arr.markers.append(fill)

            outline = Marker()
            outline.header.frame_id = frame_id
            outline.header.stamp = rospy.Time(0)
            outline.ns = "pregrasp_paper"
            outline.id = 11
            outline.type = Marker.LINE_STRIP
            outline.action = Marker.ADD
            outline.pose.orientation.w = 1.0
            outline.scale.x = 0.0015

            outline.color.r = 1.0
            outline.color.g = 1.0
            outline.color.b = 1.0
            outline.color.a = 1.0

            for p in paper_points:
                outline.points.append(
                    self.make_point(
                        np.array([
                            p.x,
                            p.y,
                            p.z + 0.0006,
                        ])
                    )
                )

            p = paper_points[0]
            outline.points.append(
                self.make_point(
                    np.array([
                        p.x,
                        p.y,
                        p.z + 0.0006,
                    ])
                )
            )

            arr.markers.append(outline)

        line = Marker()
        line.header.frame_id = frame_id
        line.header.stamp = rospy.Time(0)
        line.ns = "pregrasp_path"
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.pose.orientation.w = 1.0
        line.scale.x = 0.002

        line.color.r = 1.0
        line.color.g = 1.0
        line.color.b = 0.0
        line.color.a = 1.0

        line.points = [
            self.make_point(pre),
            self.make_point(edge),
            self.make_point(p0),
        ]

        arr.markers.append(line)

        for marker_id, position, name in (
            (1, pre, "PRE-GRASP"),
            (2, edge, "PAPER EDGE"),
            (3, p0, "P0"),
        ):
            sphere = Marker()
            sphere.header.frame_id = frame_id
            sphere.header.stamp = rospy.Time(0)
            sphere.ns = "pregrasp_points"
            sphere.id = marker_id
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position = self.make_point(
                position
            )
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = 0.007
            sphere.scale.y = 0.007
            sphere.scale.z = 0.007
            sphere.color.a = 1.0

            if name == "PRE-GRASP":
                sphere.color.r = 1.0
                sphere.color.g = 1.0
            elif name == "PAPER EDGE":
                sphere.color.r = 1.0
                sphere.color.g = 0.5
            else:
                sphere.color.g = 1.0

            arr.markers.append(sphere)

            label = Marker()
            label.header.frame_id = frame_id
            label.header.stamp = rospy.Time(0)
            label.ns = "pregrasp_labels"
            label.id = 200 + marker_id
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD

            label.pose.position.x = float(position[0])
            label.pose.position.y = float(position[1])
            label.pose.position.z = float(position[2] + 0.015)
            label.pose.orientation.w = 1.0

            label.scale.z = 0.010
            label.color.r = 1.0
            label.color.g = 1.0
            label.color.b = 1.0
            label.color.a = 1.0
            label.text = name

            arr.markers.append(label)

        return arr

    def try_build(self):
        if self.built:
            return

        if (
            self.paper_msg is None
            or self.p0_msg is None
            or self.pose_a_msg is None
        ):
            return

        if (
            self.p0_index
            >= len(self.p0_msg.poses)
        ):
            rospy.logerr(
                "P0 index out of range: %d / %d",
                self.p0_index,
                len(self.p0_msg.poses),
            )
            return

        p = self.p0_msg.poses[
            self.p0_index
        ].position

        p0 = np.array([
            p.x,
            p.y,
            p.z,
        ])

        orientation = (
            self.pose_a_msg.pose.orientation
        )

        q = np.array([
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        ])

        rotation = quaternion_matrix(
            q
        )[:3, :3]

        z_tool = rotation[:, 2]
        z_tool = z_tool / np.linalg.norm(
            z_tool
        )

        # P0から-Z_tool方向へ紙端を探す
        ray_out = -z_tool

        (
            d_edge,
            edge_point,
        ) = self.ray_polygon_intersection(
            p0,
            ray_out,
        )

        total_retreat = (
            d_edge
            + self.finger_front
            + self.clearance
        )

        pre = (
            p0
            - total_retreat * z_tool
        )

        frame_id = self.p0_msg.header.frame_id

        rospy.loginfo(
            "P0[%d] = (%.3f, %.3f, %.3f) mm",
            self.p0_index,
            *(p0 * 1000.0)
        )

        rospy.loginfo(
            "P0 -> paper edge = %.3f mm",
            d_edge * 1000.0,
        )

        rospy.loginfo(
            "finger front = %.3f mm",
            self.finger_front * 1000.0,
        )

        rospy.loginfo(
            "clearance = %.3f mm",
            self.clearance * 1000.0,
        )

        rospy.loginfo(
            "P0 -> PRE-GRASP = %.3f mm",
            total_retreat * 1000.0,
        )

        rospy.loginfo(
            "PRE-GRASP = (%.3f, %.3f, %.3f) mm",
            *(pre * 1000.0)
        )

        self.marker_pub.publish(
            self.make_markers(
                p0,
                edge_point,
                pre,
                frame_id,
            )
        )

        # まずP0のIKを求める。
        # そこから逆向きにPRE-GRASPまで連続IKし、
        # 最後に逆順にして実際の進入アニメーションにする。
        current_seed = (
            self.robot.get_current_state()
        )

        p0_pose = self.make_tool_pose(
            p0,
            rotation,
            orientation,
            frame_id,
        )

        p0_state = self.solve_ik(
            p0_pose,
            current_seed,
        )

        if p0_state is None:
            rospy.logerr(
                "P0 Candidate-A IK failed."
            )
            return

        backward_states = [
            p0_state
        ]

        seed = p0_state

        for i in range(
            1,
            self.approach_steps + 1,
        ):
            alpha = (
                float(i)
                / float(self.approach_steps)
            )

            grasp_point = (
                p0
                - alpha
                * total_retreat
                * z_tool
            )

            target = self.make_tool_pose(
                grasp_point,
                rotation,
                orientation,
                frame_id,
            )

            state = self.solve_ik(
                target,
                seed,
            )

            if state is None:
                rospy.logerr(
                    "Approach IK failed at "
                    "backward step %d/%d",
                    i,
                    self.approach_steps,
                )
                return

            backward_states.append(
                state
            )

            seed = state

        approach_states = list(
            reversed(backward_states)
        )

        # PRE-GRASPで少し停止
        for _ in range(10):
            self.frames.append(
                self.set_gripper(
                    approach_states[0],
                    0.015,
                )
            )

        # 横侵入中は最大OPEN
        for state in approach_states:
            self.frames.append(
                self.set_gripper(
                    state,
                    0.015,
                )
            )

        # P0で15 mm -> 0 mmへCLOSE
        final_state = approach_states[-1]

        # 3つの確認用静止状態をPublish
        pre_open_state = self.set_gripper(
            approach_states[0],
            0.015,
        )

        fold_start_open_state = self.set_gripper(
            final_state,
            0.015,
        )

        fold_start_closed_state = self.set_gripper(
            final_state,
            0.0,
        )

        msg = DisplayRobotState()
        msg.state = copy.deepcopy(pre_open_state)
        self.pregrasp_state_pub.publish(msg)

        msg = DisplayRobotState()
        msg.state = copy.deepcopy(fold_start_open_state)
        self.fold_start_open_state_pub.publish(msg)

        msg = DisplayRobotState()
        msg.state = copy.deepcopy(fold_start_closed_state)
        self.fold_start_closed_state_pub.publish(msg)

        for i in range(
            self.close_steps + 1
        ):
            ratio = (
                float(i)
                / float(self.close_steps)
            )

            opening = (
                0.015
                * (1.0 - ratio)
            )

            self.frames.append(
                self.set_gripper(
                    final_state,
                    opening,
                )
            )

        # CLOSE状態で停止
        for _ in range(15):
            self.frames.append(
                self.set_gripper(
                    final_state,
                    0.0,
                )
            )

        self.built = True

        rospy.loginfo(
            "Animation ready: %d frames",
            len(self.frames),
        )

    def timer_cb(self, _event):
        if not self.frames:
            return

        state = self.frames[
            self.frame_index
        ]

        msg = DisplayRobotState()
        msg.state = copy.deepcopy(
            state
        )

        self.state_pub.publish(msg)

        self.frame_index += 1

        if self.frame_index >= len(
            self.frames
        ):
            self.frame_index = 0


def main():
    rospy.init_node(
        "cobotta_pregrasp_gripper_animation"
    )

    PreGraspAnimation()

    rospy.spin()


if __name__ == "__main__":
    main()
