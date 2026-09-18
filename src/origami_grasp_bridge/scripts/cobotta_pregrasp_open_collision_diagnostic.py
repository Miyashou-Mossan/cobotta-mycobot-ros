#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import math
import sys

import moveit_commander
import numpy as np
import rospy

from geometry_msgs.msg import PolygonStamped, PoseArray, PoseStamped
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import (
    GetPositionIK,
    GetPositionIKRequest,
    GetStateValidity,
    GetStateValidityRequest,
)
from tf.transformations import quaternion_matrix


GROUP = "cobotta_arm"
TIP = "cobotta_tool_link"

PAPER_TOPIC = "/origami/active_folding_paper_t0_ros"
P0_TOPIC = "/origami/cobotta_p0_candidates_lower"
POSE_A_TOPIC = "/origami/debug/cobotta_tool_target_pose_a"

LOCAL_PAPER_OBJECT = "cobotta_local_grasp_paper_collision"

# cobotta_tool_link -> actual_grasp_point [m]
R_GRASP = np.array(
    [0.002000, 0.000000, -0.004894],
    dtype=float,
)

# P0より進行方向側へfingerが張り出す距離 [m]
FINGER_FRONT = 0.009894


class OpenApproachDiagnostic:

    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)

        self.p0_index = int(
            rospy.get_param("~p0_index", 0)
        )

        self.steps = int(
            rospy.get_param("~steps", 30)
        )

        self.clearance = (
            float(
                rospy.get_param(
                    "~clearance_mm",
                    0.0,
                )
            )
            / 1000.0
        )

        self.paper_msg = None
        self.p0_msg = None
        self.pose_a_msg = None
        self.done = False

        self.robot = moveit_commander.RobotCommander()

        rospy.wait_for_service(
            "/compute_ik",
            timeout=30.0,
        )

        rospy.wait_for_service(
            "/check_state_validity",
            timeout=30.0,
        )

        self.compute_ik = rospy.ServiceProxy(
            "/compute_ik",
            GetPositionIK,
            persistent=True,
        )

        self.check_validity = rospy.ServiceProxy(
            "/check_state_validity",
            GetStateValidity,
            persistent=True,
        )

        rospy.Subscriber(
            PAPER_TOPIC,
            PolygonStamped,
            self.paper_cb,
            queue_size=1,
        )

        rospy.Subscriber(
            P0_TOPIC,
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

        rospy.loginfo(
            "OPEN PRE-GRASP collision diagnostic started."
        )

    def paper_cb(self, msg):
        self.paper_msg = msg
        self.try_run()

    def p0_cb(self, msg):
        self.p0_msg = msg
        self.try_run()

    def pose_a_cb(self, msg):
        self.pose_a_msg = msg
        self.try_run()

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

        distance_3d = (
            best_t / ray_xy_norm
        )

        return distance_3d

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
        seed_state,
    ):
        req = GetPositionIKRequest()

        req.ik_request.group_name = GROUP
        req.ik_request.ik_link_name = TIP
        req.ik_request.pose_stamped = pose

        req.ik_request.robot_state = copy.deepcopy(
            seed_state
        )

        # IKとCollision判定は分離する
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
    def set_gripper_open(state):
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

        # 最大OPEN = 15 mm
        if "cobotta_joint_gripper" in lookup:
            positions[
                lookup["cobotta_joint_gripper"]
            ] = 0.015

        if (
            "cobotta_joint_gripper_mimic"
            in lookup
        ):
            positions[
                lookup[
                    "cobotta_joint_gripper_mimic"
                ]
            ] = -0.015

        state.joint_state.position = positions
        state.joint_state.header.stamp = rospy.Time(0)

        return state

    def validity(self, state):
        req = GetStateValidityRequest()

        req.robot_state = state
        req.group_name = GROUP

        res = self.check_validity(req)

        pairs = []

        for contact in res.contacts:
            pair = tuple(sorted([
                contact.contact_body_1,
                contact.contact_body_2,
            ]))

            if pair not in pairs:
                pairs.append(pair)

        return bool(res.valid), pairs

    @staticmethod
    def pair_text(pairs):
        if not pairs:
            return "-"

        return "; ".join(
            "{}<->{}".format(a, b)
            for a, b in pairs
        )

    def try_run(self):
        if self.done:
            return

        if (
            self.paper_msg is None
            or self.p0_msg is None
            or self.pose_a_msg is None
        ):
            return

        if self.p0_index >= len(
            self.p0_msg.poses
        ):
            rospy.logerr(
                "P0 index out of range: %d / %d",
                self.p0_index,
                len(self.p0_msg.poses),
            )
            return

        self.done = True

        p = self.p0_msg.poses[
            self.p0_index
        ].position

        p0 = np.array([
            p.x,
            p.y,
            p.z,
        ], dtype=float)

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
        z_tool = (
            z_tool / np.linalg.norm(z_tool)
        )

        # P0から -Z_tool 側へ紙端を探す
        d_edge = self.ray_polygon_intersection(
            p0,
            -z_tool,
        )

        total_retreat = (
            d_edge
            + FINGER_FRONT
            + self.clearance
        )

        pre = (
            p0
            - total_retreat * z_tool
        )

        frame_id = self.p0_msg.header.frame_id

        print()
        print(
            "===== OPEN PRE-GRASP Collision Diagnostic ====="
        )

        print(
            "P0 index           :",
            self.p0_index,
        )

        print(
            "steps              :",
            self.steps,
        )

        print(
            "P0 -> paper edge   : "
            "{:.3f} mm".format(
                d_edge * 1000.0
            )
        )

        print(
            "finger front       : "
            "{:.3f} mm".format(
                FINGER_FRONT * 1000.0
            )
        )

        print(
            "clearance          : "
            "{:.3f} mm".format(
                self.clearance * 1000.0
            )
        )

        print(
            "P0 -> PRE-GRASP    : "
            "{:.3f} mm".format(
                total_retreat * 1000.0
            )
        )

        print()

        # 前回のアニメーションと同じ枝を使うため、
        # P0から逆向きにPRE-GRASPまでIKを解く。
        seed = self.robot.get_current_state()

        p0_pose = self.make_tool_pose(
            p0,
            rotation,
            orientation,
            frame_id,
        )

        state = self.solve_ik(
            p0_pose,
            seed,
        )

        if state is None:
            print("P0 IK FAILED")
            return

        backward_states = [
            self.set_gripper_open(state)
        ]

        seed = state

        for i in range(
            1,
            self.steps + 1,
        ):
            ratio = (
                float(i)
                / float(self.steps)
            )

            grasp_point = (
                p0
                - ratio
                * total_retreat
                * z_tool
            )

            pose = self.make_tool_pose(
                grasp_point,
                rotation,
                orientation,
                frame_id,
            )

            state = self.solve_ik(
                pose,
                seed,
            )

            if state is None:
                print(
                    "Backward IK FAILED: "
                    "step {}/{}".format(
                        i,
                        self.steps,
                    )
                )
                return

            backward_states.append(
                self.set_gripper_open(
                    state
                )
            )

            seed = state

        # 実際の進入順 PRE-GRASP -> P0
        states = list(
            reversed(backward_states)
        )

        valid_count = 0
        collision_count = 0
        local_paper_collision_count = 0

        first_collision = None

        print(
            "step | progress | result     | contact"
        )
        print(
            "-----+----------+------------+------------------------------"
        )

        for step, state in enumerate(states):
            progress = (
                float(step)
                / float(self.steps)
            )

            valid, pairs = self.validity(
                state
            )

            local_hit = any(
                LOCAL_PAPER_OBJECT in pair
                for pair in pairs
            )

            if valid:
                result = "VALID"
                valid_count += 1

            else:
                result = "COLLISION"
                collision_count += 1

                if first_collision is None:
                    first_collision = step

            if local_hit:
                local_paper_collision_count += 1

            print(
                "{:4d} | {:8.3f} | {:10s} | {}".format(
                    step,
                    progress,
                    result,
                    self.pair_text(pairs),
                )
            )

        print()
        print(
            "===== Summary ====="
        )

        print(
            "states                    :",
            len(states),
        )

        print(
            "VALID                     :",
            valid_count,
        )

        print(
            "COLLISION                 :",
            collision_count,
        )

        print(
            "local paper collision     :",
            local_paper_collision_count,
        )

        print(
            "first collision step      :",
            (
                first_collision
                if first_collision is not None
                else "NONE"
            ),
        )

        if collision_count == 0:
            print()
            print(
                "RESULT: OPEN lateral approach is Collision-free."
            )

        elif (
            local_paper_collision_count
            == collision_count
        ):
            print()
            print(
                "RESULT: Collision is caused by "
                "the local paper object."
            )

        else:
            print()
            print(
                "RESULT: Other Collision objects "
                "also affect the approach."
            )


def main():
    rospy.init_node(
        "cobotta_pregrasp_open_collision_diagnostic"
    )

    OpenApproachDiagnostic()

    rospy.spin()


if __name__ == "__main__":
    main()
