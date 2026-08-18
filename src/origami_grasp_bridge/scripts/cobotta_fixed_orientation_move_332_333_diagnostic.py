#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import csv
import math
import os
import sys
import yaml

import moveit_commander
import rospy

from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import (
    GetPositionIK,
    GetPositionIKRequest,
    GetStateValidity,
    GetStateValidityRequest,
)


INPUT_YAML = os.path.expanduser(
    "~/directionA_reverse_local_z_m25_multidof.yaml"
)

ORIENTATION_DIAG_CSV = os.path.expanduser(
    "~/cobotta_fixed_grasp_orientation_332_diagnostic.csv"
)

FULL_PATH_CSV = os.path.expanduser(
    "~/cobotta_branch_preserving_full_path_0_340_edge_fixed.csv"
)

OUTPUT_CSV = os.path.expanduser(
    "~/cobotta_fixed_orientation_move_332_333_diagnostic.csv"
)

GROUP = "cobotta_arm"
TIP = "cobotta_tool_link"
FRAME = "paper_center"

FROM_INDEX = 332
TO_INDEX = 333

LOCAL_X_DEG = -15.0
LOCAL_Y_DEG = +30.0

SUBDIVISIONS = 8

R_GRASP = [
    +0.000401,
    -0.001507,
    -0.004894,
]

IK_TIMEOUT = 0.10
MAX_JOINT_STEP_RAD = 0.010


def normalize(q):
    n = math.sqrt(sum(v*v for v in q))
    if n <= 1.0e-12:
        raise RuntimeError("zero quaternion")
    return [v/n for v in q]


def multiply(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b

    return [
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    ]


def axis_q(axis, deg):
    h = math.radians(deg) * 0.5
    s = math.sin(h)
    c = math.cos(h)

    if axis == "x":
        return [s, 0.0, 0.0, c]

    if axis == "y":
        return [0.0, s, 0.0, c]

    raise ValueError(axis)


def rotate_vector(q, v):
    x, y, z, w = normalize(q)
    vx, vy, vz = v

    return [
        (1 - 2*(y*y + z*z))*vx
        + 2*(x*y - z*w)*vy
        + 2*(x*z + y*w)*vz,

        2*(x*y + z*w)*vx
        + (1 - 2*(x*x + z*z))*vy
        + 2*(y*z - x*w)*vz,

        2*(x*z - y*w)*vx
        + 2*(y*z + x*w)*vy
        + (1 - 2*(x*x + y*y))*vz,
    ]


def slerp(q0, q1, t):
    q0 = normalize(q0)
    q1 = normalize(q1)

    dot = sum(
        a*b
        for a, b in zip(q0, q1)
    )

    if dot < 0.0:
        q1 = [-v for v in q1]
        dot = -dot

    dot = max(
        -1.0,
        min(1.0, dot),
    )

    if dot > 0.9995:
        return normalize([
            (1-t)*a + t*b
            for a, b in zip(q0, q1)
        ])

    theta = math.acos(dot)
    sin_theta = math.sin(theta)

    a = math.sin((1-t)*theta) / sin_theta
    b = math.sin(t*theta) / sin_theta

    return normalize([
        a*x + b*y
        for x, y in zip(q0, q1)
    ])


class Diagnostic:

    def __init__(self):
        moveit_commander.roscpp_initialize(
            sys.argv
        )

        self.group = (
            moveit_commander.MoveGroupCommander(
                GROUP,
                wait_for_servers=20.0,
            )
        )

        self.active_joints = list(
            self.group.get_active_joints()
        )

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

        with open(
            INPUT_YAML,
            "r",
            encoding="utf-8",
        ) as f:
            self.paper_points = (
                yaml.safe_load(f)["points"]
            )

        with open(
            ORIENTATION_DIAG_CSV,
            newline="",
            encoding="utf-8-sig",
        ) as f:
            orientation_rows = list(
                csv.DictReader(f)
            )

        if len(orientation_rows) != 8:
            raise RuntimeError(
                "Expected 8 orientation diagnostic rows, got {}".format(
                    len(orientation_rows)
                )
            )

        self.orientation_final = (
            orientation_rows[-1]
        )

        with open(
            FULL_PATH_CSV,
            newline="",
            encoding="utf-8-sig",
        ) as f:
            self.full_rows = list(
                csv.DictReader(f)
            )

        self.saved_finish_row = (
            self.full_rows[340]
        )

        if int(
            self.saved_finish_row[
                "paper_index"
            ]
        ) != TO_INDEX:
            raise RuntimeError(
                "path_point 340 is not paper index 333"
            )

    def active_positions(self, state):
        lookup = dict(zip(
            state.joint_state.name,
            state.joint_state.position,
        ))

        return [
            float(lookup[name])
            for name in self.active_joints
        ]

    def state_from_values(self, values):
        state = self.group.get_current_state()

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

        for name, value in zip(
            self.active_joints,
            values,
        ):
            positions[
                lookup[name]
            ] = float(value)

        state.joint_state.position = (
            positions
        )
        state.joint_state.header.stamp = (
            rospy.Time(0)
        )
        state.is_diff = False

        return state

    def state_from_row(self, row):
        return self.state_from_values([
            float(row[name])
            for name in self.active_joints
        ])

    def paper_pose(self, index):
        tf = self.paper_points[
            index
        ]["transforms"][0]

        t = tf["translation"]
        r = tf["rotation"]

        return (
            [
                float(t["x"]),
                float(t["y"]),
                float(t["z"]),
            ],
            normalize([
                float(r["x"]),
                float(r["y"]),
                float(r["z"]),
                float(r["w"]),
            ]),
        )

    def make_tool_pose(self, ratio):
        p0, q0 = self.paper_pose(
            FROM_INDEX
        )

        p1, q1 = self.paper_pose(
            TO_INDEX
        )

        paper_p = [
            (1-ratio)*a + ratio*b
            for a, b in zip(p0, p1)
        ]

        paper_q = slerp(
            q0,
            q1,
            ratio,
        )

        q_delta = multiply(
            axis_q(
                "x",
                LOCAL_X_DEG,
            ),
            axis_q(
                "y",
                LOCAL_Y_DEG,
            ),
        )

        tool_q = normalize(
            multiply(
                paper_q,
                q_delta,
            )
        )

        offset_world = rotate_vector(
            tool_q,
            R_GRASP,
        )

        tool_p = [
            paper_p[i]
            - offset_world[i]
            for i in range(3)
        ]

        pose = PoseStamped()
        pose.header.frame_id = FRAME
        pose.header.stamp = rospy.Time(0)

        pose.pose.position.x = tool_p[0]
        pose.pose.position.y = tool_p[1]
        pose.pose.position.z = tool_p[2]

        pose.pose.orientation.x = tool_q[0]
        pose.pose.orientation.y = tool_q[1]
        pose.pose.orientation.z = tool_q[2]
        pose.pose.orientation.w = tool_q[3]

        return pose

    def contacts_text(self, result):
        return "; ".join(
            "{}<->{} depth={:.6f}".format(
                c.contact_body_1,
                c.contact_body_2,
                c.depth,
            )
            for c in result.contacts
        )

    def check_state(self, state):
        req = GetStateValidityRequest()
        req.robot_state = state
        req.group_name = GROUP

        return self.check_validity(req)

    def solve(self, pose, seed):
        req = GetPositionIKRequest()

        req.ik_request.group_name = GROUP
        req.ik_request.ik_link_name = TIP
        req.ik_request.pose_stamped = pose
        req.ik_request.robot_state = (
            copy.deepcopy(seed)
        )
        req.ik_request.avoid_collisions = False
        req.ik_request.timeout = (
            rospy.Duration(
                IK_TIMEOUT
            )
        )

        res = self.compute_ik(req)

        if (
            res.error_code.val
            != MoveItErrorCodes.SUCCESS
        ):
            return None, "IK_FAILED"

        validity = self.check_state(
            res.solution
        )

        if not validity.valid:
            return (
                None,
                "ENDPOINT_COLLISION "
                + self.contacts_text(
                    validity
                ),
            )

        return res.solution, "VALID"

    def interpolate_state(
        self,
        a,
        b,
        ratio,
    ):
        qa = self.active_positions(a)
        qb = self.active_positions(b)

        q = [
            x + ratio*(y-x)
            for x, y in zip(qa, qb)
        ]

        state = copy.deepcopy(a)

        lookup = {
            name: i
            for i, name in enumerate(
                state.joint_state.name
            )
        }

        positions = list(
            state.joint_state.position
        )

        for name, value in zip(
            self.active_joints,
            q,
        ):
            positions[
                lookup[name]
            ] = value

        state.joint_state.position = (
            positions
        )
        state.joint_state.header.stamp = (
            rospy.Time(0)
        )

        return state

    def validate_edge(self, a, b):
        qa = self.active_positions(a)
        qb = self.active_positions(b)

        max_delta = max(
            abs(y-x)
            for x, y in zip(qa, qb)
        )

        steps = max(
            1,
            int(math.ceil(
                max_delta
                / MAX_JOINT_STEP_RAD
            ))
        )

        for step in range(
            1,
            steps + 1,
        ):
            ratio = (
                float(step)
                / float(steps)
            )

            state = self.interpolate_state(
                a,
                b,
                ratio,
            )

            result = self.check_state(
                state
            )

            if not result.valid:
                return (
                    False,
                    "EDGE_COLLISION "
                    "step={}/{} "
                    "ratio={:.6f} {}".format(
                        step,
                        steps,
                        ratio,
                        self.contacts_text(
                            result
                        ),
                    ),
                )

        return True, "VALID"

    def run(self):
        current_state = (
            self.state_from_row(
                self.orientation_final
            )
        )

        rows = []

        print("=" * 90)
        print(
            "FIXED-ORIENTATION PAPER MOVE"
        )
        print("=" * 90)

        print(
            "paper              : "
            "{} -> {}".format(
                FROM_INDEX,
                TO_INDEX,
            )
        )

        print(
            "local orientation  : "
            "({:+.3f},{:+.3f}) FIXED".format(
                LOCAL_X_DEG,
                LOCAL_Y_DEG,
            )
        )

        print(
            "start seed         : "
            "fixed-grasp orientation diagnostic final state"
        )

        print(
            "subdivisions       :",
            SUBDIVISIONS,
        )

        print()

        for step in range(
            1,
            SUBDIVISIONS + 1,
        ):
            ratio = (
                float(step)
                / float(SUBDIVISIONS)
            )

            pose = self.make_tool_pose(
                ratio
            )

            solution, endpoint_status = (
                self.solve(
                    pose,
                    current_state,
                )
            )

            if solution is None:
                print(
                    "step {}/{} "
                    "ratio={:.3f} "
                    "endpoint={}".format(
                        step,
                        SUBDIVISIONS,
                        ratio,
                        endpoint_status,
                    )
                )

                print()
                print(
                    "FIXED-ORIENTATION PAPER MOVE: FAIL"
                )
                return

            edge_valid, edge_status = (
                self.validate_edge(
                    current_state,
                    solution,
                )
            )

            qa = self.active_positions(
                current_state
            )

            qb = self.active_positions(
                solution
            )

            max_delta_deg = max(
                abs(math.degrees(b-a))
                for a, b in zip(qa, qb)
            )

            row = {
                "step": step,
                "ratio": ratio,
                "from_paper_index":
                    FROM_INDEX,
                "to_paper_index":
                    TO_INDEX,
                "local_x_deg":
                    LOCAL_X_DEG,
                "local_y_deg":
                    LOCAL_Y_DEG,
                "endpoint_status":
                    endpoint_status,
                "edge_status":
                    edge_status,
                "max_joint_delta_deg":
                    max_delta_deg,
            }

            for name, value in zip(
                self.active_joints,
                qb,
            ):
                row[name] = value

            rows.append(row)

            print(
                "step {}/{} "
                "ratio={:.3f} "
                "endpoint={} "
                "edge={} "
                "maxΔ={:.6f} deg".format(
                    step,
                    SUBDIVISIONS,
                    ratio,
                    endpoint_status,
                    edge_status,
                    max_delta_deg,
                )
            )

            if not edge_valid:
                print()
                print(
                    "FIXED-ORIENTATION PAPER MOVE: FAIL"
                )
                return

            current_state = (
                copy.deepcopy(
                    solution
                )
            )

        # --------------------------------------------
        # 元のpaper333 / (-15,+30)状態との再接続確認
        # --------------------------------------------

        saved_state = self.state_from_row(
            self.saved_finish_row
        )

        q_new = self.active_positions(
            current_state
        )

        q_saved = self.active_positions(
            saved_state
        )

        reconnect_deg = max(
            abs(math.degrees(a-b))
            for a, b in zip(
                q_new,
                q_saved,
            )
        )

        reconnect_valid, reconnect_status = (
            self.validate_edge(
                current_state,
                saved_state,
            )
        )

        fields = [
            "step",
            "ratio",
            "from_paper_index",
            "to_paper_index",
            "local_x_deg",
            "local_y_deg",
            "endpoint_status",
            "edge_status",
            "max_joint_delta_deg",
            *self.active_joints,
        ]

        with open(
            OUTPUT_CSV,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fields,
            )

            writer.writeheader()
            writer.writerows(rows)

        print()
        print("=" * 90)
        print("FINAL RESULT")
        print("=" * 90)

        print(
            "successful move steps :",
            len(rows),
        )

        print()
        print(
            "paper333 reconnect:"
        )

        print(
            "  max joint difference: "
            "{:.9f} deg".format(
                reconnect_deg
            )
        )

        print(
            "  edge                :",
            reconnect_status,
        )

        print()
        print("OUTPUT:")
        print(
            " ",
            OUTPUT_CSV,
        )

        print()

        if reconnect_valid:
            print(
                "FIXED-ORIENTATION PAPER MOVE: PASS"
            )
        else:
            print(
                "FIXED-ORIENTATION PAPER MOVE: "
                "MOVE PASS / RECONNECT FAIL"
            )


def main():
    rospy.init_node(
        "cobotta_fixed_orientation_move_332_333_diagnostic",
        anonymous=True,
    )

    Diagnostic().run()


if __name__ == "__main__":
    main()
