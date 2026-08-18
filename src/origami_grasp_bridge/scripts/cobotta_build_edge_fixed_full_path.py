#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import csv
import math
import os
import sys

import moveit_commander
import rospy
import yaml

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

FULL_PATH_CSV = os.path.expanduser(
    "~/cobotta_branch_preserving_full_path_0_340.csv"
)

OUTPUT_CSV = os.path.expanduser(
    "~/cobotta_branch_preserving_full_path_0_340_edge_fixed.csv"
)

GROUP = "cobotta_arm"
TIP = "cobotta_tool_link"
FRAME = "paper_center"

FROM_INDEX = 311
TO_INDEX = 312
SUBDIVISIONS = 8

FROM_X_DEG = -5.0
FROM_Y_DEG = +5.0
TO_X_DEG = -5.0
TO_Y_DEG = +10.0

R_GRASP = [
    +0.000401,
    -0.001507,
    -0.004894,
]

IK_TIMEOUT = 0.10
MAX_JOINT_STEP_RAD = 0.010

JOINTS = [
    "cobotta_joint_1",
    "cobotta_joint_2",
    "cobotta_joint_3",
    "cobotta_joint_4",
    "cobotta_joint_5",
    "cobotta_joint_6",
]


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
    a = math.radians(deg) * 0.5
    s = math.sin(a)
    c = math.cos(a)

    if axis == "x":
        return [s, 0.0, 0.0, c]

    if axis == "y":
        return [0.0, s, 0.0, c]

    raise ValueError(axis)


def rotate_vector(q, v):
    x, y, z, w = normalize(q)
    vx, vy, vz = v

    r00 = 1.0 - 2.0*(y*y + z*z)
    r01 = 2.0*(x*y - z*w)
    r02 = 2.0*(x*z + y*w)

    r10 = 2.0*(x*y + z*w)
    r11 = 1.0 - 2.0*(x*x + z*z)
    r12 = 2.0*(y*z - x*w)

    r20 = 2.0*(x*z - y*w)
    r21 = 2.0*(y*z + x*w)
    r22 = 1.0 - 2.0*(x*x + y*y)

    return [
        r00*vx + r01*vy + r02*vz,
        r10*vx + r11*vy + r12*vz,
        r20*vx + r21*vy + r22*vz,
    ]


def slerp(q0, q1, t):
    q0 = normalize(q0)
    q1 = normalize(q1)

    dot = sum(a*b for a, b in zip(q0, q1))

    if dot < 0.0:
        q1 = [-v for v in q1]
        dot = -dot

    dot = max(-1.0, min(1.0, dot))

    if dot > 0.9995:
        return normalize([
            (1.0-t)*a + t*b
            for a, b in zip(q0, q1)
        ])

    theta = math.acos(dot)
    sin_theta = math.sin(theta)

    a = math.sin((1.0-t)*theta) / sin_theta
    b = math.sin(t*theta) / sin_theta

    return normalize([
        a*x + b*y
        for x, y in zip(q0, q1)
    ])


class Builder:

    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)

        self.group = moveit_commander.MoveGroupCommander(
            GROUP,
            wait_for_servers=20.0,
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
            self.points = yaml.safe_load(f)["points"]

        with open(
            FULL_PATH_CSV,
            newline="",
            encoding="utf-8-sig",
        ) as f:
            self.full_rows = list(csv.DictReader(f))

        if len(self.full_rows) != 341:
            raise RuntimeError(
                "Expected 341 input points, got {}".format(
                    len(self.full_rows)
                )
            )

        self.by_index = {
            int(r["index"]): r
            for r in self.full_rows
        }

    def state_from_row(self, row):
        state = self.group.get_current_state()

        names = list(state.joint_state.name)
        positions = list(state.joint_state.position)

        lookup = {
            name: i
            for i, name in enumerate(names)
        }

        for name in self.active_joints:
            positions[lookup[name]] = float(row[name])

        state.joint_state.position = positions
        state.joint_state.header.stamp = rospy.Time(0)
        state.is_diff = False

        return state

    def active_positions(self, state):
        lookup = dict(zip(
            state.joint_state.name,
            state.joint_state.position,
        ))

        return [
            float(lookup[name])
            for name in self.active_joints
        ]

    def paper_pose(self, index):
        tf = self.points[index]["transforms"][0]

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

    def tool_pose(self, ratio):
        p0, q0 = self.paper_pose(FROM_INDEX)
        p1, q1 = self.paper_pose(TO_INDEX)

        paper_p = [
            (1.0-ratio)*a + ratio*b
            for a, b in zip(p0, p1)
        ]

        paper_q = slerp(q0, q1, ratio)

        x_deg = (
            (1.0-ratio)*FROM_X_DEG
            + ratio*TO_X_DEG
        )

        y_deg = (
            (1.0-ratio)*FROM_Y_DEG
            + ratio*TO_Y_DEG
        )

        q_delta = multiply(
            axis_q("x", x_deg),
            axis_q("y", y_deg),
        )

        tool_q = normalize(
            multiply(paper_q, q_delta)
        )

        offset_world = rotate_vector(
            tool_q,
            R_GRASP,
        )

        tool_p = [
            paper_p[i] - offset_world[i]
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

        return pose, x_deg, y_deg

    def check_state(self, state):
        req = GetStateValidityRequest()
        req.robot_state = state
        req.group_name = GROUP
        return self.check_validity(req)

    def contact_text(self, validity):
        return "; ".join(
            "{}<->{} depth={:.6f}".format(
                c.contact_body_1,
                c.contact_body_2,
                c.depth,
            )
            for c in validity.contacts
        )

    def solve(self, pose, seed_state):
        req = GetPositionIKRequest()

        req.ik_request.group_name = GROUP
        req.ik_request.ik_link_name = TIP
        req.ik_request.pose_stamped = pose
        req.ik_request.robot_state = copy.deepcopy(
            seed_state
        )
        req.ik_request.avoid_collisions = False
        req.ik_request.timeout = rospy.Duration(
            IK_TIMEOUT
        )

        res = self.compute_ik(req)

        if (
            res.error_code.val
            != MoveItErrorCodes.SUCCESS
        ):
            raise RuntimeError("IK_FAILED")

        validity = self.check_state(res.solution)

        if not validity.valid:
            raise RuntimeError(
                "ENDPOINT_COLLISION: "
                + self.contact_text(validity)
            )

        return res.solution

    def interpolate_state(
        self,
        state_a,
        state_b,
        ratio,
    ):
        qa = self.active_positions(state_a)
        qb = self.active_positions(state_b)

        q = [
            (1.0-ratio)*a + ratio*b
            for a, b in zip(qa, qb)
        ]

        state = copy.deepcopy(state_a)

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
            positions[lookup[name]] = value

        state.joint_state.position = positions
        state.joint_state.header.stamp = rospy.Time(0)

        return state

    def validate_edge(
        self,
        state_a,
        state_b,
    ):
        qa = self.active_positions(state_a)
        qb = self.active_positions(state_b)

        max_delta = max(
            abs(b-a)
            for a, b in zip(qa, qb)
        )

        steps = max(
            1,
            int(math.ceil(
                max_delta / MAX_JOINT_STEP_RAD
            ))
        )

        for k in range(1, steps + 1):
            ratio = float(k) / float(steps)

            state = self.interpolate_state(
                state_a,
                state_b,
                ratio,
            )

            validity = self.check_state(state)

            if not validity.valid:
                raise RuntimeError(
                    "EDGE_COLLISION ratio={:.6f}: {}".format(
                        ratio,
                        self.contact_text(validity),
                    )
                )

    def make_output_row(
        self,
        path_point,
        paper_index,
        sub_ratio,
        source,
        original_time,
        x_deg,
        y_deg,
        joints,
    ):
        row = {
            "path_point": path_point,
            "paper_index": paper_index,
            "sub_ratio": sub_ratio,
            "source": source,
            "original_time_from_start": original_time,
            "local_x_deg": x_deg,
            "local_y_deg": y_deg,
        }

        for name, value in zip(
            self.active_joints,
            joints,
        ):
            row[name] = "{:.15f}".format(value)

        return row

    def run(self):
        output = []

        path_point = 0

        # 0..311 をそのまま保存
        for index in range(0, FROM_INDEX + 1):
            row = self.by_index[index]

            output.append(
                self.make_output_row(
                    path_point=path_point,
                    paper_index=index,
                    sub_ratio=0.0,
                    source=row["source"],
                    original_time=row[
                        "original_time_from_start"
                    ],
                    x_deg=row["local_x_deg"],
                    y_deg=row["local_y_deg"],
                    joints=[
                        float(row[j])
                        for j in self.active_joints
                    ],
                )
            )

            path_point += 1

        current_state = self.state_from_row(
            self.by_index[FROM_INDEX]
        )

        print("=" * 90)
        print("BUILD SAFE TRANSITION 311 -> 312")
        print("=" * 90)

        # ratio 1/8 .. 7/8 の7点だけ挿入
        for step in range(1, SUBDIVISIONS):
            ratio = float(step) / float(SUBDIVISIONS)

            pose, x_deg, y_deg = self.tool_pose(
                ratio
            )

            solution = self.solve(
                pose,
                current_state,
            )

            self.validate_edge(
                current_state,
                solution,
            )

            joints = self.active_positions(
                solution
            )

            t0 = float(
                self.by_index[FROM_INDEX][
                    "original_time_from_start"
                ]
            )
            t1 = float(
                self.by_index[TO_INDEX][
                    "original_time_from_start"
                ]
            )

            original_time = (
                (1.0-ratio)*t0
                + ratio*t1
            )

            output.append(
                self.make_output_row(
                    path_point=path_point,
                    paper_index=FROM_INDEX,
                    sub_ratio=ratio,
                    source="edge_refinement_311_312",
                    original_time=original_time,
                    x_deg=x_deg,
                    y_deg=y_deg,
                    joints=joints,
                )
            )

            print(
                "insert {:d}/7 ratio={:.3f} "
                "X={:+.3f} Y={:+.3f} VALID".format(
                    step,
                    ratio,
                    x_deg,
                    y_deg,
                )
            )

            path_point += 1
            current_state = copy.deepcopy(
                solution
            )

        # 最後は既存の正式index312へ再接続
        saved312 = self.state_from_row(
            self.by_index[TO_INDEX]
        )

        self.validate_edge(
            current_state,
            saved312,
        )

        q_last = self.active_positions(
            current_state
        )
        q_saved = self.active_positions(
            saved312
        )

        reconnect_deg = max(
            abs(math.degrees(a-b))
            for a, b in zip(q_last, q_saved)
        )

        print()
        print(
            "last inserted -> saved index312:"
        )
        print(
            "  max joint difference = "
            "{:.9f} deg".format(
                reconnect_deg
            )
        )
        print("  edge validation = PASS")

        # 312..340 は元の正式pathへ復帰
        for index in range(
            TO_INDEX,
            341,
        ):
            row = self.by_index[index]

            output.append(
                self.make_output_row(
                    path_point=path_point,
                    paper_index=index,
                    sub_ratio=0.0,
                    source=row["source"],
                    original_time=row[
                        "original_time_from_start"
                    ],
                    x_deg=row["local_x_deg"],
                    y_deg=row["local_y_deg"],
                    joints=[
                        float(row[j])
                        for j in self.active_joints
                    ],
                )
            )

            path_point += 1

        if len(output) != 348:
            raise RuntimeError(
                "Unexpected output count: {}".format(
                    len(output)
                )
            )

        fields = [
            "path_point",
            "paper_index",
            "sub_ratio",
            "source",
            "original_time_from_start",
            "local_x_deg",
            "local_y_deg",
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
            writer.writerows(output)

        print()
        print("=" * 90)
        print("BUILD RESULT")
        print("=" * 90)
        print("original points :", 341)
        print("inserted points :", 7)
        print("output points   :", len(output))
        print("FINISH paper index: 340")
        print("OUTPUT:", OUTPUT_CSV)
        print()
        print("EDGE-FIXED PATH BUILD: SUCCESS")


def main():
    rospy.init_node(
        "cobotta_build_edge_fixed_full_path",
        anonymous=True,
    )

    Builder().run()


if __name__ == "__main__":
    main()
