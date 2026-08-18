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

BEST_PATH_CSV = os.path.expanduser(
    "~/cobotta_branch_preserving_best_path.csv"
)

OUTPUT_CSV = os.path.expanduser(
    "~/cobotta_transition_311_312_diagnostic.csv"
)

GROUP = "cobotta_arm"
TIP = "cobotta_tool_link"
FRAME = "paper_center"

FROM_INDEX = 311
TO_INDEX = 312

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
MAX_JUMP_DEG = 10.0
MAX_JUMP_RAD = math.radians(MAX_JUMP_DEG)

SUBDIVISION_CANDIDATES = [8, 16, 32, 64]

ACTIVE_JOINTS = [
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
        q = [
            (1.0-t)*a + t*b
            for a, b in zip(q0, q1)
        ]
        return normalize(q)

    theta0 = math.acos(dot)
    sin_theta0 = math.sin(theta0)

    a = math.sin((1.0-t)*theta0) / sin_theta0
    b = math.sin(t*theta0) / sin_theta0

    return normalize([
        a*x + b*y
        for x, y in zip(q0, q1)
    ])


class TransitionDiagnostic:

    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)

        self.group = moveit_commander.MoveGroupCommander(
            GROUP,
            wait_for_servers=20.0,
        )

        self.active_joints = list(
            self.group.get_active_joints()
        )

        if self.active_joints != ACTIVE_JOINTS:
            print("active joints:", self.active_joints)

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
            doc = yaml.safe_load(f)

        self.points = doc["points"]

        with open(
            BEST_PATH_CSV,
            newline="",
            encoding="utf-8-sig",
        ) as f:
            rows = list(csv.DictReader(f))

        self.best = {
            int(row["index"]): row
            for row in rows
        }

    def paper_pose(self, index):
        tf = self.points[index]["transforms"][0]

        t = tf["translation"]
        r = tf["rotation"]

        p = [
            float(t["x"]),
            float(t["y"]),
            float(t["z"]),
        ]

        q = normalize([
            float(r["x"]),
            float(r["y"]),
            float(r["z"]),
            float(r["w"]),
        ])

        return p, q

    def interpolated_tool_pose(self, ratio):
        p0, q0 = self.paper_pose(FROM_INDEX)
        p1, q1 = self.paper_pose(TO_INDEX)

        paper_p = [
            (1.0-ratio)*a + ratio*b
            for a, b in zip(p0, p1)
        ]

        paper_q = slerp(
            q0,
            q1,
            ratio,
        )

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

    def state_from_saved_index(self, index):
        row = self.best[index]

        state = self.group.get_current_state()

        names = list(state.joint_state.name)
        positions = list(state.joint_state.position)

        lookup = {
            name: i
            for i, name in enumerate(names)
        }

        joints = []

        for name in self.active_joints:
            value = float(row[name])

            positions[lookup[name]] = value
            joints.append(value)

        state.joint_state.position = positions
        state.joint_state.header.stamp = rospy.Time(0)
        state.is_diff = False

        return state, joints

    def active_positions(self, state):
        lookup = dict(zip(
            state.joint_state.name,
            state.joint_state.position,
        ))

        return [
            float(lookup[name])
            for name in self.active_joints
        ]

    def state_validity(self, state):
        req = GetStateValidityRequest()
        req.robot_state = state
        req.group_name = GROUP

        return self.check_validity(req)

    def contact_text(self, validity):
        values = []

        for c in validity.contacts:
            values.append(
                "{}<->{} depth={:.6f}".format(
                    c.contact_body_1,
                    c.contact_body_2,
                    c.depth,
                )
            )

        return "; ".join(values)

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
            return "IK_FAILED", None, ""

        validity = self.state_validity(
            res.solution
        )

        if not validity.valid:
            return (
                "ENDPOINT_COLLISION",
                None,
                self.contact_text(validity),
            )

        return "VALID", res.solution, ""

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

        names = list(state.joint_state.name)
        positions = list(state.joint_state.position)

        lookup = {
            name: i
            for i, name in enumerate(names)
        }

        for name, value in zip(
            self.active_joints,
            q,
        ):
            positions[lookup[name]] = value

        state.joint_state.position = positions
        state.joint_state.header.stamp = rospy.Time(0)
        state.is_diff = False

        return state

    def validate_edge(
        self,
        state_a,
        state_b,
    ):
        qa = self.active_positions(state_a)
        qb = self.active_positions(state_b)

        deltas = [
            abs(b-a)
            for a, b in zip(qa, qb)
        ]

        max_delta = max(deltas)

        if max_delta > MAX_JUMP_RAD:
            return (
                False,
                "JOINT_JUMP",
                0.0,
                max_delta,
                "",
            )

        steps = max(
            1,
            int(math.ceil(
                max_delta
                / MAX_JOINT_STEP_RAD
            ))
        )

        for k in range(1, steps + 1):
            ratio = float(k) / float(steps)

            state = self.interpolate_state(
                state_a,
                state_b,
                ratio,
            )

            validity = self.state_validity(
                state
            )

            if not validity.valid:
                return (
                    False,
                    "EDGE_COLLISION",
                    ratio,
                    max_delta,
                    self.contact_text(validity),
                )

        return (
            True,
            "VALID",
            1.0,
            max_delta,
            "",
        )

    def run_one(self, subdivisions):
        state, joints = self.state_from_saved_index(
            FROM_INDEX
        )

        start_validity = self.state_validity(
            state
        )

        if not start_validity.valid:
            raise RuntimeError(
                "Saved index311 itself is invalid: "
                + self.contact_text(start_validity)
            )

        rows = []

        print()
        print("=" * 90)
        print(
            "TEST subdivisions={}".format(
                subdivisions
            )
        )
        print("=" * 90)

        success = True

        for step in range(1, subdivisions + 1):
            ratio = float(step) / float(subdivisions)

            pose, x_deg, y_deg = (
                self.interpolated_tool_pose(
                    ratio
                )
            )

            status, solution, contacts = (
                self.solve(
                    pose,
                    state,
                )
            )

            if solution is None:
                print(
                    "step {}/{} ratio={:.6f} "
                    "X={:+.3f} Y={:+.3f} "
                    "{}".format(
                        step,
                        subdivisions,
                        ratio,
                        x_deg,
                        y_deg,
                        status,
                    )
                )

                if contacts:
                    print(
                        "  contacts:",
                        contacts,
                    )

                rows.append({
                    "subdivisions": subdivisions,
                    "step": step,
                    "ratio": ratio,
                    "local_x_deg": x_deg,
                    "local_y_deg": y_deg,
                    "status": status,
                    "edge_ratio": "",
                    "max_edge_delta_deg": "",
                    "contacts": contacts,
                })

                success = False
                break

            edge_ok, edge_status, edge_ratio, \
                max_edge_delta, edge_contacts = \
                self.validate_edge(
                    state,
                    solution,
                )

            print(
                "step {}/{} ratio={:.6f} "
                "X={:+.3f} Y={:+.3f} "
                "endpoint={} edge={} "
                "maxΔ={:.6f} deg".format(
                    step,
                    subdivisions,
                    ratio,
                    x_deg,
                    y_deg,
                    status,
                    edge_status,
                    math.degrees(max_edge_delta),
                )
            )

            if edge_contacts:
                print(
                    "  contacts:",
                    edge_contacts,
                )

            rows.append({
                "subdivisions": subdivisions,
                "step": step,
                "ratio": ratio,
                "local_x_deg": x_deg,
                "local_y_deg": y_deg,
                "status": status,
                "edge_ratio": edge_ratio,
                "max_edge_delta_deg":
                    math.degrees(max_edge_delta),
                "contacts": edge_contacts,
            })

            if not edge_ok:
                success = False
                break

            state = copy.deepcopy(solution)
            joints = self.active_positions(
                solution
            )

        if success:
            saved312_state, saved312_q = (
                self.state_from_saved_index(
                    TO_INDEX
                )
            )

            final_q = self.active_positions(
                state
            )

            reconnect = [
                abs(a-b)
                for a, b in zip(
                    final_q,
                    saved312_q,
                )
            ]

            max_reconnect = max(reconnect)

            print()
            print(
                "index312 reconnect difference:"
            )

            print(
                "  max = {:.9f} deg".format(
                    math.degrees(
                        max_reconnect
                    )
                )
            )

            print(
                "TRANSITION PASS "
                "(subdivisions={})".format(
                    subdivisions
                )
            )

            return True, rows, state

        print(
            "TRANSITION FAIL "
            "(subdivisions={})".format(
                subdivisions
            )
        )

        return False, rows, None

    def run(self):
        all_rows = []

        selected = None

        for subdivisions in (
            SUBDIVISION_CANDIDATES
        ):
            ok, rows, final_state = (
                self.run_one(
                    subdivisions
                )
            )

            all_rows.extend(rows)

            if ok:
                selected = subdivisions
                break

        fields = [
            "subdivisions",
            "step",
            "ratio",
            "local_x_deg",
            "local_y_deg",
            "status",
            "edge_ratio",
            "max_edge_delta_deg",
            "contacts",
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
            writer.writerows(all_rows)

        print()
        print("=" * 90)
        print("FINAL RESULT")
        print("=" * 90)

        if selected is None:
            print(
                "SAFE TRANSITION: NOT FOUND"
            )
        else:
            print(
                "SAFE TRANSITION: FOUND"
            )
            print(
                "selected subdivisions:",
                selected,
            )

        print("OUTPUT:", OUTPUT_CSV)


def main():
    rospy.init_node(
        "cobotta_transition_311_312_diagnostic",
        anonymous=True,
    )

    diagnostic = TransitionDiagnostic()
    diagnostic.run()


if __name__ == "__main__":
    main()
