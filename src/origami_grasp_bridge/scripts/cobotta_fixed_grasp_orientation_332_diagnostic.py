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

FULL_PATH_CSV = os.path.expanduser(
    "~/cobotta_branch_preserving_full_path_0_340_edge_fixed.csv"
)

OUTPUT_CSV = os.path.expanduser(
    "~/cobotta_fixed_grasp_orientation_332_diagnostic.csv"
)

GROUP = "cobotta_arm"
TIP = "cobotta_tool_link"
FRAME = "paper_center"

PATH_POINT_START = 339
PAPER_INDEX = 332

START_X_DEG = -10.0
END_X_DEG = -15.0
Y_DEG = +30.0

SUBDIVISIONS = 8

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


class Diagnostic:

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
            self.paper_points = yaml.safe_load(f)["points"]

        with open(
            FULL_PATH_CSV,
            newline="",
            encoding="utf-8-sig",
        ) as f:
            self.path_rows = list(csv.DictReader(f))

        if PATH_POINT_START >= len(self.path_rows):
            raise RuntimeError(
                "path_point {} does not exist".format(
                    PATH_POINT_START
                )
            )

        row = self.path_rows[PATH_POINT_START]

        if int(row["paper_index"]) != PAPER_INDEX:
            raise RuntimeError(
                "path_point {} paper_index is {}, expected {}".format(
                    PATH_POINT_START,
                    row["paper_index"],
                    PAPER_INDEX,
                )
            )

        self.start_row = row

    def active_positions(self, state):
        lookup = dict(zip(
            state.joint_state.name,
            state.joint_state.position,
        ))

        return [
            float(lookup[name])
            for name in self.active_joints
        ]

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

    def paper_pose(self):
        tf = self.paper_points[
            PAPER_INDEX
        ]["transforms"][0]

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

    def make_tool_pose(self, x_deg):
        paper_p, paper_q = self.paper_pose()

        q_delta = multiply(
            axis_q("x", x_deg),
            axis_q("y", Y_DEG),
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

        # 紙側の実把持点 paper_p は固定。
        # 工具姿勢に応じてtool originだけを補正する。
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

        return pose

    def check_state(self, state):
        req = GetStateValidityRequest()
        req.robot_state = state
        req.group_name = GROUP

        return self.check_validity(req)

    def contacts_text(self, result):
        return "; ".join(
            "{}<->{} depth={:.6f}".format(
                c.contact_body_1,
                c.contact_body_2,
                c.depth,
            )
            for c in result.contacts
        )

    def solve(self, pose, seed_state):
        req = GetPositionIKRequest()

        req.ik_request.group_name = GROUP
        req.ik_request.ik_link_name = TIP
        req.ik_request.pose_stamped = pose
        req.ik_request.robot_state = copy.deepcopy(seed_state)
        req.ik_request.avoid_collisions = False
        req.ik_request.timeout = rospy.Duration(IK_TIMEOUT)

        res = self.compute_ik(req)

        if res.error_code.val != MoveItErrorCodes.SUCCESS:
            return None, "IK_FAILED"

        validity = self.check_state(res.solution)

        if not validity.valid:
            return (
                None,
                "ENDPOINT_COLLISION "
                + self.contacts_text(validity),
            )

        return res.solution, "VALID"

    def interpolate_state(self, a, b, ratio):
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
            positions[lookup[name]] = value

        state.joint_state.position = positions
        state.joint_state.header.stamp = rospy.Time(0)

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

        for step in range(1, steps + 1):
            ratio = float(step) / float(steps)

            state = self.interpolate_state(
                a,
                b,
                ratio,
            )

            result = self.check_validity(
                GetStateValidityRequest(
                    robot_state=state,
                    group_name=GROUP,
                )
            )

            if not result.valid:
                return (
                    False,
                    "EDGE_COLLISION step={}/{} ratio={:.6f} {}".format(
                        step,
                        steps,
                        ratio,
                        self.contacts_text(result),
                    ),
                )

        return True, "VALID"

    def run(self):
        current_state = self.state_from_row(
            self.start_row
        )

        output_rows = []

        print("=" * 90)
        print(
            "FIXED-GRASP ORIENTATION TRANSITION"
        )
        print("=" * 90)

        print(
            "paper index       :",
            PAPER_INDEX,
        )

        print(
            "paper grasp point : FIXED"
        )

        print(
            "local Y           : {:+.3f} deg FIXED".format(
                Y_DEG
            )
        )

        print(
            "local X           : {:+.3f} -> {:+.3f} deg".format(
                START_X_DEG,
                END_X_DEG,
            )
        )

        print(
            "subdivisions      :",
            SUBDIVISIONS,
        )

        print()

        previous_x = START_X_DEG

        for step in range(
            1,
            SUBDIVISIONS + 1,
        ):
            ratio = (
                float(step)
                / float(SUBDIVISIONS)
            )

            x_deg = (
                START_X_DEG
                + ratio
                * (END_X_DEG - START_X_DEG)
            )

            pose = self.make_tool_pose(
                x_deg
            )

            solution, endpoint_status = self.solve(
                pose,
                current_state,
            )

            if solution is None:
                print(
                    "step {}/{} X={:+.3f} endpoint={}".format(
                        step,
                        SUBDIVISIONS,
                        x_deg,
                        endpoint_status,
                    )
                )

                print()
                print(
                    "FIXED-GRASP ORIENTATION TRANSITION: FAIL"
                )
                return

            edge_valid, edge_status = self.validate_edge(
                current_state,
                solution,
            )

            qa = self.active_positions(
                current_state
            )

            qb = self.active_positions(
                solution
            )

            deltas_deg = [
                abs(math.degrees(b-a))
                for a, b in zip(qa, qb)
            ]

            max_delta_deg = max(
                deltas_deg
            )

            joints = self.active_positions(
                solution
            )

            output_row = {
                "step": step,
                "ratio": ratio,
                "paper_index": PAPER_INDEX,
                "local_x_deg": x_deg,
                "local_y_deg": Y_DEG,
                "endpoint_status": endpoint_status,
                "edge_status": edge_status,
                "max_joint_delta_deg": max_delta_deg,
            }

            for name, value in zip(
                self.active_joints,
                joints,
            ):
                output_row[name] = value

            output_rows.append(
                output_row
            )

            print(
                "step {}/{} "
                "X={:+.3f} "
                "endpoint={} "
                "edge={} "
                "maxΔ={:.6f} deg".format(
                    step,
                    SUBDIVISIONS,
                    x_deg,
                    endpoint_status,
                    edge_status,
                    max_delta_deg,
                )
            )

            if not edge_valid:
                print()
                print(
                    "FIXED-GRASP ORIENTATION TRANSITION: FAIL"
                )
                return

            current_state = copy.deepcopy(
                solution
            )

            previous_x = x_deg

        fields = [
            "step",
            "ratio",
            "paper_index",
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
            writer.writerows(
                output_rows
            )

        print()
        print("=" * 90)
        print("FINAL RESULT")
        print("=" * 90)

        print(
            "paper grasp point     : FIXED"
        )

        print(
            "orientation transition: "
            "({:+.3f},{:+.3f}) -> ({:+.3f},{:+.3f})".format(
                START_X_DEG,
                Y_DEG,
                END_X_DEG,
                Y_DEG,
            )
        )

        print(
            "successful steps      :",
            len(output_rows),
        )

        print(
            "OUTPUT:"
        )
        print(
            " ",
            OUTPUT_CSV,
        )

        print()
        print(
            "FIXED-GRASP ORIENTATION TRANSITION: PASS"
        )


def main():
    rospy.init_node(
        "cobotta_fixed_grasp_orientation_332_diagnostic",
        anonymous=True,
    )

    Diagnostic().run()


if __name__ == "__main__":
    main()
