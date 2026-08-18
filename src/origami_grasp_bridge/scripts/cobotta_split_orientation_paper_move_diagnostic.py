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

SUMMARY_CSV = os.path.expanduser(
    "~/cobotta_split_motion_diagnostic_summary.csv"
)

DETAIL_CSV = os.path.expanduser(
    "~/cobotta_split_motion_diagnostic_detail.csv"
)

GROUP = "cobotta_arm"
TIP = "cobotta_tool_link"
FRAME = "paper_center"

R_GRASP = [
    +0.000401,
    -0.001507,
    -0.004894,
]

IK_TIMEOUT = 0.10
MAX_JOINT_STEP_RAD = 0.010

DEFAULT_SUBDIVISIONS = 8
DEFAULT_MOVE_THRESHOLD_DEG = 2.0


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

    dot = max(-1.0, min(1.0, dot))

    if dot > 0.9995:
        return normalize([
            (1.0-t)*a + t*b
            for a, b in zip(q0, q1)
        ])

    theta = math.acos(dot)
    sin_theta = math.sin(theta)

    a = (
        math.sin((1.0-t)*theta)
        / sin_theta
    )

    b = (
        math.sin(t*theta)
        / sin_theta
    )

    return normalize([
        a*x + b*y
        for x, y in zip(q0, q1)
    ])


class Diagnostic:

    def __init__(self):
        moveit_commander.roscpp_initialize(
            sys.argv
        )

        self.subdivisions = int(
            rospy.get_param(
                "~subdivisions",
                DEFAULT_SUBDIVISIONS,
            )
        )

        self.move_threshold_deg = float(
            rospy.get_param(
                "~move_threshold_deg",
                DEFAULT_MOVE_THRESHOLD_DEG,
            )
        )

        if self.subdivisions < 1:
            raise ValueError(
                "subdivisions must be >= 1"
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
            FULL_PATH_CSV,
            newline="",
            encoding="utf-8-sig",
        ) as f:
            rows = list(
                csv.DictReader(f)
            )

        self.rows = sorted(
            rows,
            key=lambda r: int(
                r["path_point"]
            ),
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

    def state_from_row(self, row):
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

        for name in self.active_joints:
            if name not in lookup:
                raise RuntimeError(
                    "RobotState missing {}".format(
                        name
                    )
                )

            positions[
                lookup[name]
            ] = float(row[name])

        state.joint_state.position = positions
        state.joint_state.header.stamp = (
            rospy.Time(0)
        )
        state.is_diff = False

        return state

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

    def make_tool_pose(
        self,
        paper_p,
        paper_q,
        x_deg,
        y_deg,
    ):
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

    def fixed_paper_pose(
        self,
        paper_index,
        x_deg,
        y_deg,
    ):
        p, q = self.paper_pose(
            paper_index
        )

        return self.make_tool_pose(
            p,
            q,
            x_deg,
            y_deg,
        )

    def moving_paper_pose(
        self,
        from_index,
        to_index,
        ratio,
        x_deg,
        y_deg,
    ):
        p0, q0 = self.paper_pose(
            from_index
        )

        p1, q1 = self.paper_pose(
            to_index
        )

        p = [
            (1.0-ratio)*a + ratio*b
            for a, b in zip(p0, p1)
        ]

        q = slerp(
            q0,
            q1,
            ratio,
        )

        return self.make_tool_pose(
            p,
            q,
            x_deg,
            y_deg,
        )

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

        req.ik_request.robot_state = (
            copy.deepcopy(
                seed_state
            )
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
        state_a,
        state_b,
        ratio,
    ):
        qa = self.active_positions(
            state_a
        )

        qb = self.active_positions(
            state_b
        )

        q = [
            a + ratio*(b-a)
            for a, b in zip(qa, qb)
        ]

        state = copy.deepcopy(
            state_a
        )

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

    def validate_edge(
        self,
        state_a,
        state_b,
    ):
        qa = self.active_positions(
            state_a
        )

        qb = self.active_positions(
            state_b
        )

        max_delta = max(
            abs(b-a)
            for a, b in zip(qa, qb)
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
                state_a,
                state_b,
                ratio,
            )

            validity = self.check_state(
                state
            )

            if not validity.valid:
                return (
                    False,
                    "EDGE_COLLISION "
                    "step={}/{} ratio={:.6f} {}".format(
                        step,
                        steps,
                        ratio,
                        self.contacts_text(
                            validity
                        ),
                    ),
                )

        return True, "VALID"

    def max_delta_deg(
        self,
        state_a,
        state_b,
    ):
        qa = self.active_positions(
            state_a
        )

        qb = self.active_positions(
            state_b
        )

        return max(
            abs(math.degrees(b-a))
            for a, b in zip(qa, qb)
        )

    def find_targets(self):
        targets = []

        for i in range(
            len(self.rows) - 1
        ):
            a = self.rows[i]
            b = self.rows[i + 1]

            xa = float(
                a["local_x_deg"]
            )
            ya = float(
                a["local_y_deg"]
            )

            xb = float(
                b["local_x_deg"]
            )
            yb = float(
                b["local_y_deg"]
            )

            orientation_changed = (
                abs(xb-xa) > 1.0e-9
                or
                abs(yb-ya) > 1.0e-9
            )

            if not orientation_changed:
                continue

            qa = [
                float(a[name])
                for name in self.active_joints
            ]

            qb = [
                float(b[name])
                for name in self.active_joints
            ]

            deltas_deg = [
                abs(math.degrees(y-x))
                for x, y in zip(qa, qb)
            ]

            max_delta_deg = max(
                deltas_deg
            )

            if (
                max_delta_deg
                < self.move_threshold_deg
            ):
                continue

            targets.append({
                "row_from": a,
                "row_to": b,

                "path_from":
                    int(a["path_point"]),
                "path_to":
                    int(b["path_point"]),

                "paper_from":
                    int(a["paper_index"]),
                "paper_to":
                    int(b["paper_index"]),

                "x_from": xa,
                "y_from": ya,
                "x_to": xb,
                "y_to": yb,

                "original_max_delta_deg":
                    max_delta_deg,

                "dominant_joint":
                    self.active_joints[
                        deltas_deg.index(
                            max_delta_deg
                        )
                    ],
            })

        return targets

    def detail_row(
        self,
        target,
        phase,
        step,
        ratio,
        x_deg,
        y_deg,
        endpoint_status,
        edge_status,
        max_delta_deg,
        state,
    ):
        row = {
            "path_from":
                target["path_from"],
            "path_to":
                target["path_to"],

            "paper_from":
                target["paper_from"],
            "paper_to":
                target["paper_to"],

            "phase": phase,
            "step": step,
            "ratio": ratio,

            "local_x_deg": x_deg,
            "local_y_deg": y_deg,

            "endpoint_status":
                endpoint_status,

            "edge_status":
                edge_status,

            "max_joint_delta_deg":
                max_delta_deg,
        }

        if state is not None:
            q = self.active_positions(
                state
            )

            for name, value in zip(
                self.active_joints,
                q,
            ):
                row[name] = value

        return row

    def run_transition(
        self,
        target,
        detail_rows,
    ):
        state = self.state_from_row(
            target["row_from"]
        )

        start_validity = self.check_state(
            state
        )

        if not start_validity.valid:
            return {
                "status": "START_INVALID",
                "reason":
                    self.contacts_text(
                        start_validity
                    ),
            }

        max_orientation_step = 0.0
        max_paper_step = 0.0

        # ====================================================
        # PHASE 1
        # paper_from の把持点を固定して
        # local X/Yだけを変更
        # ====================================================

        for step in range(
            1,
            self.subdivisions + 1,
        ):
            ratio = (
                float(step)
                / float(self.subdivisions)
            )

            x_deg = (
                target["x_from"]
                + ratio
                * (
                    target["x_to"]
                    - target["x_from"]
                )
            )

            y_deg = (
                target["y_from"]
                + ratio
                * (
                    target["y_to"]
                    - target["y_from"]
                )
            )

            pose = self.fixed_paper_pose(
                target["paper_from"],
                x_deg,
                y_deg,
            )

            solution, endpoint_status = (
                self.solve(
                    pose,
                    state,
                )
            )

            if solution is None:
                detail_rows.append(
                    self.detail_row(
                        target,
                        "ORIENTATION",
                        step,
                        ratio,
                        x_deg,
                        y_deg,
                        endpoint_status,
                        "",
                        "",
                        None,
                    )
                )

                return {
                    "status":
                        "ORIENTATION_FAIL",
                    "reason":
                        endpoint_status,
                }

            edge_valid, edge_status = (
                self.validate_edge(
                    state,
                    solution,
                )
            )

            delta = self.max_delta_deg(
                state,
                solution,
            )

            max_orientation_step = max(
                max_orientation_step,
                delta,
            )

            detail_rows.append(
                self.detail_row(
                    target,
                    "ORIENTATION",
                    step,
                    ratio,
                    x_deg,
                    y_deg,
                    endpoint_status,
                    edge_status,
                    delta,
                    solution,
                )
            )

            if not edge_valid:
                return {
                    "status":
                        "ORIENTATION_EDGE_FAIL",
                    "reason":
                        edge_status,
                }

            state = copy.deepcopy(
                solution
            )

        # ====================================================
        # PHASE 2
        # 新しいlocal X/Yを固定したまま
        # paper_from -> paper_to
        # ====================================================

        for step in range(
            1,
            self.subdivisions + 1,
        ):
            ratio = (
                float(step)
                / float(self.subdivisions)
            )

            pose = self.moving_paper_pose(
                target["paper_from"],
                target["paper_to"],
                ratio,
                target["x_to"],
                target["y_to"],
            )

            solution, endpoint_status = (
                self.solve(
                    pose,
                    state,
                )
            )

            if solution is None:
                detail_rows.append(
                    self.detail_row(
                        target,
                        "PAPER_MOVE",
                        step,
                        ratio,
                        target["x_to"],
                        target["y_to"],
                        endpoint_status,
                        "",
                        "",
                        None,
                    )
                )

                return {
                    "status":
                        "PAPER_MOVE_FAIL",
                    "reason":
                        endpoint_status,
                }

            edge_valid, edge_status = (
                self.validate_edge(
                    state,
                    solution,
                )
            )

            delta = self.max_delta_deg(
                state,
                solution,
            )

            max_paper_step = max(
                max_paper_step,
                delta,
            )

            detail_rows.append(
                self.detail_row(
                    target,
                    "PAPER_MOVE",
                    step,
                    ratio,
                    target["x_to"],
                    target["y_to"],
                    endpoint_status,
                    edge_status,
                    delta,
                    solution,
                )
            )

            if not edge_valid:
                return {
                    "status":
                        "PAPER_MOVE_EDGE_FAIL",
                    "reason":
                        edge_status,
                }

            state = copy.deepcopy(
                solution
            )

        # ====================================================
        # 元の次点へ再接続
        # ====================================================

        saved_next = self.state_from_row(
            target["row_to"]
        )

        q_new = self.active_positions(
            state
        )

        q_saved = self.active_positions(
            saved_next
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
                state,
                saved_next,
            )
        )

        if not reconnect_valid:
            return {
                "status":
                    "RECONNECT_FAIL",
                "reason":
                    reconnect_status,
                "reconnect_deg":
                    reconnect_deg,
                "max_orientation_step":
                    max_orientation_step,
                "max_paper_step":
                    max_paper_step,
            }

        return {
            "status": "PASS",
            "reason": "",
            "reconnect_deg":
                reconnect_deg,
            "max_orientation_step":
                max_orientation_step,
            "max_paper_step":
                max_paper_step,
        }

    def run(self):
        targets = self.find_targets()

        print("=" * 100)
        print(
            "COBOTTA SPLIT ORIENTATION / PAPER MOVE DIAGNOSTIC"
        )
        print("=" * 100)

        print(
            "subdivisions         :",
            self.subdivisions,
        )

        print(
            "move threshold       : "
            "{:.3f} deg".format(
                self.move_threshold_deg
            )
        )

        print(
            "detected transitions :",
            len(targets),
        )

        print()

        for t in targets:
            print(
                "pp {}->{}  paper {}->{}  "
                "({:+.1f},{:+.1f}) -> "
                "({:+.1f},{:+.1f})  "
                "original maxΔ={:.3f}° {}".format(
                    t["path_from"],
                    t["path_to"],
                    t["paper_from"],
                    t["paper_to"],
                    t["x_from"],
                    t["y_from"],
                    t["x_to"],
                    t["y_to"],
                    t[
                        "original_max_delta_deg"
                    ],
                    t["dominant_joint"],
                )
            )

        print()

        detail_rows = []
        summary_rows = []

        for number, target in enumerate(
            targets,
            1,
        ):
            print("-" * 100)

            print(
                "[{}/{}] pp {}->{} / paper {}->{}".format(
                    number,
                    len(targets),
                    target["path_from"],
                    target["path_to"],
                    target["paper_from"],
                    target["paper_to"],
                )
            )

            result = self.run_transition(
                target,
                detail_rows,
            )

            summary = {
                "path_from":
                    target["path_from"],
                "path_to":
                    target["path_to"],

                "paper_from":
                    target["paper_from"],
                "paper_to":
                    target["paper_to"],

                "x_from":
                    target["x_from"],
                "y_from":
                    target["y_from"],
                "x_to":
                    target["x_to"],
                "y_to":
                    target["y_to"],

                "original_max_delta_deg":
                    target[
                        "original_max_delta_deg"
                    ],

                "dominant_joint":
                    target[
                        "dominant_joint"
                    ],

                "status":
                    result["status"],

                "reason":
                    result.get(
                        "reason",
                        "",
                    ),

                "max_orientation_step_deg":
                    result.get(
                        "max_orientation_step",
                        "",
                    ),

                "max_paper_step_deg":
                    result.get(
                        "max_paper_step",
                        "",
                    ),

                "reconnect_difference_deg":
                    result.get(
                        "reconnect_deg",
                        "",
                    ),
            }

            summary_rows.append(
                summary
            )

            print(
                "RESULT:",
                result["status"],
            )

            if (
                "max_orientation_step"
                in result
            ):
                print(
                    "  orientation max step : "
                    "{:.6f} deg".format(
                        result[
                            "max_orientation_step"
                        ]
                    )
                )

            if "max_paper_step" in result:
                print(
                    "  paper move max step  : "
                    "{:.6f} deg".format(
                        result[
                            "max_paper_step"
                        ]
                    )
                )

            if "reconnect_deg" in result:
                print(
                    "  reconnect difference : "
                    "{:.9f} deg".format(
                        result[
                            "reconnect_deg"
                        ]
                    )
                )

            if result.get("reason"):
                print(
                    "  reason:",
                    result["reason"],
                )

        # ====================================================
        # SAVE
        # ====================================================

        summary_fields = [
            "path_from",
            "path_to",
            "paper_from",
            "paper_to",
            "x_from",
            "y_from",
            "x_to",
            "y_to",
            "original_max_delta_deg",
            "dominant_joint",
            "status",
            "reason",
            "max_orientation_step_deg",
            "max_paper_step_deg",
            "reconnect_difference_deg",
        ]

        with open(
            SUMMARY_CSV,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=summary_fields,
            )

            writer.writeheader()
            writer.writerows(
                summary_rows
            )

        detail_fields = [
            "path_from",
            "path_to",
            "paper_from",
            "paper_to",
            "phase",
            "step",
            "ratio",
            "local_x_deg",
            "local_y_deg",
            "endpoint_status",
            "edge_status",
            "max_joint_delta_deg",
            *self.active_joints,
        ]

        with open(
            DETAIL_CSV,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=detail_fields,
            )

            writer.writeheader()
            writer.writerows(
                detail_rows
            )

        pass_count = sum(
            1
            for r in summary_rows
            if r["status"] == "PASS"
        )

        print()
        print("=" * 100)
        print("FINAL RESULT")
        print("=" * 100)

        print(
            "target transitions :",
            len(targets),
        )

        print(
            "PASS               :",
            pass_count,
        )

        print(
            "FAIL               :",
            len(targets) - pass_count,
        )

        print()
        print(
            "SUMMARY:",
            SUMMARY_CSV,
        )

        print(
            "DETAIL :",
            DETAIL_CSV,
        )

        print()

        if (
            targets
            and
            pass_count == len(targets)
        ):
            print(
                "ALL SPLIT TRANSITIONS: PASS"
            )
        else:
            print(
                "SPLIT TRANSITION DIAGNOSTIC: "
                "SOME TRANSITIONS FAILED"
            )


def main():
    rospy.init_node(
        "cobotta_split_orientation_paper_move_diagnostic",
        anonymous=True,
    )

    Diagnostic().run()


if __name__ == "__main__":
    main()
