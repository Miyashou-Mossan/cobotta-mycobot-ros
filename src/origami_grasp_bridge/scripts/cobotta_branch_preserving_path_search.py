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


DEFAULT_INPUT_YAML = os.path.expanduser(
    "~/directionA_reverse_local_z_m25_multidof.yaml"
)

DEFAULT_BASE_SCAN_CSV = os.path.expanduser(
    "~/directionA_reverse_local_z_m25_actual_grasp_collision_scan.csv"
)

DEFAULT_OUTPUT_CSV = os.path.expanduser(
    "~/cobotta_branch_preserving_path_search.csv"
)

DEFAULT_BEST_PATH_CSV = os.path.expanduser(
    "~/cobotta_branch_preserving_path_best.csv"
)

DEFAULT_GROUP = "cobotta_arm"
DEFAULT_TIP = "cobotta_tool_link"
DEFAULT_FRAME = "paper_center"

# 探索区間
DEFAULT_SEARCH_START = 260
DEFAULT_SEARCH_END = 340

DEFAULT_INITIAL_X_DEG = 0.0
DEFAULT_INITIAL_Y_DEG = 0.0

# actual_grasp_point offset:
# cobotta_tool_link -> actual_grasp_point [m]
DEFAULT_R_GRASP = [
    +0.000401,
    -0.001507,
    -0.004894,
]

# branch-preserving探索のデフォルト設定
DEFAULT_X_ANGLE_LIMIT = 30.0
DEFAULT_Y_ANGLE_MIN = -30.0
DEFAULT_Y_ANGLE_MAX = +45.0
DEFAULT_ANGLE_STEP = 5.0
DEFAULT_BEAM_WIDTH = 20
DEFAULT_MAX_JUMP_DEG = 10.0
DEFAULT_IK_TIMEOUT = 0.10

DEFAULT_TILT_WEIGHT = 0.0003
DEFAULT_ORIENTATION_CHANGE_WEIGHT = 0.001
DEFAULT_DEDUP_JOINT_RESOLUTION_DEG = 10.0


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


class Node:
    def __init__(
        self,
        index,
        x_deg,
        y_deg,
        state,
        joints,
        cost,
        parent,
    ):
        self.index = index
        self.x_deg = x_deg
        self.y_deg = y_deg
        self.state = state
        self.joints = joints
        self.cost = cost
        self.parent = parent


class BranchPreservingSearch:

    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)

        self.group_name = str(
            rospy.get_param(
                "~group",
                DEFAULT_GROUP,
            )
        )

        self.tip = str(
            rospy.get_param(
                "~tip",
                DEFAULT_TIP,
            )
        )

        self.frame = str(
            rospy.get_param(
                "~frame",
                DEFAULT_FRAME,
            )
        )

        grasp_offset = rospy.get_param(
            "~actual_grasp_offset",
            DEFAULT_R_GRASP,
        )

        if (
            not isinstance(grasp_offset, (list, tuple))
            or len(grasp_offset) != 3
        ):
            raise RuntimeError(
                "actual_grasp_offset must contain "
                "exactly 3 values"
            )

        self.r_grasp = [
            float(v)
            for v in grasp_offset
        ]

        self.x_angle_limit = float(
            rospy.get_param(
                "~x_angle_limit",
                DEFAULT_X_ANGLE_LIMIT,
            )
        )

        self.y_angle_min = float(
            rospy.get_param(
                "~y_angle_min",
                DEFAULT_Y_ANGLE_MIN,
            )
        )

        self.y_angle_max = float(
            rospy.get_param(
                "~y_angle_max",
                DEFAULT_Y_ANGLE_MAX,
            )
        )

        self.angle_step = float(
            rospy.get_param(
                "~angle_step",
                DEFAULT_ANGLE_STEP,
            )
        )

        self.beam_width = int(
            rospy.get_param(
                "~beam_width",
                DEFAULT_BEAM_WIDTH,
            )
        )

        self.max_jump_deg = float(
            rospy.get_param(
                "~max_jump_deg",
                DEFAULT_MAX_JUMP_DEG,
            )
        )
        self.max_jump_rad = math.radians(
            self.max_jump_deg
        )

        self.ik_timeout = float(
            rospy.get_param(
                "~ik_timeout",
                DEFAULT_IK_TIMEOUT,
            )
        )

        self.tilt_weight = float(
            rospy.get_param(
                "~tilt_weight",
                DEFAULT_TILT_WEIGHT,
            )
        )

        self.orientation_change_weight = float(
            rospy.get_param(
                "~orientation_change_weight",
                DEFAULT_ORIENTATION_CHANGE_WEIGHT,
            )
        )

        self.dedup_joint_resolution_deg = float(
            rospy.get_param(
                "~dedup_joint_resolution_deg",
                DEFAULT_DEDUP_JOINT_RESOLUTION_DEG,
            )
        )

        if self.x_angle_limit < 0.0:
            raise RuntimeError(
                "x_angle_limit must be >= 0"
            )

        if self.y_angle_min > self.y_angle_max:
            raise RuntimeError(
                "y_angle_min must be <= y_angle_max"
            )

        if self.angle_step <= 0.0:
            raise RuntimeError(
                "angle_step must be > 0"
            )

        if self.beam_width < 1:
            raise RuntimeError(
                "beam_width must be >= 1"
            )

        if self.max_jump_deg <= 0.0:
            raise RuntimeError(
                "max_jump_deg must be > 0"
            )

        if self.ik_timeout <= 0.0:
            raise RuntimeError(
                "ik_timeout must be > 0"
            )

        if self.dedup_joint_resolution_deg <= 0.0:
            raise RuntimeError(
                "dedup_joint_resolution_deg must be > 0"
            )

        self.group = moveit_commander.MoveGroupCommander(
            self.group_name,
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

        self.input_yaml = os.path.expanduser(
            rospy.get_param(
                "~input_yaml",
                DEFAULT_INPUT_YAML,
            )
        )

        self.base_scan_csv = os.path.expanduser(
            rospy.get_param(
                "~base_scan_csv",
                DEFAULT_BASE_SCAN_CSV,
            )
        )

        self.output_csv = os.path.expanduser(
            rospy.get_param(
                "~output_csv",
                DEFAULT_OUTPUT_CSV,
            )
        )

        self.best_path_csv = os.path.expanduser(
            rospy.get_param(
                "~best_path_csv",
                DEFAULT_BEST_PATH_CSV,
            )
        )

        with open(
            self.input_yaml,
            "r",
            encoding="utf-8",
        ) as f:
            doc = yaml.safe_load(f)

        self.points = doc["points"]

        self.search_start = int(
            rospy.get_param(
                "~search_start",
                DEFAULT_SEARCH_START,
            )
        )

        self.search_end = int(
            rospy.get_param(
                "~search_end",
                DEFAULT_SEARCH_END,
            )
        )

        self.initial_x_deg = float(
            rospy.get_param(
                "~initial_x_deg",
                DEFAULT_INITIAL_X_DEG,
            )
        )

        self.initial_y_deg = float(
            rospy.get_param(
                "~initial_y_deg",
                DEFAULT_INITIAL_Y_DEG,
            )
        )

        if not (
            0 <= self.search_start
            <= self.search_end
            < len(self.points)
        ):
            raise RuntimeError(
                "invalid search range: "
                "{}..{} for trajectory 0..{}"
                .format(
                    self.search_start,
                    self.search_end,
                    len(self.points) - 1,
                )
            )

        with open(
            self.base_scan_csv,
            newline="",
            encoding="utf-8-sig",
        ) as f:
            self.base_rows = list(
                csv.DictReader(f)
            )

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

    def make_tool_pose(
        self,
        index,
        x_deg,
        y_deg,
    ):
        paper_p, base_q = self.paper_pose(index)

        q_delta = multiply(
            axis_q("x", x_deg),
            axis_q("y", y_deg),
        )

        tool_q = normalize(
            multiply(
                base_q,
                q_delta,
            )
        )

        # 姿勢ごとにactual_grasp補正
        offset_world = rotate_vector(
            tool_q,
            self.r_grasp,
        )

        tool_p = [
            paper_p[i] - offset_world[i]
            for i in range(3)
        ]

        pose = PoseStamped()
        pose.header.frame_id = self.frame
        pose.header.stamp = rospy.Time(0)

        pose.pose.position.x = tool_p[0]
        pose.pose.position.y = tool_p[1]
        pose.pose.position.z = tool_p[2]

        pose.pose.orientation.x = tool_q[0]
        pose.pose.orientation.y = tool_q[1]
        pose.pose.orientation.z = tool_q[2]
        pose.pose.orientation.w = tool_q[3]

        return pose

    def initial_state(self):
        matching_rows = [
            row
            for row in self.base_rows
            if int(float(row["index"]))
            == self.search_start
        ]

        if len(matching_rows) != 1:
            raise RuntimeError(
                "expected exactly one base-scan row "
                "for index {}, found {}"
                .format(
                    self.search_start,
                    len(matching_rows),
                )
            )

        row = matching_rows[0]

        state = self.group.get_current_state()

        names = list(state.joint_state.name)
        positions = list(
            state.joint_state.position
        )

        lookup = {
            name: i
            for i, name in enumerate(names)
        }

        for name in self.active_joints:
            positions[
                lookup[name]
            ] = float(row[name])

        state.joint_state.position = positions
        state.joint_state.header.stamp = rospy.Time(0)
        state.is_diff = False

        joints = [
            float(row[name])
            for name in self.active_joints
        ]

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

    def solve(
        self,
        pose,
        seed_state,
    ):
        req = GetPositionIKRequest()

        req.ik_request.group_name = self.group_name
        req.ik_request.ik_link_name = self.tip
        req.ik_request.pose_stamped = pose
        req.ik_request.robot_state = copy.deepcopy(
            seed_state
        )
        req.ik_request.avoid_collisions = False
        req.ik_request.timeout = rospy.Duration(
            self.ik_timeout
        )

        res = self.compute_ik(req)

        if (
            res.error_code.val
            != MoveItErrorCodes.SUCCESS
        ):
            return None

        vreq = GetStateValidityRequest()
        vreq.robot_state = res.solution
        vreq.group_name = self.group_name

        validity = self.check_validity(vreq)

        if not validity.valid:
            return None

        return res.solution

    def candidate_offsets(self, node):
        values = []

        for dx in [-self.angle_step, 0.0, self.angle_step]:
            for dy in [-self.angle_step, 0.0, self.angle_step]:

                x = node.x_deg + dx
                y = node.y_deg + dy

                if abs(x) > self.x_angle_limit:
                    continue

                if y < self.y_angle_min or y > self.y_angle_max:
                    continue

                values.append((x, y))

        return values

    def score(
        self,
        parent,
        x_deg,
        y_deg,
        joints,
    ):
        deltas = [
            abs(a-b)
            for a, b in zip(
                joints,
                parent.joints,
            )
        ]

        max_delta = max(deltas)

        # できるだけ
        # ・関節変化を小さく
        # ・工具傾きを小さく
        # ・角度変化を小さく
        joint_cost = sum(
            d*d
            for d in deltas
        )

        tilt_cost = (
            self.tilt_weight
            * (
                x_deg*x_deg
                + y_deg*y_deg
            )
        )

        orientation_change = (
            self.orientation_change_weight
            * (
                (x_deg-parent.x_deg)**2
                + (y_deg-parent.y_deg)**2
            )
        )

        return (
            parent.cost
            + joint_cost
            + tilt_cost
            + orientation_change
        ), max_delta

    def deduplicate(self, nodes):
        # 同じX/Yでも別IK枝があり得るため、
        # 関節角も含めて粗く分類
        best = {}

        for node in nodes:
            joint_key = tuple(
                int(round(math.degrees(q) / self.dedup_joint_resolution_deg))
                for q in node.joints
            )

            key = (
                int(round(node.x_deg)),
                int(round(node.y_deg)),
                joint_key,
            )

            if (
                key not in best
                or node.cost < best[key].cost
            ):
                best[key] = node

        nodes = list(best.values())

        nodes.sort(
            key=lambda n: n.cost
        )

        return nodes[:self.beam_width]

    def reconstruct(self, node):
        path = []

        while node is not None:
            path.append(node)
            node = node.parent

        path.reverse()
        return path

    def run(self):
        initial_state, initial_joints = (
            self.initial_state()
        )

        # base scan CSVは基準姿勢
        # (local X=0, Y=0 deg) の関節状態を与える。
        #
        # 初期local角が0/0なら従来の状態をそのまま使い、
        # 今回の成功経路の再現性を維持する。
        #
        # 0/0以外を指定した場合だけ、
        # base scan状態をseedとしてsearch_start上で
        # 指定姿勢のIKを解き直し、
        # rootの姿勢と関節状態を一致させる。
        if (
            abs(self.initial_x_deg) > 1.0e-12
            or abs(self.initial_y_deg) > 1.0e-12
        ):
            root_pose = self.make_tool_pose(
                self.search_start,
                self.initial_x_deg,
                self.initial_y_deg,
            )

            root_solution = self.solve(
                root_pose,
                initial_state,
            )

            if root_solution is None:
                raise RuntimeError(
                    "initial root pose is not valid: "
                    "index={} X={:+.1f} Y={:+.1f} deg"
                    .format(
                        self.search_start,
                        self.initial_x_deg,
                        self.initial_y_deg,
                    )
                )

            initial_state = copy.deepcopy(
                root_solution
            )

            initial_joints = self.active_positions(
                root_solution
            )

            print(
                "initial root IK: recomputed "
                "for X={:+.1f} Y={:+.1f} deg"
                .format(
                    self.initial_x_deg,
                    self.initial_y_deg,
                )
            )
        else:
            print(
                "initial root IK: base scan state "
                "(X=0.0 Y=0.0 deg)"
            )

        root = Node(
            index=self.search_start,
            x_deg=self.initial_x_deg,
            y_deg=self.initial_y_deg,
            state=initial_state,
            joints=initial_joints,
            cost=0.0,
            parent=None,
        )

        beam = [root]
        best_node = root

        log_rows = []

        print(
            "===== Branch-preserving FINISH search ====="
        )
        print("search start :", self.search_start)
        print("search end   :", self.search_end)
        print(
            "initial local : "
            "X={:+.1f} Y={:+.1f} deg".format(
                self.initial_x_deg,
                self.initial_y_deg,
            )
        )
        print("beam width   :", self.beam_width)
        print(
            "X angle range : +/-{} deg".format(
                self.x_angle_limit
            )
        )
        print(
            "Y angle range : {:+.1f} .. {:+.1f} deg".format(
                self.y_angle_min,
                self.y_angle_max,
            )
        )
        print("angle step   :", self.angle_step, "deg")
        print("jump limit   :", self.max_jump_deg, "deg")
        print()

        for index in range(
            self.search_start + 1,
            self.search_end + 1,
        ):
            next_nodes = []

            ik_attempts = 0
            ik_valid = 0
            jump_reject = 0

            for parent in beam:

                for x_deg, y_deg in (
                    self.candidate_offsets(parent)
                ):
                    ik_attempts += 1

                    pose = self.make_tool_pose(
                        index,
                        x_deg,
                        y_deg,
                    )

                    solution = self.solve(
                        pose,
                        parent.state,
                    )

                    if solution is None:
                        continue

                    ik_valid += 1

                    joints = self.active_positions(
                        solution
                    )

                    cost, max_delta = self.score(
                        parent,
                        x_deg,
                        y_deg,
                        joints,
                    )

                    if max_delta > self.max_jump_rad:
                        jump_reject += 1
                        continue

                    node = Node(
                        index=index,
                        x_deg=x_deg,
                        y_deg=y_deg,
                        state=copy.deepcopy(
                            solution
                        ),
                        joints=joints,
                        cost=cost,
                        parent=parent,
                    )

                    next_nodes.append(node)

            beam = self.deduplicate(
                next_nodes
            )

            print(
                "index={:3d}  "
                "attempts={:3d}  "
                "IK/valid={:3d}  "
                "jump_reject={:3d}  "
                "beam={:2d}".format(
                    index,
                    ik_attempts,
                    ik_valid,
                    jump_reject,
                    len(beam),
                )
            )

            log_rows.append({
                "index": index,
                "attempts": ik_attempts,
                "ik_valid": ik_valid,
                "jump_reject": jump_reject,
                "beam_size": len(beam),
            })

            if not beam:
                print()
                print(
                    "Search stopped before index",
                    index,
                )
                break

            best_node = min(
                beam,
                key=lambda n: n.cost
            )

        path = self.reconstruct(
            best_node
        )

        print()
        print("===== Search summary =====")
        print(
            "highest continuous FINISH:",
            best_node.index,
        )
        print(
            "FINISH local X:",
            "{:+.1f} deg".format(
                best_node.x_deg
            ),
        )
        print(
            "FINISH local Y:",
            "{:+.1f} deg".format(
                best_node.y_deg
            ),
        )
        print(
            "path points:",
            len(path),
        )

        print()
        print("===== Last 15 path states =====")

        for node in path[-15:]:
            print(
                "index={:3d} X={:+5.1f} Y={:+5.1f}"
                .format(
                    node.index,
                    node.x_deg,
                    node.y_deg,
                )
            )

        with open(
            self.output_csv,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "index",
                    "attempts",
                    "ik_valid",
                    "jump_reject",
                    "beam_size",
                ],
            )
            writer.writeheader()
            writer.writerows(
                log_rows
            )

        fields = [
            "index",
            "local_x_deg",
            "local_y_deg",
            "cost",
        ] + self.active_joints

        with open(
            self.best_path_csv,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fields,
            )

            writer.writeheader()

            for node in path:
                row = {
                    "index": node.index,
                    "local_x_deg":
                        node.x_deg,
                    "local_y_deg":
                        node.y_deg,
                    "cost":
                        node.cost,
                }

                for name, value in zip(
                    self.active_joints,
                    node.joints,
                ):
                    row[name] = value

                writer.writerow(row)

        print()
        print("search log :", self.output_csv)
        print("best path  :", self.best_path_csv)


if __name__ == "__main__":
    rospy.init_node(
        "cobotta_branch_preserving_finish_search"
    )

    BranchPreservingSearch().run()
