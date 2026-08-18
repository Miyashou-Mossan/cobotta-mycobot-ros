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


# ============================================================
# 固定する最良候補
# ============================================================

FINISH_INDEX = 342
LOCAL_X_DEG = -15.0
LOCAL_Y_DEG = +15.0
TILT_START_INDEX = 282

GROUP = "cobotta_arm"
TIP = "cobotta_tool_link"
FRAME = "paper_center"

INPUT_YAML = os.path.expanduser(
    "~/directionA_reverse_local_z_m25_multidof.yaml"
)

SEED_CSV = os.path.expanduser(
    "~/directionA_reverse_local_z_m25_actual_grasp_collision_scan.csv"
)

ENDPOINT_CSV = os.path.expanduser(
    "~/cobotta_finish_endpoint_candidates_v2.csv"
)

OUTPUT_CSV = os.path.expanduser(
    "~/cobotta_bidirectional_ik_bridge_diagnostic.csv"
)

# cobotta_tool_link -> actual_grasp_point [m]
R_GRASP = [
    +0.000401,
    -0.001507,
    -0.004894,
]

IK_TIMEOUT = 0.20
MAX_JUMP_DEG = 10.0
MAX_JUMP_RAD = math.radians(MAX_JUMP_DEG)


def normalize(q):
    n = math.sqrt(sum(v * v for v in q))

    if n <= 1.0e-12:
        raise RuntimeError("Quaternion norm is zero")

    return [v / n for v in q]


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


def smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t*t*(3.0 - 2.0*t)


def slerp_identity(q, t):
    q = normalize(q)

    if q[3] < 0.0:
        q = [-v for v in q]

    w = max(-1.0, min(1.0, q[3]))
    theta = math.acos(w)

    if theta < 1.0e-10:
        return [0.0, 0.0, 0.0, 1.0]

    sin_theta = math.sin(theta)

    scale_q = math.sin(t * theta) / sin_theta
    scale_i = math.sin((1.0 - t) * theta) / sin_theta

    return normalize([
        q[0] * scale_q,
        q[1] * scale_q,
        q[2] * scale_q,
        scale_i + q[3] * scale_q,
    ])


class BidirectionalDiagnostic:

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
            doc = yaml.safe_load(f)

        self.points = doc["points"]

        with open(
            SEED_CSV,
            newline="",
            encoding="utf-8-sig",
        ) as f:
            self.seed_rows = list(csv.DictReader(f))

        with open(
            ENDPOINT_CSV,
            newline="",
            encoding="utf-8-sig",
        ) as f:
            endpoint_rows = list(csv.DictReader(f))

        matches = [
            r for r in endpoint_rows
            if int(r["finish_index"]) == FINISH_INDEX
            and abs(float(r["local_x_deg"]) - LOCAL_X_DEG) < 1.0e-9
            and abs(float(r["local_y_deg"]) - LOCAL_Y_DEG) < 1.0e-9
        ]

        if not matches:
            raise RuntimeError(
                "指定したFINISH候補がendpoint CSVにありません。"
            )

        self.endpoint_row = matches[0]

        self.q_delta_full = multiply(
            axis_q("x", LOCAL_X_DEG),
            axis_q("y", LOCAL_Y_DEG),
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

    def desired_pose(self, index):
        paper_p, base_q = self.paper_pose(index)

        if index <= TILT_START_INDEX:
            alpha = 0.0
        else:
            alpha = (
                index - TILT_START_INDEX
            ) / float(
                FINISH_INDEX - TILT_START_INDEX
            )

        alpha = smoothstep(alpha)

        q_delta = slerp_identity(
            self.q_delta_full,
            alpha,
        )

        tool_q = normalize(
            multiply(base_q, q_delta)
        )

        # 姿勢に応じてactual_grasp補正を毎点再計算
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

        return pose

    def state_from_values(self, values_by_name):
        state = self.group.get_current_state()

        names = list(state.joint_state.name)
        positions = list(state.joint_state.position)

        lookup = {
            name: i
            for i, name in enumerate(names)
        }

        for name in self.active_joints:
            if name not in values_by_name:
                raise RuntimeError(
                    "関節値がありません: {}".format(name)
                )

            positions[
                lookup[name]
            ] = float(values_by_name[name])

        state.joint_state.position = positions
        state.joint_state.header.stamp = rospy.Time(0)
        state.is_diff = False

        return state

    def forward_initial_state(self):
        row = self.seed_rows[0]

        values = {
            name: float(row[name])
            for name in self.active_joints
        }

        return self.state_from_values(values)

    def backward_initial_state(self):
        values = {
            name: float(self.endpoint_row[name])
            for name in self.active_joints
        }

        return self.state_from_values(values)

    def active_positions(self, state):
        lookup = dict(zip(
            state.joint_state.name,
            state.joint_state.position,
        ))

        return [
            float(lookup[name])
            for name in self.active_joints
        ]

    def request_ik(
        self,
        pose,
        seed,
        avoid_collisions,
    ):
        req = GetPositionIKRequest()

        req.ik_request.group_name = GROUP
        req.ik_request.ik_link_name = TIP
        req.ik_request.pose_stamped = pose
        req.ik_request.robot_state = copy.deepcopy(seed)
        req.ik_request.avoid_collisions = avoid_collisions
        req.ik_request.timeout = rospy.Duration(
            IK_TIMEOUT
        )

        return self.compute_ik(req)

    def request_validity(self, state):
        req = GetStateValidityRequest()
        req.robot_state = state
        req.group_name = GROUP

        return self.check_validity(req)

    def solve(self, pose, seed):
        res = self.request_ik(
            pose,
            seed,
            False,
        )

        if (
            res.error_code.val
            != MoveItErrorCodes.SUCCESS
        ):
            return None, "IK_FAILED"

        validity = self.request_validity(
            res.solution
        )

        if validity.valid:
            return res.solution, "VALID"

        # Collision-aware別枝も試す
        res2 = self.request_ik(
            pose,
            seed,
            True,
        )

        if (
            res2.error_code.val
            == MoveItErrorCodes.SUCCESS
        ):
            validity2 = self.request_validity(
                res2.solution
            )

            if validity2.valid:
                return (
                    res2.solution,
                    "VALID_ALTERNATIVE",
                )

        return None, "COLLISION"

    def track(self, indices, initial_state, label):
        results = {}

        seed = copy.deepcopy(initial_state)
        previous = None
        previous_index = None

        fail_index = None
        fail_reason = ""

        for index in indices:
            pose = self.desired_pose(index)

            solution, result = self.solve(
                pose,
                seed,
            )

            if solution is None:
                fail_index = index
                fail_reason = result
                break

            joints = self.active_positions(
                solution
            )

            max_delta = 0.0
            max_joint = ""

            if previous is not None:
                deltas = [
                    abs(a - b)
                    for a, b in zip(
                        joints,
                        previous,
                    )
                ]

                j = max(
                    range(len(deltas)),
                    key=lambda k: deltas[k],
                )

                max_delta = deltas[j]
                max_joint = self.active_joints[j]

                if max_delta > MAX_JUMP_RAD:
                    fail_index = index
                    fail_reason = (
                        "JOINT_JUMP {} {:.3f} deg"
                        .format(
                            max_joint,
                            math.degrees(max_delta),
                        )
                    )
                    break

            results[index] = {
                "joints": joints,
                "result": result,
                "max_delta_rad": max_delta,
                "max_delta_joint": max_joint,
            }

            previous = joints
            previous_index = index
            seed = copy.deepcopy(solution)

        print()
        print("===== {} tracking =====".format(label))
        print("success points :", len(results))

        if results:
            print(
                "index range    : {} -> {}".format(
                    min(results.keys()),
                    max(results.keys()),
                )
            )

        if fail_index is None:
            print("failure        : none")
        else:
            print(
                "failure        : index {}  {}".format(
                    fail_index,
                    fail_reason,
                )
            )

        return results, fail_index, fail_reason

    def compare_overlap(
        self,
        forward,
        backward,
    ):
        overlap = sorted(
            set(forward.keys())
            & set(backward.keys())
        )

        print()
        print("===== Forward / Backward overlap =====")

        if not overlap:
            print("overlap : NONE")
            return None, []

        print(
            "overlap : {} - {} ({} points)".format(
                overlap[0],
                overlap[-1],
                len(overlap),
            )
        )

        comparisons = []

        for index in overlap:
            fj = forward[index]["joints"]
            bj = backward[index]["joints"]

            diffs = [
                abs(a - b)
                for a, b in zip(fj, bj)
            ]

            max_i = max(
                range(len(diffs)),
                key=lambda i: diffs[i],
            )

            rms = math.sqrt(
                sum(d*d for d in diffs)
                / len(diffs)
            )

            comparisons.append({
                "index": index,
                "max_diff_rad": diffs[max_i],
                "max_diff_deg": math.degrees(
                    diffs[max_i]
                ),
                "max_diff_joint":
                    self.active_joints[max_i],
                "rms_diff_rad": rms,
                "rms_diff_deg":
                    math.degrees(rms),
                "diffs": diffs,
            })

        comparisons.sort(
            key=lambda r: (
                r["max_diff_rad"],
                r["rms_diff_rad"],
            )
        )

        best = comparisons[0]

        print()
        print("===== BEST BRIDGE INDEX =====")
        print(
            "index          :",
            best["index"],
        )
        print(
            "max joint diff : {:.3f} deg ({})".format(
                best["max_diff_deg"],
                best["max_diff_joint"],
            )
        )
        print(
            "RMS joint diff : {:.3f} deg".format(
                best["rms_diff_deg"]
            )
        )

        print()
        print("per-joint difference:")

        for name, value in zip(
            self.active_joints,
            best["diffs"],
        ):
            print(
                "  {:16s} {:8.3f} deg".format(
                    name,
                    math.degrees(value),
                )
            )

        return best, comparisons

    def save_csv(
        self,
        forward,
        backward,
        comparisons,
    ):
        compare_lookup = {
            r["index"]: r
            for r in comparisons
        }

        fields = [
            "index",
            "forward_valid",
            "backward_valid",
            "branch_max_diff_deg",
            "branch_max_diff_joint",
            "branch_rms_diff_deg",
        ]

        for name in self.active_joints:
            fields.append(
                "forward_" + name
            )

        for name in self.active_joints:
            fields.append(
                "backward_" + name
            )

        rows = []

        for index in range(
            0,
            FINISH_INDEX + 1,
        ):
            row = {
                "index": index,
                "forward_valid":
                    index in forward,
                "backward_valid":
                    index in backward,
                "branch_max_diff_deg": "",
                "branch_max_diff_joint": "",
                "branch_rms_diff_deg": "",
            }

            if index in forward:
                for name, value in zip(
                    self.active_joints,
                    forward[index]["joints"],
                ):
                    row[
                        "forward_" + name
                    ] = value

            if index in backward:
                for name, value in zip(
                    self.active_joints,
                    backward[index]["joints"],
                ):
                    row[
                        "backward_" + name
                    ] = value

            if index in compare_lookup:
                c = compare_lookup[index]

                row["branch_max_diff_deg"] = (
                    c["max_diff_deg"]
                )
                row["branch_max_diff_joint"] = (
                    c["max_diff_joint"]
                )
                row["branch_rms_diff_deg"] = (
                    c["rms_diff_deg"]
                )

            rows.append(row)

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
        print("output :", OUTPUT_CSV)

    def run(self):
        print("===== Bidirectional IK bridge diagnostic =====")
        print("FINISH     :", FINISH_INDEX)
        print("local X    :", LOCAL_X_DEG, "deg")
        print("local Y    :", LOCAL_Y_DEG, "deg")
        print("tilt start :", TILT_START_INDEX)
        print("jump limit :", MAX_JUMP_DEG, "deg")

        forward_indices = range(
            0,
            FINISH_INDEX + 1,
        )

        backward_indices = range(
            FINISH_INDEX,
            -1,
            -1,
        )

        forward, _, _ = self.track(
            forward_indices,
            self.forward_initial_state(),
            "FORWARD",
        )

        backward, _, _ = self.track(
            backward_indices,
            self.backward_initial_state(),
            "BACKWARD",
        )

        best, comparisons = (
            self.compare_overlap(
                forward,
                backward,
            )
        )

        self.save_csv(
            forward,
            backward,
            comparisons,
        )

        print()
        print("===== Judgment =====")

        if best is None:
            print(
                "No forward/backward overlap."
            )
            print(
                "The two continuous IK branches "
                "do not reach a common index."
            )

        elif best["max_diff_deg"] <= 10.0:
            print(
                "BRIDGE CANDIDATE FOUND."
            )
            print(
                "Forward/backward branches are "
                "within 10 deg at index {}."
                .format(best["index"])
            )

        else:
            print(
                "Overlap exists, but branches "
                "are still separated."
            )
            print(
                "Next step: bridge the overlap "
                "with a local posture/IK search."
            )


if __name__ == "__main__":
    rospy.init_node(
        "cobotta_bidirectional_ik_bridge_diagnostic"
    )

    BidirectionalDiagnostic().run()
