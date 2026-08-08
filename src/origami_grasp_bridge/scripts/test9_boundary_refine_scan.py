#!/usr/bin/env python3

import copy
import csv
import math
import sys

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


CSV_PATH = "/home/maeda/test9_pf_ik_collision_scan.csv"
GROUP_NAME = "cobotta_arm"
IK_LINK_NAME = "cobotta_tool_link"
REFERENCE_FRAME = "paper_center"

JOINT_NAMES = [
    "cobotta_joint_1",
    "cobotta_joint_2",
    "cobotta_joint_3",
    "cobotta_joint_4",
    "cobotta_joint_5",
    "cobotta_joint_6",
]


def load_rows():
    with open(CSV_PATH, newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    by_index = {
        int(row["index"]): row
        for row in rows
    }

    if 10 not in by_index or 11 not in by_index:
        raise RuntimeError("CSVにindex 10または11がありません。")

    return rows, by_index[10], by_index[11]


def normalize_quaternion(q):
    norm = math.sqrt(sum(value * value for value in q))

    if norm <= 1.0e-12:
        raise ValueError("Quaternion norm is zero.")

    return [value / norm for value in q]


def quaternion_from_row(row):
    return normalize_quaternion([
        float(row["qx"]),
        float(row["qy"]),
        float(row["qz"]),
        float(row["qw"]),
    ])


def quaternion_angle_deg(q0, q1):
    q0 = normalize_quaternion(q0)
    q1 = normalize_quaternion(q1)

    dot = abs(sum(a * b for a, b in zip(q0, q1)))
    dot = max(-1.0, min(1.0, dot))

    return math.degrees(2.0 * math.acos(dot))


def quaternion_slerp(q0, q1, t):
    q0 = normalize_quaternion(q0)
    q1 = normalize_quaternion(q1)

    dot = sum(a * b for a, b in zip(q0, q1))

    if dot < 0.0:
        q1 = [-value for value in q1]
        dot = -dot

    dot = max(-1.0, min(1.0, dot))

    if dot > 0.9995:
        result = [
            (1.0 - t) * a + t * b
            for a, b in zip(q0, q1)
        ]
        return normalize_quaternion(result)

    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)

    theta = theta_0 * t

    scale_0 = math.sin(theta_0 - theta) / sin_theta_0
    scale_1 = math.sin(theta) / sin_theta_0

    return [
        scale_0 * a + scale_1 * b
        for a, b in zip(q0, q1)
    ]


def position_from_row(row):
    return [
        float(row["x"]),
        float(row["y"]),
        float(row["z"]),
    ]


def interpolate_position(p0, p1, t):
    return [
        (1.0 - t) * a + t * b
        for a, b in zip(p0, p1)
    ]


def make_pose(position, quaternion):
    pose = PoseStamped()
    pose.header.frame_id = REFERENCE_FRAME
    pose.header.stamp = rospy.Time(0)

    pose.pose.position.x = position[0]
    pose.pose.position.y = position[1]
    pose.pose.position.z = position[2]

    pose.pose.orientation.x = quaternion[0]
    pose.pose.orientation.y = quaternion[1]
    pose.pose.orientation.z = quaternion[2]
    pose.pose.orientation.w = quaternion[3]

    return pose


def seed_from_row(robot, row):
    state = copy.deepcopy(robot.get_current_state())

    names = list(state.joint_state.name)
    positions = list(state.joint_state.position)

    name_to_index = {
        name: index
        for index, name in enumerate(names)
    }

    for joint_name in JOINT_NAMES:
        value = float(row[joint_name])

        if joint_name in name_to_index:
            positions[name_to_index[joint_name]] = value
        else:
            names.append(joint_name)
            positions.append(value)
            name_to_index[joint_name] = len(positions) - 1

    state.joint_state.name = names
    state.joint_state.position = positions
    state.joint_state.header.stamp = rospy.Time.now()

    return state


def contact_summary(contacts):
    pairs = []

    for contact in contacts:
        pair = "{} <-> {}".format(
            contact.contact_body_1,
            contact.contact_body_2,
        )

        if pair not in pairs:
            pairs.append(pair)

    return "; ".join(pairs)


def request_ik(
    compute_ik,
    check_validity,
    pose,
    seed_state,
    timeout=1.0,
):
    request = GetPositionIKRequest()

    request.ik_request.group_name = GROUP_NAME
    request.ik_request.robot_state = copy.deepcopy(seed_state)
    request.ik_request.avoid_collisions = False
    request.ik_request.ik_link_name = IK_LINK_NAME
    request.ik_request.pose_stamped = pose
    request.ik_request.timeout = rospy.Duration(timeout)

    response = compute_ik(request)
    code = int(response.error_code.val)

    if code != MoveItErrorCodes.SUCCESS:
        return {
            "success": False,
            "code": code,
            "state": None,
            "valid": None,
            "contacts": "",
        }

    validity_request = GetStateValidityRequest()
    validity_request.robot_state = response.solution
    validity_request.group_name = GROUP_NAME

    validity_response = check_validity(validity_request)

    return {
        "success": True,
        "code": code,
        "state": response.solution,
        "valid": bool(validity_response.valid),
        "contacts": contact_summary(
            validity_response.contacts
        ),
    }


def print_result(label, result):
    if result["success"]:
        state_text = (
            "VALID"
            if result["valid"]
            else "COLLISION"
        )
    else:
        state_text = "-"

    print(
        "{:<34s} IK={:<7s} code={:<4d} "
        "state={:<9s} contacts={}".format(
            label,
            "SUCCESS" if result["success"] else "FAILED",
            result["code"],
            state_text,
            result["contacts"] or "-",
        )
    )


def main():
    moveit_commander.roscpp_initialize(sys.argv)

    rospy.init_node(
        "test9_boundary_refine_scan",
        anonymous=True,
    )

    rows, row10, row11 = load_rows()

    rospy.wait_for_service("/compute_ik", timeout=20.0)
    rospy.wait_for_service(
        "/check_state_validity",
        timeout=20.0,
    )

    compute_ik = rospy.ServiceProxy(
        "/compute_ik",
        GetPositionIK,
    )

    check_validity = rospy.ServiceProxy(
        "/check_state_validity",
        GetStateValidity,
    )

    robot = moveit_commander.RobotCommander()

    p10 = position_from_row(row10)
    p11 = position_from_row(row11)

    q10 = quaternion_from_row(row10)
    q11 = quaternion_from_row(row11)

    position_difference = [
        b - a for a, b in zip(p10, p11)
    ]

    position_distance = math.sqrt(
        sum(value * value for value in position_difference)
    )

    orientation_distance = quaternion_angle_deg(
        q10,
        q11,
    )

    pose_a = make_pose(p10, q10)

    print("\n===== Full difference P10/Q10 -> P11/Q11 =====")
    print(
        "dx                  : {:.6f} mm".format(
            position_difference[0] * 1000.0
        )
    )
    print(
        "dy                  : {:.6f} mm".format(
            position_difference[1] * 1000.0
        )
    )
    print(
        "dz                  : {:.6f} mm".format(
            position_difference[2] * 1000.0
        )
    )
    print(
        "position distance   : {:.6f} mm".format(
            position_distance * 1000.0
        )
    )
    print(
        "orientation distance: {:.6f} deg".format(
            orientation_distance
        )
    )

    print("\n===== Test 1: A with multiple seeds =====")

    seed_candidates = []

    # 現在のRobotState
    seed_candidates.append(
        ("current state", robot.get_current_state())
    )

    # 軌道上の複数の有効解をシードとして使用
    for index in range(11, len(rows), 10):
        row = rows[index]

        if row.get("valid", "").lower() != "true":
            continue

        seed_candidates.append(
            (
                "CSV index {}".format(index),
                seed_from_row(robot, row),
            )
        )

    any_seed_success = False

    for label, seed_state in seed_candidates:
        result = request_ik(
            compute_ik,
            check_validity,
            pose_a,
            seed_state,
            timeout=2.0,
        )

        print_result(label, result)

        if result["success"]:
            any_seed_success = True

    print(
        "A succeeded with any seed:",
        any_seed_success,
    )

    print(
        "\n===== Test 2: P10 fixed, Q10 -> Q11 ====="
    )
    print(
        "t=0.00 is Q10, t=1.00 is Q11"
    )

    # 成功側Q11からQ10へ近づけ、同じIK分岐を追跡する。
    orientation_seed = seed_from_row(robot, row11)

    for step in range(20, -1, -1):
        t = step / 20.0

        quaternion = quaternion_slerp(
            q10,
            q11,
            t,
        )

        pose = make_pose(
            p10,
            quaternion,
        )

        result = request_ik(
            compute_ik,
            check_validity,
            pose,
            orientation_seed,
            timeout=1.0,
        )

        changed_angle = orientation_distance * t

        label = (
            "t={:.2f}, Q10から{:.6f} deg".format(
                t,
                changed_angle,
            )
        )

        print_result(label, result)

        if result["success"]:
            orientation_seed = result["state"]

    print(
        "\n===== Test 3: Q10 fixed, P10 -> P11 ====="
    )
    print(
        "t=0.00 is P10, t=1.00 is P11"
    )

    position_seed = seed_from_row(robot, row11)

    for step in range(20, -1, -1):
        t = step / 20.0

        position = interpolate_position(
            p10,
            p11,
            t,
        )

        pose = make_pose(
            position,
            q10,
        )

        result = request_ik(
            compute_ik,
            check_validity,
            pose,
            position_seed,
            timeout=1.0,
        )

        moved_distance = position_distance * t

        label = (
            "t={:.2f}, P10から{:.6f} mm".format(
                t,
                moved_distance * 1000.0,
            )
        )

        print_result(label, result)

        if result["success"]:
            position_seed = result["state"]

    print("\n===== Interpretation =====")

    if any_seed_success:
        print(
            "Aに別シードで解が出たため、"
            "主因はシード依存または数値IK探索です。"
        )
    else:
        print(
            "複数シードでもAに解が出なければ、"
            "Aは実際の6自由度可到達境界に近い可能性があります。"
        )

    print(
        "姿勢スキャンでは、P10を固定したまま"
        "必要な最小姿勢修正量を確認します。"
    )
    print(
        "位置スキャンでは、Q10を固定したまま"
        "必要な最小位置修正量を確認します。"
    )


if __name__ == "__main__":
    main()
