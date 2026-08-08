#!/usr/bin/env python3

import copy
import csv
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

    rows_by_index = {
        int(row["index"]): row
        for row in rows
    }

    return rows_by_index[10], rows_by_index[11]


def make_pose(position_row, orientation_row):
    pose = PoseStamped()
    pose.header.frame_id = REFERENCE_FRAME
    pose.header.stamp = rospy.Time(0)

    pose.pose.position.x = float(position_row["x"])
    pose.pose.position.y = float(position_row["y"])
    pose.pose.position.z = float(position_row["z"])

    pose.pose.orientation.x = float(orientation_row["qx"])
    pose.pose.orientation.y = float(orientation_row["qy"])
    pose.pose.orientation.z = float(orientation_row["qz"])
    pose.pose.orientation.w = float(orientation_row["qw"])

    return pose


def make_seed_state(robot, row11):
    state = copy.deepcopy(robot.get_current_state())

    names = list(state.joint_state.name)
    positions = list(state.joint_state.position)

    name_to_index = {
        name: index
        for index, name in enumerate(names)
    }

    for joint_name in JOINT_NAMES:
        value = float(row11[joint_name])

        if joint_name in name_to_index:
            positions[name_to_index[joint_name]] = value
        else:
            names.append(joint_name)
            positions.append(value)

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


def run_test(compute_ik, check_validity, pose, seed_state):
    request = GetPositionIKRequest()

    request.ik_request.group_name = GROUP_NAME
    request.ik_request.robot_state = copy.deepcopy(seed_state)
    request.ik_request.avoid_collisions = False
    request.ik_request.ik_link_name = IK_LINK_NAME
    request.ik_request.pose_stamped = pose
    request.ik_request.timeout = rospy.Duration(1.0)

    response = compute_ik(request)
    code = int(response.error_code.val)

    if code != MoveItErrorCodes.SUCCESS:
        return {
            "ik": "FAILED",
            "code": code,
            "state": "-",
            "contacts": "-",
        }

    validity_request = GetStateValidityRequest()
    validity_request.robot_state = response.solution
    validity_request.group_name = GROUP_NAME

    validity_response = check_validity(validity_request)

    return {
        "ik": "SUCCESS",
        "code": code,
        "state": (
            "VALID"
            if validity_response.valid
            else "COLLISION"
        ),
        "contacts": (
            contact_summary(validity_response.contacts)
            or "-"
        ),
    }


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node(
        "test9_boundary_cross_check",
        anonymous=True,
    )

    row10, row11 = load_rows()

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
    seed_state = make_seed_state(robot, row11)

    tests = [
        ("A: P10 + Q10", row10, row10),
        ("B: P10 + Q11", row10, row11),
        ("C: P11 + Q10", row11, row10),
        ("D: P11 + Q11", row11, row11),
    ]

    print("\n===== Test9 boundary cross check =====")
    print("Common seed: joint solution recorded at index 11")
    print("IK timeout: 1.0 s")
    print("avoid_collisions: False")

    for label, position_row, orientation_row in tests:
        pose = make_pose(
            position_row,
            orientation_row,
        )

        result = run_test(
            compute_ik,
            check_validity,
            pose,
            seed_state,
        )

        print(
            "\n{}".format(label)
        )
        print(
            "  xyz      : ({:.6f}, {:.6f}, {:.6f})".format(
                pose.pose.position.x,
                pose.pose.position.y,
                pose.pose.position.z,
            )
        )
        print("  IK       :", result["ik"])
        print("  code     :", result["code"])
        print("  state    :", result["state"])
        print("  contacts :", result["contacts"])

    print("\n===== Interpretation =====")
    print("B success / C failure : orientation Q10 is influential")
    print("B failure / C success : position P10 is influential")
    print("B failure / C failure : both or their combination")
    print("B success / C success : only P10 + Q10 combination fails")


if __name__ == "__main__":
    main()
