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


POSE_YAML = os.path.expanduser(
    "~/directionA_reverse_local_z_m25_multidof.yaml"
)

SEED_CSV = os.path.expanduser(
    "~/directionA_reverse_local_z_m25_scan.csv"
)

OUTPUT_CSV = os.path.expanduser(
    "~/directionA_goal_orientation_xy_scan.csv"
)

POINT_INDEX = 399
GROUP_NAME = "cobotta_arm"
TIP_LINK = "cobotta_tool_link"
FRAME_ID = "paper_center"

OFFSETS = list(range(-20, 21, 5))


def normalize(q):
    n = math.sqrt(sum(v * v for v in q))
    return [v / n for v in q]


def multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    return [
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ]


def axis_quaternion(axis, deg):
    a = math.radians(deg) / 2.0
    s = math.sin(a)
    c = math.cos(a)

    if axis == "x":
        return [s, 0.0, 0.0, c]

    if axis == "y":
        return [0.0, s, 0.0, c]

    raise ValueError(axis)


def contact_summary(contacts):
    values = []
    used = set()

    for contact in contacts:
        pair = tuple(sorted([
            contact.contact_body_1,
            contact.contact_body_2,
        ]))

        if pair in used:
            continue

        used.add(pair)

        values.append(
            "{}<->{} depth={:.6f}".format(
                contact.contact_body_1,
                contact.contact_body_2,
                contact.depth,
            )
        )

    return "; ".join(values)


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node(
        "cobotta_goal_orientation_grid_scan"
    )

    group = moveit_commander.MoveGroupCommander(
        GROUP_NAME,
        wait_for_servers=20.0,
    )

    active_joints = list(
        group.get_active_joints()
    )

    # ---------------- Pose ----------------

    with open(
        POSE_YAML,
        "r",
        encoding="utf-8",
    ) as f:
        doc = yaml.safe_load(f)

    point = doc["points"][POINT_INDEX]
    tf = point["transforms"][0]

    t = tf["translation"]
    r = tf["rotation"]

    q_base = normalize([
        float(r["x"]),
        float(r["y"]),
        float(r["z"]),
        float(r["w"]),
    ])

    # ---------------- Known GOAL IK seed ----------------

    with open(
        SEED_CSV,
        newline="",
        encoding="utf-8-sig",
    ) as f:
        rows = list(csv.DictReader(f))

    seed_row = rows[POINT_INDEX]

    seed_state = group.get_current_state()

    names = list(seed_state.joint_state.name)
    positions = list(
        seed_state.joint_state.position
    )

    name_to_index = {
        name: i
        for i, name in enumerate(names)
    }

    for joint_name in active_joints:
        positions[
            name_to_index[joint_name]
        ] = float(
            seed_row[joint_name]
        )

    seed_state.joint_state.position = positions
    seed_state.joint_state.header.stamp = rospy.Time(0)
    seed_state.is_diff = False

    # ---------------- Services ----------------

    rospy.wait_for_service(
        "/compute_ik",
        timeout=20.0,
    )

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

    results = []

    for x_deg in OFFSETS:
        for y_deg in OFFSETS:

            qx = axis_quaternion(
                "x",
                x_deg,
            )

            qy = axis_quaternion(
                "y",
                y_deg,
            )

            # base姿勢にlocal X, local Yを追加
            q_new = normalize(
                multiply(
                    multiply(q_base, qx),
                    qy,
                )
            )

            pose = PoseStamped()
            pose.header.frame_id = FRAME_ID
            pose.header.stamp = rospy.Time(0)

            pose.pose.position.x = float(t["x"])
            pose.pose.position.y = float(t["y"])
            pose.pose.position.z = float(t["z"])

            pose.pose.orientation.x = q_new[0]
            pose.pose.orientation.y = q_new[1]
            pose.pose.orientation.z = q_new[2]
            pose.pose.orientation.w = q_new[3]

            req = GetPositionIKRequest()
            req.ik_request.group_name = GROUP_NAME
            req.ik_request.ik_link_name = TIP_LINK
            req.ik_request.pose_stamped = pose
            req.ik_request.robot_state = copy.deepcopy(
                seed_state
            )
            req.ik_request.timeout = rospy.Duration(
                0.20
            )
            req.ik_request.avoid_collisions = False

            ik = compute_ik(req)

            result = "IK_FAILED"
            contacts = ""
            solution = None

            if (
                ik.error_code.val
                == MoveItErrorCodes.SUCCESS
            ):
                solution = ik.solution

                vreq = GetStateValidityRequest()
                vreq.robot_state = solution
                vreq.group_name = GROUP_NAME

                validity = check_validity(vreq)

                contacts = contact_summary(
                    validity.contacts
                )

                if validity.valid:
                    result = "VALID"

                else:
                    result = "COLLISION"

                    # Collision-aware IKも試す
                    req.ik_request.avoid_collisions = True

                    ik2 = compute_ik(req)

                    if (
                        ik2.error_code.val
                        == MoveItErrorCodes.SUCCESS
                    ):
                        vreq.robot_state = ik2.solution

                        validity2 = check_validity(
                            vreq
                        )

                        if validity2.valid:
                            solution = ik2.solution
                            result = "VALID_ALTERNATIVE"
                            contacts = ""

            row = {
                "local_x_deg": x_deg,
                "local_y_deg": y_deg,
                "result": result,
                "contacts": contacts,
            }

            if solution is not None:
                sol_map = dict(zip(
                    solution.joint_state.name,
                    solution.joint_state.position,
                ))

                for joint_name in active_joints:
                    row[joint_name] = sol_map.get(
                        joint_name,
                        ""
                    )
            else:
                for joint_name in active_joints:
                    row[joint_name] = ""

            results.append(row)

            print(
                "X={:+3d} Y={:+3d} : {}".format(
                    x_deg,
                    y_deg,
                    result,
                )
            )

    fields = [
        "local_x_deg",
        "local_y_deg",
        "result",
        "contacts",
    ] + active_joints

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
        writer.writerows(results)

    valid_rows = [
        row
        for row in results
        if row["result"].startswith("VALID")
    ]

    print()
    print("===== GOAL orientation scan =====")
    print("tested :", len(results))
    print("valid  :", len(valid_rows))
    print("output :", OUTPUT_CSV)

    if valid_rows:
        print()
        print("Collision-free candidates:")

        valid_rows.sort(
            key=lambda row:
            abs(row["local_x_deg"])
            + abs(row["local_y_deg"])
        )

        for row in valid_rows[:20]:
            print(
                "X={:+3d} deg  Y={:+3d} deg  {}".format(
                    row["local_x_deg"],
                    row["local_y_deg"],
                    row["result"],
                )
            )


if __name__ == "__main__":
    main()
