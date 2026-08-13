#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import rospy
import numpy as np
import moveit_commander
import tf.transformations as tft

from moveit_msgs.srv import GetPositionFK, GetPositionFKRequest


# --------------------------------------------------
# LEFT FULL PASS姿勢
# --------------------------------------------------

TARGET_JOINTS = {
    "mycobot_joint1":  0.742824352,
    "mycobot_joint2": -0.843296686,
    "mycobot_joint3": -2.403899620,
    "mycobot_joint4":  1.514111985,
    "mycobot_joint5": -0.765822056,
    "mycobot_joint6": -0.104575353,
}


# --------------------------------------------------
# mycobot_tool_link基準の接触辺
# --------------------------------------------------

A_LOCAL = np.array([
    0.000,
    -0.014,
    0.043,
    1.0
])

B_LOCAL = np.array([
    0.000,
    +0.014,
    0.043,
    1.0
])


# --------------------------------------------------
# 仮想四つ折り折り筋
#
# 元の二つ折り三角形
# P0 -------- P1
#      MID
#
# そのCOBOTTA側半分を残しているため
# 折り筋は P0 -> MID
# --------------------------------------------------

CREASE_P0 = np.array([
    -0.075003,
    +0.074997,
    +0.000115
])

CREASE_P1 = np.array([
    -0.000003,
    -0.000003,
    +0.000115
])


def closest_points_between_segments(p1, q1, p2, q2):
    """
    3D線分 p1-q1 と p2-q2 の最短距離を求める。

    Returns:
        distance
        closest_on_seg1
        closest_on_seg2
        s
        t
    """

    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2

    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)

    eps = 1e-12

    if a <= eps and e <= eps:
        return (
            np.linalg.norm(p1 - p2),
            p1,
            p2,
            0.0,
            0.0
        )

    if a <= eps:
        s = 0.0
        t = np.clip(
            f / e,
            0.0,
            1.0
        )

    else:
        c = np.dot(d1, r)

        if e <= eps:
            t = 0.0
            s = np.clip(
                -c / a,
                0.0,
                1.0
            )

        else:
            b = np.dot(d1, d2)
            denom = a * e - b * b

            if abs(denom) > eps:
                s = np.clip(
                    (b * f - c * e) / denom,
                    0.0,
                    1.0
                )
            else:
                s = 0.0

            t = (b * s + f) / e

            if t < 0.0:
                t = 0.0
                s = np.clip(
                    -c / a,
                    0.0,
                    1.0
                )

            elif t > 1.0:
                t = 1.0
                s = np.clip(
                    (b - c) / a,
                    0.0,
                    1.0
                )

    c1 = p1 + d1 * s
    c2 = p2 + d2 * t

    distance = np.linalg.norm(
        c1 - c2
    )

    return (
        distance,
        c1,
        c2,
        s,
        t
    )


def main():
    moveit_commander.roscpp_initialize(
        sys.argv
    )

    rospy.init_node(
        "mycobot_contact_edge_crease_diagnostic",
        anonymous=True
    )

    robot = moveit_commander.RobotCommander()

    state = robot.get_current_state()

    names = list(
        state.joint_state.name
    )

    positions = list(
        state.joint_state.position
    )

    # LEFT FULL PASS姿勢をRobotStateへ反映
    for joint_name, joint_value in TARGET_JOINTS.items():

        if joint_name not in names:
            raise RuntimeError(
                "joint not found: {}".format(
                    joint_name
                )
            )

        idx = names.index(
            joint_name
        )

        positions[idx] = joint_value

    state.joint_state.position = positions

    rospy.wait_for_service(
        "/compute_fk"
    )

    fk_srv = rospy.ServiceProxy(
        "/compute_fk",
        GetPositionFK
    )

    req = GetPositionFKRequest()

    req.header.frame_id = "paper_center"

    req.fk_link_names = [
        "mycobot_tool_link"
    ]

    req.robot_state = state

    res = fk_srv(req)

    if res.error_code.val != 1:
        raise RuntimeError(
            "FK failed: {}".format(
                res.error_code.val
            )
        )

    pose = res.pose_stamped[0].pose

    # tool_link -> paper_center
    T = tft.quaternion_matrix([
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w
    ])

    T[0:3, 3] = [
        pose.position.x,
        pose.position.y,
        pose.position.z
    ]

    A = T.dot(
        A_LOCAL
    )[:3]

    B = T.dot(
        B_LOCAL
    )[:3]

    (
        distance,
        contact_closest,
        crease_closest,
        contact_ratio,
        crease_ratio
    ) = closest_points_between_segments(
        A,
        B,
        CREASE_P0,
        CREASE_P1
    )

    print("")
    print(
        "=============================================="
    )
    print(
        " MyCobot contact-edge / crease diagnostic"
    )
    print(
        "=============================================="
    )

    print("")
    print("===== Contact edge =====")

    print(
        "A = ({:+.6f}, {:+.6f}, {:+.6f})".format(
            A[0],
            A[1],
            A[2]
        )
    )

    print(
        "B = ({:+.6f}, {:+.6f}, {:+.6f})".format(
            B[0],
            B[1],
            B[2]
        )
    )

    print(
        "edge length = {:.3f} mm".format(
            np.linalg.norm(
                B - A
            ) * 1000.0
        )
    )

    print("")
    print("===== Crease =====")

    print(
        "P0 = ({:+.6f}, {:+.6f}, {:+.6f})".format(
            CREASE_P0[0],
            CREASE_P0[1],
            CREASE_P0[2]
        )
    )

    print(
        "P1 = ({:+.6f}, {:+.6f}, {:+.6f})".format(
            CREASE_P1[0],
            CREASE_P1[1],
            CREASE_P1[2]
        )
    )

    print(
        "crease length = {:.3f} mm".format(
            np.linalg.norm(
                CREASE_P1 - CREASE_P0
            ) * 1000.0
        )
    )

    print("")
    print("===== Closest points =====")

    print(
        "contact point = "
        "({:+.6f}, {:+.6f}, {:+.6f})".format(
            contact_closest[0],
            contact_closest[1],
            contact_closest[2]
        )
    )

    print(
        "crease point  = "
        "({:+.6f}, {:+.6f}, {:+.6f})".format(
            crease_closest[0],
            crease_closest[1],
            crease_closest[2]
        )
    )

    print(
        "contact-edge ratio = {:.4f}".format(
            contact_ratio
        )
    )

    print(
        "crease ratio       = {:.4f}".format(
            crease_ratio
        )
    )

    print("")
    print(
        "minimum distance = {:.3f} mm".format(
            distance * 1000.0
        )
    )

    print("")
    print(
        "NOTE: contact threshold is not defined yet."
    )


if __name__ == "__main__":
    main()
