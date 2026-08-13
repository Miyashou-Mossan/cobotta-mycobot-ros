#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import math
import rospy
import numpy as np
import moveit_commander
import tf.transformations as tft

from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import (
    GetPositionFK, GetPositionFKRequest,
    GetPositionIK, GetPositionIKRequest,
    GetStateValidity, GetStateValidityRequest
)

TARGET_JOINTS = {
    "mycobot_joint1":  0.742824352,
    "mycobot_joint2": -0.843296686,
    "mycobot_joint3": -2.403899620,
    "mycobot_joint4":  1.514111985,
    "mycobot_joint5": -0.765822056,
    "mycobot_joint6": -0.104575353,
}

A_LOCAL = np.array([0.000, -0.014, 0.043, 1.0])
B_LOCAL = np.array([0.000, +0.014, 0.043, 1.0])

CREASE_P0 = np.array([-0.075003, +0.074997, +0.000115])
CREASE_P1 = np.array([-0.000003, -0.000003, +0.000115])

BASE_LOCAL_Y_DEG = -5.0
LOCAL_X_DEG = -2.0
SHIFT_MM = 3.6


def closest_points_between_segments(p1, q1, p2, q2):
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2

    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    eps = 1e-12

    if a <= eps and e <= eps:
        return np.linalg.norm(p1 - p2), p1, p2

    if a <= eps:
        s = 0.0
        t = np.clip(f / e, 0.0, 1.0)

    else:
        c = np.dot(d1, r)

        if e <= eps:
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)

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
                s = np.clip(-c / a, 0.0, 1.0)

            elif t > 1.0:
                t = 1.0
                s = np.clip((b - c) / a, 0.0, 1.0)

    c1 = p1 + d1 * s
    c2 = p2 + d2 * t

    return np.linalg.norm(c1 - c2), c1, c2


moveit_commander.roscpp_initialize(sys.argv)

rospy.init_node(
    "mycobot_contact_localy_refine_at_36mm",
    anonymous=True
)

robot = moveit_commander.RobotCommander()
seed_state = robot.get_current_state()

names = list(seed_state.joint_state.name)
positions = list(seed_state.joint_state.position)

for name, value in TARGET_JOINTS.items():
    positions[names.index(name)] = value

seed_state.joint_state.position = positions

rospy.wait_for_service("/compute_fk")
rospy.wait_for_service("/compute_ik")
rospy.wait_for_service("/check_state_validity")

fk_srv = rospy.ServiceProxy(
    "/compute_fk",
    GetPositionFK
)

ik_srv = rospy.ServiceProxy(
    "/compute_ik",
    GetPositionIK
)

valid_srv = rospy.ServiceProxy(
    "/check_state_validity",
    GetStateValidity
)

# --------------------------------------------------
# 元LEFT FULL PASS姿勢
# --------------------------------------------------

fk_req = GetPositionFKRequest()
fk_req.header.frame_id = "paper_center"
fk_req.fk_link_names = ["mycobot_tool_link"]
fk_req.robot_state = seed_state

fk_res = fk_srv(fk_req)

if fk_res.error_code.val != 1:
    raise RuntimeError("base FK failed")

base_pose = fk_res.pose_stamped[0].pose

base_T = tft.quaternion_matrix([
    base_pose.orientation.x,
    base_pose.orientation.y,
    base_pose.orientation.z,
    base_pose.orientation.w
])

# --------------------------------------------------
# 現在の基準条件
# local_y=-5°, local_x=-2°
# このときの3.6 mm shift位置を固定する
# --------------------------------------------------

Ry_base = tft.rotation_matrix(
    math.radians(BASE_LOCAL_Y_DEG),
    (0, 1, 0)
)

Rx_base = tft.rotation_matrix(
    math.radians(LOCAL_X_DEG),
    (1, 0, 0)
)

T_base_orient = np.dot(
    np.dot(base_T, Ry_base),
    Rx_base
)

q_base = tft.quaternion_from_matrix(
    T_base_orient
)

base_candidate = PoseStamped()
base_candidate.header.frame_id = "paper_center"
base_candidate.header.stamp = rospy.Time.now()
base_candidate.pose.position = base_pose.position

base_candidate.pose.orientation.x = q_base[0]
base_candidate.pose.orientation.y = q_base[1]
base_candidate.pose.orientation.z = q_base[2]
base_candidate.pose.orientation.w = q_base[3]

req = GetPositionIKRequest()
req.ik_request.group_name = "mycobot_arm"
req.ik_request.ik_link_name = "mycobot_tool_link"
req.ik_request.pose_stamped = base_candidate
req.ik_request.robot_state = seed_state
req.ik_request.avoid_collisions = False
req.ik_request.timeout = rospy.Duration(1.0)

base_ik = ik_srv(req)

if base_ik.error_code.val != 1:
    raise RuntimeError("base orientation IK failed")

fk2 = GetPositionFKRequest()
fk2.header.frame_id = "paper_center"
fk2.fk_link_names = ["mycobot_tool_link"]
fk2.robot_state = base_ik.solution

fk2_res = fk_srv(fk2)
pose_ref = fk2_res.pose_stamped[0].pose

T_ref = tft.quaternion_matrix([
    pose_ref.orientation.x,
    pose_ref.orientation.y,
    pose_ref.orientation.z,
    pose_ref.orientation.w
])

T_ref[:3, 3] = [
    pose_ref.position.x,
    pose_ref.position.y,
    pose_ref.position.z
]

A_ref = T_ref.dot(A_LOCAL)[:3]
B_ref = T_ref.dot(B_LOCAL)[:3]

_, contact_ref, crease_ref = closest_points_between_segments(
    A_ref,
    B_ref,
    CREASE_P0,
    CREASE_P1
)

approach_direction = crease_ref - contact_ref
approach_direction /= np.linalg.norm(
    approach_direction
)

shift = approach_direction * (
    SHIFT_MM / 1000.0
)

fixed_position = np.array([
    pose_ref.position.x,
    pose_ref.position.y,
    pose_ref.position.z
]) + shift

print("")
print("===== local_y refine at fixed 3.6 mm position =====")
print("local_x = -2.0 deg")
print("reference local_y = -5.0 deg")
print("shift = 3.6 mm")
print(
    "fixed tool position = "
    "({:+.6f}, {:+.6f}, {:+.6f})".format(
        *fixed_position
    )
)

print("")
print(
    "local_y   IK       valid    contact_dist(mm)   collisions"
)
print(
    "--------------------------------------------------------------------------"
)

for local_y_deg in range(-8, -1):

    Ry = tft.rotation_matrix(
        math.radians(local_y_deg),
        (0, 1, 0)
    )

    Rx = tft.rotation_matrix(
        math.radians(LOCAL_X_DEG),
        (1, 0, 0)
    )

    T_candidate = np.dot(
        np.dot(base_T, Ry),
        Rx
    )

    q = tft.quaternion_from_matrix(
        T_candidate
    )

    target = PoseStamped()
    target.header.frame_id = "paper_center"
    target.header.stamp = rospy.Time.now()

    target.pose.position.x = fixed_position[0]
    target.pose.position.y = fixed_position[1]
    target.pose.position.z = fixed_position[2]

    target.pose.orientation.x = q[0]
    target.pose.orientation.y = q[1]
    target.pose.orientation.z = q[2]
    target.pose.orientation.w = q[3]

    ik_req = GetPositionIKRequest()
    ik_req.ik_request.group_name = "mycobot_arm"
    ik_req.ik_request.ik_link_name = "mycobot_tool_link"
    ik_req.ik_request.pose_stamped = target
    ik_req.ik_request.robot_state = base_ik.solution
    ik_req.ik_request.avoid_collisions = False
    ik_req.ik_request.timeout = rospy.Duration(1.0)

    ik_res = ik_srv(ik_req)

    if ik_res.error_code.val != 1:
        print(
            "{:+4d}      FAIL     ---       ---             ---".format(
                local_y_deg
            )
        )
        continue

    check_fk = GetPositionFKRequest()
    check_fk.header.frame_id = "paper_center"
    check_fk.fk_link_names = ["mycobot_tool_link"]
    check_fk.robot_state = ik_res.solution

    check_fk_res = fk_srv(check_fk)

    pose = check_fk_res.pose_stamped[0].pose

    T = tft.quaternion_matrix([
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w
    ])

    T[:3, 3] = [
        pose.position.x,
        pose.position.y,
        pose.position.z
    ]

    A = T.dot(A_LOCAL)[:3]
    B = T.dot(B_LOCAL)[:3]

    dist, _, _ = closest_points_between_segments(
        A,
        B,
        CREASE_P0,
        CREASE_P1
    )

    valid_req = GetStateValidityRequest()
    valid_req.robot_state = ik_res.solution
    valid_req.group_name = "mycobot_arm"

    valid_res = valid_srv(valid_req)

    contacts = []

    for c in valid_res.contacts:
        pair = "{}<->{}".format(
            c.contact_body_1,
            c.contact_body_2
        )

        if pair not in contacts:
            contacts.append(pair)

    collision_text = (
        "; ".join(contacts)
        if contacts
        else "-"
    )

    print(
        "{:+4d}      SUCCESS  {:<5s}      {:8.3f}         {}".format(
            local_y_deg,
            str(valid_res.valid),
            dist * 1000.0,
            collision_text
        )
    )
