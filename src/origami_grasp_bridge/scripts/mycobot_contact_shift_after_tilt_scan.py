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

LOCAL_Y_DEG = -5.0


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
                    0.0, 1.0
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
    "mycobot_contact_shift_after_tilt_scan",
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

fk_srv = rospy.ServiceProxy("/compute_fk", GetPositionFK)
ik_srv = rospy.ServiceProxy("/compute_ik", GetPositionIK)
valid_srv = rospy.ServiceProxy(
    "/check_state_validity",
    GetStateValidity
)

# 基準LEFT姿勢のFK
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

# local_y=-5deg
R_local = tft.rotation_matrix(
    math.radians(LOCAL_Y_DEG),
    (0, 1, 0)
)

tilted_T = np.dot(
    base_T,
    R_local
)

tilted_q = tft.quaternion_from_matrix(
    tilted_T
)

# まず傾けただけの姿勢をIK
tilted_pose = PoseStamped()
tilted_pose.header.frame_id = "paper_center"
tilted_pose.header.stamp = rospy.Time.now()

tilted_pose.pose.position = base_pose.position
tilted_pose.pose.orientation.x = tilted_q[0]
tilted_pose.pose.orientation.y = tilted_q[1]
tilted_pose.pose.orientation.z = tilted_q[2]
tilted_pose.pose.orientation.w = tilted_q[3]

ik_req = GetPositionIKRequest()
ik_req.ik_request.group_name = "mycobot_arm"
ik_req.ik_request.ik_link_name = "mycobot_tool_link"
ik_req.ik_request.pose_stamped = tilted_pose
ik_req.ik_request.robot_state = seed_state
ik_req.ik_request.avoid_collisions = False
ik_req.ik_request.timeout = rospy.Duration(1.0)

tilted_ik = ik_srv(ik_req)

if tilted_ik.error_code.val != 1:
    raise RuntimeError("tilted IK failed")

# 実際の傾斜姿勢をFK
fk_tilt_req = GetPositionFKRequest()
fk_tilt_req.header.frame_id = "paper_center"
fk_tilt_req.fk_link_names = ["mycobot_tool_link"]
fk_tilt_req.robot_state = tilted_ik.solution

fk_tilt_res = fk_srv(fk_tilt_req)

tilt_pose = fk_tilt_res.pose_stamped[0].pose

T = tft.quaternion_matrix([
    tilt_pose.orientation.x,
    tilt_pose.orientation.y,
    tilt_pose.orientation.z,
    tilt_pose.orientation.w
])

T[:3, 3] = [
    tilt_pose.position.x,
    tilt_pose.position.y,
    tilt_pose.position.z
]

A0 = T.dot(A_LOCAL)[:3]
B0 = T.dot(B_LOCAL)[:3]

dist0, contact0, crease0 = closest_points_between_segments(
    A0,
    B0,
    CREASE_P0,
    CREASE_P1
)

# local_y=-5deg時点の最短接近方向
direction = crease0 - contact0
direction /= np.linalg.norm(direction)

print("")
print("===== local_y = -5 deg =====")
print(
    "initial contact distance = {:.3f} mm".format(
        dist0 * 1000.0
    )
)
print(
    "approach direction = "
    "({:+.6f}, {:+.6f}, {:+.6f})".format(
        direction[0],
        direction[1],
        direction[2]
    )
)

print("")
print("shift   IK       valid    contact_dist(mm)   collisions")
print(" mm")
print("--------------------------------------------------------------------------")

current_seed = tilted_ik.solution

for shift_mm in np.arange(0.0, 6.01, 0.5):

    shift = direction * (shift_mm / 1000.0)

    target = PoseStamped()
    target.header.frame_id = "paper_center"
    target.header.stamp = rospy.Time.now()

    target.pose.position.x = tilt_pose.position.x + shift[0]
    target.pose.position.y = tilt_pose.position.y + shift[1]
    target.pose.position.z = tilt_pose.position.z + shift[2]

    target.pose.orientation = tilt_pose.orientation

    req = GetPositionIKRequest()
    req.ik_request.group_name = "mycobot_arm"
    req.ik_request.ik_link_name = "mycobot_tool_link"
    req.ik_request.pose_stamped = target
    req.ik_request.robot_state = current_seed
    req.ik_request.avoid_collisions = False
    req.ik_request.timeout = rospy.Duration(1.0)

    res = ik_srv(req)

    if res.error_code.val != 1:
        print(
            "{:4.1f}   FAIL     ---       ---             ---".format(
                shift_mm
            )
        )
        continue

    check_fk = GetPositionFKRequest()
    check_fk.header.frame_id = "paper_center"
    check_fk.fk_link_names = ["mycobot_tool_link"]
    check_fk.robot_state = res.solution

    check_fk_res = fk_srv(check_fk)
    pose = check_fk_res.pose_stamped[0].pose

    TT = tft.quaternion_matrix([
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w
    ])

    TT[:3, 3] = [
        pose.position.x,
        pose.position.y,
        pose.position.z
    ]

    A = TT.dot(A_LOCAL)[:3]
    B = TT.dot(B_LOCAL)[:3]

    dist, _, _ = closest_points_between_segments(
        A,
        B,
        CREASE_P0,
        CREASE_P1
    )

    valid_req = GetStateValidityRequest()
    valid_req.robot_state = res.solution
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
        "{:4.1f}   SUCCESS  {:<5s}      {:8.3f}         {}".format(
            shift_mm,
            str(valid_res.valid),
            dist * 1000.0,
            collision_text
        )
    )

    current_seed = res.solution
