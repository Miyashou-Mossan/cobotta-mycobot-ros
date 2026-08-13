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

from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory

TARGET_JOINTS = {
    "mycobot_joint1":  0.742824352,
    "mycobot_joint2": -0.843296686,
    "mycobot_joint3": -2.403899620,
    "mycobot_joint4":  1.514111985,
    "mycobot_joint5": -0.765822056,
    "mycobot_joint6": -0.104575353,
}

A_LOCAL = np.array([0.000, -0.014, 0.043])
B_LOCAL = np.array([0.000, +0.014, 0.043])

CREASE_P0 = np.array([-0.075003, +0.074997, +0.000115])
CREASE_P1 = np.array([-0.000003, -0.000003, +0.000115])

LOCAL_Y_DEG = -6.0
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
                    (b*f - c*e) / denom,
                    0.0, 1.0
                )
            else:
                s = 0.0

            t = (b*s + f) / e

            if t < 0.0:
                t = 0.0
                s = np.clip(-c/a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((b-c)/a, 0.0, 1.0)

    c1 = p1 + d1*s
    c2 = p2 + d2*t

    return np.linalg.norm(c1-c2), c1, c2


def point_to_segment_distance(p, a, b):
    ab = b - a
    denom = np.dot(ab, ab)

    if denom < 1e-12:
        return np.linalg.norm(p-a), 0.0

    t = np.clip(
        np.dot(p-a, ab) / denom,
        0.0,
        1.0
    )

    closest = a + t*ab
    return np.linalg.norm(p-closest), t


moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node(
    "mycobot_yminus6_collision_location",
    anonymous=True
)

robot = moveit_commander.RobotCommander()
seed = robot.get_current_state()

names = list(seed.joint_state.name)
positions = list(seed.joint_state.position)

for name, value in TARGET_JOINTS.items():
    positions[names.index(name)] = value

seed.joint_state.position = positions

rospy.wait_for_service("/compute_fk")
rospy.wait_for_service("/compute_ik")
rospy.wait_for_service("/check_state_validity")

fk_srv = rospy.ServiceProxy("/compute_fk", GetPositionFK)
ik_srv = rospy.ServiceProxy("/compute_ik", GetPositionIK)
valid_srv = rospy.ServiceProxy(
    "/check_state_validity",
    GetStateValidity
)

# 基準FK
fk_req = GetPositionFKRequest()
fk_req.header.frame_id = "paper_center"
fk_req.fk_link_names = ["mycobot_tool_link"]
fk_req.robot_state = seed

fk_res = fk_srv(fk_req)
base_pose = fk_res.pose_stamped[0].pose

base_T = tft.quaternion_matrix([
    base_pose.orientation.x,
    base_pose.orientation.y,
    base_pose.orientation.z,
    base_pose.orientation.w
])

Ry = tft.rotation_matrix(
    math.radians(LOCAL_Y_DEG),
    (0,1,0)
)

Rx = tft.rotation_matrix(
    math.radians(LOCAL_X_DEG),
    (1,0,0)
)

Txy = np.dot(
    np.dot(base_T, Ry),
    Rx
)

qxy = tft.quaternion_from_matrix(Txy)

pose_xy = PoseStamped()
pose_xy.header.frame_id = "paper_center"
pose_xy.pose.position = base_pose.position
pose_xy.pose.orientation.x = qxy[0]
pose_xy.pose.orientation.y = qxy[1]
pose_xy.pose.orientation.z = qxy[2]
pose_xy.pose.orientation.w = qxy[3]

ik_req = GetPositionIKRequest()
ik_req.ik_request.group_name = "mycobot_arm"
ik_req.ik_request.ik_link_name = "mycobot_tool_link"
ik_req.ik_request.pose_stamped = pose_xy
ik_req.ik_request.robot_state = seed
ik_req.ik_request.avoid_collisions = False
ik_req.ik_request.timeout = rospy.Duration(1.0)

ik_res = ik_srv(ik_req)

if ik_res.error_code.val != 1:
    raise RuntimeError("orientation IK failed")

# 傾斜姿勢FK
fk2 = GetPositionFKRequest()
fk2.header.frame_id = "paper_center"
fk2.fk_link_names = ["mycobot_tool_link"]
fk2.robot_state = ik_res.solution

pose0 = fk_srv(fk2).pose_stamped[0].pose

T0 = tft.quaternion_matrix([
    pose0.orientation.x,
    pose0.orientation.y,
    pose0.orientation.z,
    pose0.orientation.w
])

T0[:3,3] = [
    pose0.position.x,
    pose0.position.y,
    pose0.position.z
]

A0 = T0.dot(np.r_[A_LOCAL,1.0])[:3]
B0 = T0.dot(np.r_[B_LOCAL,1.0])[:3]

_, cp0, cr0 = closest_points_between_segments(
    A0, B0,
    CREASE_P0, CREASE_P1
)

direction = cr0 - cp0
direction /= np.linalg.norm(direction)

shift = direction * (SHIFT_MM/1000.0)

target = PoseStamped()
target.header.frame_id = "paper_center"

target.pose.position.x = pose0.position.x + shift[0]
target.pose.position.y = pose0.position.y + shift[1]
target.pose.position.z = pose0.position.z + shift[2]
target.pose.orientation = pose0.orientation

req = GetPositionIKRequest()
req.ik_request.group_name = "mycobot_arm"
req.ik_request.ik_link_name = "mycobot_tool_link"
req.ik_request.pose_stamped = target
req.ik_request.robot_state = ik_res.solution
req.ik_request.avoid_collisions = False
req.ik_request.timeout = rospy.Duration(1.0)

res = ik_srv(req)

if res.error_code.val != 1:
    raise RuntimeError("shift IK failed")

# 最終tool poseをworldとpaper_centerで取得
fk_paper = GetPositionFKRequest()
fk_paper.header.frame_id = "paper_center"
fk_paper.fk_link_names = ["mycobot_tool_link"]
fk_paper.robot_state = res.solution

pose_paper = fk_srv(fk_paper).pose_stamped[0].pose

T_paper_tool = tft.quaternion_matrix([
    pose_paper.orientation.x,
    pose_paper.orientation.y,
    pose_paper.orientation.z,
    pose_paper.orientation.w
])

T_paper_tool[:3,3] = [
    pose_paper.position.x,
    pose_paper.position.y,
    pose_paper.position.z
]

T_tool_paper = np.linalg.inv(T_paper_tool)

valid_req = GetStateValidityRequest()
valid_req.robot_state = res.solution
valid_req.group_name = "mycobot_arm"

valid_res = valid_srv(valid_req)

print("")
print("===== y=-6 x=-2 shift=3.6 collision location =====")
print("valid =", valid_res.valid)

for i, c in enumerate(valid_res.contacts):

    p_world = np.array([
        c.position.x,
        c.position.y,
        c.position.z
    ])

    # world -> paper_center
    origin = np.array([0.260, -0.070, 0.107])
    yaw = math.radians(45.0)

    Rwp = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw),  math.cos(yaw), 0],
        [0, 0, 1]
    ])

    p_paper = Rwp.T.dot(
        p_world - origin
    )

    p_tool_h = T_tool_paper.dot(
        np.r_[p_paper, 1.0]
    )

    p_tool = p_tool_h[:3]

    edge_dist, edge_ratio = \
        point_to_segment_distance(
            p_tool,
            A_LOCAL,
            B_LOCAL
        )

    print("")
    print("contact", i)
    print(
        " pair = {} <-> {}".format(
            c.contact_body_1,
            c.contact_body_2
        )
    )

    print(
        " world = ({:+.6f}, {:+.6f}, {:+.6f})".format(
            *p_world
        )
    )

    print(
        " paper = ({:+.6f}, {:+.6f}, {:+.6f})".format(
            *p_paper
        )
    )

    print(
        " tool  = ({:+.6f}, {:+.6f}, {:+.6f})".format(
            *p_tool
        )
    )

    print(
        " distance to A-B = {:.3f} mm".format(
            edge_dist * 1000.0
        )
    )

    print(
        " edge ratio A->B = {:.4f}".format(
            edge_ratio
        )
    )


# --------------------------------------------------
# RVizへ衝突姿勢を表示
# --------------------------------------------------

display_pub = rospy.Publisher(
    "/move_group/display_planned_path",
    DisplayTrajectory,
    queue_size=1,
    latch=True
)

rospy.sleep(1.0)

display = DisplayTrajectory()

# trajectory_startに、MyCobot以外も含む最終RobotStateを入れる
display.trajectory_start = res.solution

traj = RobotTrajectory()

# 1点だけの軌道として表示
traj.joint_trajectory.joint_names = [
    "mycobot_joint1",
    "mycobot_joint2",
    "mycobot_joint3",
    "mycobot_joint4",
    "mycobot_joint5",
    "mycobot_joint6"
]

solution_names = list(
    res.solution.joint_state.name
)

solution_pos = list(
    res.solution.joint_state.position
)

point = __import__(
    "trajectory_msgs.msg",
    fromlist=["JointTrajectoryPoint"]
).JointTrajectoryPoint()

point.positions = [
    solution_pos[
        solution_names.index(name)
    ]
    for name in traj.joint_trajectory.joint_names
]

point.time_from_start = rospy.Duration(0.1)

traj.joint_trajectory.points.append(
    point
)

display.trajectory.append(
    traj
)

display_pub.publish(
    display
)

print("")
print("===== RViz collision state =====")
print("Published to /move_group/display_planned_path")
print("")
input(
    "RVizで衝突姿勢を確認したら Enter..."
)
