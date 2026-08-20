#!/usr/bin/env python3

import math
import yaml
import rospy
import moveit_commander

from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState, Constraints
from moveit_msgs.srv import GetStateValidity


OUTPUT = "/home/maeda/cobotta_safezero_to_finish340_start_robot_trajectory.yaml"

GROUP_NAME = "cobotta_arm"

COBOTTA_JOINTS = [
    "cobotta_joint_1",
    "cobotta_joint_2",
    "cobotta_joint_3",
    "cobotta_joint_4",
    "cobotta_joint_5",
    "cobotta_joint_6",
]

SAFE_ZERO = [
    0.0,
    0.0,
    0.35,
    0.0,
    0.0,
    0.0,
]

FINISH340_POINT0 = [
    0.108349353929664,
    1.535845917214651,
    0.795688366531443,
    -1.371947126080195,
    1.502693101760984,
    -1.139483138438110,
]

MAX_INTERPOLATION_STEP = 0.01


def duration_to_dict(d):
    return {
        "secs": int(d.secs),
        "nsecs": int(d.nsecs),
    }


def make_full_state(base_joint_state, cobotta_positions):
    state = RobotState()
    state.joint_state.name = list(base_joint_state.name)
    state.joint_state.position = list(base_joint_state.position)

    for name, value in zip(COBOTTA_JOINTS, cobotta_positions):
        if name not in state.joint_state.name:
            raise RuntimeError("Joint not found in /joint_states: " + name)

        idx = state.joint_state.name.index(name)
        state.joint_state.position[idx] = float(value)

    return state


def main():
    rospy.init_node(
        "cobotta_generate_safezero_to_finish340_start",
        anonymous=True
    )
    moveit_commander.roscpp_initialize([])

    print("===== SAFE-ZERO -> FINISH340 POINT0 GENERATOR =====")

    current = rospy.wait_for_message(
        "/joint_states",
        JointState,
        timeout=5.0
    )

    group = moveit_commander.MoveGroupCommander(GROUP_NAME)

    # ----------------------------
    # 明示的な safe-zero start state
    # ----------------------------
    start_state = make_full_state(current, SAFE_ZERO)

    group.set_start_state(start_state)
    group.set_joint_value_target(FINISH340_POINT0)
    group.set_planning_time(10.0)
    group.set_num_planning_attempts(20)

    result = group.plan()

    if isinstance(result, tuple):
        success = result[0]
        plan = result[1]
    else:
        plan = result
        success = len(plan.joint_trajectory.points) > 0

    points = plan.joint_trajectory.points

    print("plan success :", success)
    print("plan points  :", len(points))

    if not success or not points:
        raise RuntimeError("Planning failed")

    # ----------------------------
    # 終点確認
    # ----------------------------
    final = list(points[-1].positions)

    final_diff = [
        abs(a - b)
        for a, b in zip(final, FINISH340_POINT0)
    ]

    print("duration     :",
          points[-1].time_from_start.to_sec())
    print("max end error:",
          max(final_diff))

    if max(final_diff) >= 0.01:
        raise RuntimeError(
            "Final position error is too large"
        )

    # ----------------------------
    # Dense Collision Validation
    # ----------------------------
    rospy.wait_for_service(
        "/check_state_validity",
        timeout=5.0
    )

    check = rospy.ServiceProxy(
        "/check_state_validity",
        GetStateValidity
    )

    dense_states = []

    # safe-zeroそのものを先頭に含める
    previous = list(SAFE_ZERO)
    dense_states.append(previous)

    for p in points:
        current_pos = list(p.positions)

        max_delta = max(
            abs(b - a)
            for a, b in zip(previous, current_pos)
        )

        subdivisions = max(
            1,
            int(math.ceil(
                max_delta / MAX_INTERPOLATION_STEP
            ))
        )

        for k in range(1, subdivisions + 1):
            t = float(k) / float(subdivisions)

            q = [
                a + t * (b - a)
                for a, b in zip(previous, current_pos)
            ]

            dense_states.append(q)

        previous = current_pos

    invalid = 0

    for i, q in enumerate(dense_states):
        state = make_full_state(current, q)

        res = check(
            state,
            GROUP_NAME,
            Constraints()
        )

        if not res.valid:
            invalid += 1
            print("INVALID dense state:", i)

            for c in res.contacts:
                print(
                    "  CONTACT:",
                    c.contact_body_1,
                    "<->",
                    c.contact_body_2,
                    "depth =",
                    c.depth
                )

    print()
    print("===== DENSE COLLISION VALIDATION =====")
    print("checked :", len(dense_states))
    print("invalid :", invalid)

    if invalid != 0:
        raise RuntimeError(
            "Collision validation failed; YAML will NOT be saved"
        )

    # ----------------------------
    # trajectory_start
    # ----------------------------
    start_names = list(current.name)
    start_positions = list(current.position)

    for name, value in zip(COBOTTA_JOINTS, SAFE_ZERO):
        idx = start_names.index(name)
        start_positions[idx] = float(value)

    # ----------------------------
    # YAML
    # ----------------------------
    data = {
        "metadata": {
            "generator":
                "cobotta_generate_safezero_to_finish340_start.py",
            "group_name": GROUP_NAME,
            "purpose":
                "fixed safe-zero to FINISH340 main trajectory point0",
            "safe_zero": SAFE_ZERO,
            "finish340_point0": FINISH340_POINT0,
            "planning_time": 10.0,
            "planning_attempts": 20,
            "dense_collision_max_joint_step":
                MAX_INTERPOLATION_STEP,
            "dense_collision_checked_states":
                len(dense_states),
            "dense_collision_invalid_states":
                invalid,
        },
        "trajectory_start": {
            "joint_names": start_names,
            "positions": start_positions,
        },
        "joint_trajectory": {
            "frame_id": (
                plan.joint_trajectory.header.frame_id
                if plan.joint_trajectory.header.frame_id
                else "world"
            ),
            "joint_names":
                list(plan.joint_trajectory.joint_names),
            "points": [],
        },
    }

    for p in points:
        data["joint_trajectory"]["points"].append({
            "positions": list(p.positions),
            "velocities": list(p.velocities),
            "accelerations": list(p.accelerations),
            "effort": list(p.effort),
            "time_from_start":
                duration_to_dict(p.time_from_start),
        })

    with open(OUTPUT, "w") as f:
        yaml.safe_dump(
            data,
            f,
            default_flow_style=False,
            sort_keys=False
        )

    print()
    print("===== SAVED =====")
    print("file   :", OUTPUT)
    print("points :", len(points))
    print(
        "duration:",
        points[-1].time_from_start.to_sec()
    )


if __name__ == "__main__":
    main()
