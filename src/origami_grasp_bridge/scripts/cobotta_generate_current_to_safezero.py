#!/usr/bin/env python3

import math
import yaml
import rospy
import moveit_commander

from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState, Constraints
from moveit_msgs.srv import GetStateValidity


OUTPUT = "/home/maeda/cobotta_current_to_safezero_robot_trajectory.yaml"

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

REAL_CURRENT = [
    0.09255779235263528,
    0.12032862181758125,
    0.41035122937826213,
    -0.2121904542296417,
    -0.7928083875176896,
    -1.5089877302015249,
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
        "cobotta_generate_current_to_safezero",
        anonymous=True
    )
    moveit_commander.roscpp_initialize([])

    print("===== CURRENT -> SAFE-ZERO GENERATOR =====")

    current = rospy.wait_for_message(
        "/joint_states",
        JointState,
        timeout=5.0
    )

    # COBOTTAについてはHiroshige側 /joint_states を使わず、
    # COBOTTA内蔵PCから直接取得した実機関節角を開始値とする
    current_cobotta = list(REAL_CURRENT)

    print("REAL robot current:")
    print(current_cobotta)

    print("safe-zero:")
    print(SAFE_ZERO)

    group = moveit_commander.MoveGroupCommander(GROUP_NAME)

    # 実機 /joint_states から取得した現在姿勢を
    # MoveIt の開始状態として明示的に設定する
    start_state = make_full_state(current, current_cobotta)
    group.set_start_state(start_state)

    group.set_joint_value_target(SAFE_ZERO)
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

    final = list(points[-1].positions)

    final_diff = [
        abs(a - b)
        for a, b in zip(final, SAFE_ZERO)
    ]

    print("duration      :",
          points[-1].time_from_start.to_sec())
    print("max end error :",
          max(final_diff))

    if max(final_diff) >= 0.01:
        raise RuntimeError(
            "Final position error is too large"
        )

    rospy.wait_for_service(
        "/check_state_validity",
        timeout=5.0
    )

    check = rospy.ServiceProxy(
        "/check_state_validity",
        GetStateValidity
    )

    dense_states = []

    previous = list(current_cobotta)
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

    data = {
        "metadata": {
            "generator":
                "cobotta_generate_current_to_safezero.py",
            "group_name": GROUP_NAME,
            "purpose":
                "current real robot pose to COBOTTA safe-zero",
            "safe_zero": SAFE_ZERO,
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
            "joint_names": list(current.name),
            "positions": [
                (
                    float(REAL_CURRENT[COBOTTA_JOINTS.index(name)])
                    if name in COBOTTA_JOINTS
                    else float(current.position[list(current.name).index(name)])
                )
                for name in current.name
            ],
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
