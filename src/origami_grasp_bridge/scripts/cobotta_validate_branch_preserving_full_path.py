#!/usr/bin/env python3

import csv
import math
from pathlib import Path

import rospy

from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState, Constraints
from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest


INPUT_FILE = Path(
    "/home/maeda/cobotta_branch_preserving_full_path_0_340.csv"
)

OUTPUT_FILE = Path(
    "/home/maeda/"
    "cobotta_branch_preserving_full_path_0_340_collision_validation.csv"
)

JOINTS = [
    "cobotta_joint_1",
    "cobotta_joint_2",
    "cobotta_joint_3",
    "cobotta_joint_4",
    "cobotta_joint_5",
    "cobotta_joint_6",
]

GROUP_NAME = "cobotta_arm"

# 各隣接joint state間をこの値以下になるまで補間する
MAX_JOINT_STEP_RAD = 0.010


def load_path():
    if not INPUT_FILE.exists():
        raise RuntimeError(
            "Input path does not exist: {}".format(INPUT_FILE)
        )

    with INPUT_FILE.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 341:
        raise RuntimeError(
            "Unexpected path point count: {} (expected 341)".format(
                len(rows)
            )
        )

    expected = list(range(0, 341))
    actual = [int(r["index"]) for r in rows]

    if actual != expected:
        raise RuntimeError(
            "Path index sequence is not exactly 0..340"
        )

    return rows


def row_to_q(row):
    q = []

    for joint in JOINTS:
        value = float(row[joint])

        if not math.isfinite(value):
            raise RuntimeError(
                "NaN/Inf: index={} joint={}".format(
                    row["index"], joint
                )
            )

        q.append(value)

    return q


def interpolate(q0, q1):
    max_delta = max(
        abs(b - a)
        for a, b in zip(q0, q1)
    )

    steps = max(
        1,
        int(math.ceil(max_delta / MAX_JOINT_STEP_RAD))
    )

    result = []

    for k in range(1, steps + 1):
        t = float(k) / float(steps)

        q = [
            a + t * (b - a)
            for a, b in zip(q0, q1)
        ]

        result.append((k, steps, t, q))

    return result


def build_robot_state(base_joint_state, cobotta_q):
    names = list(base_joint_state.name)
    positions = list(base_joint_state.position)

    name_to_index = {
        name: i
        for i, name in enumerate(names)
    }

    for joint, value in zip(JOINTS, cobotta_q):
        if joint in name_to_index:
            positions[name_to_index[joint]] = value
        else:
            names.append(joint)
            positions.append(value)

    joint_state = JointState()
    joint_state.header.stamp = rospy.Time.now()
    joint_state.name = names
    joint_state.position = positions

    robot_state = RobotState()
    robot_state.joint_state = joint_state

    return robot_state


def contact_text(result):
    texts = []

    for contact in result.contacts:
        pair = "{}<->{}".format(
            contact.contact_body_1,
            contact.contact_body_2,
        )

        depth = contact.depth

        texts.append(
            "{} depth={:.6f}".format(pair, depth)
        )

    return "; ".join(texts)


def check_state(service, base_joint_state, q):
    robot_state = build_robot_state(
        base_joint_state,
        q,
    )

    request = GetStateValidityRequest()
    request.robot_state = robot_state
    request.group_name = GROUP_NAME
    request.constraints = Constraints()

    return service(request)


def main():
    rospy.init_node(
        "cobotta_validate_branch_preserving_full_path",
        anonymous=True,
    )

    rows = load_path()

    print("=" * 90)
    print("COBOTTA full path Collision validation")
    print("=" * 90)

    print("INPUT :", INPUT_FILE)
    print("OUTPUT:", OUTPUT_FILE)
    print("points:", len(rows))
    print("range : 0 -> 340")
    print(
        "max interpolation joint step:",
        MAX_JOINT_STEP_RAD,
        "rad",
    )

    print()
    print("Waiting for /joint_states ...")

    current_joint_state = rospy.wait_for_message(
        "/joint_states",
        JointState,
        timeout=10.0,
    )

    print("joint_states received")

    missing = [
        j for j in JOINTS
        if j not in current_joint_state.name
    ]

    if missing:
        raise RuntimeError(
            "Missing COBOTTA joints in /joint_states: {}".format(
                missing
            )
        )

    print("Waiting for /check_state_validity ...")

    rospy.wait_for_service(
        "/check_state_validity",
        timeout=10.0,
    )

    check = rospy.ServiceProxy(
        "/check_state_validity",
        GetStateValidity,
    )

    print("service ready")
    print()

    output_rows = []

    total_states = 0
    invalid_states = 0

    first_invalid = None

    # ------------------------------------------------------
    # index0を最初に検査
    # ------------------------------------------------------

    q0 = row_to_q(rows[0])

    result = check_state(
        check,
        current_joint_state,
        q0,
    )

    total_states += 1

    if not result.valid:
        invalid_states += 1
        first_invalid = (
            0,
            0,
            0,
            contact_text(result),
        )

    output_rows.append({
        "from_index": 0,
        "to_index": 0,
        "interpolation_step": 0,
        "interpolation_steps": 0,
        "ratio": 0.0,
        "valid": result.valid,
        "contacts": contact_text(result),
    })

    # ------------------------------------------------------
    # index0 -> 1 -> ... -> 340
    # ------------------------------------------------------

    for i in range(len(rows) - 1):

        from_index = int(rows[i]["index"])
        to_index = int(rows[i + 1]["index"])

        qa = row_to_q(rows[i])
        qb = row_to_q(rows[i + 1])

        interpolated = interpolate(qa, qb)

        for step, steps, ratio, q in interpolated:

            result = check_state(
                check,
                current_joint_state,
                q,
            )

            total_states += 1

            contacts = contact_text(result)

            if not result.valid:
                invalid_states += 1

                if first_invalid is None:
                    first_invalid = (
                        from_index,
                        to_index,
                        step,
                        contacts,
                    )

            output_rows.append({
                "from_index": from_index,
                "to_index": to_index,
                "interpolation_step": step,
                "interpolation_steps": steps,
                "ratio": ratio,
                "valid": result.valid,
                "contacts": contacts,
            })

        if (
            to_index % 25 == 0
            or to_index == 340
        ):
            print(
                "checked through index {:3d} / 340 "
                "| total states={} invalid={}".format(
                    to_index,
                    total_states,
                    invalid_states,
                )
            )

    # ------------------------------------------------------
    # 保存
    # ------------------------------------------------------

    fields = [
        "from_index",
        "to_index",
        "interpolation_step",
        "interpolation_steps",
        "ratio",
        "valid",
        "contacts",
    ]

    with OUTPUT_FILE.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(output_rows)

    # ------------------------------------------------------
    # 最終結果
    # ------------------------------------------------------

    print()
    print("=" * 90)
    print("VALIDATION RESULT")
    print("=" * 90)

    print("original path points :", len(rows))
    print("checked robot states :", total_states)
    print("invalid states       :", invalid_states)

    if first_invalid is None:

        print()
        print("first invalid         : NONE")
        print()
        print("FULL PATH COLLISION CHECK: PASS")

    else:

        print()
        print(
            "first invalid         : {} -> {} "
            "interpolation step {}".format(
                first_invalid[0],
                first_invalid[1],
                first_invalid[2],
            )
        )

        print(
            "contacts              : {}".format(
                first_invalid[3]
            )
        )

        print()
        print("FULL PATH COLLISION CHECK: FAIL")

    print()
    print("OUTPUT:")
    print(" ", OUTPUT_FILE)


if __name__ == "__main__":
    main()
