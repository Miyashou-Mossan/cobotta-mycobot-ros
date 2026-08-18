#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import math
from pathlib import Path

import rospy

from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState, Constraints
from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest


INPUT_FILE = Path(
    "/home/maeda/"
    "cobotta_branch_preserving_full_path_0_340_edge_fixed.csv"
)

OUTPUT_FILE = Path(
    "/home/maeda/"
    "cobotta_branch_preserving_full_path_0_340_edge_fixed_collision_validation.csv"
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

MAX_JOINT_STEP_RAD = 0.010
EXPECTED_POINTS = 348


def load_path():
    if not INPUT_FILE.exists():
        raise RuntimeError(
            "Input file does not exist: {}".format(INPUT_FILE)
        )

    with INPUT_FILE.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != EXPECTED_POINTS:
        raise RuntimeError(
            "Unexpected point count: {} (expected {})".format(
                len(rows),
                EXPECTED_POINTS,
            )
        )

    path_points = [
        int(r["path_point"])
        for r in rows
    ]

    if path_points != list(range(EXPECTED_POINTS)):
        raise RuntimeError(
            "path_point sequence is not exactly 0..{}".format(
                EXPECTED_POINTS - 1
            )
        )

    return rows


def row_to_q(row):
    values = []

    for joint in JOINTS:
        value = float(row[joint])

        if not math.isfinite(value):
            raise RuntimeError(
                "NaN/Inf: path_point={} joint={}".format(
                    row["path_point"],
                    joint,
                )
            )

        values.append(value)

    return values


def interpolate(q0, q1):
    max_delta = max(
        abs(b - a)
        for a, b in zip(q0, q1)
    )

    steps = max(
        1,
        int(math.ceil(
            max_delta / MAX_JOINT_STEP_RAD
        ))
    )

    result = []

    for k in range(1, steps + 1):
        ratio = float(k) / float(steps)

        q = [
            a + ratio * (b - a)
            for a, b in zip(q0, q1)
        ]

        result.append(
            (k, steps, ratio, q)
        )

    return result


def build_robot_state(base_joint_state, q):
    names = list(base_joint_state.name)
    positions = list(base_joint_state.position)

    lookup = {
        name: i
        for i, name in enumerate(names)
    }

    for joint, value in zip(JOINTS, q):
        if joint in lookup:
            positions[lookup[joint]] = value
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
    values = []

    for contact in result.contacts:
        values.append(
            "{}<->{} depth={:.6f}".format(
                contact.contact_body_1,
                contact.contact_body_2,
                contact.depth,
            )
        )

    return "; ".join(values)


def check_state(
    service,
    base_joint_state,
    q,
):
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
        "cobotta_validate_edge_fixed_full_path",
        anonymous=True,
    )

    rows = load_path()

    print("=" * 90)
    print("COBOTTA edge-fixed full path Collision validation")
    print("=" * 90)

    print("INPUT :", INPUT_FILE)
    print("OUTPUT:", OUTPUT_FILE)
    print("points:", len(rows))
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

    missing = [
        joint
        for joint in JOINTS
        if joint not in current_joint_state.name
    ]

    if missing:
        raise RuntimeError(
            "Missing COBOTTA joints: {}".format(
                missing
            )
        )

    print("joint_states received")
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

    # --------------------------------------------------------
    # 最初の保存点
    # --------------------------------------------------------

    result = check_state(
        check,
        current_joint_state,
        row_to_q(rows[0]),
    )

    total_states += 1

    contacts = contact_text(result)

    if not result.valid:
        invalid_states += 1
        first_invalid = {
            "from_path_point": 0,
            "to_path_point": 0,
            "from_paper_index": rows[0]["paper_index"],
            "to_paper_index": rows[0]["paper_index"],
            "step": 0,
            "ratio": 0.0,
            "contacts": contacts,
        }

    output_rows.append({
        "from_path_point": 0,
        "to_path_point": 0,
        "from_paper_index": rows[0]["paper_index"],
        "to_paper_index": rows[0]["paper_index"],
        "from_sub_ratio": rows[0]["sub_ratio"],
        "to_sub_ratio": rows[0]["sub_ratio"],
        "interpolation_step": 0,
        "interpolation_steps": 0,
        "ratio": 0.0,
        "valid": result.valid,
        "contacts": contacts,
    })

    # --------------------------------------------------------
    # 全edge
    # --------------------------------------------------------

    worst_joint_delta = -1.0
    worst_from = None
    worst_to = None
    worst_joint = None

    for i in range(len(rows) - 1):

        qa = row_to_q(rows[i])
        qb = row_to_q(rows[i + 1])

        joint_deltas = [
            abs(b - a)
            for a, b in zip(qa, qb)
        ]

        max_delta = max(joint_deltas)

        if max_delta > worst_joint_delta:
            worst_joint_delta = max_delta
            worst_from = i
            worst_to = i + 1
            worst_joint = JOINTS[
                joint_deltas.index(max_delta)
            ]

        interpolated = interpolate(
            qa,
            qb,
        )

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
                    first_invalid = {
                        "from_path_point": i,
                        "to_path_point": i + 1,
                        "from_paper_index":
                            rows[i]["paper_index"],
                        "to_paper_index":
                            rows[i + 1]["paper_index"],
                        "step": step,
                        "ratio": ratio,
                        "contacts": contacts,
                    }

            output_rows.append({
                "from_path_point": i,
                "to_path_point": i + 1,
                "from_paper_index":
                    rows[i]["paper_index"],
                "to_paper_index":
                    rows[i + 1]["paper_index"],
                "from_sub_ratio":
                    rows[i]["sub_ratio"],
                "to_sub_ratio":
                    rows[i + 1]["sub_ratio"],
                "interpolation_step": step,
                "interpolation_steps": steps,
                "ratio": ratio,
                "valid": result.valid,
                "contacts": contacts,
            })

        if (
            (i + 1) % 25 == 0
            or i + 1 == len(rows) - 1
        ):
            print(
                "checked through path_point "
                "{:3d} / {} | total states={} invalid={}".format(
                    i + 1,
                    len(rows) - 1,
                    total_states,
                    invalid_states,
                )
            )

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    fields = [
        "from_path_point",
        "to_path_point",
        "from_paper_index",
        "to_paper_index",
        "from_sub_ratio",
        "to_sub_ratio",
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

    # --------------------------------------------------------
    # 結果
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("VALIDATION RESULT")
    print("=" * 90)

    print(
        "saved path points    :",
        len(rows),
    )

    print(
        "checked robot states :",
        total_states,
    )

    print(
        "invalid states       :",
        invalid_states,
    )

    print()
    print("maximum adjacent saved-point change")

    print(
        "  path_point : {} -> {}".format(
            worst_from,
            worst_to,
        )
    )

    print(
        "  paper      : {} -> {}".format(
            rows[worst_from]["paper_index"],
            rows[worst_to]["paper_index"],
        )
    )

    print(
        "  joint      :",
        worst_joint,
    )

    print(
        "  delta      : {:.6f} deg".format(
            math.degrees(
                worst_joint_delta
            )
        )
    )

    if first_invalid is None:

        print()
        print("first invalid         : NONE")
        print()
        print(
            "EDGE-FIXED FULL PATH COLLISION CHECK: PASS"
        )

    else:

        print()
        print(
            "first invalid         : "
            "path_point {} -> {}".format(
                first_invalid[
                    "from_path_point"
                ],
                first_invalid[
                    "to_path_point"
                ],
            )
        )

        print(
            "paper transition      : "
            "{} -> {}".format(
                first_invalid[
                    "from_paper_index"
                ],
                first_invalid[
                    "to_paper_index"
                ],
            )
        )

        print(
            "interpolation         : "
            "step {} ratio={:.6f}".format(
                first_invalid["step"],
                first_invalid["ratio"],
            )
        )

        print(
            "contacts              :",
            first_invalid["contacts"],
        )

        print()
        print(
            "EDGE-FIXED FULL PATH COLLISION CHECK: FAIL"
        )

    print()
    print("OUTPUT:")
    print(" ", OUTPUT_FILE)


if __name__ == "__main__":
    main()
