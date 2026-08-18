#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import math
import os

import rospy

from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState, Constraints
from moveit_msgs.srv import (
    GetStateValidity,
    GetStateValidityRequest,
)


DEFAULT_INPUT = os.path.expanduser(
    "~/cobotta_branch_preserving_full_path_0_340_split_motion.csv"
)

DEFAULT_OUTPUT = os.path.expanduser(
    "~/cobotta_branch_preserving_full_path_0_340_split_motion_collision_validation.csv"
)

GROUP_NAME = "cobotta_arm"

JOINTS = [
    "cobotta_joint_1",
    "cobotta_joint_2",
    "cobotta_joint_3",
    "cobotta_joint_4",
    "cobotta_joint_5",
    "cobotta_joint_6",
]

MAX_JOINT_STEP_RAD = 0.010


def load_path(filename):
    if not os.path.exists(filename):
        raise RuntimeError(
            "Input file does not exist: {}".format(
                filename
            )
        )

    with open(
        filename,
        newline="",
        encoding="utf-8-sig",
    ) as f:
        rows = list(csv.DictReader(f))

    if len(rows) < 2:
        raise RuntimeError(
            "Path must contain at least 2 points"
        )

    for expected, row in enumerate(rows):
        pp = int(row["path_point"])

        if pp != expected:
            raise RuntimeError(
                "path_point mismatch: expected {}, got {}".format(
                    expected,
                    pp,
                )
            )

        for joint in JOINTS:
            if joint not in row:
                raise RuntimeError(
                    "Missing joint column: {}".format(
                        joint
                    )
                )

            value = float(row[joint])

            if not math.isfinite(value):
                raise RuntimeError(
                    "NaN/Inf at path_point {} {}".format(
                        pp,
                        joint,
                    )
                )

    return rows


def row_to_q(row):
    return [
        float(row[joint])
        for joint in JOINTS
    ]


def build_robot_state(
    base_joint_state,
    q,
):
    names = list(
        base_joint_state.name
    )

    positions = list(
        base_joint_state.position
    )

    lookup = {
        name: i
        for i, name in enumerate(names)
    }

    for joint, value in zip(
        JOINTS,
        q,
    ):
        if joint not in lookup:
            raise RuntimeError(
                "joint_states missing {}".format(
                    joint
                )
            )

        positions[
            lookup[joint]
        ] = value

    joint_state = JointState()

    joint_state.header.stamp = (
        rospy.Time(0)
    )

    joint_state.name = names
    joint_state.position = positions

    robot_state = RobotState()
    robot_state.joint_state = joint_state
    robot_state.is_diff = False

    return robot_state


def check_state(
    service,
    base_joint_state,
    q,
):
    request = GetStateValidityRequest()

    request.robot_state = (
        build_robot_state(
            base_joint_state,
            q,
        )
    )

    request.group_name = (
        GROUP_NAME
    )

    request.constraints = (
        Constraints()
    )

    return service(request)


def contact_text(result):
    return "; ".join(
        "{}<->{} depth={:.6f}".format(
            c.contact_body_1,
            c.contact_body_2,
            c.depth,
        )
        for c in result.contacts
    )


def interpolate_edge(q0, q1):
    max_delta = max(
        abs(b-a)
        for a, b in zip(q0, q1)
    )

    steps = max(
        1,
        int(math.ceil(
            max_delta
            / MAX_JOINT_STEP_RAD
        ))
    )

    for step in range(
        1,
        steps + 1,
    ):
        ratio = (
            float(step)
            / float(steps)
        )

        q = [
            a + ratio*(b-a)
            for a, b in zip(q0, q1)
        ]

        yield (
            step,
            steps,
            ratio,
            q,
        )


def main():
    rospy.init_node(
        "cobotta_validate_joint_path_collision",
        anonymous=True,
    )

    input_file = os.path.expanduser(
        rospy.get_param(
            "~input_file",
            DEFAULT_INPUT,
        )
    )

    output_file = os.path.expanduser(
        rospy.get_param(
            "~output_file",
            DEFAULT_OUTPUT,
        )
    )

    rows = load_path(
        input_file
    )

    print("=" * 90)
    print(
        "COBOTTA JOINT PATH COLLISION VALIDATION"
    )
    print("=" * 90)

    print(
        "input                :",
        input_file,
    )

    print(
        "path points          :",
        len(rows),
    )

    print(
        "max interpolation step:",
        MAX_JOINT_STEP_RAD,
        "rad",
    )

    print()
    print(
        "Waiting for /joint_states ..."
    )

    base_joint_state = (
        rospy.wait_for_message(
            "/joint_states",
            JointState,
            timeout=10.0,
        )
    )

    print(
        "Waiting for /check_state_validity ..."
    )

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

    # ----------------------------------------
    # Initial point
    # ----------------------------------------

    result = check_state(
        check,
        base_joint_state,
        row_to_q(rows[0]),
    )

    total_states += 1

    if not result.valid:
        invalid_states += 1

        first_invalid = {
            "from": 0,
            "to": 0,
            "step": 0,
            "ratio": 0.0,
            "contacts":
                contact_text(result),
        }

    output_rows.append({
        "from_path_point": 0,
        "to_path_point": 0,
        "interpolation_step": 0,
        "interpolation_steps": 0,
        "ratio": 0.0,
        "valid": result.valid,
        "contacts":
            contact_text(result),
    })

    # ----------------------------------------
    # All edges
    # ----------------------------------------

    worst_delta = -1.0
    worst_from = None
    worst_joint = None

    for i in range(
        len(rows) - 1
    ):
        qa = row_to_q(
            rows[i]
        )

        qb = row_to_q(
            rows[i + 1]
        )

        deltas = [
            abs(b-a)
            for a, b in zip(qa, qb)
        ]

        edge_max = max(
            deltas
        )

        if edge_max > worst_delta:
            worst_delta = edge_max
            worst_from = i

            worst_joint = JOINTS[
                deltas.index(
                    edge_max
                )
            ]

        for (
            step,
            steps,
            ratio,
            q,
        ) in interpolate_edge(
            qa,
            qb,
        ):
            result = check_state(
                check,
                base_joint_state,
                q,
            )

            total_states += 1

            contacts = (
                contact_text(result)
            )

            if not result.valid:
                invalid_states += 1

                if first_invalid is None:
                    first_invalid = {
                        "from": i,
                        "to": i + 1,
                        "step": step,
                        "ratio": ratio,
                        "contacts":
                            contacts,
                    }

            output_rows.append({
                "from_path_point":
                    i,

                "to_path_point":
                    i + 1,

                "interpolation_step":
                    step,

                "interpolation_steps":
                    steps,

                "ratio":
                    ratio,

                "valid":
                    result.valid,

                "contacts":
                    contacts,
            })

        if (
            (i + 1) % 50 == 0
            or
            i + 1 == len(rows) - 1
        ):
            print(
                "checked through "
                "{:3d} / {} | "
                "states={} invalid={}".format(
                    i + 1,
                    len(rows) - 1,
                    total_states,
                    invalid_states,
                )
            )

    # ----------------------------------------
    # Save
    # ----------------------------------------

    fields = [
        "from_path_point",
        "to_path_point",
        "interpolation_step",
        "interpolation_steps",
        "ratio",
        "valid",
        "contacts",
    ]

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(
            output_rows
        )

    # ----------------------------------------
    # Result
    # ----------------------------------------

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
    print(
        "maximum adjacent saved-point change"
    )

    print(
        "  path_point : {} -> {}".format(
            worst_from,
            worst_from + 1,
        )
    )

    print(
        "  joint      :",
        worst_joint,
    )

    print(
        "  delta      : {:.6f} deg".format(
            math.degrees(
                worst_delta
            )
        )
    )

    print()

    if first_invalid is None:
        print(
            "first invalid         : NONE"
        )

        print()
        print(
            "JOINT PATH COLLISION CHECK: PASS"
        )

    else:
        print(
            "first invalid         : "
            "{} -> {}".format(
                first_invalid["from"],
                first_invalid["to"],
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
            "JOINT PATH COLLISION CHECK: FAIL"
        )

    print()
    print("OUTPUT:")
    print(
        " ",
        output_file,
    )


if __name__ == "__main__":
    main()
