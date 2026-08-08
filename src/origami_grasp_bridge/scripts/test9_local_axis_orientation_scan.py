#!/usr/bin/env python3

import copy
import csv
import importlib.util
import math
import os
import sys

import moveit_commander
import rospy


MODULE_PATH = (
    "/home/maeda/catkin_ws/src/"
    "origami_grasp_bridge/scripts/"
    "test9_boundary_refine_scan.py"
)

OUTPUT_CSV = (
    "/home/maeda/"
    "test9_local_axis_orientation_scan.csv"
)

ANGLE_MIN_DEG = -12.0
ANGLE_MAX_DEG = 12.0
ANGLE_STEP_DEG = 1.0


def load_module():
    spec = importlib.util.spec_from_file_location(
        "boundary_refine",
        MODULE_PATH,
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def normalize_quaternion(q):
    norm = math.sqrt(sum(value * value for value in q))

    if norm <= 1.0e-12:
        raise ValueError("Quaternion norm is zero")

    return [value / norm for value in q]


def quaternion_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    return [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]


def axis_angle_quaternion(axis, angle_rad):
    axis_norm = math.sqrt(
        sum(value * value for value in axis)
    )

    axis = [
        value / axis_norm
        for value in axis
    ]

    half_angle = angle_rad / 2.0
    scale = math.sin(half_angle)

    return [
        axis[0] * scale,
        axis[1] * scale,
        axis[2] * scale,
        math.cos(half_angle),
    ]


def joint_dictionary(robot_state):
    return dict(
        zip(
            robot_state.joint_state.name,
            robot_state.joint_state.position,
        )
    )


def angle_values():
    count = int(
        round(
            (ANGLE_MAX_DEG - ANGLE_MIN_DEG)
            / ANGLE_STEP_DEG
        )
    )

    return [
        ANGLE_MIN_DEG + index * ANGLE_STEP_DEG
        for index in range(count + 1)
    ]


def main():
    module = load_module()

    moveit_commander.roscpp_initialize(sys.argv)

    rospy.init_node(
        "test9_local_axis_orientation_scan",
        anonymous=True,
    )

    rows, row10, row11 = module.load_rows()

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
        module.GetPositionIK,
    )

    check_validity = rospy.ServiceProxy(
        "/check_state_validity",
        module.GetStateValidity,
    )

    robot = moveit_commander.RobotCommander()

    p10 = module.position_from_row(row10)
    q10 = module.quaternion_from_row(row10)

    common_seed = module.seed_from_row(
        robot,
        row11,
    )

    j5_lower, _ = robot.get_joint(
        "cobotta_joint_5"
    ).bounds()

    local_axes = {
        "local_x": [1.0, 0.0, 0.0],
        "local_y": [0.0, 1.0, 0.0],
        "local_z": [0.0, 0.0, 1.0],
    }

    output_rows = []

    print("\n===== Test9 local-axis orientation scan =====")
    print(
        "P10 position: "
        "({:.6f}, {:.6f}, {:.6f})".format(
            p10[0],
            p10[1],
            p10[2],
        )
    )
    print(
        "angle range: {:.1f} to {:.1f} deg, "
        "step {:.1f} deg".format(
            ANGLE_MIN_DEG,
            ANGLE_MAX_DEG,
            ANGLE_STEP_DEG,
        )
    )

    for axis_name, axis in local_axes.items():
        print("\n" + "=" * 72)
        print(axis_name)
        print("=" * 72)
        print(
            "angle_deg,IK,state,J5_deg,"
            "J5_margin_deg,contacts"
        )

        for angle_deg in angle_values():
            delta_q = axis_angle_quaternion(
                axis,
                math.radians(angle_deg),
            )

            # 工具ローカル軸回りの回転なので、
            # 基準姿勢の右側から回転を掛ける。
            target_q = quaternion_multiply(
                q10,
                delta_q,
            )

            target_q = normalize_quaternion(
                target_q
            )

            pose = module.make_pose(
                p10,
                target_q,
            )

            result = module.request_ik(
                compute_ik,
                check_validity,
                pose,
                copy.deepcopy(common_seed),
                timeout=1.0,
            )

            record = {
                "axis": axis_name,
                "angle_deg": angle_deg,
                "ik_success": result["success"],
                "state_valid": (
                    result["valid"]
                    if result["success"]
                    else ""
                ),
                "error_code": result["code"],
                "j5_deg": "",
                "j5_margin_deg": "",
                "contacts": result["contacts"],
            }

            for joint_name in module.JOINT_NAMES:
                record[joint_name] = ""

            if not result["success"]:
                print(
                    "{:.1f},FAILED,-,-,-,-".format(
                        angle_deg
                    )
                )

                output_rows.append(record)
                continue

            joint_values = joint_dictionary(
                result["state"]
            )

            for joint_name in module.JOINT_NAMES:
                record[joint_name] = joint_values[
                    joint_name
                ]

            j5_value = joint_values[
                "cobotta_joint_5"
            ]

            j5_margin = math.degrees(
                j5_value - j5_lower
            )

            record["j5_deg"] = math.degrees(
                j5_value
            )
            record["j5_margin_deg"] = j5_margin

            state_text = (
                "VALID"
                if result["valid"]
                else "COLLISION"
            )

            print(
                "{:.1f},SUCCESS,{},{:.6f},"
                "{:.6f},{}".format(
                    angle_deg,
                    state_text,
                    math.degrees(j5_value),
                    j5_margin,
                    result["contacts"] or "-",
                )
            )

            output_rows.append(record)

    fieldnames = [
        "axis",
        "angle_deg",
        "ik_success",
        "state_valid",
        "error_code",
        "j5_deg",
        "j5_margin_deg",
        "contacts",
    ] + list(module.JOINT_NAMES)

    output_directory = os.path.dirname(
        OUTPUT_CSV
    )

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True,
        )

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(output_rows)

    valid_rows = [
        row
        for row in output_rows
        if row["ik_success"]
        and row["state_valid"]
    ]

    print("\n===== Recommended candidates =====")

    if not valid_rows:
        print("No valid orientation was found.")
        print("CSV:", OUTPUT_CSV)
        return

    best_margin = max(
        valid_rows,
        key=lambda row: row["j5_margin_deg"],
    )

    print(
        "largest J5 margin:"
    )
    print(
        "  axis       :",
        best_margin["axis"],
    )
    print(
        "  angle      : {:.1f} deg".format(
            best_margin["angle_deg"]
        )
    )
    print(
        "  J5         : {:.6f} deg".format(
            best_margin["j5_deg"]
        )
    )
    print(
        "  J5 margin  : {:.6f} deg".format(
            best_margin["j5_margin_deg"]
        )
    )

    for target_margin in [1.0, 2.0, 3.0, 5.0, 10.0]:
        candidates = [
            row
            for row in valid_rows
            if row["j5_margin_deg"] >= target_margin
        ]

        if not candidates:
            print(
                "margin >= {:.1f} deg: "
                "not found".format(
                    target_margin
                )
            )
            continue

        smallest_correction = min(
            candidates,
            key=lambda row: (
                abs(row["angle_deg"]),
                -row["j5_margin_deg"],
            ),
        )

        print(
            "margin >= {:.1f} deg: "
            "{} {:+.1f} deg "
            "(margin {:.6f} deg)".format(
                target_margin,
                smallest_correction["axis"],
                smallest_correction["angle_deg"],
                smallest_correction[
                    "j5_margin_deg"
                ],
            )
        )

    print("\nCSV:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
