#!/usr/bin/env python3

import copy
import csv
import importlib.util
import math
import sys

import moveit_commander
import rospy


BOUNDARY_MODULE_PATH = (
    "/home/maeda/catkin_ws/src/"
    "origami_grasp_bridge/scripts/"
    "test9_boundary_refine_scan.py"
)

FULL_SCAN_MODULE_PATH = (
    "/home/maeda/catkin_ws/src/"
    "origami_grasp_bridge/scripts/"
    "test9_full_local_y_correction_scan.py"
)

TRANSITION_MODULE_PATH = (
    "/home/maeda/catkin_ws/src/"
    "origami_grasp_bridge/scripts/"
    "test9_paper_plus_x_transition_tilt40_scan.py"
)

ORIGINAL_CSV = (
    "/home/maeda/test9_pf_ik_collision_scan.csv"
)

TILT_BACK_CSV = (
    "/home/maeda/test9_paper_plus_x_tilt_back_scan.csv"
)

OUTPUT_CSV = (
    "/home/maeda/"
    "cobotta_current_model_low_height_continuous_scan.csv"
)

POINT_INDEX = 0

TARGET_TILT_DEG = 17.0
TARGET_AZIMUTH_DEG = 0.0

START_Z_MM = 60.35
END_Z_MM = 0.35
STEP_Z_MM = 1.0

MAX_CONTINUOUS_JOINT_STEP_DEG = 5.0


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def joint_dictionary(robot_state):
    return dict(zip(
        robot_state.joint_state.name,
        robot_state.joint_state.position,
    ))


def generate_z_values():
    values = []
    z_mm = START_Z_MM

    while z_mm >= END_Z_MM - 1.0e-9:
        values.append(round(z_mm, 6))
        z_mm -= STEP_Z_MM

    return values


def main():
    boundary = load_module(
        BOUNDARY_MODULE_PATH,
        "boundary_module",
    )

    full_scan = load_module(
        FULL_SCAN_MODULE_PATH,
        "full_scan_module",
    )

    transition = load_module(
        TRANSITION_MODULE_PATH,
        "transition_module",
    )

    moveit_commander.roscpp_initialize(sys.argv)

    rospy.init_node(
        "test9_paper_plus_x_tilt17_height_scan",
        anonymous=True,
    )

    point_index = int(
        rospy.get_param(
            "~point_index",
            POINT_INDEX,
        )
    )

    print(
        "target point index    : {}".format(
            point_index
        ),
        flush=True,
    )

    with open(ORIGINAL_CSV, newline="") as f:
        original_rows = list(csv.DictReader(f))

    with open(TILT_BACK_CSV, newline="") as f:
        tilt_back_rows = list(csv.DictReader(f))

    original_by_index = {
        int(row["index"]): row
        for row in original_rows
    }

    if point_index not in original_by_index:
        raise RuntimeError(
            "point index {} が入力CSVにありません。".format(
                point_index
            )
        )

    original_row = original_by_index[point_index]

    # 関節ジャンプが起きる前までを同一分岐とみなす。
    continuous_rows = []

    for row in sorted(
        tilt_back_rows,
        key=lambda item: int(item["step"]),
    ):
        if row.get("valid", "").lower() != "true":
            break

        adjacent_text = row.get(
            "adjacent_delta_deg",
            "",
        )

        if adjacent_text:
            adjacent_deg = float(adjacent_text)

            if adjacent_deg > MAX_CONTINUOUS_JOINT_STEP_DEG:
                break

        continuous_rows.append(row)

    if not continuous_rows:
        raise RuntimeError(
            "関節ジャンプ前の連続解が見つかりません。"
        )

    # 連続分岐上で17°に最も近い行を選択する。
    selected_row = min(
        continuous_rows,
        key=lambda row: abs(
            float(row["tilt_deg"])
            - TARGET_TILT_DEG
        ),
    )

    selected_tilt_deg = float(
        selected_row["tilt_deg"]
    )

    print(
        "\n===== paper_plus_x low-height scan =====",
        flush=True,
    )

    print(
        "requested tilt      : {:.3f} deg".format(
            TARGET_TILT_DEG
        ),
        flush=True,
    )

    print(
        "selected safe tilt  : {:.3f} deg".format(
            selected_tilt_deg
        ),
        flush=True,
    )

    print(
        "selected source step:",
        selected_row["step"],
        flush=True,
    )

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
        boundary.GetPositionIK,
    )

    check_validity = rospy.ServiceProxy(
        "/check_state_validity",
        boundary.GetStateValidity,
    )

    robot = moveit_commander.RobotCommander()

    seed_state = boundary.seed_from_row(
        robot,
        selected_row,
    )

    previous_positions = [
        float(selected_row[name])
        for name in boundary.JOINT_NAMES
    ]

    original_q = transition.normalize_quaternion(
        boundary.quaternion_from_row(original_row)
    )

    local_y_q = transition.axis_angle_quaternion(
        [0.0, 1.0, 0.0],
        math.radians(selected_tilt_deg),
    )

    base_q = transition.normalize_quaternion(
        transition.quaternion_multiply(
            original_q,
            local_y_q,
        )
    )

    tool_axis = transition.rotate_vector(
        base_q,
        [0.0, 0.0, 1.0],
    )

    current_azimuth_deg = math.degrees(
        math.atan2(
            tool_axis[1],
            tool_axis[0],
        )
    )

    yaw_delta_deg = transition.wrap_angle_deg(
        TARGET_AZIMUTH_DEG
        - current_azimuth_deg
    )

    yaw_q = transition.axis_angle_quaternion(
        [0.0, 0.0, 1.0],
        math.radians(yaw_delta_deg),
    )

    target_q = transition.normalize_quaternion(
        transition.quaternion_multiply(
            yaw_q,
            base_q,
        )
    )

    position = boundary.position_from_row(
        original_row
    )

    j5_lower, j5_upper = robot.get_joint(
        "cobotta_joint_5"
    ).bounds()

    output_rows = []

    lowest_valid_z_mm = None
    first_failure_z_mm = None
    first_failure_result = ""
    first_failure_contacts = ""

    minimum_j5_lower_margin_deg = None
    minimum_j5_upper_margin_deg = None

    maximum_adjacent_delta_deg = 0.0
    maximum_adjacent_joint = ""
    maximum_adjacent_segment = ""

    for scan_index, z_mm in enumerate(
        generate_z_values()
    ):
        pose = boundary.make_pose(
            position,
            target_q,
        )

        pose.pose.position.z = z_mm / 1000.0

        result, result_name = transition.solve_pose(
            boundary,
            full_scan,
            compute_ik,
            check_validity,
            pose,
            seed_state,
        )

        valid = (
            result["success"]
            and result["valid"]
        )

        if not valid:
            first_failure_z_mm = z_mm
            first_failure_result = result_name
            first_failure_contacts = (
                result["contacts"] or "-"
            )

            print(
                "Z={:6.2f} mm : {} / {}".format(
                    z_mm,
                    result_name,
                    first_failure_contacts,
                ),
                flush=True,
            )

            output_rows.append({
                "scan_index": scan_index,
                "tilt_deg": selected_tilt_deg,
                "yaw_delta_deg": yaw_delta_deg,
                "z_mm": z_mm,
                "result": result_name,
                "valid": False,
                "j5_deg": "",
                "j5_lower_margin_deg": "",
                "j5_upper_margin_deg": "",
                "adjacent_delta_deg": "",
                "adjacent_joint": "",
                "contacts": result["contacts"],
                **{
                    name: ""
                    for name in boundary.JOINT_NAMES
                },
            })

            break

        values = joint_dictionary(
            result["state"]
        )

        positions = [
            values[name]
            for name in boundary.JOINT_NAMES
        ]

        deltas = [
            abs(current - previous)
            for current, previous in zip(
                positions,
                previous_positions,
            )
        ]

        max_index = max(
            range(len(deltas)),
            key=lambda index: deltas[index],
        )

        adjacent_delta_deg = math.degrees(
            deltas[max_index]
        )

        adjacent_joint = boundary.JOINT_NAMES[
            max_index
        ]

        if adjacent_delta_deg > MAX_CONTINUOUS_JOINT_STEP_DEG:
            first_failure_z_mm = z_mm
            first_failure_result = "JOINT_JUMP"
            first_failure_contacts = "-"

            print(
                "Z={:6.2f} mm : JOINT_JUMP "
                "{:.3f} deg / {}".format(
                    z_mm,
                    adjacent_delta_deg,
                    adjacent_joint,
                ),
                flush=True,
            )

            break

        if (
            adjacent_delta_deg
            > maximum_adjacent_delta_deg
        ):
            maximum_adjacent_delta_deg = (
                adjacent_delta_deg
            )
            maximum_adjacent_joint = adjacent_joint
            maximum_adjacent_segment = (
                "{}->{}".format(
                    scan_index - 1,
                    scan_index,
                )
                if scan_index > 0
                else "seed->0"
            )

        j5_value = values["cobotta_joint_5"]

        j5_deg = math.degrees(j5_value)

        j5_lower_margin_deg = math.degrees(
            j5_value - j5_lower
        )

        j5_upper_margin_deg = math.degrees(
            j5_upper - j5_value
        )

        if minimum_j5_lower_margin_deg is None:
            minimum_j5_lower_margin_deg = (
                j5_lower_margin_deg
            )
            minimum_j5_upper_margin_deg = (
                j5_upper_margin_deg
            )
        else:
            minimum_j5_lower_margin_deg = min(
                minimum_j5_lower_margin_deg,
                j5_lower_margin_deg,
            )
            minimum_j5_upper_margin_deg = min(
                minimum_j5_upper_margin_deg,
                j5_upper_margin_deg,
            )

        lowest_valid_z_mm = z_mm

        print(
            "Z={:6.2f} mm : VALID "
            "J5={:+8.3f} deg "
            "step={:.3f} deg / {}".format(
                z_mm,
                j5_deg,
                adjacent_delta_deg,
                adjacent_joint,
            ),
            flush=True,
        )

        output_rows.append({
            "scan_index": scan_index,
            "tilt_deg": selected_tilt_deg,
            "yaw_delta_deg": yaw_delta_deg,
            "z_mm": z_mm,
            "result": result_name,
            "valid": True,
            "j5_deg": j5_deg,
            "j5_lower_margin_deg": (
                j5_lower_margin_deg
            ),
            "j5_upper_margin_deg": (
                j5_upper_margin_deg
            ),
            "adjacent_delta_deg": (
                adjacent_delta_deg
            ),
            "adjacent_joint": adjacent_joint,
            "contacts": result["contacts"],
            **{
                name: values[name]
                for name in boundary.JOINT_NAMES
            },
        })

        previous_positions = list(positions)

        seed_state = copy.deepcopy(
            result["state"]
        )

    fieldnames = [
        "scan_index",
        "tilt_deg",
        "yaw_delta_deg",
        "z_mm",
        "result",
        "valid",
        "j5_deg",
        "j5_lower_margin_deg",
        "j5_upper_margin_deg",
        "adjacent_delta_deg",
        "adjacent_joint",
        "contacts",
    ] + list(boundary.JOINT_NAMES)

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(output_rows)

    print("\n===== Low-height summary =====")

    print(
        "paper direction       : paper_plus_x"
    )

    print(
        "selected local_y tilt : "
        "{:.6f} deg".format(
            selected_tilt_deg
        )
    )

    print(
        "lowest continuous Z   :",
        (
            "{:.6f} mm".format(
                lowest_valid_z_mm
            )
            if lowest_valid_z_mm is not None
            else "-"
        ),
    )

    if minimum_j5_lower_margin_deg is not None:
        print(
            "minimum J5 lower margin: "
            "{:.6f} deg".format(
                minimum_j5_lower_margin_deg
            )
        )

        print(
            "minimum J5 upper margin: "
            "{:.6f} deg".format(
                minimum_j5_upper_margin_deg
            )
        )

    print(
        "maximum adjacent delta: "
        "{:.6f} deg, {} at {}".format(
            maximum_adjacent_delta_deg,
            maximum_adjacent_joint or "-",
            maximum_adjacent_segment or "-",
        )
    )

    if first_failure_z_mm is not None:
        print(
            "first failure Z       : "
            "{:.6f} mm".format(
                first_failure_z_mm
            )
        )
        print(
            "first failure result  :",
            first_failure_result,
        )
        print(
            "first failure contacts:",
            first_failure_contacts,
        )
    else:
        print(
            "first failure          : "
            "none within scan range"
        )

    if (
        lowest_valid_z_mm is not None
        and lowest_valid_z_mm < 10.0
    ):
        print(
            "below-10-mm judgment  : PASS"
        )
    else:
        print(
            "below-10-mm judgment  : FAIL"
        )

    print("CSV                    :", OUTPUT_CSV)
    print("execution              : DISABLED")


if __name__ == "__main__":
    main()
