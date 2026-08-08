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

ORIGINAL_CSV = (
    "/home/maeda/test9_pf_ik_collision_scan.csv"
)

CORRECTED_CSV = (
    "/home/maeda/test9_local_y_5p0deg_forward.csv"
)

OUTPUT_CSV = (
    "/home/maeda/"
    "test9_paper_aligned_yaw_height_scan.csv"
)

POINT_INDEX = 0

# 垂直方向の傾きは小さく保つ。
LOCAL_Y_TILT_DEG = 5.0

START_Z_MM = 60.35
END_Z_MM = 5.35
STEP_Z_MM = 1.0

# paper_center基準で工具前方軸を向ける方位。
TARGET_DIRECTIONS = [
    ("paper_plus_x", 0.0),
    ("paper_plus_y", 90.0),
    ("paper_minus_x", 180.0),
    ("paper_minus_y", -90.0),
]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_quaternion(q):
    norm = math.sqrt(sum(value * value for value in q))

    if norm < 1.0e-12:
        raise RuntimeError("Quaternion norm is zero")

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


def quaternion_conjugate(q):
    return [-q[0], -q[1], -q[2], q[3]]


def axis_angle_quaternion(axis, angle_rad):
    norm = math.sqrt(sum(value * value for value in axis))
    axis = [value / norm for value in axis]

    half = angle_rad / 2.0
    scale = math.sin(half)

    return [
        axis[0] * scale,
        axis[1] * scale,
        axis[2] * scale,
        math.cos(half),
    ]


def rotate_vector(q, vector):
    vector_q = [
        vector[0],
        vector[1],
        vector[2],
        0.0,
    ]

    result = quaternion_multiply(
        quaternion_multiply(q, vector_q),
        quaternion_conjugate(q),
    )

    return result[:3]


def wrap_angle_deg(angle_deg):
    while angle_deg > 180.0:
        angle_deg -= 360.0

    while angle_deg <= -180.0:
        angle_deg += 360.0

    return angle_deg


def joint_dictionary(robot_state):
    return dict(zip(
        robot_state.joint_state.name,
        robot_state.joint_state.position,
    ))


def solve_pose(
    boundary,
    full_scan,
    compute_ik,
    check_validity,
    pose,
    seed_state,
):
    free_result = full_scan.call_ik(
        boundary,
        compute_ik,
        check_validity,
        pose,
        seed_state,
        avoid_collisions=False,
    )

    if (
        free_result["success"]
        and free_result["valid"]
    ):
        return free_result, "VALID"

    collision_result = full_scan.call_ik(
        boundary,
        compute_ik,
        check_validity,
        pose,
        seed_state,
        avoid_collisions=True,
    )

    if (
        collision_result["success"]
        and collision_result["valid"]
    ):
        return collision_result, "VALID_ALTERNATIVE"

    if free_result["success"]:
        return free_result, "COLLISION"

    return free_result, "IK_FAILED"


def generate_z_values():
    z_values = []
    z_mm = START_Z_MM

    while z_mm >= END_Z_MM - 1.0e-9:
        z_values.append(round(z_mm, 6))
        z_mm -= STEP_Z_MM

    return z_values


def main():
    boundary = load_module(
        BOUNDARY_MODULE_PATH,
        "boundary_module",
    )

    full_scan = load_module(
        FULL_SCAN_MODULE_PATH,
        "full_scan_module",
    )

    moveit_commander.roscpp_initialize(sys.argv)

    rospy.init_node(
        "test9_paper_aligned_yaw_height_scan",
        anonymous=True,
    )

    with open(ORIGINAL_CSV, newline="") as f:
        original_rows = list(csv.DictReader(f))

    with open(CORRECTED_CSV, newline="") as f:
        corrected_rows = list(csv.DictReader(f))

    original_by_index = {
        int(row["index"]): row
        for row in original_rows
    }

    corrected_by_index = {
        int(row["index"]): row
        for row in corrected_rows
    }

    original_row = original_by_index[POINT_INDEX]
    seed_row = corrected_by_index[POINT_INDEX]

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

    j5_lower, _ = robot.get_joint(
        "cobotta_joint_5"
    ).bounds()

    original_q = normalize_quaternion(
        boundary.quaternion_from_row(original_row)
    )

    # 既存のlocal_y +5°を保持する。
    tilt_q = axis_angle_quaternion(
        [0.0, 1.0, 0.0],
        math.radians(LOCAL_Y_TILT_DEG),
    )

    base_q = normalize_quaternion(
        quaternion_multiply(
            original_q,
            tilt_q,
        )
    )

    # URDF上でTCPはlocal Z方向に伸びているため、
    # local +Zを工具前方軸として扱う。
    current_tool_axis = rotate_vector(
        base_q,
        [0.0, 0.0, 1.0],
    )

    current_azimuth_deg = math.degrees(
        math.atan2(
            current_tool_axis[1],
            current_tool_axis[0],
        )
    )

    print(
        "\n===== Paper-aligned yaw scan =====",
        flush=True,
    )

    print(
        "current tool-axis azimuth in paper_center: "
        "{:.3f} deg".format(current_azimuth_deg),
        flush=True,
    )

    output_rows = []
    summaries = []

    for direction_name, target_azimuth_deg in TARGET_DIRECTIONS:
        yaw_delta_deg = wrap_angle_deg(
            target_azimuth_deg
            - current_azimuth_deg
        )

        # paper_center Z軸回りの回転なので、
        # クォータニオンを左側から掛ける。
        yaw_q = axis_angle_quaternion(
            [0.0, 0.0, 1.0],
            math.radians(yaw_delta_deg),
        )

        target_q = normalize_quaternion(
            quaternion_multiply(
                yaw_q,
                base_q,
            )
        )

        seed_state = boundary.seed_from_row(
            robot,
            seed_row,
        )

        previous_positions = None
        lowest_continuous_z = None
        first_failure_z = None
        first_failure_result = ""
        first_failure_contacts = ""
        minimum_j5_margin = None
        maximum_joint_step = 0.0
        continuous = True

        print(
            "\n--- {} ---".format(direction_name),
            flush=True,
        )

        print(
            "target azimuth={:.1f} deg, "
            "parent-Z rotation={:+.3f} deg".format(
                target_azimuth_deg,
                yaw_delta_deg,
            ),
            flush=True,
        )

        for z_mm in generate_z_values():
            position = boundary.position_from_row(
                original_row
            )

            pose = boundary.make_pose(
                position,
                target_q,
            )

            pose.pose.position.z = z_mm / 1000.0

            result, result_name = solve_pose(
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

            print(
                "  Z={:6.2f} mm : {}{}".format(
                    z_mm,
                    result_name,
                    (
                        " / " + result["contacts"]
                        if result["contacts"]
                        else ""
                    ),
                ),
                flush=True,
            )

            j5_margin_deg = ""
            joint_step_deg = ""

            if valid:
                values = joint_dictionary(
                    result["state"]
                )

                positions = [
                    values[name]
                    for name in boundary.JOINT_NAMES
                ]

                j5_margin_deg = math.degrees(
                    values["cobotta_joint_5"]
                    - j5_lower
                )

                if previous_positions is not None:
                    joint_step_deg = max(
                        math.degrees(
                            abs(current - previous)
                        )
                        for current, previous in zip(
                            positions,
                            previous_positions,
                        )
                    )

                    maximum_joint_step = max(
                        maximum_joint_step,
                        joint_step_deg,
                    )

                if continuous:
                    lowest_continuous_z = z_mm

                    if (
                        minimum_j5_margin is None
                        or j5_margin_deg < minimum_j5_margin
                    ):
                        minimum_j5_margin = j5_margin_deg

                previous_positions = positions
                seed_state = copy.deepcopy(
                    result["state"]
                )

            else:
                if continuous:
                    continuous = False
                    first_failure_z = z_mm
                    first_failure_result = result_name
                    first_failure_contacts = (
                        result["contacts"] or "-"
                    )

                previous_positions = None

            output_rows.append({
                "direction": direction_name,
                "target_azimuth_deg": (
                    target_azimuth_deg
                ),
                "yaw_delta_deg": yaw_delta_deg,
                "z_mm": z_mm,
                "result": result_name,
                "valid": valid,
                "j5_margin_deg": j5_margin_deg,
                "joint_step_deg": joint_step_deg,
                "contacts": result["contacts"],
            })

        summaries.append({
            "direction": direction_name,
            "target_azimuth_deg": (
                target_azimuth_deg
            ),
            "yaw_delta_deg": yaw_delta_deg,
            "lowest_continuous_z_mm": (
                lowest_continuous_z
            ),
            "minimum_j5_margin_deg": (
                minimum_j5_margin
            ),
            "maximum_joint_step_deg": (
                maximum_joint_step
            ),
            "first_failure_z_mm": (
                first_failure_z
            ),
            "first_failure_result": (
                first_failure_result
            ),
            "first_failure_contacts": (
                first_failure_contacts
            ),
        })

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "direction",
                "target_azimuth_deg",
                "yaw_delta_deg",
                "z_mm",
                "result",
                "valid",
                "j5_margin_deg",
                "joint_step_deg",
                "contacts",
            ],
        )

        writer.writeheader()
        writer.writerows(output_rows)

    print(
        "\n===== Paper-aligned yaw summary ====="
    )

    print(
        "direction | yaw rotation | lowest Z | "
        "min J5 margin | max joint step | first failure"
    )

    print("-" * 110)

    for summary in summaries:
        lowest_text = (
            "{:.2f} mm".format(
                summary["lowest_continuous_z_mm"]
            )
            if summary["lowest_continuous_z_mm"]
            is not None
            else "-"
        )

        margin_text = (
            "{:.3f} deg".format(
                summary["minimum_j5_margin_deg"]
            )
            if summary["minimum_j5_margin_deg"]
            is not None
            else "-"
        )

        failure_text = (
            "{:.2f} mm / {} / {}".format(
                summary["first_failure_z_mm"],
                summary["first_failure_result"],
                summary["first_failure_contacts"],
            )
            if summary["first_failure_z_mm"]
            is not None
            else "none"
        )

        print(
            "{:>13} | {:+10.3f} deg | "
            "{:>9} | {:>14} | "
            "{:>14.3f} deg | {}".format(
                summary["direction"],
                summary["yaw_delta_deg"],
                lowest_text,
                margin_text,
                summary["maximum_joint_step_deg"],
                failure_text,
            )
        )

    print("\nCSV:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
