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
    "/home/maeda/test9_angle_height_clearance_scan.csv"
)

POINT_INDEX = 0

ANGLES_DEG = list(range(5, 61, 5))

START_Z_MM = 60.35
END_Z_MM = 5.35
STEP_Z_MM = 1.0

IK_TIMEOUT = 1.0


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

    return collision_result, "IK_FAILED"


def generate_z_values():
    values = []
    current = START_Z_MM

    while current >= END_Z_MM - 1.0e-9:
        values.append(round(current, 6))
        current -= STEP_Z_MM

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

    moveit_commander.roscpp_initialize(sys.argv)

    rospy.init_node(
        "test9_angle_height_clearance_scan",
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

    output_rows = []
    summaries = []

    for angle_deg in ANGLES_DEG:
        seed_state = boundary.seed_from_row(
            robot,
            seed_row,
        )

        previous_positions = None
        continuous = True

        continuous_lowest_z = None
        minimum_j5_margin = None
        maximum_adjacent_delta = 0.0
        first_failure_z = None
        first_failure_result = ""
        first_failure_contacts = ""

        for z_mm in generate_z_values():
            pose = full_scan.make_corrected_pose(
                boundary,
                original_row,
                float(angle_deg),
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

            j5_margin_deg = ""
            adjacent_delta_deg = ""

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
                    adjacent_delta_deg = max(
                        math.degrees(
                            abs(current - previous)
                        )
                        for current, previous in zip(
                            positions,
                            previous_positions,
                        )
                    )

                    maximum_adjacent_delta = max(
                        maximum_adjacent_delta,
                        adjacent_delta_deg,
                    )

                if continuous:
                    continuous_lowest_z = z_mm

                    if (
                        minimum_j5_margin is None
                        or j5_margin_deg
                        < minimum_j5_margin
                    ):
                        minimum_j5_margin = (
                            j5_margin_deg
                        )

                seed_state = copy.deepcopy(
                    result["state"]
                )
                previous_positions = positions

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
                "angle_deg": angle_deg,
                "z_mm": z_mm,
                "result": result_name,
                "valid": valid,
                "continuous_from_start": (
                    continuous and valid
                ),
                "j5_margin_deg": j5_margin_deg,
                "adjacent_delta_deg": (
                    adjacent_delta_deg
                ),
                "contacts": result["contacts"],
            })

        summaries.append({
            "angle_deg": angle_deg,
            "lowest_continuous_z_mm": (
                continuous_lowest_z
            ),
            "minimum_j5_margin_deg": (
                minimum_j5_margin
            ),
            "maximum_adjacent_delta_deg": (
                maximum_adjacent_delta
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
                "angle_deg",
                "z_mm",
                "result",
                "valid",
                "continuous_from_start",
                "j5_margin_deg",
                "adjacent_delta_deg",
                "contacts",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print("\n===== Angle-height clearance summary =====")
    print(
        "angle | lowest continuous Z | "
        "min J5 margin | max joint step | "
        "first failure"
    )
    print("-" * 100)

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
            "{:>5.1f} | {:>19} | {:>14} | "
            "{:>14.3f} deg | {}".format(
                summary["angle_deg"],
                lowest_text,
                margin_text,
                summary[
                    "maximum_adjacent_delta_deg"
                ],
                failure_text,
            )
        )

    candidates = [
        summary
        for summary in summaries
        if (
            summary["lowest_continuous_z_mm"]
            is not None
            and summary[
                "lowest_continuous_z_mm"
            ] < 10.0
        )
    ]

    print("\n===== Candidates below 10 mm =====")

    if candidates:
        for summary in candidates:
            print(
                "local_y +{:.1f} deg: "
                "lowest Z={:.2f} mm, "
                "minimum J5 margin={:.3f} deg, "
                "max step={:.3f} deg".format(
                    summary["angle_deg"],
                    summary[
                        "lowest_continuous_z_mm"
                    ],
                    summary[
                        "minimum_j5_margin_deg"
                    ],
                    summary[
                        "maximum_adjacent_delta_deg"
                    ],
                )
            )
    else:
        print(
            "No local_y-only candidate reached "
            "below 10 mm."
        )

    print("\nCSV:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
