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

CORRECTED_CSV = (
    "/home/maeda/test9_local_y_5p0deg_forward.csv"
)

OUTPUT_CSV = (
    "/home/maeda/"
    "test9_tilt5_to40_transition_scan.csv"
)

POINT_INDEX = 0
FIXED_Z_MM = 60.35

START_TILT_DEG = 5.0
END_TILT_DEG = 40.0
TRANSITION_STEPS = 51

MAX_ACCEPTABLE_ADJACENT_DELTA_DEG = 5.0


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
        "test9_tilt5_to40_transition_scan",
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
    corrected_row = corrected_by_index[POINT_INDEX]

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
        corrected_row,
    )

    original_q = transition.normalize_quaternion(
        boundary.quaternion_from_row(original_row)
    )

    position = boundary.position_from_row(
        original_row
    )

    j5_lower, j5_upper = robot.get_joint(
        "cobotta_joint_5"
    ).bounds()

    records = []
    previous_positions = None

    completed = True
    failure_step = None
    failure_result = ""
    failure_contacts = ""

    maximum_adjacent_delta_deg = 0.0
    maximum_adjacent_joint = ""
    maximum_adjacent_segment = ""

    minimum_j5_deg = None
    maximum_j5_deg = None

    print(
        "\n===== local_y +5 to +40 transition =====",
        flush=True,
    )

    print(
        "fixed TCP Z       : {:.3f} mm".format(
            FIXED_Z_MM
        ),
        flush=True,
    )

    print(
        "local_y tilt      : {:.1f} -> {:.1f} deg".format(
            START_TILT_DEG,
            END_TILT_DEG,
        ),
        flush=True,
    )

    for step in range(TRANSITION_STEPS):
        ratio = step / float(
            TRANSITION_STEPS - 1
        )

        tilt_deg = (
            START_TILT_DEG
            + (
                END_TILT_DEG
                - START_TILT_DEG
            )
            * ratio
        )

        local_y_q = transition.axis_angle_quaternion(
            [0.0, 1.0, 0.0],
            math.radians(tilt_deg),
        )

        target_q = transition.normalize_quaternion(
            transition.quaternion_multiply(
                original_q,
                local_y_q,
            )
        )

        pose = boundary.make_pose(
            position,
            target_q,
        )

        pose.pose.position.z = (
            FIXED_Z_MM / 1000.0
        )

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
            completed = False
            failure_step = step
            failure_result = result_name
            failure_contacts = (
                result["contacts"] or "-"
            )

            print(
                "[step {:02d}] "
                "tilt={:7.3f} deg : {} / {}".format(
                    step,
                    tilt_deg,
                    failure_result,
                    failure_contacts,
                ),
                flush=True,
            )
            break

        values = joint_dictionary(
            result["state"]
        )

        positions = [
            values[name]
            for name in boundary.JOINT_NAMES
        ]

        adjacent_delta_deg = ""
        adjacent_joint = ""

        if previous_positions is not None:
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

            if (
                adjacent_delta_deg
                > maximum_adjacent_delta_deg
            ):
                maximum_adjacent_delta_deg = (
                    adjacent_delta_deg
                )
                maximum_adjacent_joint = (
                    adjacent_joint
                )
                maximum_adjacent_segment = (
                    "{}->{}".format(
                        step - 1,
                        step,
                    )
                )

        j5_value = values["cobotta_joint_5"]
        j5_deg = math.degrees(j5_value)

        j5_lower_margin_deg = math.degrees(
            j5_value - j5_lower
        )

        j5_upper_margin_deg = math.degrees(
            j5_upper - j5_value
        )

        if minimum_j5_deg is None:
            minimum_j5_deg = j5_deg
            maximum_j5_deg = j5_deg
        else:
            minimum_j5_deg = min(
                minimum_j5_deg,
                j5_deg,
            )
            maximum_j5_deg = max(
                maximum_j5_deg,
                j5_deg,
            )

        records.append({
            "step": step,
            "ratio": ratio,
            "tilt_deg": tilt_deg,
            "result": result_name,
            "valid": valid,
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

        print(
            "[step {:02d}] "
            "tilt={:7.3f} deg "
            "J5={:+8.3f} deg "
            "step={}".format(
                step,
                tilt_deg,
                j5_deg,
                (
                    "{:.3f} deg / {}".format(
                        adjacent_delta_deg,
                        adjacent_joint,
                    )
                    if adjacent_delta_deg != ""
                    else "-"
                ),
            ),
            flush=True,
        )

        previous_positions = list(positions)
        seed_state = copy.deepcopy(
            result["state"]
        )

    fieldnames = [
        "step",
        "ratio",
        "tilt_deg",
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
        writer.writerows(records)

    print("\n===== Tilt transition summary =====")

    print(
        "completed             :",
        completed,
    )

    print(
        "valid transition steps:",
        len(records),
        "/",
        TRANSITION_STEPS,
    )

    if completed:
        print(
            "maximum adjacent delta: "
            "{:.6f} deg, {} at {}".format(
                maximum_adjacent_delta_deg,
                maximum_adjacent_joint or "-",
                maximum_adjacent_segment or "-",
            )
        )

        print(
            "J5 range              : "
            "{:.6f} to {:.6f} deg".format(
                minimum_j5_deg,
                maximum_j5_deg,
            )
        )

        if (
            maximum_adjacent_delta_deg
            <= MAX_ACCEPTABLE_ADJACENT_DELTA_DEG
        ):
            print(
                "continuity judgment   : PASS"
            )
        else:
            print(
                "continuity judgment   : "
                "FAIL - joint jump detected"
            )

    else:
        print(
            "first failed step      :",
            failure_step,
        )
        print(
            "failure result         :",
            failure_result,
        )
        print(
            "failure contacts       :",
            failure_contacts,
        )

    print("CSV                   :", OUTPUT_CSV)
    print("execution             : DISABLED")


if __name__ == "__main__":
    main()
