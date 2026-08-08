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
    "test9_paper_plus_x_transition_tilt40_scan.csv"
)

POINT_INDEX = 0

FIXED_Z_MM = 60.35
LOCAL_Y_TILT_DEG = 40.0

# 約1°刻みになるよう十分細かく補間する。
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


def normalize_quaternion(q):
    norm = math.sqrt(
        sum(value * value for value in q)
    )

    if norm < 1.0e-12:
        raise RuntimeError(
            "Quaternion norm is zero"
        )

    return [
        value / norm
        for value in q
    ]


def quaternion_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    return [
        w1 * x2 + x1 * w2
        + y1 * z2 - z1 * y2,

        w1 * y2 - x1 * z2
        + y1 * w2 + z1 * x2,

        w1 * z2 + x1 * y2
        - y1 * x2 + z1 * w2,

        w1 * w2
        - x1 * x2
        - y1 * y2
        - z1 * z2,
    ]


def quaternion_conjugate(q):
    return [
        -q[0],
        -q[1],
        -q[2],
        q[3],
    ]


def axis_angle_quaternion(axis, angle_rad):
    norm = math.sqrt(
        sum(value * value for value in axis)
    )

    axis = [
        value / norm
        for value in axis
    ]

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
        quaternion_multiply(
            q,
            vector_q,
        ),
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
        return (
            collision_result,
            "VALID_ALTERNATIVE",
        )

    if free_result["success"]:
        return free_result, "COLLISION"

    return collision_result, "IK_FAILED"


def main():
    boundary = load_module(
        BOUNDARY_MODULE_PATH,
        "boundary_module",
    )

    full_scan = load_module(
        FULL_SCAN_MODULE_PATH,
        "full_scan_module",
    )

    moveit_commander.roscpp_initialize(
        sys.argv
    )

    rospy.init_node(
        "test9_paper_plus_x_transition_tilt40_scan",
        anonymous=True,
    )

    with open(
        ORIGINAL_CSV,
        newline="",
    ) as f:
        original_rows = list(
            csv.DictReader(f)
        )

    with open(
        CORRECTED_CSV,
        newline="",
    ) as f:
        corrected_rows = list(
            csv.DictReader(f)
        )

    original_by_index = {
        int(row["index"]): row
        for row in original_rows
    }

    corrected_by_index = {
        int(row["index"]): row
        for row in corrected_rows
    }

    original_row = original_by_index[
        POINT_INDEX
    ]

    seed_row = corrected_by_index[
        POINT_INDEX
    ]

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
        seed_row,
    )

    original_q = normalize_quaternion(
        boundary.quaternion_from_row(
            original_row
        )
    )

    local_y_q = axis_angle_quaternion(
        [0.0, 1.0, 0.0],
        math.radians(
            LOCAL_Y_TILT_DEG
        ),
    )

    # 従来のlocal_y +5°姿勢。
    start_q = normalize_quaternion(
        quaternion_multiply(
            original_q,
            local_y_q,
        )
    )

    # 工具前方軸はlocal +Zとして扱う。
    start_tool_axis = rotate_vector(
        start_q,
        [0.0, 0.0, 1.0],
    )

    start_azimuth_deg = math.degrees(
        math.atan2(
            start_tool_axis[1],
            start_tool_axis[0],
        )
    )

    # paper_plus_xの方位は0°。
    target_azimuth_deg = 0.0

    yaw_delta_deg = wrap_angle_deg(
        target_azimuth_deg
        - start_azimuth_deg
    )

    j5_lower, j5_upper = robot.get_joint(
        "cobotta_joint_5"
    ).bounds()

    position = boundary.position_from_row(
        original_row
    )

    records = []

    previous_positions = None
    start_positions = None
    final_positions = None

    max_adjacent_delta_deg = 0.0
    max_adjacent_joint = ""
    max_adjacent_segment = ""

    minimum_j5_deg = None
    maximum_j5_deg = None

    completed = True
    failure_step = None
    failure_result = ""
    failure_contacts = ""

    print(
        "\n===== paper_plus_x transition scan =====",
        flush=True,
    )

    print(
        "fixed TCP Z          : "
        "{:.3f} mm".format(FIXED_Z_MM),
        flush=True,
    )

    print(
        "start azimuth        : "
        "{:.6f} deg".format(
            start_azimuth_deg
        ),
        flush=True,
    )

    print(
        "target azimuth       : "
        "{:.6f} deg".format(
            target_azimuth_deg
        ),
        flush=True,
    )

    print(
        "required yaw rotation: "
        "{:+.6f} deg".format(
            yaw_delta_deg
        ),
        flush=True,
    )

    for step in range(
        TRANSITION_STEPS
    ):
        ratio = (
            step
            / float(
                TRANSITION_STEPS - 1
            )
        )

        current_yaw_deg = (
            yaw_delta_deg * ratio
        )

        parent_yaw_q = (
            axis_angle_quaternion(
                [0.0, 0.0, 1.0],
                math.radians(
                    current_yaw_deg
                ),
            )
        )

        target_q = normalize_quaternion(
            quaternion_multiply(
                parent_yaw_q,
                start_q,
            )
        )

        pose = boundary.make_pose(
            position,
            target_q,
        )

        pose.pose.position.z = (
            FIXED_Z_MM / 1000.0
        )

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

        if not valid:
            completed = False
            failure_step = step
            failure_result = result_name
            failure_contacts = (
                result["contacts"] or "-"
            )

            print(
                "[step {:02d}] "
                "yaw={:+8.3f} deg : "
                "{} / {}".format(
                    step,
                    current_yaw_deg,
                    result_name,
                    failure_contacts,
                ),
                flush=True,
            )

            break

        values = joint_dictionary(
            result["state"]
        )

        positions = [
            values[joint_name]
            for joint_name
            in boundary.JOINT_NAMES
        ]

        j5_value = values[
            "cobotta_joint_5"
        ]

        j5_deg = math.degrees(
            j5_value
        )

        j5_lower_margin_deg = (
            math.degrees(
                j5_value - j5_lower
            )
        )

        j5_upper_margin_deg = (
            math.degrees(
                j5_upper - j5_value
            )
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

        adjacent_delta_deg = ""
        adjacent_joint = ""

        if previous_positions is not None:
            deltas = [
                abs(current - previous)
                for current, previous
                in zip(
                    positions,
                    previous_positions,
                )
            ]

            max_index = max(
                range(len(deltas)),
                key=lambda index: (
                    deltas[index]
                ),
            )

            adjacent_delta_deg = (
                math.degrees(
                    deltas[max_index]
                )
            )

            adjacent_joint = (
                boundary.JOINT_NAMES[
                    max_index
                ]
            )

            if (
                adjacent_delta_deg
                > max_adjacent_delta_deg
            ):
                max_adjacent_delta_deg = (
                    adjacent_delta_deg
                )
                max_adjacent_joint = (
                    adjacent_joint
                )
                max_adjacent_segment = (
                    "{}->{}".format(
                        step - 1,
                        step,
                    )
                )

        if start_positions is None:
            start_positions = list(
                positions
            )

        final_positions = list(
            positions
        )

        records.append({
            "step": step,
            "ratio": ratio,
            "yaw_rotation_deg": (
                current_yaw_deg
            ),
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
            "adjacent_joint": (
                adjacent_joint
            ),
            "contacts": (
                result["contacts"]
            ),
            **{
                joint_name: values[
                    joint_name
                ]
                for joint_name
                in boundary.JOINT_NAMES
            },
        })

        print(
            "[step {:02d}] "
            "yaw={:+8.3f} deg "
            "J5={:+8.3f} deg "
            "step={}".format(
                step,
                current_yaw_deg,
                j5_deg,
                (
                    "{:.3f} deg / {}".format(
                        adjacent_delta_deg,
                        adjacent_joint,
                    )
                    if adjacent_delta_deg
                    != ""
                    else "-"
                ),
            ),
            flush=True,
        )

        previous_positions = list(
            positions
        )

        seed_state = copy.deepcopy(
            result["state"]
        )

    fieldnames = [
        "step",
        "ratio",
        "yaw_rotation_deg",
        "result",
        "valid",
        "j5_deg",
        "j5_lower_margin_deg",
        "j5_upper_margin_deg",
        "adjacent_delta_deg",
        "adjacent_joint",
        "contacts",
    ] + list(boundary.JOINT_NAMES)

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(records)

    print(
        "\n===== Transition summary ====="
    )

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
                max_adjacent_delta_deg,
                max_adjacent_joint or "-",
                max_adjacent_segment or "-",
            )
        )

        print(
            "J5 range              : "
            "{:.6f} to {:.6f} deg".format(
                minimum_j5_deg,
                maximum_j5_deg,
            )
        )

        print(
            "\n===== Start-to-end joint changes ====="
        )

        total_deltas = []

        for index, joint_name in enumerate(
            boundary.JOINT_NAMES
        ):
            delta_deg = math.degrees(
                final_positions[index]
                - start_positions[index]
            )

            total_deltas.append(
                abs(delta_deg)
            )

            print(
                "{:<20} "
                "{:+10.6f} deg".format(
                    joint_name,
                    delta_deg,
                )
            )

        print(
            "maximum total change  : "
            "{:.6f} deg".format(
                max(total_deltas)
            )
        )

        if (
            max_adjacent_delta_deg
            <= MAX_ACCEPTABLE_ADJACENT_DELTA_DEG
        ):
            print(
                "continuity judgment   : "
                "PASS"
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

        if records:
            print(
                "last successful yaw   : "
                "{:+.6f} deg".format(
                    records[-1][
                        "yaw_rotation_deg"
                    ]
                )
            )

    print("CSV                   :", OUTPUT_CSV)
    print("execution             : DISABLED")


if __name__ == "__main__":
    main()
