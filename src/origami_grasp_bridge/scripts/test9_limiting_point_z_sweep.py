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
    "test9_local_y_5deg_limiting_point_z_sweep.csv"
)

CORRECTION_DEG = 5.0
MAX_DROP_MM = 25
STEP_MM = 1


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def joint_values(robot_state):
    return dict(
        zip(
            robot_state.joint_state.name,
            robot_state.joint_state.position,
        )
    )


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
        "test9_limiting_point_z_sweep",
        anonymous=True,
    )

    with open(ORIGINAL_CSV, newline="") as f:
        original_rows = list(csv.DictReader(f))

    with open(CORRECTED_CSV, newline="") as f:
        corrected_rows = list(csv.DictReader(f))

    valid_corrected_rows = [
        row
        for row in corrected_rows
        if row.get("valid", "").lower() == "true"
        and row.get("j5_margin_deg", "") != ""
    ]

    if not valid_corrected_rows:
        raise RuntimeError(
            "補正後CSVに有効点がありません。"
        )

    limiting_row = min(
        valid_corrected_rows,
        key=lambda row: float(
            row["j5_margin_deg"]
        ),
    )

    limiting_index = int(
        limiting_row["index"]
    )

    original_by_index = {
        int(row["index"]): row
        for row in original_rows
    }

    original_row = original_by_index[
        limiting_index
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

    # 0 mm補正時に既に得られている有効な関節解を
    # 最初のシードとして使用する。
    seed_state = boundary.seed_from_row(
        robot,
        limiting_row,
    )

    j5_lower, _ = robot.get_joint(
        "cobotta_joint_5"
    ).bounds()

    base_pose = full_scan.make_corrected_pose(
        boundary,
        original_row,
        CORRECTION_DEG,
    )

    base_z = base_pose.pose.position.z

    print(
        "\n===== Test9 limiting-point Z sweep ====="
    )
    print("limiting index       :", limiting_index)
    print(
        "original J5 margin  : {:.6f} deg".format(
            float(limiting_row["j5_margin_deg"])
        )
    )
    print(
        "base paper-frame Z  : {:.6f} mm".format(
            base_z * 1000.0
        )
    )
    print(
        "orientation correction: "
        "local_y +{:.1f} deg".format(
            CORRECTION_DEG
        )
    )

    print(
        "\ndrop_mm,target_z_mm,IK,state,"
        "J5_deg,J5_margin_deg,contacts"
    )

    output_rows = []

    for drop_mm in range(
        0,
        MAX_DROP_MM + 1,
        STEP_MM,
    ):
        pose = copy.deepcopy(base_pose)

        pose.pose.position.z = (
            base_z - drop_mm / 1000.0
        )

        free_result = full_scan.call_ik(
            boundary,
            compute_ik,
            check_validity,
            pose,
            seed_state,
            avoid_collisions=False,
        )

        final_result = free_result
        result_name = "IK_FAILED"

        if free_result["success"]:
            if free_result["valid"]:
                result_name = "VALID"
            else:
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
                    result_name = "VALID_ALTERNATIVE"
                    final_result = collision_result
                else:
                    result_name = "COLLISION"

                    if collision_result["success"]:
                        final_result = collision_result

        j5_deg = ""
        j5_margin_deg = ""

        is_valid = (
            final_result["success"]
            and final_result["valid"]
        )

        if is_valid:
            seed_state = copy.deepcopy(
                final_result["state"]
            )

            values = joint_values(
                final_result["state"]
            )

            j5_value = values[
                "cobotta_joint_5"
            ]

            j5_deg = math.degrees(j5_value)
            j5_margin_deg = math.degrees(
                j5_value - j5_lower
            )

        print(
            "{:02d},{:.6f},{},{},{},{},{}".format(
                drop_mm,
                pose.pose.position.z * 1000.0,
                (
                    "SUCCESS"
                    if final_result["success"]
                    else "FAILED"
                ),
                result_name,
                (
                    "{:.6f}".format(j5_deg)
                    if j5_deg != ""
                    else "-"
                ),
                (
                    "{:.6f}".format(j5_margin_deg)
                    if j5_margin_deg != ""
                    else "-"
                ),
                final_result["contacts"] or "-",
            )
        )

        output_rows.append({
            "index": limiting_index,
            "correction_deg": CORRECTION_DEG,
            "drop_mm": drop_mm,
            "target_z_m": pose.pose.position.z,
            "target_z_mm": (
                pose.pose.position.z * 1000.0
            ),
            "result": result_name,
            "valid": is_valid,
            "free_ik_code": free_result["code"],
            "j5_deg": j5_deg,
            "j5_margin_deg": j5_margin_deg,
            "contacts": final_result["contacts"],
        })

    fieldnames = [
        "index",
        "correction_deg",
        "drop_mm",
        "target_z_m",
        "target_z_mm",
        "result",
        "valid",
        "free_ik_code",
        "j5_deg",
        "j5_margin_deg",
        "contacts",
    ]

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
        writer.writerows(output_rows)

    valid_rows = [
        row
        for row in output_rows
        if row["valid"]
    ]

    print("\n===== Summary =====")

    if valid_rows:
        lowest_valid = min(
            valid_rows,
            key=lambda row: row["target_z_mm"],
        )

        print(
            "lowest valid Z:",
            "{:.6f} mm".format(
                lowest_valid["target_z_mm"]
            )
        )
        print(
            "drop from original:",
            "{} mm".format(
                lowest_valid["drop_mm"]
            )
        )
        print(
            "J5 margin:",
            "{:.6f} deg".format(
                lowest_valid["j5_margin_deg"]
            )
        )
    else:
        print("No valid point was found.")

    print("CSV:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
