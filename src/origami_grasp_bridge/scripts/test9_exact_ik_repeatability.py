#!/usr/bin/env python3

import copy
import csv
import importlib.util
import math
import statistics
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
    "test9_exact_p10_q10_repeatability.csv"
)

TRIALS_PER_TIMEOUT = 30
TIMEOUTS = [0.20, 1.00, 2.00]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "boundary_refine",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def joint_values(robot_state, joint_names):
    values_by_name = dict(
        zip(
            robot_state.joint_state.name,
            robot_state.joint_state.position,
        )
    )

    return [
        values_by_name[name]
        for name in joint_names
    ]


def main():
    module = load_module()

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node(
        "test9_exact_ik_repeatability",
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

    joint_names = list(module.JOINT_NAMES)

    # 毎回同じindex 11の関節状態をシードにする。
    common_seed = module.seed_from_row(
        robot,
        row11,
    )

    p10 = module.position_from_row(row10)
    q10 = module.quaternion_from_row(row10)

    exact_pose = module.make_pose(
        p10,
        q10,
    )

    j5_joint = robot.get_joint(
        "cobotta_joint_5"
    )
    j5_lower, _ = j5_joint.bounds()

    output_rows = []

    print("\n===== Exact P10 + Q10 repeatability =====")
    print(
        "target xyz:",
        "({:.9f}, {:.9f}, {:.9f})".format(
            p10[0],
            p10[1],
            p10[2],
        ),
    )
    print(
        "target quaternion:",
        "({:.9f}, {:.9f}, {:.9f}, {:.9f})".format(
            q10[0],
            q10[1],
            q10[2],
            q10[3],
        ),
    )
    print(
        "common seed: CSV index 11"
    )

    for timeout in TIMEOUTS:
        success_count = 0
        valid_count = 0
        collision_count = 0

        successful_j5 = []
        branches = {}

        print(
            "\n===== timeout {:.2f} s =====".format(
                timeout
            )
        )

        for trial in range(1, TRIALS_PER_TIMEOUT + 1):
            result = module.request_ik(
                compute_ik,
                check_validity,
                exact_pose,
                copy.deepcopy(common_seed),
                timeout=timeout,
            )

            row = {
                "timeout": timeout,
                "trial": trial,
                "ik_success": result["success"],
                "state_valid": (
                    result["valid"]
                    if result["success"]
                    else ""
                ),
                "error_code": result["code"],
                "contacts": result["contacts"],
            }

            if not result["success"]:
                for joint_name in joint_names:
                    row[joint_name] = ""

                row["j5_margin_deg"] = ""
                row["branch"] = ""

                output_rows.append(row)

                print(
                    "trial {:02d}: FAILED code={}".format(
                        trial,
                        result["code"],
                    )
                )
                continue

            success_count += 1

            if result["valid"]:
                valid_count += 1
            else:
                collision_count += 1

            values = joint_values(
                result["state"],
                joint_names,
            )

            for joint_name, value in zip(
                joint_names,
                values,
            ):
                row[joint_name] = value

            j5_value = values[
                joint_names.index("cobotta_joint_5")
            ]

            j5_margin_deg = math.degrees(
                j5_value - j5_lower
            )

            successful_j5.append(
                math.degrees(j5_value)
            )

            # 0.001 rad単位で関節解の分岐を分類する。
            branch = tuple(
                round(value, 3)
                for value in values
            )

            branch_text = ",".join(
                "{:.3f}".format(value)
                for value in branch
            )

            branches[branch_text] = (
                branches.get(branch_text, 0) + 1
            )

            row["j5_margin_deg"] = j5_margin_deg
            row["branch"] = branch_text

            output_rows.append(row)

            print(
                "trial {:02d}: SUCCESS {} "
                "J5={:.6f} deg "
                "margin={:.6f} deg".format(
                    trial,
                    (
                        "VALID"
                        if result["valid"]
                        else "COLLISION"
                    ),
                    math.degrees(j5_value),
                    j5_margin_deg,
                )
            )

        print("\n----- summary -----")
        print(
            "success:",
            "{}/{}".format(
                success_count,
                TRIALS_PER_TIMEOUT,
            ),
        )
        print("valid:", valid_count)
        print("collision:", collision_count)
        print("failure:", TRIALS_PER_TIMEOUT - success_count)
        print("branch count:", len(branches))

        for branch, count in sorted(
            branches.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            print(
                "  branch count={:2d}: {}".format(
                    count,
                    branch,
                )
            )

        if successful_j5:
            print(
                "J5 deg min/median/max:",
                "{:.6f} / {:.6f} / {:.6f}".format(
                    min(successful_j5),
                    statistics.median(successful_j5),
                    max(successful_j5),
                ),
            )

    fieldnames = [
        "timeout",
        "trial",
        "ik_success",
        "state_valid",
        "error_code",
        "contacts",
    ] + joint_names + [
        "j5_margin_deg",
        "branch",
    ]

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

    print("\nCSV:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
