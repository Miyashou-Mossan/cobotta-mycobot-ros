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

CORRECTIONS_DEG = [3.0, 5.0]
IK_TIMEOUT = 1.0


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


def joint_dictionary(robot_state):
    return dict(zip(
        robot_state.joint_state.name,
        robot_state.joint_state.position,
    ))


def make_corrected_pose(module, row, correction_deg):
    position = module.position_from_row(row)
    quaternion = module.quaternion_from_row(row)

    delta_q = axis_angle_quaternion(
        [0.0, 1.0, 0.0],
        math.radians(correction_deg),
    )

    # 工具ローカルY軸回りなので右側から掛ける。
    corrected_q = quaternion_multiply(
        quaternion,
        delta_q,
    )

    corrected_q = normalize_quaternion(corrected_q)

    return module.make_pose(
        position,
        corrected_q,
    )


def contact_summary(contacts):
    pairs = []

    for contact in contacts:
        pair = "{} <-> {}".format(
            contact.contact_body_1,
            contact.contact_body_2,
        )

        if pair not in pairs:
            pairs.append(pair)

    return "; ".join(pairs)


def call_ik(
    module,
    compute_ik,
    check_validity,
    pose,
    seed_state,
    avoid_collisions,
):
    request = module.GetPositionIKRequest()

    request.ik_request.group_name = module.GROUP_NAME
    request.ik_request.robot_state = copy.deepcopy(
        seed_state
    )
    request.ik_request.avoid_collisions = avoid_collisions
    request.ik_request.ik_link_name = module.IK_LINK_NAME
    request.ik_request.pose_stamped = pose
    request.ik_request.timeout = rospy.Duration(
        IK_TIMEOUT
    )

    response = compute_ik(request)
    code = int(response.error_code.val)

    if code != module.MoveItErrorCodes.SUCCESS:
        return {
            "success": False,
            "code": code,
            "valid": False,
            "state": None,
            "contacts": "",
        }

    validity_request = module.GetStateValidityRequest()
    validity_request.robot_state = response.solution
    validity_request.group_name = module.GROUP_NAME

    validity_response = check_validity(
        validity_request
    )

    return {
        "success": True,
        "code": code,
        "valid": bool(validity_response.valid),
        "state": response.solution,
        "contacts": contact_summary(
            validity_response.contacts
        ),
    }


def scan_pass(
    module,
    rows,
    correction_deg,
    direction,
    robot,
    compute_ik,
    check_validity,
    j5_lower,
):
    point_count = len(rows)

    if direction == "forward":
        order = list(range(point_count))
        seed_index = 11
    elif direction == "reverse":
        order = list(reversed(range(point_count)))
        seed_index = point_count - 1
    else:
        raise ValueError("Unknown direction")

    seed_state = module.seed_from_row(
        robot,
        rows[seed_index],
    )

    output_rows = []

    previous_index = None
    previous_positions = None

    max_adjacent_delta = 0.0
    max_adjacent_joint = ""
    max_adjacent_segment = ""

    for index in order:
        row = rows[index]

        pose = make_corrected_pose(
            module,
            row,
            correction_deg,
        )

        free_result = call_ik(
            module,
            compute_ik,
            check_validity,
            pose,
            seed_state,
            avoid_collisions=False,
        )

        result_name = "IK_FAILED"
        final_result = free_result

        if free_result["success"]:
            if free_result["valid"]:
                result_name = "VALID"
            else:
                collision_result = call_ik(
                    module,
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

        joint_positions = []
        j5_deg = ""
        j5_margin_deg = ""
        adjacent_delta_deg = ""
        adjacent_joint = ""

        is_valid = (
            final_result["success"]
            and final_result["valid"]
        )

        if is_valid:
            seed_state = copy.deepcopy(
                final_result["state"]
            )

            values = joint_dictionary(
                final_result["state"]
            )

            joint_positions = [
                values[name]
                for name in module.JOINT_NAMES
            ]

            j5_value = values["cobotta_joint_5"]

            j5_deg = math.degrees(j5_value)
            j5_margin_deg = math.degrees(
                j5_value - j5_lower
            )

            if (
                previous_index is not None
                and previous_positions is not None
                and abs(index - previous_index) == 1
            ):
                deltas = [
                    abs(current - previous)
                    for current, previous in zip(
                        joint_positions,
                        previous_positions,
                    )
                ]

                max_index = max(
                    range(len(deltas)),
                    key=lambda i: deltas[i],
                )

                adjacent_delta_deg = math.degrees(
                    deltas[max_index]
                )
                adjacent_joint = module.JOINT_NAMES[
                    max_index
                ]

                if deltas[max_index] > max_adjacent_delta:
                    max_adjacent_delta = deltas[max_index]
                    max_adjacent_joint = adjacent_joint
                    max_adjacent_segment = (
                        "{}->{}".format(
                            previous_index,
                            index,
                        )
                    )

            previous_index = index
            previous_positions = list(
                joint_positions
            )
        else:
            previous_index = None
            previous_positions = None

        output = {
            "index": index,
            "direction": direction,
            "correction_deg": correction_deg,
            "x": pose.pose.position.x,
            "y": pose.pose.position.y,
            "z": pose.pose.position.z,
            "result": result_name,
            "valid": is_valid,
            "free_ik_code": free_result["code"],
            "contacts": final_result["contacts"],
            "j5_deg": j5_deg,
            "j5_margin_deg": j5_margin_deg,
            "max_adjacent_delta_deg": adjacent_delta_deg,
            "max_adjacent_joint": adjacent_joint,
        }

        for joint_name in module.JOINT_NAMES:
            output[joint_name] = (
                joint_positions[
                    module.JOINT_NAMES.index(joint_name)
                ]
                if joint_positions
                else ""
            )

        output_rows.append(output)

    output_rows.sort(
        key=lambda row: int(row["index"])
    )

    valid_rows = [
        row for row in output_rows
        if row["valid"]
    ]

    summary = {
        "point_count": len(output_rows),
        "valid_count": len(valid_rows),
        "ik_failed_count": sum(
            row["result"] == "IK_FAILED"
            for row in output_rows
        ),
        "collision_count": sum(
            row["result"] == "COLLISION"
            for row in output_rows
        ),
        "alternative_count": sum(
            row["result"] == "VALID_ALTERNATIVE"
            for row in output_rows
        ),
        "minimum_j5_margin_deg": (
            min(
                float(row["j5_margin_deg"])
                for row in valid_rows
            )
            if valid_rows
            else None
        ),
        "maximum_j5_margin_deg": (
            max(
                float(row["j5_margin_deg"])
                for row in valid_rows
            )
            if valid_rows
            else None
        ),
        "max_adjacent_delta_deg": math.degrees(
            max_adjacent_delta
        ),
        "max_adjacent_joint": max_adjacent_joint,
        "max_adjacent_segment": max_adjacent_segment,
        "failed_indices": [
            int(row["index"])
            for row in output_rows
            if not row["valid"]
        ],
    }

    return output_rows, summary


def save_csv(path, rows, joint_names):
    fieldnames = [
        "index",
        "direction",
        "correction_deg",
        "x",
        "y",
        "z",
        "result",
        "valid",
        "free_ik_code",
        "contacts",
        "j5_deg",
        "j5_margin_deg",
        "max_adjacent_delta_deg",
        "max_adjacent_joint",
    ] + list(joint_names)

    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def print_summary(
    correction_deg,
    direction,
    summary,
    output_path,
):
    print("\n" + "=" * 76)
    print(
        "local_y {:+.1f} deg / {}".format(
            correction_deg,
            direction,
        )
    )
    print("=" * 76)

    print("points              :", summary["point_count"])
    print("valid               :", summary["valid_count"])
    print("IK failed           :", summary["ik_failed_count"])
    print("collision           :", summary["collision_count"])
    print("valid alternatives  :", summary["alternative_count"])

    if summary["minimum_j5_margin_deg"] is not None:
        print(
            "minimum J5 margin   : {:.6f} deg".format(
                summary["minimum_j5_margin_deg"]
            )
        )
        print(
            "maximum J5 margin   : {:.6f} deg".format(
                summary["maximum_j5_margin_deg"]
            )
        )

    print(
        "max adjacent delta  : {:.6f} deg, {} at {}".format(
            summary["max_adjacent_delta_deg"],
            summary["max_adjacent_joint"] or "-",
            summary["max_adjacent_segment"] or "-",
        )
    )

    print(
        "failed indices      :",
        summary["failed_indices"],
    )
    print("CSV                 :", output_path)


def main():
    module = load_module()

    moveit_commander.roscpp_initialize(sys.argv)

    rospy.init_node(
        "test9_full_local_y_correction_scan",
        anonymous=True,
    )

    rows, _, _ = module.load_rows()

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

    j5_lower, _ = robot.get_joint(
        "cobotta_joint_5"
    ).bounds()

    summaries = {}

    for correction_deg in CORRECTIONS_DEG:
        summaries[correction_deg] = {}

        for direction in ["forward", "reverse"]:
            output_rows, summary = scan_pass(
                module=module,
                rows=rows,
                correction_deg=correction_deg,
                direction=direction,
                robot=robot,
                compute_ik=compute_ik,
                check_validity=check_validity,
                j5_lower=j5_lower,
            )

            angle_text = str(correction_deg).replace(
                ".", "p"
            )

            output_path = (
                "/home/maeda/"
                "test9_local_y_{}deg_{}.csv".format(
                    angle_text,
                    direction,
                )
            )

            save_csv(
                output_path,
                output_rows,
                module.JOINT_NAMES,
            )

            summaries[correction_deg][
                direction
            ] = (output_rows, summary)

            print_summary(
                correction_deg,
                direction,
                summary,
                output_path,
            )

        forward_rows = {
            int(row["index"]): row
            for row in summaries[
                correction_deg
            ]["forward"][0]
        }

        reverse_rows = {
            int(row["index"]): row
            for row in summaries[
                correction_deg
            ]["reverse"][0]
        }

        mismatches = []

        for index in sorted(forward_rows):
            forward_row = forward_rows[index]
            reverse_row = reverse_rows[index]

            if (
                forward_row["result"]
                != reverse_row["result"]
                or forward_row["valid"]
                != reverse_row["valid"]
            ):
                mismatches.append(index)

        print(
            "\nlocal_y {:+.1f} deg "
            "forward/reverse mismatches: {}".format(
                correction_deg,
                mismatches,
            )
        )


if __name__ == "__main__":
    main()
