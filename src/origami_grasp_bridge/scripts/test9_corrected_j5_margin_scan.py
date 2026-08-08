#!/usr/bin/env python3

import copy
import importlib.util
import math
import sys

import moveit_commander
import rospy


MODULE_PATH = (
    "/home/maeda/catkin_ws/src/"
    "origami_grasp_bridge/scripts/"
    "test9_boundary_refine_scan.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "boundary_refine",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_joint_value(robot_state, joint_name):
    values = dict(zip(
        robot_state.joint_state.name,
        robot_state.joint_state.position,
    ))
    return values[joint_name]


def quaternion_error(q1, q2):
    return max(
        abs(a - b)
        for a, b in zip(q1, q2)
    )


def scan(
    label,
    make_target,
    display_amount,
    module,
    compute_ik,
    check_validity,
    initial_seed,
    j5_lower,
):
    print("\n" + "=" * 75)
    print(label)
    print("=" * 75)

    seed = copy.deepcopy(initial_seed)
    records = []

    # 既知の成功側t=1.00から、
    # 失敗側t=0.00へ連続的に追跡する。
    for step in range(50, -1, -1):
        t = step / 50.0

        pose = make_target(t)

        result = module.request_ik(
            compute_ik,
            check_validity,
            pose,
            copy.deepcopy(seed),
            timeout=0.50,
        )

        record = {
            "t": t,
            "amount": display_amount(t),
            "success": result["success"],
            "valid": (
                result["valid"]
                if result["success"]
                else False
            ),
            "j5_deg": None,
            "margin_deg": None,
            "contacts": result["contacts"],
        }

        if result["success"]:
            j5 = get_joint_value(
                result["state"],
                "cobotta_joint_5",
            )

            record["j5_deg"] = math.degrees(j5)
            record["margin_deg"] = math.degrees(
                j5 - j5_lower
            )

            # 成功解を次の近接点のシードへ引き継ぐ。
            seed = copy.deepcopy(result["state"])

        records.append(record)

    print(
        "t,correction,IK,state,J5_deg,"
        "J5_margin_deg,contacts"
    )

    previous_status = None

    for i, record in enumerate(records):
        status = (
            "VALID"
            if record["success"] and record["valid"]
            else "COLLISION"
            if record["success"]
            else "FAILED"
        )

        # 成否が切り替わる部分と端点だけ表示する。
        changed = status != previous_status
        near_change = (
            i > 0
            and status != (
                "VALID"
                if records[i - 1]["success"]
                and records[i - 1]["valid"]
                else "COLLISION"
                if records[i - 1]["success"]
                else "FAILED"
            )
        )

        if changed or near_change or record["t"] in (1.0, 0.0):
            print(
                "{:.2f},{:.9f},{},{},{},{},{}".format(
                    record["t"],
                    record["amount"],
                    (
                        "SUCCESS"
                        if record["success"]
                        else "FAILED"
                    ),
                    status,
                    (
                        "{:.9f}".format(record["j5_deg"])
                        if record["j5_deg"] is not None
                        else "-"
                    ),
                    (
                        "{:.9f}".format(
                            record["margin_deg"]
                        )
                        if record["margin_deg"] is not None
                        else "-"
                    ),
                    record["contacts"] or "-",
                )
            )

        previous_status = status

    valid_records = [
        record
        for record in records
        if record["success"] and record["valid"]
    ]

    print("\n----- summary -----")

    if not valid_records:
        print("No valid solution")
        return

    nearest_to_target = min(
        valid_records,
        key=lambda record: record["t"],
    )

    print(
        "minimum successful correction:",
        "{:.9f}".format(
            nearest_to_target["amount"]
        ),
    )
    print(
        "J5 at boundary:",
        "{:.9f} deg".format(
            nearest_to_target["j5_deg"]
        ),
    )
    print(
        "J5 margin at boundary:",
        "{:.9f} deg".format(
            nearest_to_target["margin_deg"]
        ),
    )

    for required_margin in (0.5, 1.0, 2.0, 3.0):
        candidates = [
            record
            for record in valid_records
            if record["margin_deg"] >= required_margin
        ]

        if candidates:
            candidate = min(
                candidates,
                key=lambda record: record["amount"],
            )

            print(
                "minimum correction for "
                "J5 margin >= {:.1f} deg: "
                "{:.9f}".format(
                    required_margin,
                    candidate["amount"],
                )
            )
        else:
            print(
                "J5 margin >= {:.1f} deg: "
                "not reached".format(
                    required_margin
                )
            )


def main():
    module = load_module()

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node(
        "test9_corrected_j5_margin_scan",
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
    p11 = module.position_from_row(row11)

    q10 = module.quaternion_from_row(row10)
    q11 = module.quaternion_from_row(row11)

    seed = module.seed_from_row(robot, row11)

    j5_lower, _ = robot.get_joint(
        "cobotta_joint_5"
    ).bounds()

    orientation_difference = (
        module.quaternion_angle_deg(q10, q11)
    )

    position_vector = [
        b - a
        for a, b in zip(p10, p11)
    ]

    position_distance = math.sqrt(
        sum(value * value for value in position_vector)
    )

    q_at_zero = module.quaternion_slerp(
        q10,
        q11,
        0.0,
    )
    q_at_one = module.quaternion_slerp(
        q10,
        q11,
        1.0,
    )

    print("\n===== Interpolation verification =====")
    print(
        "SLERP t=0 versus Q10 max error:",
        "{:.12e}".format(
            quaternion_error(q_at_zero, q10)
        ),
    )
    print(
        "SLERP t=1 versus Q11 max error:",
        "{:.12e}".format(
            quaternion_error(q_at_one, q11)
        ),
    )
    print(
        "Q10 -> Q11 angle:",
        "{:.9f} deg".format(
            orientation_difference
        ),
    )
    print(
        "P10 -> P11 distance:",
        "{:.9f} mm".format(
            position_distance * 1000.0
        ),
    )

    scan(
        label=(
            "Orientation correction scan "
            "(P10 fixed)"
        ),
        make_target=lambda t: module.make_pose(
            p10,
            module.quaternion_slerp(
                q10,
                q11,
                t,
            ),
        ),
        display_amount=lambda t: (
            orientation_difference * t
        ),
        module=module,
        compute_ik=compute_ik,
        check_validity=check_validity,
        initial_seed=seed,
        j5_lower=j5_lower,
    )

    scan(
        label=(
            "Position correction scan "
            "(Q10 fixed)"
        ),
        make_target=lambda t: module.make_pose(
            module.interpolate_position(
                p10,
                p11,
                t,
            ),
            q10,
        ),
        display_amount=lambda t: (
            position_distance * t * 1000.0
        ),
        module=module,
        compute_ik=compute_ik,
        check_validity=check_validity,
        initial_seed=seed,
        j5_lower=j5_lower,
    )


if __name__ == "__main__":
    main()
