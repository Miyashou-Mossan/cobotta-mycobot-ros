#!/usr/bin/env python3

import copy
import math

import rospy
import moveit_commander

from geometry_msgs.msg import PoseStamped, Quaternion
from moveit_msgs.srv import (
    GetPositionFK,
    GetPositionFKRequest,
    GetPositionIK,
    GetPositionIKRequest,
    GetStateValidity,
    GetStateValidityRequest,
)

from tf.transformations import (
    quaternion_from_euler,
    quaternion_multiply,
)


GROUP_NAME = "mycobot_arm"
TIP_LINK = "mycobot_crease_tcp"
FRAME_ID = "paper_center"

NUM_POINTS = 41
CENTER_INDEX = 20

START = {
    "x": -0.075,
    "y": -0.050,
    "z": 0.030,
}

END = {
    "x": 0.075,
    "y": 0.050,
    "z": 0.030,
}

GOOD_SEED_JOINTS = {
    "mycobot_joint1":  3.081166,
    "mycobot_joint2":  1.203978,
    "mycobot_joint3":  0.807778,
    "mycobot_joint4":  1.129844,
    "mycobot_joint5": -0.060434,
    "mycobot_joint6":  3.141590,
}

JOINT_LIMITS = {
    "mycobot_joint1": (-3.14, 3.14159),
    "mycobot_joint2": (-3.14, 3.14159),
    "mycobot_joint3": (-3.14, 3.14159),
    "mycobot_joint4": (-3.14, 3.14159),
    "mycobot_joint5": (-3.14, 3.14159),
    "mycobot_joint6": (-3.14, 3.14159),
}

ANGLE_VALUES_DEG = [
    -10.0,
    -5.0,
    -3.0,
    0.0,
    3.0,
    5.0,
    10.0,
]

# LEFT用CENTER候補探索
J1_OFFSETS_DEG = [
    -150,
    -100,
    -50,
    0,
    50,
    100,
    150,
]

J2_OFFSETS_DEG = [
    -90,
    -45,
    0,
    45,
    90,
]

DUPLICATE_THRESHOLD_DEG = 5.0
JUMP_LIMIT_DEG = 30.0


def wrap_to_pi(angle):

    return math.atan2(
        math.sin(angle),
        math.cos(angle)
    )


def clamp(value, lower, upper):

    return max(
        lower,
        min(upper, value)
    )


def make_robot_state(robot, joint_dict):

    state = copy.deepcopy(
        robot.get_current_state()
    )

    names = list(
        state.joint_state.name
    )

    positions = list(
        state.joint_state.position
    )

    for i, name in enumerate(names):

        if name in joint_dict:
            positions[i] = joint_dict[name]

    state.joint_state.position = positions

    return state


def get_joint_dict(state, joint_names):

    all_values = dict(
        zip(
            state.joint_state.name,
            state.joint_state.position
        )
    )

    return {
        name: all_values[name]
        for name in joint_names
        if name in all_values
    }


def point_position(index):

    ratio = (
        float(index)
        / float(NUM_POINTS - 1)
    )

    x = (
        START["x"]
        + ratio
        * (
            END["x"]
            - START["x"]
        )
    )

    y = (
        START["y"]
        + ratio
        * (
            END["y"]
            - START["y"]
        )
    )

    z = (
        START["z"]
        + ratio
        * (
            END["z"]
            - START["z"]
        )
    )

    return x, y, z


def solve_ik(
    compute_ik,
    position,
    orientation,
    seed_state
):

    x, y, z = position

    target = PoseStamped()

    target.header.frame_id = FRAME_ID
    target.header.stamp = rospy.Time.now()

    target.pose.position.x = x
    target.pose.position.y = y
    target.pose.position.z = z

    target.pose.orientation = copy.deepcopy(
        orientation
    )

    req = GetPositionIKRequest()

    req.ik_request.group_name = GROUP_NAME
    req.ik_request.ik_link_name = TIP_LINK

    req.ik_request.pose_stamped = target

    req.ik_request.robot_state = copy.deepcopy(
        seed_state
    )

    req.ik_request.timeout = rospy.Duration(
        1.0
    )

    return compute_ik(req)


def check_state(
    check_validity,
    state
):

    req = GetStateValidityRequest()

    req.robot_state = copy.deepcopy(
        state
    )

    req.group_name = GROUP_NAME

    return check_validity(req)


def rotate_local(
    base_orientation,
    axis,
    angle_deg
):

    angle = math.radians(
        angle_deg
    )

    if axis == "x":

        q_delta = quaternion_from_euler(
            angle,
            0.0,
            0.0
        )

    elif axis == "y":

        q_delta = quaternion_from_euler(
            0.0,
            angle,
            0.0
        )

    elif axis == "z":

        q_delta = quaternion_from_euler(
            0.0,
            0.0,
            angle
        )

    else:

        raise ValueError(
            "Unknown axis: %s" % axis
        )

    q_base = [
        base_orientation.x,
        base_orientation.y,
        base_orientation.z,
        base_orientation.w,
    ]

    # local-axis rotation
    q_new = quaternion_multiply(
        q_base,
        q_delta
    )

    result = Quaternion()

    result.x = q_new[0]
    result.y = q_new[1]
    result.z = q_new[2]
    result.w = q_new[3]

    return result


def joint_distance_deg(
    joints_a,
    joints_b,
    joint_names
):

    max_delta = 0.0

    for name in joint_names:

        if (
            name not in joints_a
            or name not in joints_b
        ):
            continue

        delta = wrap_to_pi(
            joints_a[name]
            - joints_b[name]
        )

        max_delta = max(
            max_delta,
            abs(
                math.degrees(delta)
            )
        )

    return max_delta


def is_duplicate(
    joints,
    candidate_list,
    joint_names
):

    for candidate in candidate_list:

        distance = joint_distance_deg(
            joints,
            candidate["center_joints"],
            joint_names
        )

        if (
            distance
            < DUPLICATE_THRESHOLD_DEG
        ):
            return True

    return False


def min_margin_from_joints(
    joints,
    joint_names
):

    best_margin = float("inf")
    best_joint = None

    for name in joint_names:

        if (
            name not in joints
            or name not in JOINT_LIMITS
        ):
            continue

        lower, upper = JOINT_LIMITS[name]
        value = joints[name]

        margin = min(
            value - lower,
            upper - value
        )

        margin_deg = math.degrees(
            margin
        )

        if margin_deg < best_margin:

            best_margin = margin_deg
            best_joint = name

    return (
        best_margin,
        best_joint
    )


def evaluate_branch(
    indices,
    center_state,
    orientation,
    compute_ik,
    check_validity,
    joint_names
):

    center_validity = check_state(
        check_validity,
        center_state
    )

    if not center_validity.valid:

        return {
            "pass": False,
            "reason": "CENTER_INVALID",
        }

    previous_state = copy.deepcopy(
        center_state
    )

    previous_joints = get_joint_dict(
        center_state,
        joint_names
    )

    max_delta_deg = 0.0

    min_margin_deg = float("inf")
    limiting_joint = None

    center_margin, center_joint = (
        min_margin_from_joints(
            previous_joints,
            joint_names
        )
    )

    min_margin_deg = center_margin
    limiting_joint = center_joint

    for index in indices:

        ik_res = solve_ik(
            compute_ik,
            point_position(index),
            orientation,
            previous_state
        )

        if ik_res.error_code.val != 1:

            return {
                "pass": False,
                "reason":
                    "IK_FAIL_P%d" % index,
            }

        state = copy.deepcopy(
            ik_res.solution
        )

        validity = check_state(
            check_validity,
            state
        )

        if not validity.valid:

            return {
                "pass": False,
                "reason":
                    "COLLISION_P%d" % index,
            }

        joints = get_joint_dict(
            state,
            joint_names
        )

        for name in joint_names:

            delta = wrap_to_pi(
                joints[name]
                - previous_joints[name]
            )

            delta_deg = abs(
                math.degrees(delta)
            )

            max_delta_deg = max(
                max_delta_deg,
                delta_deg
            )

        margin_deg, margin_joint = (
            min_margin_from_joints(
                joints,
                joint_names
            )
        )

        if margin_deg < min_margin_deg:

            min_margin_deg = margin_deg
            limiting_joint = margin_joint

        previous_state = copy.deepcopy(
            state
        )

        previous_joints = copy.deepcopy(
            joints
        )

    smooth = (
        max_delta_deg < JUMP_LIMIT_DEG
    )

    return {
        "pass": True,
        "smooth": smooth,
        "reason": "PASS",
        "max_delta_deg": max_delta_deg,
        "min_margin_deg": min_margin_deg,
        "limiting_joint": limiting_joint,
    }


def find_left_center_candidate(
    robot,
    orientation,
    compute_ik,
    check_validity,
    joint_names
):

    center_position = point_position(
        CENTER_INDEX
    )

    candidates = []

    for j1_offset in J1_OFFSETS_DEG:

        for j2_offset in J2_OFFSETS_DEG:

            seed_joints = copy.deepcopy(
                GOOD_SEED_JOINTS
            )

            for name, offset in [
                (
                    "mycobot_joint1",
                    j1_offset
                ),
                (
                    "mycobot_joint2",
                    j2_offset
                ),
            ]:

                lower, upper = (
                    JOINT_LIMITS[name]
                )

                value = (
                    GOOD_SEED_JOINTS[name]
                    + math.radians(offset)
                )

                seed_joints[name] = clamp(
                    value,
                    lower + 0.01,
                    upper - 0.01
                )

            seed_state = make_robot_state(
                robot,
                seed_joints
            )

            ik_res = solve_ik(
                compute_ik,
                center_position,
                orientation,
                seed_state
            )

            if ik_res.error_code.val != 1:
                continue

            center_state = copy.deepcopy(
                ik_res.solution
            )

            validity = check_state(
                check_validity,
                center_state
            )

            if not validity.valid:
                continue

            center_joints = get_joint_dict(
                center_state,
                joint_names
            )

            if is_duplicate(
                center_joints,
                candidates,
                joint_names
            ):
                continue

            candidates.append(
                {
                    "center_state":
                        copy.deepcopy(
                            center_state
                        ),

                    "center_joints":
                        copy.deepcopy(
                            center_joints
                        ),
                }
            )

    if len(candidates) == 0:
        return None

    left_indices = list(
        range(
            CENTER_INDEX - 1,
            -1,
            -1
        )
    )

    best = None

    for candidate in candidates:

        result = evaluate_branch(
            left_indices,
            candidate["center_state"],
            orientation,
            compute_ik,
            check_validity,
            joint_names
        )

        if (
            not result.get("pass", False)
            or not result.get(
                "smooth",
                False
            )
        ):
            continue

        candidate["result"] = result

        if best is None:

            best = candidate

        else:

            # margin優先、次に最大delta
            current_key = (
                result[
                    "min_margin_deg"
                ],
                -result[
                    "max_delta_deg"
                ],
            )

            best_key = (
                best["result"][
                    "min_margin_deg"
                ],
                -best["result"][
                    "max_delta_deg"
                ],
            )

            if current_key > best_key:
                best = candidate

    if best is None:
        return None

    return copy.deepcopy(
        best["center_state"]
    )


def test_orientation(
    robot,
    orientation,
    compute_ik,
    check_validity,
    joint_names
):

    left_center_state = (
        find_left_center_candidate(
            robot,
            orientation,
            compute_ik,
            check_validity,
            joint_names
        )
    )

    if left_center_state is None:

        return {
            "pass": False,
            "reason": "LEFT_NO_CANDIDATE",
        }

    left_indices = list(
        range(
            CENTER_INDEX - 1,
            -1,
            -1
        )
    )

    left_result = evaluate_branch(
        left_indices,
        left_center_state,
        orientation,
        compute_ik,
        check_validity,
        joint_names
    )

    if (
        not left_result.get("pass", False)
        or not left_result.get(
            "smooth",
            False
        )
    ):

        return {
            "pass": False,
            "reason": "LEFT_FAIL",
        }

    # RIGHTは通常seedからCENTER IK
    good_seed_state = make_robot_state(
        robot,
        GOOD_SEED_JOINTS
    )

    center_ik = solve_ik(
        compute_ik,
        point_position(CENTER_INDEX),
        orientation,
        good_seed_state
    )

    if center_ik.error_code.val != 1:

        return {
            "pass": False,
            "reason": "RIGHT_CENTER_IK_FAIL",
        }

    right_center_state = copy.deepcopy(
        center_ik.solution
    )

    right_indices = list(
        range(
            CENTER_INDEX + 1,
            NUM_POINTS
        )
    )

    right_result = evaluate_branch(
        right_indices,
        right_center_state,
        orientation,
        compute_ik,
        check_validity,
        joint_names
    )

    if (
        not right_result.get("pass", False)
        or not right_result.get(
            "smooth",
            False
        )
    ):

        return {
            "pass": False,
            "reason": "RIGHT_FAIL",
        }

    overall_margin = min(
        left_result[
            "min_margin_deg"
        ],
        right_result[
            "min_margin_deg"
        ]
    )

    overall_max_delta = max(
        left_result[
            "max_delta_deg"
        ],
        right_result[
            "max_delta_deg"
        ]
    )

    return {
        "pass": True,

        "left":
            left_result,

        "right":
            right_result,

        "overall_margin_deg":
            overall_margin,

        "overall_max_delta_deg":
            overall_max_delta,
    }


def main():

    moveit_commander.roscpp_initialize([])

    rospy.init_node(
        "mycobot_crease_orientation_sweep"
    )

    robot = moveit_commander.RobotCommander()

    group = moveit_commander.MoveGroupCommander(
        GROUP_NAME
    )

    joint_names = group.get_active_joints()

    rospy.loginfo(
        "Waiting for MoveIt services ..."
    )

    rospy.wait_for_service(
        "/compute_fk"
    )

    rospy.wait_for_service(
        "/compute_ik"
    )

    rospy.wait_for_service(
        "/check_state_validity"
    )

    compute_fk = rospy.ServiceProxy(
        "/compute_fk",
        GetPositionFK
    )

    compute_ik = rospy.ServiceProxy(
        "/compute_ik",
        GetPositionIK
    )

    check_validity = rospy.ServiceProxy(
        "/check_state_validity",
        GetStateValidity
    )

    # ========================================================
    # Current reference orientation
    # ========================================================

    good_seed_state = make_robot_state(
        robot,
        GOOD_SEED_JOINTS
    )

    fk_req = GetPositionFKRequest()

    fk_req.header.frame_id = FRAME_ID
    fk_req.fk_link_names = [
        TIP_LINK
    ]

    fk_req.robot_state = copy.deepcopy(
        good_seed_state
    )

    fk_res = compute_fk(
        fk_req
    )

    if fk_res.error_code.val != 1:

        print(
            "Reference FK failed."
        )

        return

    base_orientation = copy.deepcopy(
        fk_res.pose_stamped[0]
        .pose.orientation
    )

    print("")
    print(
        "===== MyCobot crease orientation sweep ====="
    )

    print(
        "TCP        : %s"
        % TIP_LINK
    )

    print(
        "height     : 30 mm"
    )

    print(
        "fold line  : P0 <-> P20 <-> P40"
    )

    print("")

    print(
        "axis angle | result | "
        "LEFT margin | RIGHT margin | "
        "overall margin | max delta"
    )

    print(
        "------------------------------------------------------------------"
    )

    successful = []

    for axis in [
        "x",
        "y",
        "z"
    ]:

        for angle_deg in (
            ANGLE_VALUES_DEG
        ):

            orientation = rotate_local(
                base_orientation,
                axis,
                angle_deg
            )

            result = test_orientation(
                robot,
                orientation,
                compute_ik,
                check_validity,
                joint_names
            )

            if not result.get(
                "pass",
                False
            ):

                print(
                    "%s %+5.1f | FAIL   | "
                    "%11s | %12s | "
                    "%14s | %s"
                    % (
                        axis.upper(),
                        angle_deg,
                        "-",
                        "-",
                        "-",
                        result[
                            "reason"
                        ]
                    )
                )

                continue

            left = result["left"]
            right = result["right"]

            print(
                "%s %+5.1f | PASS   | "
                "%8.3f %-2s | "
                "%8.3f %-2s | "
                "%12.3f | %8.3f"
                % (
                    axis.upper(),
                    angle_deg,

                    left[
                        "min_margin_deg"
                    ],
                    left[
                        "limiting_joint"
                    ].replace(
                        "mycobot_joint",
                        "J"
                    ),

                    right[
                        "min_margin_deg"
                    ],
                    right[
                        "limiting_joint"
                    ].replace(
                        "mycobot_joint",
                        "J"
                    ),

                    result[
                        "overall_margin_deg"
                    ],

                    result[
                        "overall_max_delta_deg"
                    ]
                )
            )

            successful.append(
                {
                    "axis": axis,
                    "angle_deg":
                        angle_deg,
                    "result":
                        result,
                }
            )

    print("")
    print(
        "===== Best orientation ====="
    )

    if len(successful) == 0:

        print(
            "No orientation passed both branches."
        )
        return

    best = max(
        successful,
        key=lambda item:
            (
                item["result"][
                    "overall_margin_deg"
                ],
                -item["result"][
                    "overall_max_delta_deg"
                ],
            )
    )

    print(
        "axis        : local_%s"
        % best["axis"]
    )

    print(
        "angle       : %+5.1f deg"
        % best["angle_deg"]
    )

    print(
        "LEFT margin : %.6f deg"
        % best["result"][
            "left"
        ][
            "min_margin_deg"
        ]
    )

    print(
        "RIGHT margin: %.6f deg"
        % best["result"][
            "right"
        ][
            "min_margin_deg"
        ]
    )

    print(
        "overall     : %.6f deg"
        % best["result"][
            "overall_margin_deg"
        ]
    )

    print(
        "max delta   : %.6f deg"
        % best["result"][
            "overall_max_delta_deg"
        ]
    )


if __name__ == "__main__":
    main()
