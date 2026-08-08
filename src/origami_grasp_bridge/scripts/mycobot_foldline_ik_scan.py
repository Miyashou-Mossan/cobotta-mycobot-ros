#!/usr/bin/env python3

import copy
import math

import rospy
import tf2_ros

from moveit_msgs.srv import (
    GetPositionIK,
    GetPositionIKRequest,
    GetStateValidity,
    GetStateValidityRequest,
)
from geometry_msgs.msg import PoseStamped


GROUP_NAME = "mycobot_arm"
TIP_LINK = "mycobot_tool_link"
FRAME_ID = "paper_center"

NUM_POINTS = 11

X_START = -0.075
X_END = 0.075

Y = 0.0
Z = 0.030


def main():
    rospy.init_node("mycobot_foldline_ik_scan")

    rospy.loginfo("Waiting for /compute_ik ...")
    rospy.wait_for_service("/compute_ik")
    compute_ik = rospy.ServiceProxy(
        "/compute_ik",
        GetPositionIK
    )

    rospy.loginfo("Waiting for /check_state_validity ...")
    rospy.wait_for_service("/check_state_validity")
    check_validity = rospy.ServiceProxy(
        "/check_state_validity",
        GetStateValidity
    )

    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    rospy.loginfo(
        "Waiting for TF: %s -> %s",
        FRAME_ID,
        TIP_LINK
    )

    try:
        tf = tf_buffer.lookup_transform(
            FRAME_ID,
            TIP_LINK,
            rospy.Time(0),
            rospy.Duration(5.0)
        )
    except Exception as e:
        rospy.logerr("TF lookup failed: %s", str(e))
        return

    qx = tf.transform.rotation.x
    qy = tf.transform.rotation.y
    qz = tf.transform.rotation.z
    qw = tf.transform.rotation.w

    print("")
    print("===== MyCobot fold-line IK scan =====")
    print("frame      :", FRAME_ID)
    print("group      :", GROUP_NAME)
    print("tip        :", TIP_LINK)
    print("points     :", NUM_POINTS)
    print(
        "X          : %.1f -> %.1f mm"
        % (
            X_START * 1000.0,
            X_END * 1000.0
        )
    )
    print(
        "Y          : %.1f mm"
        % (Y * 1000.0)
    )
    print(
        "Z          : %.1f mm"
        % (Z * 1000.0)
    )
    print("")

    success_count = 0
    valid_count = 0

    previous_robot_state = None
    previous_joint_values = None

    max_adjacent_delta_deg = 0.0

    for i in range(NUM_POINTS):

        ratio = float(i) / float(NUM_POINTS - 1)

        x = (
            X_START
            + ratio * (X_END - X_START)
        )

        pose = PoseStamped()
        pose.header.frame_id = FRAME_ID
        pose.header.stamp = rospy.Time.now()

        pose.pose.position.x = x
        pose.pose.position.y = Y
        pose.pose.position.z = Z

        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        req = GetPositionIKRequest()
        req.ik_request.group_name = GROUP_NAME
        req.ik_request.ik_link_name = TIP_LINK
        req.ik_request.pose_stamped = pose
        req.ik_request.timeout = rospy.Duration(1.0)

        if previous_robot_state is not None:
            req.ik_request.robot_state = copy.deepcopy(
                previous_robot_state
            )

        try:
            res = compute_ik(req)
        except rospy.ServiceException:
            print(
                "[%02d] x=%7.2f mm : SERVICE ERROR"
                % (
                    i,
                    x * 1000.0
                )
            )
            continue

        error_code = res.error_code.val

        if error_code != 1:
            print(
                "[%02d] x=%7.2f mm : FAIL (%d)"
                % (
                    i,
                    x * 1000.0,
                    error_code
                )
            )
            continue

        success_count += 1

        solution = copy.deepcopy(
            res.solution
        )

        valid_req = GetStateValidityRequest()
        valid_req.robot_state = solution
        valid_req.group_name = GROUP_NAME

        try:
            valid_res = check_validity(
                valid_req
            )
        except rospy.ServiceException:
            print(
                "[%02d] x=%7.2f mm : "
                "VALIDITY SERVICE ERROR"
                % (
                    i,
                    x * 1000.0
                )
            )
            continue

        if valid_res.valid:
            valid_count += 1
            validity_text = "VALID"
        else:
            validity_text = "COLLISION/INVALID"

            collision_pairs = set()

            for contact in valid_res.contacts:
                collision_pairs.add(
                    (
                        contact.contact_body_1,
                        contact.contact_body_2
                    )
                )

            for body1, body2 in sorted(
                collision_pairs
            ):
                print(
                    "      collision: %s <-> %s"
                    % (
                        body1,
                        body2
                    )
                )

        joint_names = solution.joint_state.name
        joint_positions = solution.joint_state.position

        current_joint_values = []

        for name, value in zip(
            joint_names,
            joint_positions
        ):
            if name.startswith("mycobot_joint"):
                current_joint_values.append(
                    (name, value)
                )

        point_max_delta_deg = 0.0

        if previous_joint_values is not None:

            previous_dict = dict(
                previous_joint_values
            )

            current_dict = dict(
                current_joint_values
            )

            common_names = sorted(
                set(previous_dict.keys())
                & set(current_dict.keys())
            )

            for name in common_names:

                delta_rad = abs(
                    current_dict[name]
                    - previous_dict[name]
                )

                delta_deg = math.degrees(
                    delta_rad
                )

                point_max_delta_deg = max(
                    point_max_delta_deg,
                    delta_deg
                )

            max_adjacent_delta_deg = max(
                max_adjacent_delta_deg,
                point_max_delta_deg
            )

        print(
            "[%02d] x=%7.2f mm : "
            "SUCCESS  %-17s  "
            "adjacent_delta=%7.3f deg"
            % (
                i,
                x * 1000.0,
                validity_text,
                point_max_delta_deg
            )
        )

        previous_robot_state = copy.deepcopy(
            solution
        )

        previous_joint_values = (
            current_joint_values
        )

    print("")
    print("===== Result =====")

    print(
        "IK success : %d / %d"
        % (
            success_count,
            NUM_POINTS
        )
    )

    print(
        "valid      : %d / %d"
        % (
            valid_count,
            NUM_POINTS
        )
    )

    print(
        "max adjacent joint delta : %.3f deg"
        % max_adjacent_delta_deg
    )

    if (
        success_count == NUM_POINTS
        and valid_count == NUM_POINTS
    ):
        print("judgment   : PASS")
    else:
        print("judgment   : PARTIAL / FAIL")


if __name__ == "__main__":
    main()
