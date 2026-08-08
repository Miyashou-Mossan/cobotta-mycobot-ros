#!/usr/bin/env python3

import copy

import rospy
import tf2_ros
import moveit_commander

from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.msg import DisplayTrajectory
from moveit_msgs.srv import (
    GetPositionIK,
    GetPositionIKRequest,
    GetStateValidity,
    GetStateValidityRequest,
)


GROUP_NAME = "mycobot_arm"
TIP_LINK = "mycobot_tool_link"
FRAME_ID = "paper_center"

NUM_POINTS = 11

X_START = -0.075
X_END = 0.075

Y = 0.0
Z = 0.030

EEF_STEP = 0.005


def extract_plan(plan_result):

    if isinstance(plan_result, tuple):
        success = plan_result[0]
        trajectory = plan_result[1]
        return success, trajectory

    trajectory = plan_result

    success = (
        hasattr(trajectory, "joint_trajectory")
        and len(trajectory.joint_trajectory.points) > 0
    )

    return success, trajectory


def get_group_joint_dict(robot_state, group_joint_names):

    state_dict = dict(
        zip(
            robot_state.joint_state.name,
            robot_state.joint_state.position
        )
    )

    result = {}

    for name in group_joint_names:
        if name in state_dict:
            result[name] = state_dict[name]

    return result


def main():

    moveit_commander.roscpp_initialize([])
    rospy.init_node("mycobot_foldline_plan")

    robot = moveit_commander.RobotCommander()

    group = moveit_commander.MoveGroupCommander(
        GROUP_NAME
    )

    group.set_pose_reference_frame(
        FRAME_ID
    )

    group.set_end_effector_link(
        TIP_LINK
    )

    group.set_planning_time(5.0)
    group.set_num_planning_attempts(10)

    group_joint_names = group.get_active_joints()

    # --------------------------------------------------
    # MoveIt services
    # --------------------------------------------------

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

    # --------------------------------------------------
    # RViz publisher
    # --------------------------------------------------

    display_pub = rospy.Publisher(
        "/move_group/display_planned_path",
        DisplayTrajectory,
        queue_size=1,
        latch=True
    )

    # --------------------------------------------------
    # Current tool orientation in paper_center
    # --------------------------------------------------

    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(
        tf_buffer
    )

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

        rospy.logerr(
            "TF lookup failed: %s",
            str(e)
        )
        return

    qx = tf.transform.rotation.x
    qy = tf.transform.rotation.y
    qz = tf.transform.rotation.z
    qw = tf.transform.rotation.w

    # --------------------------------------------------
    # Generate fold-line poses
    # --------------------------------------------------

    foldline_poses = []

    for i in range(NUM_POINTS):

        ratio = float(i) / float(NUM_POINTS - 1)

        x = (
            X_START
            + ratio * (X_END - X_START)
        )

        pose = Pose()

        pose.position.x = x
        pose.position.y = Y
        pose.position.z = Z

        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw

        foldline_poses.append(
            copy.deepcopy(pose)
        )

    print("")
    print("===== MyCobot fold-line planning =====")
    print("group      :", GROUP_NAME)
    print("tip        :", TIP_LINK)
    print("frame      :", FRAME_ID)
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

    print(
        "EEF step   : %.1f mm"
        % (EEF_STEP * 1000.0)
    )

    # ==================================================
    # STEP 1
    # Solve IK explicitly for fold-line start
    # ==================================================

    print("")
    print("===== Step 1: solve start IK =====")

    current_state = robot.get_current_state()

    start_pose_stamped = PoseStamped()
    start_pose_stamped.header.frame_id = FRAME_ID
    start_pose_stamped.header.stamp = rospy.Time.now()
    start_pose_stamped.pose = copy.deepcopy(
        foldline_poses[0]
    )

    ik_req = GetPositionIKRequest()

    ik_req.ik_request.group_name = GROUP_NAME
    ik_req.ik_request.ik_link_name = TIP_LINK

    ik_req.ik_request.pose_stamped = (
        start_pose_stamped
    )

    ik_req.ik_request.robot_state = copy.deepcopy(
        current_state
    )

    ik_req.ik_request.timeout = rospy.Duration(2.0)

    try:
        ik_res = compute_ik(ik_req)

    except rospy.ServiceException as e:

        print("IK service error:", str(e))
        return

    if ik_res.error_code.val != 1:

        print(
            "start IK     : FAIL (%d)"
            % ik_res.error_code.val
        )
        return

    start_ik_state = copy.deepcopy(
        ik_res.solution
    )

    print("start IK     : SUCCESS")

    # --------------------------------------------------
    # Check validity of start IK
    # --------------------------------------------------

    validity_req = GetStateValidityRequest()

    validity_req.robot_state = copy.deepcopy(
        start_ik_state
    )

    validity_req.group_name = GROUP_NAME

    validity_res = check_validity(
        validity_req
    )

    if not validity_res.valid:

        print("start state  : COLLISION / INVALID")

        for contact in validity_res.contacts:

            print(
                "collision    : %s <-> %s"
                % (
                    contact.contact_body_1,
                    contact.contact_body_2
                )
            )

        return

    print("start state  : VALID")

    # --------------------------------------------------
    # Print selected start joints
    # --------------------------------------------------

    start_joint_dict = get_group_joint_dict(
        start_ik_state,
        group_joint_names
    )

    print("")
    print("selected start joints:")

    for name in group_joint_names:

        if name in start_joint_dict:

            print(
                "  %-20s %.6f rad"
                % (
                    name,
                    start_joint_dict[name]
                )
            )

    # ==================================================
    # STEP 2
    # Plan approach to the SELECTED joint state
    # ==================================================

    print("")
    print("===== Step 2: approach to selected start =====")

    group.set_start_state_to_current_state()

    group.set_joint_value_target(
        start_joint_dict
    )

    approach_result = group.plan()

    approach_success, approach_plan = extract_plan(
        approach_result
    )

    approach_points = len(
        approach_plan.joint_trajectory.points
    )

    print(
        "approach planning       : %s"
        % (
            "SUCCESS"
            if approach_success
            else "FAIL"
        )
    )

    print(
        "approach point count    : %d"
        % approach_points
    )

    if (
        not approach_success
        or approach_points == 0
    ):

        print("")
        print("===== Result =====")
        print("start IK     : PASS")
        print("approach     : FAIL")
        print("fold-line    : NOT TESTED")
        print("judgment     : FAIL")
        print("Execution    : DISABLED")
        return

    # ==================================================
    # STEP 3
    # Cartesian path from EXACT SAME IK state
    # ==================================================

    group.set_start_state(
        start_ik_state
    )

    print("")
    print("===== Step 3: fold-line Cartesian path =====")

    waypoints = []

    for i in range(1, NUM_POINTS):

        waypoints.append(
            copy.deepcopy(
                foldline_poses[i]
            )
        )

        print(
            "[%02d] x=%7.2f mm"
            % (
                i,
                foldline_poses[i].position.x
                * 1000.0
            )
        )

    try:

        cartesian_plan, fraction = (
            group.compute_cartesian_path(
                waypoints,
                EEF_STEP,
                True
            )
        )

    except Exception as e:

        print("")
        print("Cartesian planning ERROR:")
        print(str(e))
        return

    cartesian_points = len(
        cartesian_plan.joint_trajectory.points
    )

    print("")
    print(
        "Cartesian path fraction : %.3f"
        % fraction
    )

    print(
        "Cartesian point count    : %d"
        % cartesian_points
    )

    # ==================================================
    # Result
    # ==================================================

    print("")
    print("===== Result =====")
    print("start IK     : PASS")
    print("approach     : PASS")

    if fraction < 0.999:

        print(
            "fold-line    : PARTIAL (%.1f %%)"
            % (fraction * 100.0)
        )

        print("judgment     : PARTIAL / FAIL")
        print("Execution    : DISABLED")

        return

    print("fold-line    : COMPLETE")
    print("judgment     : PASS")
    print("Execution    : DISABLED")

    # ==================================================
    # RViz visualization
    # ==================================================

    print("")
    print("===== RViz visualization =====")

    display_msg = DisplayTrajectory()

    display_msg.trajectory_start = (
        copy.deepcopy(current_state)
    )

    display_msg.trajectory.append(
        approach_plan
    )

    display_msg.trajectory.append(
        cartesian_plan
    )

    rospy.sleep(1.0)

    display_pub.publish(
        display_msg
    )

    print(
        "Published to /move_group/display_planned_path"
    )

    print(
        "RVizで進入動作と150 mm折り筋追従を確認してください。"
    )

    print("")
    print("Ctrl+Cで終了できます。")

    rospy.spin()


if __name__ == "__main__":
    main()
