#!/usr/bin/env python3

import copy

import rospy
import moveit_commander

from geometry_msgs.msg import Pose
from moveit_msgs.msg import DisplayTrajectory, RobotState
from moveit_msgs.srv import (
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

GOOD_START_JOINTS = {
    "mycobot_joint1":  3.081166,
    "mycobot_joint2":  1.203978,
    "mycobot_joint3":  0.807778,
    "mycobot_joint4":  1.129844,
    "mycobot_joint5": -0.060434,
    "mycobot_joint6":  3.141590,
}


def make_robot_state_from_joint_dict(robot, joint_dict):
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


def main():
    moveit_commander.roscpp_initialize([])
    rospy.init_node(
        "mycobot_foldline_fixed_branch"
    )

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

    rospy.loginfo(
        "Waiting for /check_state_validity ..."
    )

    rospy.wait_for_service(
        "/check_state_validity"
    )

    check_validity = rospy.ServiceProxy(
        "/check_state_validity",
        GetStateValidity
    )

    display_pub = rospy.Publisher(
        "/move_group/display_planned_path",
        DisplayTrajectory,
        queue_size=1,
        latch=True
    )

    # ========================================
    # Fixed start state
    # ========================================

    start_state = (
        make_robot_state_from_joint_dict(
            robot,
            GOOD_START_JOINTS
        )
    )

    print("")
    print(
        "===== MyCobot fixed-branch fold-line test ====="
    )

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

    print("")
    print(
        "===== Fixed start joints ====="
    )

    for name in [
        "mycobot_joint1",
        "mycobot_joint2",
        "mycobot_joint3",
        "mycobot_joint4",
        "mycobot_joint5",
        "mycobot_joint6",
    ]:
        print(
            "%-18s %.6f rad"
            % (
                name,
                GOOD_START_JOINTS[name]
            )
        )

    # ========================================
    # Check fixed start validity
    # ========================================

    valid_req = GetStateValidityRequest()

    valid_req.robot_state = copy.deepcopy(
        start_state
    )

    valid_req.group_name = GROUP_NAME

    valid_res = check_validity(
        valid_req
    )

    print("")
    print(
        "===== Start-state validity ====="
    )

    if valid_res.valid:
        print("start state : VALID")
    else:
        print(
            "start state : COLLISION / INVALID"
        )

        for contact in valid_res.contacts:
            print(
                "collision   : %s <-> %s"
                % (
                    contact.contact_body_1,
                    contact.contact_body_2
                )
            )

        return

    # ========================================
    # Get start TCP pose from fixed joint state
    # ========================================

    group.set_start_state(
        start_state
    )

    # We know the target orientation used when
    # this branch was originally found.
    # Use FK-equivalent current planning pose by
    # first setting the fixed state and then
    # building waypoints from the known fold line.
    #
    # Orientation is obtained by solving the
    # first point from the stored start state
    # through forward kinematics in MoveIt.
    #
    # get_current_pose() does not honor arbitrary
    # start_state, so use /compute_fk indirectly
    # through RobotState not available here.
    #
    # Therefore use the previously confirmed
    # tool orientation from this branch:
    #
    # Quaternion corresponding to the fixed
    # paper-center tool orientation is filled
    # below after FK lookup from the robot model.
    #
    # To avoid assuming it manually, we use
    # MoveIt FK service.
    # ========================================

    from moveit_msgs.srv import (
        GetPositionFK,
        GetPositionFKRequest
    )

    rospy.loginfo(
        "Waiting for /compute_fk ..."
    )

    rospy.wait_for_service(
        "/compute_fk"
    )

    compute_fk = rospy.ServiceProxy(
        "/compute_fk",
        GetPositionFK
    )

    fk_req = GetPositionFKRequest()

    fk_req.header.frame_id = FRAME_ID
    fk_req.fk_link_names = [
        TIP_LINK
    ]

    fk_req.robot_state = copy.deepcopy(
        start_state
    )

    fk_res = compute_fk(
        fk_req
    )

    if fk_res.error_code.val != 1:
        print(
            "FK failed with error code:",
            fk_res.error_code.val
        )
        return

    if len(fk_res.pose_stamped) == 0:
        print("FK returned no pose.")
        return

    start_pose = (
        fk_res.pose_stamped[0].pose
    )

    print("")
    print(
        "===== Fixed start TCP pose ====="
    )

    print(
        "position [mm] : %.3f %.3f %.3f"
        % (
            start_pose.position.x * 1000.0,
            start_pose.position.y * 1000.0,
            start_pose.position.z * 1000.0
        )
    )

    print(
        "orientation   : %.6f %.6f %.6f %.6f"
        % (
            start_pose.orientation.x,
            start_pose.orientation.y,
            start_pose.orientation.z,
            start_pose.orientation.w
        )
    )

    # ========================================
    # Check that saved branch corresponds
    # approximately to expected fold-line start
    # ========================================

    dx_mm = abs(
        start_pose.position.x
        - X_START
    ) * 1000.0

    dy_mm = abs(
        start_pose.position.y
        - Y
    ) * 1000.0

    dz_mm = abs(
        start_pose.position.z
        - Z
    ) * 1000.0

    print("")
    print(
        "start position error:"
    )

    print(
        "  dx = %.3f mm"
        % dx_mm
    )

    print(
        "  dy = %.3f mm"
        % dy_mm
    )

    print(
        "  dz = %.3f mm"
        % dz_mm
    )

    # ========================================
    # Generate remaining fold-line waypoints
    # ========================================

    waypoints = []

    for i in range(1, NUM_POINTS):

        ratio = float(i) / float(
            NUM_POINTS - 1
        )

        x = (
            X_START
            + ratio * (
                X_END - X_START
            )
        )

        pose = Pose()

        pose.position.x = x
        pose.position.y = Y
        pose.position.z = Z

        pose.orientation = copy.deepcopy(
            start_pose.orientation
        )

        waypoints.append(
            copy.deepcopy(pose)
        )

    # ========================================
    # Cartesian path from fixed branch
    # ========================================

    group.set_start_state(
        start_state
    )

    print("")
    print(
        "===== Fold-line Cartesian path ====="
    )

    for i in range(
        1,
        NUM_POINTS
    ):
        print(
            "[%02d] x=%7.2f mm"
            % (
                i,
                (
                    X_START
                    + float(i)
                    / float(
                        NUM_POINTS - 1
                    )
                    * (
                        X_END
                        - X_START
                    )
                ) * 1000.0
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
        print(
            "Cartesian planning ERROR:"
        )
        print(str(e))
        return

    point_count = len(
        cartesian_plan
        .joint_trajectory
        .points
    )

    print("")
    print(
        "Cartesian path fraction : %.3f"
        % fraction
    )

    print(
        "trajectory point count  : %d"
        % point_count
    )

    # ========================================
    # Result
    # ========================================

    print("")
    print("===== Result =====")

    if fraction >= 0.999:
        print(
            "fold-line    : COMPLETE"
        )
        print(
            "judgment     : PASS"
        )
    else:
        print(
            "fold-line    : PARTIAL (%.1f %%)"
            % (
                fraction * 100.0
            )
        )
        print(
            "judgment     : PARTIAL / FAIL"
        )
        return

    print(
        "Execution    : DISABLED"
    )

    # ========================================
    # RViz
    # Only show fold-line trajectory
    # ========================================

    display_msg = DisplayTrajectory()

    display_msg.trajectory_start = (
        copy.deepcopy(
            start_state
        )
    )

    display_msg.trajectory.append(
        cartesian_plan
    )

    rospy.sleep(1.0)

    display_pub.publish(
        display_msg
    )

    print("")
    print(
        "===== RViz visualization ====="
    )

    print(
        "Only the 150 mm fold-line path "
        "was published."
    )

    print(
        "Published to "
        "/move_group/display_planned_path"
    )

    print("")
    print(
        "Ctrl+C to exit."
    )

    rospy.spin()


if __name__ == "__main__":
    main()
