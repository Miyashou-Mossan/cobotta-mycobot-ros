#!/usr/bin/env python3

import copy
import math

import rospy
import moveit_commander

from geometry_msgs.msg import Pose
from moveit_msgs.srv import (
    GetPositionFK,
    GetPositionFKRequest,
    GetStateValidity,
    GetStateValidityRequest,
)

from trajectory_yaml_io import save_robot_trajectory


# ============================================================
# MyCobot-specific settings
# ============================================================

GROUP_NAME = "mycobot_arm"
TIP_LINK = "mycobot_tool_link"
FRAME_ID = "paper_center"

NUM_POINTS = 11

X_START = -0.075
X_END = 0.075

Y = 0.0
Z = 0.030

EEF_STEP = 0.005

OUTPUT_FILE = "/home/maeda/mycobot_foldline_trajectory.yaml"


# Day1で150 mm追従に成功したIK分岐
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


def validate_generated_trajectory(robot_trajectory):

    trajectory = robot_trajectory.joint_trajectory

    if len(trajectory.points) == 0:
        return None

    nan_inf_found = False
    monotonic_time = True

    previous_time = None
    max_adjacent_delta_rad = 0.0

    for i, point in enumerate(
        trajectory.points
    ):

        values = []

        values.extend(
            list(point.positions)
        )

        values.extend(
            list(point.velocities)
        )

        values.extend(
            list(point.accelerations)
        )

        for value in values:

            if not math.isfinite(value):
                nan_inf_found = True

        current_time = (
            point.time_from_start.to_sec()
        )

        if (
            previous_time is not None
            and current_time <= previous_time
        ):
            monotonic_time = False

        previous_time = current_time

        if i > 0:

            previous_positions = (
                trajectory.points[
                    i - 1
                ].positions
            )

            for current, previous in zip(
                point.positions,
                previous_positions
            ):

                delta = abs(
                    current - previous
                )

                max_adjacent_delta_rad = max(
                    max_adjacent_delta_rad,
                    delta
                )

    total_duration = (
        trajectory.points[-1]
        .time_from_start
        .to_sec()
    )

    max_adjacent_delta_deg = (
        math.degrees(
            max_adjacent_delta_rad
        )
    )

    return {
        "nan_inf_found": nan_inf_found,
        "monotonic_time": monotonic_time,
        "total_duration_sec": total_duration,
        "max_adjacent_joint_delta_deg":
            max_adjacent_delta_deg,
    }


def main():

    moveit_commander.roscpp_initialize([])
    rospy.init_node(
        "mycobot_save_foldline_trajectory"
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

    # ========================================================
    # MoveIt services
    # ========================================================

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

    # ========================================================
    # Fixed MyCobot start state
    # ========================================================

    start_state = (
        make_robot_state_from_joint_dict(
            robot,
            GOOD_START_JOINTS
        )
    )

    print("")
    print(
        "===== MyCobot fold-line trajectory generation ====="
    )

    print("group       :", GROUP_NAME)
    print("tip         :", TIP_LINK)
    print("frame       :", FRAME_ID)
    print("output file :", OUTPUT_FILE)

    # ========================================================
    # Start-state validity
    # ========================================================

    validity_req = GetStateValidityRequest()

    validity_req.robot_state = (
        copy.deepcopy(start_state)
    )

    validity_req.group_name = (
        GROUP_NAME
    )

    validity_res = check_validity(
        validity_req
    )

    if not validity_res.valid:

        print("")
        print(
            "start state : COLLISION / INVALID"
        )

        for contact in validity_res.contacts:

            print(
                "collision   : %s <-> %s"
                % (
                    contact.contact_body_1,
                    contact.contact_body_2
                )
            )

        return

    print("")
    print("start state : VALID")

    # ========================================================
    # FK at the fixed start state
    # ========================================================

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
            "FK failed:",
            fk_res.error_code.val
        )
        return

    if len(fk_res.pose_stamped) == 0:

        print(
            "FK returned no TCP pose."
        )
        return

    start_pose = (
        fk_res.pose_stamped[0].pose
    )

    print("")
    print(
        "===== Fixed start TCP ====="
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

    # ========================================================
    # MyCobot fold-line waypoint generation
    # ========================================================

    waypoints = []

    print("")
    print(
        "===== Fold-line waypoints ====="
    )

    for i in range(
        1,
        NUM_POINTS
    ):

        ratio = (
            float(i)
            / float(NUM_POINTS - 1)
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

        print(
            "[%02d] x=%7.2f mm"
            % (
                i,
                x * 1000.0
            )
        )

    # ========================================================
    # Cartesian trajectory generation
    # ========================================================

    group.set_start_state(
        start_state
    )

    try:

        robot_trajectory, fraction = (
            group.compute_cartesian_path(
                waypoints,
                EEF_STEP,
                True
            )
        )

    except Exception as e:

        print("")
        print(
            "Cartesian planning ERROR:"
        )
        print(str(e))
        return

    point_count = len(
        robot_trajectory
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

    if fraction < 0.999:

        print("")
        print("judgment : FAIL")

        print(
            "Trajectory will NOT be saved."
        )
        return

    # ========================================================
    # Basic generated-trajectory validation
    # ========================================================

    validation = (
        validate_generated_trajectory(
            robot_trajectory
        )
    )

    if validation is None:

        print(
            "No trajectory points."
        )
        return

    print("")
    print(
        "===== Preliminary validation ====="
    )

    print(
        "NaN / Inf                 : %s"
        % (
            "FOUND"
            if validation[
                "nan_inf_found"
            ]
            else "NONE"
        )
    )

    print(
        "time monotonic            : %s"
        % (
            "PASS"
            if validation[
                "monotonic_time"
            ]
            else "FAIL"
        )
    )

    print(
        "total duration            : %.6f s"
        % validation[
            "total_duration_sec"
        ]
    )

    print(
        "max adjacent joint delta  : %.6f deg"
        % validation[
            "max_adjacent_joint_delta_deg"
        ]
    )

    if validation["nan_inf_found"]:

        print(
            "Trajectory will NOT be saved."
        )
        return

    if not validation[
        "monotonic_time"
    ]:

        print(
            "Trajectory will NOT be saved."
        )
        return

    # ========================================================
    # Metadata
    # ========================================================

    metadata = {

        "robot_group":
            GROUP_NAME,

        "tip_link":
            TIP_LINK,

        "reference_frame":
            FRAME_ID,

        "trajectory_type":
            "cartesian_fold_line",

        "source":
            "mycobot_save_foldline_trajectory.py",

        "cartesian_fraction":
            float(fraction),

        "eef_step_m":
            float(EEF_STEP),

        "fold_line": {

            "x_start_m":
                float(X_START),

            "x_end_m":
                float(X_END),

            "y_m":
                float(Y),

            "z_m":
                float(Z),

            "input_waypoint_count":
                int(NUM_POINTS),
        },

        "trajectory_point_count":
            int(point_count),

        "total_duration_sec":
            float(
                validation[
                    "total_duration_sec"
                ]
            ),

        "max_adjacent_joint_delta_deg":
            float(
                validation[
                    "max_adjacent_joint_delta_deg"
                ]
            ),
    }

    # ========================================================
    # Common YAML save
    # ========================================================

    try:

        save_robot_trajectory(
            OUTPUT_FILE,
            robot_trajectory,
            start_state,
            metadata
        )

    except Exception as e:

        print("")
        print(
            "Common trajectory save ERROR:"
        )
        print(str(e))
        return

    # ========================================================
    # Result
    # ========================================================

    print("")
    print(
        "===== Save result ====="
    )

    print(
        "save module  : trajectory_yaml_io.py"
    )

    print(
        "saved        : SUCCESS"
    )

    print(
        "file         : %s"
        % OUTPUT_FILE
    )

    print(
        "points       : %d"
        % point_count
    )

    print(
        "duration     : %.6f s"
        % validation[
            "total_duration_sec"
        ]
    )

    print("")
    print(
        "judgment     : PASS"
    )


if __name__ == "__main__":
    main()
