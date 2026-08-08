#!/usr/bin/env python3

import copy
import yaml

import rospy

from trajectory_msgs.msg import JointTrajectoryPoint
from moveit_msgs.msg import (
    DisplayTrajectory,
    RobotTrajectory,
    RobotState,
)


INPUT_FILE = "/home/maeda/mycobot_foldline_trajectory.yaml"

DISPLAY_TOPIC = "/move_group/display_planned_path"


def dict_to_duration(time_dict):
    secs = int(time_dict.get("secs", 0))
    nsecs = int(time_dict.get("nsecs", 0))

    return rospy.Duration(
        secs=secs,
        nsecs=nsecs
    )


def main():
    rospy.init_node(
        "replay_saved_robot_trajectory"
    )

    print("")
    print(
        "===== Saved RobotTrajectory replay ====="
    )
    print("file :", INPUT_FILE)

    # ========================================
    # Load YAML
    # ========================================

    try:
        with open(INPUT_FILE, "r") as f:
            data = yaml.safe_load(f)

    except Exception as e:
        print("YAML load : FAIL")
        print(str(e))
        return

    print("YAML load : SUCCESS")

    # ========================================
    # Required sections
    # ========================================

    if "joint_trajectory" not in data:
        print("joint_trajectory : MISSING")
        return

    if "trajectory_start" not in data:
        print("trajectory_start : MISSING")
        return

    jt_data = data["joint_trajectory"]
    start_data = data["trajectory_start"]

    joint_names = jt_data.get(
        "joint_names",
        []
    )

    points_data = jt_data.get(
        "points",
        []
    )

    if len(joint_names) == 0:
        print("joint_names : EMPTY")
        return

    if len(points_data) == 0:
        print("trajectory points : EMPTY")
        return

    # ========================================
    # Rebuild RobotTrajectory
    # ========================================

    robot_trajectory = RobotTrajectory()

    robot_trajectory.joint_trajectory.header.frame_id = (
        jt_data.get(
            "frame_id",
            ""
        )
    )

    robot_trajectory.joint_trajectory.joint_names = (
        list(joint_names)
    )

    for point_data in points_data:

        point = JointTrajectoryPoint()

        point.positions = [
            float(v)
            for v in point_data.get(
                "positions",
                []
            )
        ]

        point.velocities = [
            float(v)
            for v in point_data.get(
                "velocities",
                []
            )
        ]

        point.accelerations = [
            float(v)
            for v in point_data.get(
                "accelerations",
                []
            )
        ]

        point.effort = [
            float(v)
            for v in point_data.get(
                "effort",
                []
            )
        ]

        point.time_from_start = (
            dict_to_duration(
                point_data.get(
                    "time_from_start",
                    {}
                )
            )
        )

        robot_trajectory.joint_trajectory.points.append(
            point
        )

    # ========================================
    # Rebuild trajectory_start
    # ========================================

    start_state = RobotState()

    start_state.joint_state.header.frame_id = "world"

    start_state.joint_state.name = list(
        start_data.get(
            "joint_names",
            []
        )
    )

    start_state.joint_state.position = [
        float(v)
        for v in start_data.get(
            "positions",
            []
        )
    ]

    # ========================================
    # Print loaded information
    # ========================================

    point_count = len(
        robot_trajectory
        .joint_trajectory
        .points
    )

    duration = (
        robot_trajectory
        .joint_trajectory
        .points[-1]
        .time_from_start
        .to_sec()
    )

    print("")
    print("===== Restored trajectory =====")
    print(
        "joint count  : %d"
        % len(joint_names)
    )
    print(
        "point count  : %d"
        % point_count
    )
    print(
        "duration     : %.6f s"
        % duration
    )
    print(
        "start joints : %d"
        % len(
            start_state.joint_state.name
        )
    )

    # ========================================
    # Create DisplayTrajectory
    # ========================================

    display_msg = DisplayTrajectory()

    display_msg.trajectory_start = (
        copy.deepcopy(
            start_state
        )
    )

    display_msg.trajectory.append(
        robot_trajectory
    )

    # ========================================
    # Publisher
    # ========================================

    pub = rospy.Publisher(
        DISPLAY_TOPIC,
        DisplayTrajectory,
        queue_size=1,
        latch=True
    )

    rospy.sleep(1.0)

    pub.publish(
        display_msg
    )

    print("")
    print("===== Publish result =====")
    print(
        "topic    : %s"
        % DISPLAY_TOPIC
    )
    print(
        "published: SUCCESS"
    )

    print("")
    print(
        "RVizで保存済み軌道を確認してください。"
    )
    print(
        "Ctrl+Cで終了できます。"
    )

    rospy.spin()


if __name__ == "__main__":
    main()
