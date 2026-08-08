#!/usr/bin/env python3

import copy
import yaml

from moveit_msgs.msg import RobotTrajectory, RobotState
from trajectory_msgs.msg import JointTrajectoryPoint


def duration_to_dict(duration):
    return {
        "secs": int(duration.secs),
        "nsecs": int(duration.nsecs),
    }


def dict_to_duration(rospy, time_dict):
    return rospy.Duration(
        secs=int(time_dict.get("secs", 0)),
        nsecs=int(time_dict.get("nsecs", 0)),
    )


def save_robot_trajectory(
    filename,
    robot_trajectory,
    trajectory_start,
    metadata=None
):
    """
    RobotTrajectory + RobotState を共通YAML形式で保存する。
    MyCobot / COBOTTAのどちらにも依存しない。
    """

    if metadata is None:
        metadata = {}

    trajectory = robot_trajectory.joint_trajectory

    yaml_data = {
        "metadata": copy.deepcopy(metadata),

        "trajectory_start": {
            "joint_names": list(
                trajectory_start.joint_state.name
            ),
            "positions": [
                float(v)
                for v in trajectory_start.joint_state.position
            ],
        },

        "joint_trajectory": {
            "frame_id": trajectory.header.frame_id,
            "joint_names": list(
                trajectory.joint_names
            ),
            "points": [],
        },
    }

    for point in trajectory.points:

        yaml_data[
            "joint_trajectory"
        ][
            "points"
        ].append(
            {
                "positions": [
                    float(v)
                    for v in point.positions
                ],
                "velocities": [
                    float(v)
                    for v in point.velocities
                ],
                "accelerations": [
                    float(v)
                    for v in point.accelerations
                ],
                "effort": [
                    float(v)
                    for v in point.effort
                ],
                "time_from_start":
                    duration_to_dict(
                        point.time_from_start
                    ),
            }
        )

    with open(filename, "w") as f:
        yaml.safe_dump(
            yaml_data,
            f,
            default_flow_style=False,
            sort_keys=False
        )

    return yaml_data


def load_robot_trajectory(filename, rospy):
    """
    共通YAMLから
    RobotTrajectory + RobotState + metadata
    を復元する。
    """

    with open(filename, "r") as f:
        data = yaml.safe_load(f)

    if "joint_trajectory" not in data:
        raise ValueError(
            "joint_trajectory is missing"
        )

    if "trajectory_start" not in data:
        raise ValueError(
            "trajectory_start is missing"
        )

    jt_data = data["joint_trajectory"]
    start_data = data["trajectory_start"]

    # --------------------------------
    # RobotTrajectory
    # --------------------------------

    robot_trajectory = RobotTrajectory()

    robot_trajectory.joint_trajectory.header.frame_id = (
        jt_data.get("frame_id", "")
    )

    robot_trajectory.joint_trajectory.joint_names = list(
        jt_data.get("joint_names", [])
    )

    for point_data in jt_data.get(
        "points",
        []
    ):

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
                rospy,
                point_data.get(
                    "time_from_start",
                    {}
                )
            )
        )

        robot_trajectory.joint_trajectory.points.append(
            point
        )

    # --------------------------------
    # RobotState
    # --------------------------------

    trajectory_start = RobotState()

    trajectory_start.joint_state.name = list(
        start_data.get(
            "joint_names",
            []
        )
    )

    trajectory_start.joint_state.position = [
        float(v)
        for v in start_data.get(
            "positions",
            []
        )
    ]

    metadata = data.get(
        "metadata",
        {}
    )

    return (
        robot_trajectory,
        trajectory_start,
        metadata
    )
