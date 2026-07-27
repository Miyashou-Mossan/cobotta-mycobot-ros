#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import os
import sys

import yaml


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "保存されたRobotTrajectoryの関節数、配列長、"
            "時刻、速度、加速度、NaN/Infを検証します。"
        )
    )

    parser.add_argument(
        "trajectory_file",
        nargs="?",
        default="~/combined_robot_trajectory_0_19.yaml",
    )

    return parser.parse_args()


def load_trajectory(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"軌道ファイルがありません: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "RobotTrajectoryのYAML形式ではありません。"
        )

    joint_trajectory = data.get("joint_trajectory")

    if not isinstance(joint_trajectory, dict):
        raise ValueError(
            "joint_trajectoryがありません。"
        )

    return joint_trajectory


def duration_to_seconds(duration):
    if not isinstance(duration, dict):
        raise ValueError(
            "time_from_startの形式が不正です。"
        )

    secs = int(duration.get("secs", 0))
    nsecs = int(duration.get("nsecs", 0))

    return secs + nsecs * 1.0e-9


def all_finite(values):
    return all(
        math.isfinite(float(value))
        for value in values
    )


def main():
    args = parse_arguments()

    trajectory_path = os.path.expanduser(
        args.trajectory_file
    )

    trajectory = load_trajectory(
        trajectory_path
    )

    joint_names = trajectory.get(
        "joint_names",
        [],
    )

    points = trajectory.get(
        "points",
        [],
    )

    if not joint_names:
        raise ValueError(
            "joint_namesが空です。"
        )

    if not points:
        raise ValueError(
            "軌道点が0点です。"
        )

    joint_count = len(joint_names)

    times = []
    maximum_velocities = [0.0] * joint_count
    maximum_accelerations = [0.0] * joint_count
    maximum_position_steps = [0.0] * joint_count

    previous_positions = None

    for point_index, point in enumerate(points):
        positions = point.get("positions", [])
        velocities = point.get("velocities", [])
        accelerations = point.get(
            "accelerations",
            [],
        )

        if len(positions) != joint_count:
            raise ValueError(
                f"point {point_index}: positionsの要素数が"
                f"{joint_count}ではありません。"
            )

        if len(velocities) != joint_count:
            raise ValueError(
                f"point {point_index}: velocitiesの要素数が"
                f"{joint_count}ではありません。"
            )

        if len(accelerations) != joint_count:
            raise ValueError(
                f"point {point_index}: accelerationsの要素数が"
                f"{joint_count}ではありません。"
            )

        if not all_finite(positions):
            raise ValueError(
                f"point {point_index}: positionsにNaN/Infがあります。"
            )

        if not all_finite(velocities):
            raise ValueError(
                f"point {point_index}: velocitiesにNaN/Infがあります。"
            )

        if not all_finite(accelerations):
            raise ValueError(
                f"point {point_index}: accelerationsにNaN/Infがあります。"
            )

        current_time = duration_to_seconds(
            point.get("time_from_start")
        )

        times.append(current_time)

        for joint_index in range(joint_count):
            maximum_velocities[joint_index] = max(
                maximum_velocities[joint_index],
                abs(float(velocities[joint_index])),
            )

            maximum_accelerations[joint_index] = max(
                maximum_accelerations[joint_index],
                abs(float(accelerations[joint_index])),
            )

        if previous_positions is not None:
            for joint_index in range(joint_count):
                position_step = abs(
                    float(positions[joint_index])
                    - float(
                        previous_positions[joint_index]
                    )
                )

                maximum_position_steps[joint_index] = max(
                    maximum_position_steps[joint_index],
                    position_step,
                )

        previous_positions = positions

    if abs(times[0]) > 1.0e-9:
        raise ValueError(
            f"先頭時刻が0ではありません: {times[0]}"
        )

    non_increasing = [
        point_index
        for point_index in range(1, len(times))
        if times[point_index] <= times[point_index - 1]
    ]

    if non_increasing:
        raise ValueError(
            "time_from_startが単調増加していません。"
            f" 該当点={non_increasing}"
        )

    intervals = [
        times[point_index] - times[point_index - 1]
        for point_index in range(1, len(times))
    ]

    first_point = points[0]
    final_point = points[-1]

    print("============ 統合軌道内部検証 ============")
    print(f"ファイル: {trajectory_path}")
    print(f"関節数: {joint_count}")
    print(f"軌道点数: {len(points)}")
    print(f"総軌道時間: {times[-1]:.9f} s")
    print(f"最小時間間隔: {min(intervals):.9f} s")
    print(f"最大時間間隔: {max(intervals):.9f} s")
    print("time_from_start: 単調増加")
    print("NaN/Inf: なし")
    print("全点の配列長: 正常")
    print()

    for joint_index, joint_name in enumerate(
        joint_names
    ):
        print(
            f"{joint_name}: "
            f"最大速度={maximum_velocities[joint_index]:.6f} rad/s, "
            f"最大加速度={maximum_accelerations[joint_index]:.6f} rad/s^2, "
            f"最大1点間変化={maximum_position_steps[joint_index]:.6f} rad"
        )

    print()
    print(
        "開始速度:",
        [
            round(float(value), 9)
            for value in first_point["velocities"]
        ],
    )

    print(
        "終了速度:",
        [
            round(float(value), 9)
            for value in final_point["velocities"]
        ],
    )

    print(
        "開始加速度:",
        [
            round(float(value), 9)
            for value in first_point["accelerations"]
        ],
    )

    print(
        "終了加速度:",
        [
            round(float(value), 9)
            for value in final_point["accelerations"]
        ],
    )

    print("===========================================")


if __name__ == "__main__":
    try:
        main()

    except (
        FileNotFoundError,
        ValueError,
        KeyError,
        TypeError,
    ) as error:
        print(
            f"エラー: {error}",
            file=sys.stderr,
        )
        sys.exit(1)
