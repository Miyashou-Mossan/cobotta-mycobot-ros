#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import math
import os
import re
import sys


RESULT_CSV = os.path.expanduser(
    "~/unity_trajectory_plan_results_0_19.csv"
)

MOVEIT_LOG = os.path.expanduser(
    "~/grasp_pose_to_moveit_0_19_rosout.log"
)

OUTPUT_CSV = os.path.expanduser(
    "~/unity_trajectory_plan_analysis_0_19.csv"
)

PAPER_CENTER_X = 0.260
PAPER_CENTER_Y = -0.070
PAPER_CENTER_Z = 0.107
PAPER_CENTER_YAW = math.radians(45.0)

LARGE_JOINT_CHANGE_RAD = 0.5


JOINT_PATTERN = re.compile(
    r"cobotta_joint_([1-6]):\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[eE][-+]?\d+)?)\s*rad"
)

TCP_PATTERN = re.compile(
    r"Final TCP position:\s*frame=world,\s*"
    r"x=([-+0-9.eE]+),\s*"
    r"y=([-+0-9.eE]+),\s*"
    r"z=([-+0-9.eE]+)"
)


def load_plan_csv():
    if not os.path.isfile(RESULT_CSV):
        raise FileNotFoundError(
            f"計画結果CSVがありません: {RESULT_CSV}"
        )

    with open(
        RESULT_CSV,
        "r",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        rows = list(csv.DictReader(csv_file))

    if len(rows) != 20:
        raise ValueError(
            f"計画結果CSVが20点ではありません: {len(rows)}点"
        )

    failed = [
        row["index"]
        for row in rows
        if row["plan_success"].strip().lower()
        not in ("true", "1")
    ]

    if failed:
        raise ValueError(
            f"計画失敗点が含まれています: {failed}"
        )

    return rows


def load_moveit_log():
    if not os.path.isfile(MOVEIT_LOG):
        raise FileNotFoundError(
            f"MoveItログがありません: {MOVEIT_LOG}"
        )

    records = []
    current = None

    with open(
        MOVEIT_LOG,
        "r",
        encoding="utf-8",
        errors="replace",
    ) as log_file:
        for line in log_file:
            if "=== Planned final joint positions ===" in line:
                current = {
                    "joints": {},
                    "tcp": None,
                }
                continue

            if current is None:
                continue

            joint_match = JOINT_PATTERN.search(line)

            if joint_match:
                joint_number = int(joint_match.group(1))
                joint_value = float(joint_match.group(2))

                current["joints"][joint_number] = joint_value
                continue

            tcp_match = TCP_PATTERN.search(line)

            if tcp_match:
                current["tcp"] = (
                    float(tcp_match.group(1)),
                    float(tcp_match.group(2)),
                    float(tcp_match.group(3)),
                )

                missing = [
                    joint_number
                    for joint_number in range(1, 7)
                    if joint_number not in current["joints"]
                ]

                if missing:
                    raise ValueError(
                        f"関節角ログが不足しています: {missing}"
                    )

                records.append(current)
                current = None

    if len(records) != 20:
        raise ValueError(
            f"MoveItログから取得できた結果が20点ではありません: "
            f"{len(records)}点"
        )

    return records


def paper_center_to_world(x, y, z):
    cos_yaw = math.cos(PAPER_CENTER_YAW)
    sin_yaw = math.sin(PAPER_CENTER_YAW)

    world_x = (
        PAPER_CENTER_X
        + cos_yaw * x
        - sin_yaw * y
    )

    world_y = (
        PAPER_CENTER_Y
        + sin_yaw * x
        + cos_yaw * y
    )

    world_z = PAPER_CENTER_Z + z

    return world_x, world_y, world_z


def main():
    plan_rows = load_plan_csv()
    moveit_records = load_moveit_log()

    analysis_rows = []

    previous_joints = None
    previous_index = None

    maximum_fk_error_mm = -1.0
    maximum_fk_error_index = None

    maximum_joint_delta = -1.0
    maximum_joint_number = None
    maximum_joint_from = None
    maximum_joint_to = None

    large_change_segments = []

    fk_errors_mm = []

    for plan_row, moveit_record in zip(
        plan_rows,
        moveit_records,
    ):
        point_index = int(plan_row["index"])

        paper_x = float(plan_row["paper_center_x"])
        paper_y = float(plan_row["paper_center_y"])
        paper_z = float(plan_row["paper_center_z"])

        expected_x, expected_y, expected_z = (
            paper_center_to_world(
                paper_x,
                paper_y,
                paper_z,
            )
        )

        tcp_x, tcp_y, tcp_z = moveit_record["tcp"]

        error_x = tcp_x - expected_x
        error_y = tcp_y - expected_y
        error_z = tcp_z - expected_z

        fk_error_m = math.sqrt(
            error_x ** 2
            + error_y ** 2
            + error_z ** 2
        )

        fk_error_mm = fk_error_m * 1000.0
        fk_errors_mm.append(fk_error_mm)

        if fk_error_mm > maximum_fk_error_mm:
            maximum_fk_error_mm = fk_error_mm
            maximum_fk_error_index = point_index

        joints = [
            moveit_record["joints"][joint_number]
            for joint_number in range(1, 7)
        ]

        deltas = [""] * 6
        segment_max_delta = ""
        segment_max_joint = ""
        segment_max_delta_deg = ""

        if previous_joints is not None:
            numeric_deltas = [
                current - previous
                for current, previous in zip(
                    joints,
                    previous_joints,
                )
            ]

            absolute_deltas = [
                abs(delta)
                for delta in numeric_deltas
            ]

            segment_max_delta = max(absolute_deltas)
            segment_max_joint_number = (
                absolute_deltas.index(segment_max_delta) + 1
            )

            segment_max_joint = (
                f"J{segment_max_joint_number}"
            )

            segment_max_delta_deg = math.degrees(
                segment_max_delta
            )

            deltas = numeric_deltas

            if segment_max_delta > maximum_joint_delta:
                maximum_joint_delta = segment_max_delta
                maximum_joint_number = (
                    segment_max_joint_number
                )
                maximum_joint_from = previous_index
                maximum_joint_to = point_index

            if (
                segment_max_delta
                >= LARGE_JOINT_CHANGE_RAD
            ):
                large_change_segments.append(
                    (
                        previous_index,
                        point_index,
                        segment_max_joint_number,
                        segment_max_delta,
                    )
                )

        output_row = {
            "index": point_index,
            "paper_center_x": paper_x,
            "paper_center_y": paper_y,
            "paper_center_z": paper_z,
            "expected_world_x": expected_x,
            "expected_world_y": expected_y,
            "expected_world_z": expected_z,
            "tcp_world_x": tcp_x,
            "tcp_world_y": tcp_y,
            "tcp_world_z": tcp_z,
            "error_x_mm": error_x * 1000.0,
            "error_y_mm": error_y * 1000.0,
            "error_z_mm": error_z * 1000.0,
            "fk_error_mm": fk_error_mm,
        }

        for joint_number, joint_value in enumerate(
            joints,
            start=1,
        ):
            output_row[
                f"joint_{joint_number}_rad"
            ] = joint_value

        for joint_number, delta in enumerate(
            deltas,
            start=1,
        ):
            output_row[
                f"delta_joint_{joint_number}_rad"
            ] = delta

        output_row["max_delta_joint"] = (
            segment_max_joint
        )
        output_row["max_delta_rad"] = (
            segment_max_delta
        )
        output_row["max_delta_deg"] = (
            segment_max_delta_deg
        )

        analysis_rows.append(output_row)

        previous_joints = joints
        previous_index = point_index

    with open(
        OUTPUT_CSV,
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(analysis_rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(analysis_rows)

    mean_fk_error_mm = (
        sum(fk_errors_mm) / len(fk_errors_mm)
    )

    print("============ 20点分析結果 ============")
    print(f"分析点数: {len(analysis_rows)}点")

    print(
        "平均FK位置誤差: "
        f"{mean_fk_error_mm:.3f} mm"
    )

    print(
        "最大FK位置誤差: "
        f"{maximum_fk_error_mm:.3f} mm "
        f"(index {maximum_fk_error_index})"
    )

    print(
        "最大隣接関節角差: "
        f"{maximum_joint_delta:.6f} rad "
        f"({math.degrees(maximum_joint_delta):.3f}°), "
        f"J{maximum_joint_number}, "
        f"index {maximum_joint_from}"
        f"→{maximum_joint_to}"
    )

    if large_change_segments:
        print(
            "0.5 rad以上の隣接関節角変化: "
            f"{len(large_change_segments)}区間"
        )

        for (
            index_from,
            index_to,
            joint_number,
            joint_delta,
        ) in large_change_segments:
            print(
                f"  index {index_from}→{index_to}: "
                f"J{joint_number}, "
                f"{joint_delta:.6f} rad "
                f"({math.degrees(joint_delta):.3f}°)"
            )
    else:
        print(
            "0.5 rad以上の隣接関節角変化: なし"
        )

    print(f"分析CSV: {OUTPUT_CSV}")
    print("=======================================")


if __name__ == "__main__":
    try:
        main()

    except (
        FileNotFoundError,
        ValueError,
        KeyError,
    ) as error:
        print(
            f"エラー: {error}",
            file=sys.stderr,
        )
        sys.exit(1)
