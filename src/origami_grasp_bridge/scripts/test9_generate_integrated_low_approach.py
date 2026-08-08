#!/usr/bin/env python3

import copy
import csv
import math
import sys
import yaml

import moveit_commander
import rospy

from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory
from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest
from trajectory_msgs.msg import JointTrajectoryPoint


JOINT_NAMES = [
    "cobotta_joint_1",
    "cobotta_joint_2",
    "cobotta_joint_3",
    "cobotta_joint_4",
    "cobotta_joint_5",
    "cobotta_joint_6",
]

GROUP_NAME = "cobotta_arm"

PHASE1_CSV = (
    "/home/maeda/test9_tilt5_to40_transition_scan.csv"
)

PHASE2_CSV = (
    "/home/maeda/"
    "test9_paper_plus_x_transition_tilt40_scan.csv"
)

PHASE3_CSV = (
    "/home/maeda/"
    "test9_paper_plus_x_tilt_back_scan.csv"
)

PHASE4_CSV = (
    "/home/maeda/"
    "test9_paper_plus_x_tilt17_height_scan.csv"
)

OUTPUT_CSV = (
    "/home/maeda/"
    "test9_integrated_paper_plus_x_9p35mm.csv"
)

OUTPUT_YAML = (
    "/home/maeda/"
    "test9_integrated_paper_plus_x_9p35mm.yaml"
)

TARGET_TILT_DEG = 16.9
TARGET_Z_MM = 9.35

# RViz再生用の時間
PHASE1_DURATION = 4.0
PHASE2_DURATION = 6.0
PHASE3_DURATION = 4.0
PHASE4_DURATION = 6.0
FINAL_DWELL = 2.0

# 5度を超える隣接関節角差は分岐ジャンプとして拒否
MAX_JOINT_STEP_DEG = 5.0

# 保存点間も0.5度以下になるよう補間して衝突確認
COLLISION_CHECK_STEP_DEG = 0.5


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def is_true(value):
    return str(value).strip().lower() in (
        "true",
        "1",
        "yes",
    )


def positions_from_row(row):
    return [
        float(row[name])
        for name in JOINT_NAMES
    ]


def max_delta_deg(a, b):
    return max(
        math.degrees(abs(x - y))
        for x, y in zip(a, b)
    )


def valid_continuous_rows(rows, index_name):
    result = []

    for row in sorted(
        rows,
        key=lambda r: int(r[index_name]),
    ):
        if not is_true(row.get("valid", "")):
            break

        text = row.get(
            "adjacent_delta_deg",
            "",
        )

        if text:
            if float(text) > MAX_JOINT_STEP_DEG:
                break

        result.append(row)

    return result


def select_phases():
    # Phase 1: +5 -> +40
    phase1 = valid_continuous_rows(
        load_csv(PHASE1_CSV),
        "step",
    )

    # Phase 2: +40のままpaper_plus_xへ
    phase2 = valid_continuous_rows(
        load_csv(PHASE2_CSV),
        "step",
    )

    # Phase 3: paper_plus_xで+40 -> +16.9
    phase3_all = valid_continuous_rows(
        load_csv(PHASE3_CSV),
        "step",
    )

    if not phase3_all:
        raise RuntimeError(
            "Phase 3に連続解がありません。"
        )

    target_index = min(
        range(len(phase3_all)),
        key=lambda i: abs(
            float(phase3_all[i]["tilt_deg"])
            - TARGET_TILT_DEG
        ),
    )

    phase3 = phase3_all[
        :target_index + 1
    ]

    actual_tilt = float(
        phase3[-1]["tilt_deg"]
    )

    if abs(actual_tilt - TARGET_TILT_DEG) > 0.1:
        raise RuntimeError(
            "16.9 degの姿勢が見つかりません。"
        )

    # Phase 4: +16.9で60.35 -> 9.35 mm
    phase4_all = valid_continuous_rows(
        load_csv(PHASE4_CSV),
        "scan_index",
    )

    phase4 = []

    for row in phase4_all:
        z_mm = float(row["z_mm"])

        if z_mm >= TARGET_Z_MM - 1.0e-6:
            phase4.append(row)
        else:
            break

    if not phase4:
        raise RuntimeError(
            "Phase 4に下降軌道がありません。"
        )

    final_z = float(
        phase4[-1]["z_mm"]
    )

    if abs(final_z - TARGET_Z_MM) > 0.1:
        raise RuntimeError(
            "9.35 mmまでの点列を取得できません。"
        )

    return [
        (
            "tilt_5_to_40",
            phase1,
            PHASE1_DURATION,
        ),
        (
            "yaw_to_paper_plus_x",
            phase2,
            PHASE2_DURATION,
        ),
        (
            "tilt_40_to_16p9",
            phase3,
            PHASE3_DURATION,
        ),
        (
            "descent_to_9p35mm",
            phase4,
            PHASE4_DURATION,
        ),
    ]


def combine_phases(phases):
    records = []
    current_time = 0.0

    print("\n===== Phase boundary checks =====")

    for phase_name, rows, duration in phases:
        if len(rows) < 2:
            raise RuntimeError(
                "{}: 点数不足".format(
                    phase_name
                )
            )

        positions = [
            positions_from_row(row)
            for row in rows
        ]

        skip_first = False

        if records:
            boundary_delta = max_delta_deg(
                records[-1]["positions"],
                positions[0],
            )

            print(
                "{:<24} -> {:<24} : "
                "{:.6f} deg".format(
                    records[-1]["phase"],
                    phase_name,
                    boundary_delta,
                )
            )

            if boundary_delta > MAX_JOINT_STEP_DEG:
                raise RuntimeError(
                    "Phase境界で関節ジャンプ: "
                    "{:.6f} deg".format(
                        boundary_delta
                    )
                )

            skip_first = True

        count = len(rows)

        for i, position in enumerate(positions):
            if skip_first and i == 0:
                continue

            ratio = i / float(count - 1)

            records.append({
                "phase": phase_name,
                "phase_index": i,
                "time": (
                    current_time
                    + duration * ratio
                ),
                "positions": position,
            })

        current_time += duration

    # 最終姿勢で2秒停止
    dwell = copy.deepcopy(
        records[-1]
    )

    dwell["phase"] = "final_dwell"
    dwell["phase_index"] = 0
    dwell["time"] = (
        current_time + FINAL_DWELL
    )

    records.append(dwell)

    return records


def robot_state_from_positions(
    template,
    positions,
):
    state = copy.deepcopy(template)

    name_to_index = {
        name: i
        for i, name
        in enumerate(
            state.joint_state.name
        )
    }

    # rospy環境によってjoint_state.positionがtupleになるため、
    # 一度listへ変換してからまとめて戻す。
    joint_positions = list(
        state.joint_state.position
    )

    for name, value in zip(
        JOINT_NAMES,
        positions,
    ):
        if name not in name_to_index:
            raise RuntimeError(
                "{}がRobotStateにありません。".format(
                    name
                )
            )

        joint_positions[
            name_to_index[name]
        ] = value

    state.joint_state.position = (
        joint_positions
    )

    state.joint_state.header.stamp = (
        rospy.Time.now()
    )

    return state


def contact_text(response):
    pairs = []

    for contact in response.contacts:
        pair = "{} <-> {}".format(
            contact.contact_body_1,
            contact.contact_body_2,
        )

        if pair not in pairs:
            pairs.append(pair)

    return "; ".join(pairs)


def check_state(
    service,
    template_state,
    positions,
):
    req = GetStateValidityRequest()

    req.group_name = GROUP_NAME

    req.robot_state = (
        robot_state_from_positions(
            template_state,
            positions,
        )
    )

    res = service(req)

    return (
        bool(res.valid),
        contact_text(res),
    )


def check_joint_continuity(records):
    max_value = 0.0
    max_joint = ""
    max_segment = ""

    for i in range(1, len(records)):
        previous = records[i - 1][
            "positions"
        ]

        current = records[i][
            "positions"
        ]

        deltas = [
            abs(a - b)
            for a, b in zip(
                current,
                previous,
            )
        ]

        j = max(
            range(len(deltas)),
            key=lambda k: deltas[k],
        )

        value_deg = math.degrees(
            deltas[j]
        )

        if value_deg > max_value:
            max_value = value_deg
            max_joint = JOINT_NAMES[j]
            max_segment = "{}->{}".format(
                i - 1,
                i,
            )

    if max_value > MAX_JOINT_STEP_DEG:
        raise RuntimeError(
            "Joint jump: {:.6f} deg, "
            "{} at {}".format(
                max_value,
                max_joint,
                max_segment,
            )
        )

    return (
        max_value,
        max_joint,
        max_segment,
    )


def dense_collision_check(
    service,
    template_state,
    records,
):
    checked = 0

    print(
        "\n===== Dense collision validation ====="
    )

    # 保存されている各点
    for i, record in enumerate(records):
        valid, contacts = check_state(
            service,
            template_state,
            record["positions"],
        )

        checked += 1

        if not valid:
            raise RuntimeError(
                "Collision at point {} "
                "({}): {}".format(
                    i,
                    record["phase"],
                    contacts or "-",
                )
            )

    # 保存点と保存点の間
    for i in range(1, len(records)):
        start = records[i - 1][
            "positions"
        ]

        end = records[i][
            "positions"
        ]

        step_deg = max_delta_deg(
            start,
            end,
        )

        divisions = max(
            1,
            int(
                math.ceil(
                    step_deg
                    / COLLISION_CHECK_STEP_DEG
                )
            ),
        )

        for j in range(
            1,
            divisions,
        ):
            ratio = j / float(divisions)

            interpolated = [
                a + (b - a) * ratio
                for a, b in zip(
                    start,
                    end,
                )
            ]

            valid, contacts = check_state(
                service,
                template_state,
                interpolated,
            )

            checked += 1

            if not valid:
                raise RuntimeError(
                    "Collision between "
                    "{}->{} ratio={:.3f}: {}".format(
                        i - 1,
                        i,
                        ratio,
                        contacts or "-",
                    )
                )

    print(
        "validated states       :",
        checked,
    )

    return checked


def save_files(records):
    fieldnames = [
        "trajectory_index",
        "phase",
        "phase_index",
        "time_from_start",
    ] + JOINT_NAMES

    rows = []
    yaml_points = []

    for index, record in enumerate(
        records
    ):
        row = {
            "trajectory_index": index,
            "phase": record["phase"],
            "phase_index": (
                record["phase_index"]
            ),
            "time_from_start": (
                record["time"]
            ),
        }

        for name, value in zip(
            JOINT_NAMES,
            record["positions"],
        ):
            row[name] = value

        rows.append(row)

        duration = rospy.Duration.from_sec(
            float(record["time"])
        )

        yaml_points.append({
            "positions": list(
                record["positions"]
            ),
            "velocities": [
                0.0
                for _ in JOINT_NAMES
            ],
            "accelerations": [
                0.0
                for _ in JOINT_NAMES
            ],
            "time_from_start": {
                "secs": duration.secs,
                "nsecs": duration.nsecs,
            },
            "phase": record["phase"],
        })

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    with open(
        OUTPUT_YAML,
        "w",
    ) as f:
        yaml.safe_dump(
            {
                "joint_trajectory": {
                    "joint_names": JOINT_NAMES,
                    "points": yaml_points,
                },
                "metadata": {
                    "paper_direction":
                        "paper_plus_x",
                    "final_local_y_tilt_deg":
                        TARGET_TILT_DEG,
                    "final_paper_frame_z_mm":
                        TARGET_Z_MM,
                    "execution_enabled":
                        False,
                },
            },
            f,
            sort_keys=False,
        )


def publish_trajectory(
    template_state,
    records,
):
    trajectory = RobotTrajectory()

    trajectory.joint_trajectory.joint_names = (
        list(JOINT_NAMES)
    )

    for record in records:
        point = JointTrajectoryPoint()

        point.positions = list(
            record["positions"]
        )

        point.velocities = [
            0.0
            for _ in JOINT_NAMES
        ]

        point.accelerations = [
            0.0
            for _ in JOINT_NAMES
        ]

        point.time_from_start = (
            rospy.Duration.from_sec(
                record["time"]
            )
        )

        trajectory.joint_trajectory.points.append(
            point
        )

    publisher = rospy.Publisher(
        "/move_group/display_planned_path",
        DisplayTrajectory,
        queue_size=1,
        latch=True,
    )

    message = DisplayTrajectory()

    message.model_id = (
        "cobotta_mycobot_dual_robot"
    )

    message.trajectory_start = (
        robot_state_from_positions(
            template_state,
            records[0]["positions"],
        )
    )

    message.trajectory.append(
        trajectory
    )

    rospy.sleep(1.0)

    publisher.publish(message)

    return publisher


def main():
    moveit_commander.roscpp_initialize(
        sys.argv
    )

    rospy.init_node(
        "test9_generate_integrated_low_approach",
        anonymous=True,
    )

    phases = select_phases()

    print("\n===== Selected phases =====")

    for name, rows, duration in phases:
        print(
            "{:<25} : {:>3} points / "
            "{:.1f} s".format(
                name,
                len(rows),
                duration,
            )
        )

    records = combine_phases(
        phases
    )

    robot = (
        moveit_commander.RobotCommander()
    )

    template_state = (
        robot.get_current_state()
    )

    (
        max_delta,
        max_joint,
        max_segment,
    ) = check_joint_continuity(
        records
    )

    rospy.wait_for_service(
        "/check_state_validity",
        timeout=20.0,
    )

    check_validity = rospy.ServiceProxy(
        "/check_state_validity",
        GetStateValidity,
    )

    checked = dense_collision_check(
        check_validity,
        template_state,
        records,
    )

    save_files(records)

    publisher = publish_trajectory(
        template_state,
        records,
    )

    print(
        "\n===== Integrated trajectory summary ====="
    )

    print(
        "trajectory points      :",
        len(records),
    )

    print(
        "duration               : "
        "{:.3f} s".format(
            records[-1]["time"]
        )
    )

    print(
        "maximum adjacent delta : "
        "{:.6f} deg, {} at {}".format(
            max_delta,
            max_joint,
            max_segment,
        )
    )

    print(
        "validated states       :",
        checked,
    )

    print(
        "final paper direction  : "
        "paper_plus_x"
    )

    print(
        "final local_y tilt     : "
        "{:.3f} deg".format(
            TARGET_TILT_DEG
        )
    )

    print(
        "final paper-frame Z    : "
        "{:.3f} mm".format(
            TARGET_Z_MM
        )
    )

    print(
        "trajectory YAML        :",
        OUTPUT_YAML,
    )

    print(
        "diagnostics CSV        :",
        OUTPUT_CSV,
    )

    print(
        "execution              : DISABLED"
    )

    print(
        "\nPublished to "
        "/move_group/display_planned_path"
    )

    print(
        "RViz確認後、Ctrl+Cで終了してください。"
    )

    rospy.spin()


if __name__ == "__main__":
    main()
