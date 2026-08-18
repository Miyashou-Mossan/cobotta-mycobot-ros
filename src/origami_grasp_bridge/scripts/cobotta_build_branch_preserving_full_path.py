#!/usr/bin/env python3

import csv
import math
from pathlib import Path


SCAN_FILE = Path(
    "/home/maeda/directionA_reverse_local_z_m25_actual_grasp_collision_scan.csv"
)
BEST_FILE = Path(
    "/home/maeda/cobotta_branch_preserving_best_path.csv"
)
OUTPUT_FILE = Path(
    "/home/maeda/cobotta_branch_preserving_full_path_0_340.csv"
)

START_INDEX = 0
BRANCH_START_INDEX = 260
FINISH_INDEX = 340

JOINTS = [f"cobotta_joint_{i}" for i in range(1, 7)]

# branch-preserving searchで使用した連続性条件
MAX_JOINT_JUMP_DEG = 10.0

# index260の接続確認用
JOIN_TOLERANCE_RAD = 1.0e-10


def load_csv(path):
    if not path.exists():
        raise RuntimeError(f"Input file does not exist: {path}")

    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def joint_values(row):
    vals = []

    for name in JOINTS:
        value = float(row[name])

        if not math.isfinite(value):
            raise RuntimeError(
                f"NaN/Inf detected: index={row.get('index')} joint={name}"
            )

        vals.append(value)

    return vals


def max_joint_delta_deg(q0, q1):
    deltas = [
        abs(math.degrees(b - a))
        for a, b in zip(q0, q1)
    ]

    max_value = max(deltas)
    max_joint = JOINTS[deltas.index(max_value)]

    return max_value, max_joint, deltas


def main():

    print("=" * 80)
    print("COBOTTA branch-preserving full path builder")
    print("=" * 80)

    print("SCAN :", SCAN_FILE)
    print("BEST :", BEST_FILE)
    print("OUT  :", OUTPUT_FILE)

    scan_rows = load_csv(SCAN_FILE)
    best_rows = load_csv(BEST_FILE)

    scan = {
        int(row["index"]): row
        for row in scan_rows
    }

    best = {
        int(row["index"]): row
        for row in best_rows
    }

    # --------------------------------------------------------
    # 1. scan側確認
    # --------------------------------------------------------

    required_scan = list(
        range(START_INDEX, BRANCH_START_INDEX + 1)
    )

    missing_scan = [
        i for i in required_scan
        if i not in scan
    ]

    if missing_scan:
        raise RuntimeError(
            f"Missing scan indices: {missing_scan}"
        )

    invalid_scan = [
        i for i in required_scan
        if scan[i]["valid"].strip().lower() != "true"
    ]

    if invalid_scan:
        raise RuntimeError(
            f"Invalid scan indices: {invalid_scan}"
        )

    # --------------------------------------------------------
    # 2. best path側確認
    # --------------------------------------------------------

    expected_best = list(
        range(BRANCH_START_INDEX, FINISH_INDEX + 1)
    )

    actual_best = sorted(best.keys())

    if actual_best != expected_best:
        raise RuntimeError(
            "Best path is not contiguous "
            f"{BRANCH_START_INDEX}..{FINISH_INDEX}"
        )

    # --------------------------------------------------------
    # 3. index260の接続確認
    # --------------------------------------------------------

    q_scan_join = joint_values(scan[BRANCH_START_INDEX])
    q_best_join = joint_values(best[BRANCH_START_INDEX])

    join_diff = [
        abs(a - b)
        for a, b in zip(q_scan_join, q_best_join)
    ]

    max_join_diff = max(join_diff)

    print()
    print("JOIN CHECK")
    print(
        f"  index              : {BRANCH_START_INDEX}"
    )
    print(
        f"  max difference rad : {max_join_diff:.12e}"
    )
    print(
        "  max difference deg : "
        f"{math.degrees(max_join_diff):.12e}"
    )

    if max_join_diff > JOIN_TOLERANCE_RAD:
        raise RuntimeError(
            "index260 joint states do not match"
        )

    # --------------------------------------------------------
    # 4. full path構築
    #
    # scan : 0..259
    # best : 260..340
    # --------------------------------------------------------

    full_rows = []

    for index in range(
        START_INDEX,
        BRANCH_START_INDEX
    ):
        src = scan[index]

        row = {
            "index": index,
            "source": "baseline_scan",
            "original_time_from_start":
                src["time_from_start"],
            "local_x_deg": "0.0",
            "local_y_deg": "0.0",
        }

        for joint in JOINTS:
            row[joint] = src[joint]

        full_rows.append(row)

    for index in range(
        BRANCH_START_INDEX,
        FINISH_INDEX + 1
    ):
        src = best[index]

        # 元Unity軌道のindex/time対応だけ保持する。
        # 実機RobotTrajectoryの時刻にはまだ使用しない。
        original_time = scan[index]["time_from_start"]

        row = {
            "index": index,
            "source": "branch_preserving",
            "original_time_from_start":
                original_time,
            "local_x_deg": src["local_x_deg"],
            "local_y_deg": src["local_y_deg"],
        }

        for joint in JOINTS:
            row[joint] = src[joint]

        full_rows.append(row)

    # --------------------------------------------------------
    # 5. 点数・index確認
    # --------------------------------------------------------

    expected_count = FINISH_INDEX - START_INDEX + 1

    if len(full_rows) != expected_count:
        raise RuntimeError(
            f"Unexpected point count: {len(full_rows)} "
            f"(expected {expected_count})"
        )

    indices = [
        int(row["index"])
        for row in full_rows
    ]

    if indices != list(
        range(START_INDEX, FINISH_INDEX + 1)
    ):
        raise RuntimeError(
            "Output index sequence is not contiguous"
        )

    # --------------------------------------------------------
    # 6. 全区間joint continuity確認
    # --------------------------------------------------------

    worst_delta = -1.0
    worst_from = None
    worst_joint = None
    worst_per_joint = None

    for i in range(len(full_rows) - 1):

        qa = joint_values(full_rows[i])
        qb = joint_values(full_rows[i + 1])

        max_delta, max_joint, per_joint = \
            max_joint_delta_deg(qa, qb)

        if max_delta > worst_delta:
            worst_delta = max_delta
            worst_from = int(full_rows[i]["index"])
            worst_joint = max_joint
            worst_per_joint = per_joint

        if max_delta > MAX_JOINT_JUMP_DEG:
            raise RuntimeError(
                "Joint jump limit exceeded: "
                f"{full_rows[i]['index']} -> "
                f"{full_rows[i + 1]['index']} "
                f"{max_delta:.6f} deg "
                f"({max_joint})"
            )

    # --------------------------------------------------------
    # 7. 保存
    # --------------------------------------------------------

    fieldnames = [
        "index",
        "source",
        "original_time_from_start",
        "local_x_deg",
        "local_y_deg",
        *JOINTS,
    ]

    with OUTPUT_FILE.open(
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(full_rows)

    # --------------------------------------------------------
    # 8. Summary
    # --------------------------------------------------------

    finish = full_rows[-1]

    print()
    print("=" * 80)
    print("BUILD RESULT")
    print("=" * 80)

    print(
        f"point count : {len(full_rows)}"
    )
    print(
        f"index range : "
        f"{full_rows[0]['index']} -> "
        f"{full_rows[-1]['index']}"
    )

    print(
        f"baseline    : "
        f"{START_INDEX} -> "
        f"{BRANCH_START_INDEX - 1}"
    )

    print(
        f"branch path : "
        f"{BRANCH_START_INDEX} -> "
        f"{FINISH_INDEX}"
    )

    print()
    print("maximum adjacent joint change")
    print(
        f"  segment : "
        f"{worst_from} -> {worst_from + 1}"
    )
    print(
        f"  joint   : {worst_joint}"
    )
    print(
        f"  delta   : {worst_delta:.6f} deg"
    )
    print(
        "  all     :",
        [
            f"{v:.6f}"
            for v in worst_per_joint
        ]
    )

    print()
    print("FINISH")
    print(
        f"  index   : {finish['index']}"
    )
    print(
        f"  local X : {finish['local_x_deg']} deg"
    )
    print(
        f"  local Y : {finish['local_y_deg']} deg"
    )

    print(
        "  joints  :",
        [
            f"{float(finish[j]):.12f}"
            for j in JOINTS
        ]
    )

    print()
    print("OUTPUT")
    print(" ", OUTPUT_FILE)

    print()
    print(
        "NOTE: original_time_from_start is "
        "Unity trajectory timing reference only."
    )
    print(
        "      Robot execution timing has NOT "
        "been generated yet."
    )

    print()
    print("BUILD SUCCESS")


if __name__ == "__main__":
    main()
