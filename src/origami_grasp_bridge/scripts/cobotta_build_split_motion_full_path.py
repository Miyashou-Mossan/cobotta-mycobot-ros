#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import math
import os


INPUT_PATH = os.path.expanduser(
    "~/cobotta_branch_preserving_full_path_0_340_edge_fixed.csv"
)

SUMMARY_CSV = os.path.expanduser(
    "~/cobotta_split_motion_diagnostic_summary.csv"
)

DETAIL_CSV = os.path.expanduser(
    "~/cobotta_split_motion_diagnostic_detail.csv"
)

OUTPUT_CSV = os.path.expanduser(
    "~/cobotta_branch_preserving_full_path_0_340_split_motion.csv"
)


JOINTS = [
    "cobotta_joint_1",
    "cobotta_joint_2",
    "cobotta_joint_3",
    "cobotta_joint_4",
    "cobotta_joint_5",
    "cobotta_joint_6",
]


def load_csv(filename):
    if not os.path.exists(filename):
        raise RuntimeError(
            "File does not exist: {}".format(filename)
        )

    with open(
        filename,
        newline="",
        encoding="utf-8-sig",
    ) as f:
        return list(csv.DictReader(f))


def validate_joint_values(row, label):
    for joint in JOINTS:
        if joint not in row:
            raise RuntimeError(
                "{} missing {}".format(
                    label,
                    joint,
                )
            )

        value = float(row[joint])

        if not math.isfinite(value):
            raise RuntimeError(
                "{} {} is NaN/Inf".format(
                    label,
                    joint,
                )
            )


def main():
    original = load_csv(INPUT_PATH)
    summary = load_csv(SUMMARY_CSV)
    detail = load_csv(DETAIL_CSV)

    if not original:
        raise RuntimeError(
            "Original path is empty"
        )

    if not summary:
        raise RuntimeError(
            "Summary is empty"
        )

    if not detail:
        raise RuntimeError(
            "Detail is empty"
        )

    # --------------------------------------------------------
    # Original path validation
    # --------------------------------------------------------

    original_by_pp = {}

    for expected_pp, row in enumerate(original):
        pp = int(row["path_point"])

        if pp != expected_pp:
            raise RuntimeError(
                "Original path_point mismatch: "
                "expected {}, got {}".format(
                    expected_pp,
                    pp,
                )
            )

        validate_joint_values(
            row,
            "original pp {}".format(pp),
        )

        original_by_pp[pp] = row

    # --------------------------------------------------------
    # Diagnostic summary validation
    # --------------------------------------------------------

    targets = []

    for row in summary:
        if row["status"] != "PASS":
            raise RuntimeError(
                "Diagnostic contains non-PASS transition: "
                "{}->{} status={}".format(
                    row["path_from"],
                    row["path_to"],
                    row["status"],
                )
            )

        path_from = int(row["path_from"])
        path_to = int(row["path_to"])

        if path_to != path_from + 1:
            raise RuntimeError(
                "Target is not adjacent: {}->{}".format(
                    path_from,
                    path_to,
                )
            )

        targets.append(
            (path_from, path_to)
        )

    if len(set(targets)) != len(targets):
        raise RuntimeError(
            "Duplicate target transitions"
        )

    targets = sorted(targets)

    target_from_set = {
        a
        for a, _ in targets
    }

    # --------------------------------------------------------
    # Detail grouping
    # --------------------------------------------------------

    detail_by_target = {}

    for row in detail:
        path_from = int(row["path_from"])
        path_to = int(row["path_to"])

        key = (
            path_from,
            path_to,
        )

        if key not in targets:
            raise RuntimeError(
                "Unexpected detail target: {}->{}".format(
                    path_from,
                    path_to,
                )
            )

        if row["endpoint_status"] != "VALID":
            raise RuntimeError(
                "Invalid detail endpoint: "
                "{}->{} phase={} step={} status={}".format(
                    path_from,
                    path_to,
                    row["phase"],
                    row["step"],
                    row["endpoint_status"],
                )
            )

        if row["edge_status"] != "VALID":
            raise RuntimeError(
                "Invalid detail edge: "
                "{}->{} phase={} step={} status={}".format(
                    path_from,
                    path_to,
                    row["phase"],
                    row["step"],
                    row["edge_status"],
                )
            )

        validate_joint_values(
            row,
            "detail {}->{} {} step {}".format(
                path_from,
                path_to,
                row["phase"],
                row["step"],
            ),
        )

        detail_by_target.setdefault(
            key,
            []
        ).append(row)

    # 8 ORIENTATION + 8 PAPER_MOVE を確認
    for key in targets:
        rows = detail_by_target.get(
            key,
            [],
        )

        orientation = [
            r
            for r in rows
            if r["phase"] == "ORIENTATION"
        ]

        paper_move = [
            r
            for r in rows
            if r["phase"] == "PAPER_MOVE"
        ]

        orientation.sort(
            key=lambda r: int(r["step"])
        )

        paper_move.sort(
            key=lambda r: int(r["step"])
        )

        if len(orientation) != 8:
            raise RuntimeError(
                "{}->{} orientation count={}".format(
                    key[0],
                    key[1],
                    len(orientation),
                )
            )

        if len(paper_move) != 8:
            raise RuntimeError(
                "{}->{} paper move count={}".format(
                    key[0],
                    key[1],
                    len(paper_move),
                )
            )

        detail_by_target[key] = (
            orientation
            + paper_move
        )

    # --------------------------------------------------------
    # Output construction
    # --------------------------------------------------------

    output = []

    def append_original(row):
        output.append({
            "path_point": len(output),
            "source_path_from":
                row["path_point"],
            "source_path_to":
                row["path_point"],
            "paper_index":
                row["paper_index"],
            "motion_phase":
                "ORIGINAL",
            "phase_step":
                0,
            "phase_ratio":
                0.0,
            "source":
                row.get(
                    "source",
                    "original",
                ),
            "original_time_from_start":
                row.get(
                    "original_time_from_start",
                    "",
                ),
            "local_x_deg":
                row["local_x_deg"],
            "local_y_deg":
                row["local_y_deg"],
            **{
                joint: float(row[joint])
                for joint in JOINTS
            },
        })

    def append_detail(row):
        phase = row["phase"]

        if phase == "ORIENTATION":
            paper_index = row["paper_from"]
            source = (
                "split_fixed_grasp_orientation"
            )

            original_time = (
                original_by_pp[
                    int(row["path_from"])
                ].get(
                    "original_time_from_start",
                    "",
                )
            )

        elif phase == "PAPER_MOVE":
            paper_index = (
                "{}->{}".format(
                    row["paper_from"],
                    row["paper_to"],
                )
            )

            source = (
                "split_fixed_orientation_paper_move"
            )

            a = original_by_pp[
                int(row["path_from"])
            ]

            b = original_by_pp[
                int(row["path_to"])
            ]

            try:
                ta = float(
                    a[
                        "original_time_from_start"
                    ]
                )

                tb = float(
                    b[
                        "original_time_from_start"
                    ]
                )

                ratio = float(
                    row["ratio"]
                )

                original_time = (
                    (1.0-ratio)*ta
                    + ratio*tb
                )

            except Exception:
                original_time = ""

        else:
            raise RuntimeError(
                "Unknown phase: {}".format(
                    phase
                )
            )

        output.append({
            "path_point":
                len(output),

            "source_path_from":
                int(row["path_from"]),

            "source_path_to":
                int(row["path_to"]),

            "paper_index":
                paper_index,

            "motion_phase":
                phase,

            "phase_step":
                int(row["step"]),

            "phase_ratio":
                float(row["ratio"]),

            "source":
                source,

            "original_time_from_start":
                original_time,

            "local_x_deg":
                float(row["local_x_deg"]),

            "local_y_deg":
                float(row["local_y_deg"]),

            **{
                joint: float(row[joint])
                for joint in JOINTS
            },
        })

    for pp, row in enumerate(original):

        append_original(row)

        if pp in target_from_set:
            key = (
                pp,
                pp + 1,
            )

            for detail_row in (
                detail_by_target[key]
            ):
                append_detail(
                    detail_row
                )

    # --------------------------------------------------------
    # Expected count
    # --------------------------------------------------------

    expected_count = (
        len(original)
        + len(detail)
    )

    if len(output) != expected_count:
        raise RuntimeError(
            "Output count mismatch: "
            "{} != {}".format(
                len(output),
                expected_count,
            )
        )

    # --------------------------------------------------------
    # Joint continuity statistics
    # --------------------------------------------------------

    max_delta = -1.0
    max_from = None
    max_joint = None

    for i in range(
        len(output) - 1
    ):
        qa = [
            float(output[i][joint])
            for joint in JOINTS
        ]

        qb = [
            float(output[i + 1][joint])
            for joint in JOINTS
        ]

        deltas = [
            abs(math.degrees(b-a))
            for a, b in zip(qa, qb)
        ]

        current_max = max(
            deltas
        )

        if current_max > max_delta:
            max_delta = current_max
            max_from = i
            max_joint = JOINTS[
                deltas.index(
                    current_max
                )
            ]

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    fields = [
        "path_point",
        "source_path_from",
        "source_path_to",
        "paper_index",
        "motion_phase",
        "phase_step",
        "phase_ratio",
        "source",
        "original_time_from_start",
        "local_x_deg",
        "local_y_deg",
        *JOINTS,
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(
            output
        )

    print("=" * 90)
    print(
        "SPLIT-MOTION FULL PATH BUILD"
    )
    print("=" * 90)

    print(
        "original points      :",
        len(original),
    )

    print(
        "split transitions    :",
        len(targets),
    )

    print(
        "inserted detail points:",
        len(detail),
    )

    print(
        "output points        :",
        len(output),
    )

    print()
    print(
        "largest adjacent joint change"
    )

    print(
        "  path_point : {} -> {}".format(
            max_from,
            max_from + 1,
        )
    )

    print(
        "  joint      :",
        max_joint,
    )

    print(
        "  delta      : {:.6f} deg".format(
            max_delta
        )
    )

    print()
    print("OUTPUT:")
    print(
        " ",
        OUTPUT_CSV,
    )

    print()
    print(
        "SPLIT-MOTION FULL PATH BUILD: SUCCESS"
    )


if __name__ == "__main__":
    main()
