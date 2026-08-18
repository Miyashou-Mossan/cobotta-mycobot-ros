#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import math
import sys
from pathlib import Path

import yaml


INPUT_YAML = Path(
    "/home/maeda/"
    "directionA_reverse_full_fold_current_pose_20260817_160945_multidof.yaml"
)


def normalize(q):
    n = math.sqrt(sum(v * v for v in q))
    if n <= 1.0e-12:
        raise RuntimeError("Quaternion norm is zero")
    return [v / n for v in q]


def multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    return [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: cobotta_apply_local_z_offset.py OFFSET_DEG"
        )

    offset_deg = float(sys.argv[1])
    angle = math.radians(offset_deg)

    # 工具ローカルZ軸まわりの固定回転
    q_offset = [
        0.0,
        0.0,
        math.sin(angle / 2.0),
        math.cos(angle / 2.0),
    ]

    with INPUT_YAML.open("r", encoding="utf-8") as f:
        document = yaml.safe_load(f)

    points = document.get("points", [])

    if not points:
        raise RuntimeError("points is empty")

    output = copy.deepcopy(document)

    for point in output["points"]:
        transform = point["transforms"][0]
        rotation = transform["rotation"]

        q = normalize([
            float(rotation["x"]),
            float(rotation["y"]),
            float(rotation["z"]),
            float(rotation["w"]),
        ])

        # q_current * q_offset
        # → 現在の工具ローカル座標系Z軸まわりに回転
        q_new = normalize(
            multiply(q, q_offset)
        )

        rotation["x"] = q_new[0]
        rotation["y"] = q_new[1]
        rotation["z"] = q_new[2]
        rotation["w"] = q_new[3]

    tag = (
        "p{:02d}".format(int(round(offset_deg)))
        if offset_deg >= 0
        else "m{:02d}".format(abs(int(round(offset_deg))))
    )

    output_path = Path(
        "/home/maeda/"
        "directionA_reverse_local_z_{}_multidof.yaml".format(tag)
    )

    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            output,
            f,
            sort_keys=False,
            allow_unicode=True,
        )

    print("===== Local-Z offset trajectory =====")
    print("offset     : {:+.3f} deg".format(offset_deg))
    print("points     :", len(points))
    print("input      :", INPUT_YAML)
    print("output     :", output_path)


if __name__ == "__main__":
    main()
