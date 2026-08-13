#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import numpy as np


CSV_PATH = "/home/maeda/20230214OrigamiSim/Assets/folding_face_90.csv"


def unity_to_ros(x_u, y_u, z_u):
    return np.array([
        z_u,
        -x_u,
        y_u
    ], dtype=float)


def load_csv(path):
    fixed_center_unity = None
    data_lines = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("# fixed_paper_center_m="):
                values = line.split("=", 1)[1].split(",")

                fixed_center_unity = np.array([
                    float(values[0]),
                    float(values[1]),
                    float(values[2])
                ])

            elif not line.startswith("#") and line:
                data_lines.append(line)

    if fixed_center_unity is None:
        raise RuntimeError(
            "fixed_paper_center_m not found"
        )

    reader = csv.DictReader(data_lines)

    points = []

    for row in reader:
        p = unity_to_ros(
            float(row["x_m"]),
            float(row["y_m"]),
            float(row["z_m"])
        )

        points.append(p)

    if len(points) != 3:
        raise RuntimeError(
            "Expected 3 folding-face points, got {}".format(
                len(points)
            )
        )

    fixed_center_ros = unity_to_ros(
        fixed_center_unity[0],
        fixed_center_unity[1],
        fixed_center_unity[2]
    )

    return points, fixed_center_ros


def main():
    points, fixed_center = load_csv(
        CSV_PATH
    )

    p0, p1, p2 = points

    # 90deg moving-paper plane
    normal = np.cross(
        p1 - p0,
        p2 - p0
    )

    norm = np.linalg.norm(normal)

    if norm < 1e-9:
        raise RuntimeError(
            "Degenerate folding face"
        )

    normal = normal / norm

    # 固定側紙面中心のsigned distance
    fixed_signed_distance = np.dot(
        normal,
        fixed_center - p0
    )

    # fixedPaper側を正方向に統一する
    if fixed_signed_distance < 0.0:
        normal = -normal
        fixed_signed_distance = -fixed_signed_distance

    print("===== 90deg folding boundary =====")

    for i, p in enumerate(points):
        print(
            "P{} = ({:+.6f}, {:+.6f}, {:+.6f})".format(
                i,
                p[0],
                p[1],
                p[2]
            )
        )

    print()
    print(
        "fixed center = "
        "({:+.6f}, {:+.6f}, {:+.6f})".format(
            fixed_center[0],
            fixed_center[1],
            fixed_center[2]
        )
    )

    print()
    print(
        "boundary normal toward FIXED side = "
        "({:+.6f}, {:+.6f}, {:+.6f})".format(
            normal[0],
            normal[1],
            normal[2]
        )
    )

    print(
        "fixed-side distance = {:.3f} mm".format(
            fixed_signed_distance * 1000.0
        )
    )

    print()
    print("Definition:")
    print("  signed distance > 0 : FIXED PAPER SIDE")
    print("  signed distance = 0 : 90deg moving-paper plane")
    print("  signed distance < 0 : MOVING PAPER SIDE")


if __name__ == "__main__":
    main()
