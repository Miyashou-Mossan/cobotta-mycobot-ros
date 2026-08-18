#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import math
from pathlib import Path

import yaml


INPUT_YAML = Path(
    "/home/maeda/"
    "directionA_reverse_local_z_m25_multidof.yaml"
)

OUTPUT_YAML = Path(
    "/home/maeda/"
    "directionA_reverse_local_z_m25_actual_grasp_multidof.yaml"
)

# cobotta_tool_link -> actual_grasp_point
# tool coordinate frame [m]
R_GRASP = [
    +0.000401,
    -0.001507,
    -0.004894,
]


def normalize(q):
    n = math.sqrt(sum(v * v for v in q))
    if n <= 1.0e-12:
        raise RuntimeError("Quaternion norm is zero")
    return [v / n for v in q]


def rotate_vector(q, v):
    x, y, z, w = normalize(q)
    vx, vy, vz = v

    # Quaternion -> rotation matrix
    r00 = 1.0 - 2.0 * (y*y + z*z)
    r01 = 2.0 * (x*y - z*w)
    r02 = 2.0 * (x*z + y*w)

    r10 = 2.0 * (x*y + z*w)
    r11 = 1.0 - 2.0 * (x*x + z*z)
    r12 = 2.0 * (y*z - x*w)

    r20 = 2.0 * (x*z - y*w)
    r21 = 2.0 * (y*z + x*w)
    r22 = 1.0 - 2.0 * (x*x + y*y)

    return [
        r00*vx + r01*vy + r02*vz,
        r10*vx + r11*vy + r12*vz,
        r20*vx + r21*vy + r22*vz,
    ]


def main():
    with INPUT_YAML.open("r", encoding="utf-8") as f:
        document = yaml.safe_load(f)

    points = document.get("points", [])

    if not points:
        raise RuntimeError("points is empty")

    output = copy.deepcopy(document)

    print("===== actual grasp-point offset =====")
    print(
        "r_tool = ({:+.3f}, {:+.3f}, {:+.3f}) mm".format(
            R_GRASP[0] * 1000.0,
            R_GRASP[1] * 1000.0,
            R_GRASP[2] * 1000.0,
        )
    )

    for index, point in enumerate(output["points"]):
        tf = point["transforms"][0]

        t = tf["translation"]
        r = tf["rotation"]

        q = [
            float(r["x"]),
            float(r["y"]),
            float(r["z"]),
            float(r["w"]),
        ]

        # actual_grasp_point = Unity paper point
        #
        # p_grasp = p_tool + R_tool * r
        #
        # therefore
        #
        # p_tool = p_paper - R_tool * r
        world_offset = rotate_vector(
            q,
            R_GRASP,
        )

        old_position = [
            float(t["x"]),
            float(t["y"]),
            float(t["z"]),
        ]

        new_position = [
            old_position[i] - world_offset[i]
            for i in range(3)
        ]

        t["x"] = new_position[0]
        t["y"] = new_position[1]
        t["z"] = new_position[2]

        if index in [0, len(points) - 1]:
            print()
            print("index =", index)

            print(
                "paper/grasp : "
                "({:+.3f}, {:+.3f}, {:+.3f}) mm".format(
                    old_position[0] * 1000.0,
                    old_position[1] * 1000.0,
                    old_position[2] * 1000.0,
                )
            )

            print(
                "tool target : "
                "({:+.3f}, {:+.3f}, {:+.3f}) mm".format(
                    new_position[0] * 1000.0,
                    new_position[1] * 1000.0,
                    new_position[2] * 1000.0,
                )
            )

            print(
                "shift       : "
                "({:+.3f}, {:+.3f}, {:+.3f}) mm".format(
                    -world_offset[0] * 1000.0,
                    -world_offset[1] * 1000.0,
                    -world_offset[2] * 1000.0,
                )
            )

    with OUTPUT_YAML.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            output,
            f,
            sort_keys=False,
            allow_unicode=True,
        )

    print()
    print("===== generated =====")
    print("points :", len(points))
    print("input  :", INPUT_YAML)
    print("output :", OUTPUT_YAML)


if __name__ == "__main__":
    main()
