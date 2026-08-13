#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import struct
import numpy as np
from pathlib import Path


SRC = Path(
    "/home/maeda/catkin_ws/src/"
    "dual_robot_description/meshes/mycobot/tool/"
    "mycobot_tips.STL"
)

OUT_FORBIDDEN = SRC.parent / "mycobot_tips_forbidden.STL"
OUT_ALLOWED = SRC.parent / "mycobot_tips_allowed_contact.STL"


# 接触許容対象となる斜面triangle
SLOPE_TRIANGLES = set(range(1936, 1944))

# 接触辺は y=0
# 奥端は y=11.18 mm
# 先端側50%なので境界は y=5.59 mm
Y_SPLIT = 5.59


def read_binary_stl(path):
    triangles = []

    with open(path, "rb") as f:
        header = f.read(80)
        count = struct.unpack("<I", f.read(4))[0]

        for _ in range(count):
            data = f.read(50)
            vals = struct.unpack("<12fH", data)

            normal = np.array(
                vals[0:3],
                dtype=float
            )

            vertices = np.array([
                vals[3:6],
                vals[6:9],
                vals[9:12]
            ], dtype=float)

            triangles.append(
                (normal, vertices)
            )

    return header, triangles


def calc_normal(tri):
    a, b, c = tri

    n = np.cross(
        b - a,
        c - a
    )

    norm = np.linalg.norm(n)

    if norm > 1e-12:
        n = n / norm
    else:
        n = np.zeros(3)

    return n


def write_binary_stl(path, triangles):
    header = (
        b"MyCobot collision mesh generated automatically"
    ).ljust(80, b" ")

    with open(path, "wb") as f:
        f.write(header)
        f.write(
            struct.pack(
                "<I",
                len(triangles)
            )
        )

        for tri in triangles:
            normal = calc_normal(tri)

            vals = (
                list(normal) +
                list(tri[0]) +
                list(tri[1]) +
                list(tri[2])
            )

            f.write(
                struct.pack(
                    "<12fH",
                    *vals,
                    0
                )
            )


def clip_polygon_y(vertices, keep_less_equal):
    """
    三角形を y=Y_SPLIT で切る。

    keep_less_equal=True:
        y <= Y_SPLIT を残す（先端側）

    False:
        y >= Y_SPLIT を残す（奥側）
    """

    polygon = [
        np.array(v, dtype=float)
        for v in vertices
    ]

    result = []

    def inside(p):
        if keep_less_equal:
            return p[1] <= Y_SPLIT + 1e-9
        return p[1] >= Y_SPLIT - 1e-9

    def intersection(a, b):
        dy = b[1] - a[1]

        if abs(dy) < 1e-12:
            return a.copy()

        t = (
            Y_SPLIT - a[1]
        ) / dy

        return a + t * (b - a)

    for i in range(len(polygon)):
        current = polygon[i]
        previous = polygon[i - 1]

        current_inside = inside(current)
        previous_inside = inside(previous)

        if current_inside:

            if not previous_inside:
                result.append(
                    intersection(
                        previous,
                        current
                    )
                )

            result.append(current)

        elif previous_inside:

            result.append(
                intersection(
                    previous,
                    current
                )
            )

    return result


def triangulate_polygon(poly):
    if len(poly) < 3:
        return []

    tris = []

    for i in range(
        1,
        len(poly) - 1
    ):
        tris.append(
            np.array([
                poly[0],
                poly[i],
                poly[i + 1]
            ])
        )

    return tris


def main():
    _, source_triangles = read_binary_stl(
        SRC
    )

    forbidden = []
    allowed = []

    for index, (_, vertices) in enumerate(
        source_triangles
    ):

        # 斜面以外はすべて禁止Collisionへ
        if index not in SLOPE_TRIANGLES:
            forbidden.append(
                vertices
            )
            continue

        # 接触斜面を50%で切る
        allowed_poly = clip_polygon_y(
            vertices,
            keep_less_equal=True
        )

        forbidden_poly = clip_polygon_y(
            vertices,
            keep_less_equal=False
        )

        allowed.extend(
            triangulate_polygon(
                allowed_poly
            )
        )

        forbidden.extend(
            triangulate_polygon(
                forbidden_poly
            )
        )

    write_binary_stl(
        OUT_FORBIDDEN,
        forbidden
    )

    write_binary_stl(
        OUT_ALLOWED,
        allowed
    )

    print("")
    print("===== MyCobot collision mesh split =====")
    print("source triangles    :", len(source_triangles))
    print("forbidden triangles :", len(forbidden))
    print("allowed triangles   :", len(allowed))

    print("")
    print("split boundary:")
    print(
        "  y = {:.3f} mm".format(
            Y_SPLIT
        )
    )

    print("")
    print("forbidden:")
    print(" ", OUT_FORBIDDEN)

    print("")
    print("allowed:")
    print(" ", OUT_ALLOWED)


if __name__ == "__main__":
    main()
