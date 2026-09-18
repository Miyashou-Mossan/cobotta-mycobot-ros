#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import math

import rospkg
import rospy
import yaml

from geometry_msgs.msg import PolygonStamped


PAPER_TOPIC = "/origami/active_folding_paper_t0_ros"

RELATIVE_TOLERANCES = [
    1.0e-6,
    2.0e-6,
    5.0e-6,
    1.0e-5,
    2.0e-5,
    3.0e-5,
    5.0e-5,
    1.0e-4,
]


# ============================================================
# grasp_candidate_area.py と同じ幾何処理
# ============================================================

def polygon_signed_area(poly):
    return 0.5 * sum(
        poly[i][0] * poly[(i + 1) % len(poly)][1]
        - poly[(i + 1) % len(poly)][0] * poly[i][1]
        for i in range(len(poly))
    )


def inside(point, edge_start, edge_end, orientation):
    cross = (
        (edge_end[0] - edge_start[0])
        * (point[1] - edge_start[1])
        -
        (edge_end[1] - edge_start[1])
        * (point[0] - edge_start[0])
    )

    return orientation * cross >= -1.0e-9


def line_intersection(s, e, a, b):
    x1, y1 = s
    x2, y2 = e
    x3, y3 = a
    x4, y4 = b

    denominator = (
        (x1 - x2) * (y3 - y4)
        -
        (y1 - y2) * (x3 - x4)
    )

    if abs(denominator) < 1.0e-12:
        return e

    px = (
        (x1 * y2 - y1 * x2)
        * (x3 - x4 * 0.0)
    )

    # 上の式は使わず、元コードと同じ式で計算
    px = (
        (x1 * y2 - y1 * x2) * (x3 - x4)
        -
        (x1 - x2) * (x3 * y4 - y3 * x4)
    ) / denominator

    py = (
        (x1 * y2 - y1 * x2) * (y3 - y4)
        -
        (y1 - y2) * (x3 * y4 - y3 * x4)
    ) / denominator

    return (px, py)


def shrink_convex_polygon(poly, margin):
    if len(poly) < 3:
        return []

    ccw = polygon_signed_area(poly) > 0.0
    shifted_edges = []

    for i in range(len(poly)):
        a = poly[i]
        b = poly[(i + 1) % len(poly)]

        dx = b[0] - a[0]
        dy = b[1] - a[1]

        length = math.hypot(dx, dy)

        if length < 1.0e-12:
            return []

        if ccw:
            nx = -dy / length
            ny = dx / length
        else:
            nx = dy / length
            ny = -dx / length

        shifted_edges.append((
            (
                a[0] + nx * margin,
                a[1] + ny * margin,
            ),
            (
                b[0] + nx * margin,
                b[1] + ny * margin,
            ),
        ))

    result = []

    for i in range(len(shifted_edges)):
        prev_edge = shifted_edges[i - 1]
        this_edge = shifted_edges[i]

        point = line_intersection(
            prev_edge[0],
            prev_edge[1],
            this_edge[0],
            this_edge[1],
        )

        result.append(point)

    return result


def clip_polygon_raw(subject, clipper):
    """
    本線clip_polygon()と同じだが、
    最後のnormalize_polygon()だけ行わない。
    """
    if len(subject) < 3 or len(clipper) < 3:
        return []

    output = list(subject)

    orientation = (
        1.0
        if polygon_signed_area(clipper) >= 0.0
        else -1.0
    )

    for i in range(len(clipper)):
        a = clipper[i]
        b = clipper[(i + 1) % len(clipper)]

        input_list = output
        output = []

        if not input_list:
            break

        s = input_list[-1]

        for e in input_list:
            e_inside = inside(
                e, a, b, orientation
            )

            s_inside = inside(
                s, a, b, orientation
            )

            if e_inside:
                if not s_inside:
                    output.append(
                        line_intersection(
                            s, e, a, b
                        )
                    )

                output.append(e)

            elif s_inside:
                output.append(
                    line_intersection(
                        s, e, a, b
                    )
                )

            s = e

    return output


def characteristic_length(points):
    if len(points) < 2:
        return 0.0

    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_y = max(p[1] for p in points)

    return max(
        max_x - min_x,
        max_y - min_y,
    )


def normalize_polygon(
    points,
    relative_tolerance,
):
    if len(points) < 2:
        return list(points)

    length = characteristic_length(
        points
    )

    if length <= 0.0:
        return list(points)

    tolerance = (
        length * relative_tolerance
    )

    def distance(a, b):
        return math.hypot(
            a[0] - b[0],
            a[1] - b[1],
        )

    normalized = []

    for point in points:
        if (
            not normalized
            or distance(
                normalized[-1],
                point
            ) >= tolerance
        ):
            normalized.append(point)

    if (
        len(normalized) >= 2
        and distance(
            normalized[0],
            normalized[-1]
        ) < tolerance
    ):
        normalized.pop()

    return normalized


# ============================================================
# p0_candidates.py と同じ候補生成
# ============================================================

def polygon_centroid(points):
    if len(points) < 3:
        return None

    cross_sum = 0.0
    cx_sum = 0.0
    cy_sum = 0.0

    for i in range(len(points)):
        x0, y0 = points[i]
        x1, y1 = points[
            (i + 1) % len(points)
        ]

        cross = (
            x0 * y1
            - x1 * y0
        )

        cross_sum += cross
        cx_sum += (x0 + x1) * cross
        cy_sum += (y0 + y1) * cross

    if abs(cross_sum) < 1.0e-12:
        return (
            sum(p[0] for p in points)
            / len(points),
            sum(p[1] for p in points)
            / len(points),
        )

    return (
        cx_sum / (3.0 * cross_sum),
        cy_sum / (3.0 * cross_sum),
    )


def generate_inner_candidates(points):
    center = polygon_centroid(points)

    if center is None:
        return []

    candidates = [center]

    # 重心 → 各頂点
    for vx, vy in points:
        candidates.append((
            0.5 * (center[0] + vx),
            0.5 * (center[1] + vy),
        ))

    # 重心 → 各辺中央
    for i in range(len(points)):
        x0, y0 = points[i]
        x1, y1 = points[
            (i + 1) % len(points)
        ]

        edge_mid = (
            0.5 * (x0 + x1),
            0.5 * (y0 + y1),
        )

        candidates.append((
            0.5 * (
                center[0]
                + edge_mid[0]
            ),
            0.5 * (
                center[1]
                + edge_mid[1]
            ),
        ))

    return candidates


# ============================================================
# 診断用
# ============================================================

def distance(a, b):
    return math.hypot(
        a[0] - b[0],
        a[1] - b[1],
    )


def min_edge_length(points):
    if len(points) < 2:
        return float("nan")

    return min(
        distance(
            points[i],
            points[
                (i + 1) % len(points)
            ],
        )
        for i in range(len(points))
    )


def symmetric_hausdorff(a, b):
    """
    候補数が違っても比較できるよう、
    2つのP0集合間の最大最近傍距離を使う。
    """
    if not a or not b:
        return float("nan")

    a_to_b = max(
        min(
            distance(pa, pb)
            for pb in b
        )
        for pa in a
    )

    b_to_a = max(
        min(
            distance(pb, pa)
            for pa in a
        )
        for pb in b
    )

    return max(
        a_to_b,
        b_to_a,
    )


def main():
    rospy.init_node(
        "cobotta_polygon_tolerance_sweep_diagnostic"
    )

    package_path = Path(
        rospkg.RosPack().get_path(
            "origami_grasp_bridge"
        )
    )

    default_config = (
        package_path
        / "config"
        / "cobotta_stand_access_zones_v1.yaml"
    )

    config_path = Path(
        rospy.get_param(
            "~access_zone_config",
            str(default_config),
        )
    )

    paper_margin_mm = float(
        rospy.get_param(
            "~paper_margin_mm",
            0.0,
        )
    )

    access_margin_mm = float(
        rospy.get_param(
            "~access_margin_mm",
            10.0,
        )
    )

    with config_path.open() as f:
        config = yaml.safe_load(f)

    size_x = float(
        config["stand_top"]["size_x"]
    )

    size_y = float(
        config["stand_top"]["size_y"]
    )

    center_x = size_x / 2.0
    center_y = size_y / 2.0

    access_zones = {}

    for name, zone in (
        config["access_zones"].items()
    ):
        converted = []

        for stand_x, stand_y in zone["vertices"]:
            converted.append((
                center_x - float(stand_x),
                center_y - float(stand_y),
            ))

        access_zones[name] = converted

    rospy.loginfo(
        "Waiting for T0 paper..."
    )

    paper_msg = rospy.wait_for_message(
        PAPER_TOPIC,
        PolygonStamped,
        timeout=10.0,
    )

    paper_mm = [
        (
            p.x * 1000.0,
            p.y * 1000.0,
        )
        for p in paper_msg.polygon.points
    ]

    safe_paper = shrink_convex_polygon(
        paper_mm,
        paper_margin_mm,
    )

    safe_lower_access = (
        shrink_convex_polygon(
            access_zones["lower"],
            access_margin_mm,
        )
    )

    raw = clip_polygon_raw(
        safe_paper,
        safe_lower_access,
    )

    if len(raw) < 3:
        raise RuntimeError(
            "Raw SAFE lower polygon has <3 vertices."
        )

    length_mm = characteristic_length(
        raw
    )

    closure_gap_mm = distance(
        raw[0],
        raw[-1],
    )

    print()
    print(
        "===== Polygon Tolerance Sweep ====="
    )

    print(
        "raw vertex count        :",
        len(raw),
    )

    print(
        "bbox characteristic L   : "
        "{:.9f} mm".format(
            length_mm
        )
    )

    print(
        "first-last gap          : "
        "{:.6f} um".format(
            closure_gap_mm * 1000.0
        )
    )

    if length_mm > 0.0:
        print(
            "gap / L                : "
            "{:.9e}".format(
                closure_gap_mm
                / length_mm
            )
        )

    print()
    print("Raw vertices [mm]:")

    for i, p in enumerate(raw):
        print(
            "  [{}] ({:.9f}, {:.9f})".format(
                i,
                p[0],
                p[1],
            )
        )

    print()
    print(
        "{:<11s} | {:>10s} | {:>8s} | "
        "{:>12s} | {:>8s} | {:>14s}".format(
            "rel_tol",
            "eps[um]",
            "vertices",
            "min_edge[mm]",
            "P0",
            "dP0_prev[um]",
        )
    )

    print(
        "------------+------------+----------+"
        "--------------+----------+----------------"
    )

    previous_p0 = None

    for rel_tol in RELATIVE_TOLERANCES:
        normalized = normalize_polygon(
            raw,
            rel_tol,
        )

        p0_candidates = (
            generate_inner_candidates(
                normalized
            )
        )

        epsilon_um = (
            length_mm
            * rel_tol
            * 1000.0
        )

        minimum_edge = (
            min_edge_length(
                normalized
            )
        )

        if previous_p0 is None:
            delta_text = "-"
        else:
            delta_um = (
                symmetric_hausdorff(
                    previous_p0,
                    p0_candidates,
                )
                * 1000.0
            )

            delta_text = "{:.6f}".format(
                delta_um
            )

        print(
            "{:<11.1e} | {:>10.6f} | {:>8d} | "
            "{:>12.6f} | {:>8d} | {:>14s}".format(
                rel_tol,
                epsilon_um,
                len(normalized),
                minimum_edge,
                len(p0_candidates),
                delta_text,
            )
        )

        previous_p0 = p0_candidates

    print()
    print(
        "NOTE:"
    )
    print(
        "dP0_prev is the symmetric nearest-neighbor "
        "distance between consecutive tolerance settings."
    )
    print(
        "It is a sensitivity indicator, not an "
        "acceptance threshold."
    )


if __name__ == "__main__":
    main()
