#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import math

import numpy as np
import rospkg
import rospy
import yaml

from geometry_msgs.msg import PolygonStamped

import cobotta_grasp_candidate_area as area
import cobotta_p0_candidates as p0mod
import cobotta_pregrasp_candidates as premod


PAPER_TOPIC = "/origami/active_folding_paper_t0_ros"

TOL_A = 2.0e-5
TOL_B = 3.0e-5


def clip_polygon_raw(subject, clipper):
    """
    本線 clip_polygon() と同じ処理だが、
    最後の normalize_polygon() は行わない。
    """
    if len(subject) < 3 or len(clipper) < 3:
        return []

    output = list(subject)

    orientation = (
        1.0
        if area.polygon_signed_area(clipper) >= 0.0
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
            e_inside = area.inside(
                e, a, b, orientation
            )

            s_inside = area.inside(
                s, a, b, orientation
            )

            if e_inside:
                if not s_inside:
                    output.append(
                        area.line_intersection(
                            s, e, a, b
                        )
                    )

                output.append(e)

            elif s_inside:
                output.append(
                    area.line_intersection(
                        s, e, a, b
                    )
                )

            s = e

    return output


def rotation_difference_deg(R1, R2):
    R = R1.T.dot(R2)

    value = (
        np.trace(R) - 1.0
    ) / 2.0

    value = np.clip(
        value,
        -1.0,
        1.0,
    )

    return math.degrees(
        math.acos(value)
    )


def build_pre_candidates(p0, paper):
    normal = (
        premod.PreGraspCandidates.paper_normal(
            paper
        )
    )

    result = {}

    for edge_id in range(len(paper)):
        a = paper[edge_id]
        b = paper[
            (edge_id + 1) % len(paper)
        ]

        e, _ = (
            premod.PreGraspCandidates
            .closest_point_on_segment(
                p0,
                a,
                b,
            )
        )

        entry = p0 - e
        d_edge = np.linalg.norm(entry)

        if d_edge < 1.0e-9:
            continue

        z_direction = entry / d_edge

        for sign, sign_name in [
            (+1.0, "N+"),
            (-1.0, "N-"),
        ]:
            R = (
                premod.PreGraspCandidates
                .make_rotation(
                    sign * normal,
                    z_direction,
                )
            )

            z_tool = R[:, 2]

            d_pre = (
                d_edge
                + premod.L_FRONT
            )

            pre_grasp_point = (
                p0
                - d_pre * z_tool
            )

            pre_tool = (
                premod.PreGraspCandidates
                .tool_position_from_grasp(
                    pre_grasp_point,
                    R,
                )
            )

            result[
                (edge_id, sign_name)
            ] = {
                "pre_tool": pre_tool,
                "R": R,
            }

    return result


def make_p0_candidates(
    safe_mm,
    plane,
):
    safe_m = [
        (
            x / 1000.0,
            y / 1000.0,
        )
        for x, y in safe_mm
    ]

    candidates_2d = (
        p0mod.generate_inner_candidates(
            safe_m
        )
    )

    candidates_3d = []

    for x, y in candidates_2d:
        z = p0mod.solve_z_on_plane(
            x,
            y,
            plane,
        )

        if z is None:
            raise RuntimeError(
                "Could not reconstruct P0 z."
            )

        candidates_3d.append(
            np.array(
                [x, y, z],
                dtype=float,
            )
        )

    return candidates_3d


def nearest_index(point, candidates):
    distances = [
        np.linalg.norm(
            point - candidate
        )
        for candidate in candidates
    ]

    return int(
        np.argmin(distances)
    )


def main():
    rospy.init_node(
        "cobotta_tolerance_pose_sensitivity_diagnostic"
    )

    package_path = Path(
        rospkg.RosPack().get_path(
            "origami_grasp_bridge"
        )
    )

    config_path = (
        package_path
        / "config"
        / "cobotta_stand_access_zones_v1.yaml"
    )

    with config_path.open() as f:
        config = yaml.safe_load(f)

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

    size_x = float(
        config["stand_top"]["size_x"]
    )

    size_y = float(
        config["stand_top"]["size_y"]
    )

    center_x = size_x / 2.0
    center_y = size_y / 2.0

    lower_access = []

    for stand_x, stand_y in (
        config["access_zones"]["lower"][
            "vertices"
        ]
    ):
        lower_access.append((
            center_x - float(stand_x),
            center_y - float(stand_y),
        ))

    print(
        "Waiting for T0 paper..."
    )

    msg = rospy.wait_for_message(
        PAPER_TOPIC,
        PolygonStamped,
        timeout=10.0,
    )

    paper = np.array(
        [
            [p.x, p.y, p.z]
            for p in msg.polygon.points
        ],
        dtype=float,
    )

    paper_mm = [
        (
            p.x * 1000.0,
            p.y * 1000.0,
        )
        for p in msg.polygon.points
    ]

    safe_paper = (
        area.shrink_convex_polygon(
            paper_mm,
            paper_margin_mm,
        )
    )

    safe_lower_access = (
        area.shrink_convex_polygon(
            lower_access,
            access_margin_mm,
        )
    )

    raw = clip_polygon_raw(
        safe_paper,
        safe_lower_access,
    )

    safe_a = area.normalize_polygon(
        raw,
        relative_tolerance=TOL_A,
    )

    safe_b = area.normalize_polygon(
        raw,
        relative_tolerance=TOL_B,
    )

    plane = p0mod.find_plane(
        [
            tuple(p)
            for p in paper
        ]
    )

    if plane is None:
        raise RuntimeError(
            "Could not determine T0 paper plane."
        )

    p0_a = make_p0_candidates(
        safe_a,
        plane,
    )

    p0_b = make_p0_candidates(
        safe_b,
        plane,
    )

    print()
    print(
        "===== Tolerance Pose Sensitivity ====="
    )
    print(
        "A : {:.1e} -> {} vertices, {} P0".format(
            TOL_A,
            len(safe_a),
            len(p0_a),
        )
    )
    print(
        "B : {:.1e} -> {} vertices, {} P0".format(
            TOL_B,
            len(safe_b),
            len(p0_b),
        )
    )

    print()
    print(
        "{:<7s} {:<7s} {:>14s} {:>18s} {:>18s}".format(
            "A_P0",
            "B_P0",
            "P0_diff[um]",
            "max_PRE_diff[um]",
            "max_ori_diff[deg]",
        )
    )

    print(
        "------- ------- -------------- "
        "------------------ ------------------"
    )

    global_max_p0_um = 0.0
    global_max_pre_um = 0.0
    global_max_ori_deg = 0.0

    for i, pa in enumerate(p0_a):
        j = nearest_index(
            pa,
            p0_b,
        )

        pb = p0_b[j]

        p0_diff_um = (
            np.linalg.norm(
                pa - pb
            )
            * 1.0e6
        )

        pre_a = build_pre_candidates(
            pa,
            paper,
        )

        pre_b = build_pre_candidates(
            pb,
            paper,
        )

        common_keys = sorted(
            set(pre_a.keys())
            & set(pre_b.keys())
        )

        max_pre_um = 0.0
        max_ori_deg = 0.0

        for key in common_keys:
            pre_diff_um = (
                np.linalg.norm(
                    pre_a[key]["pre_tool"]
                    - pre_b[key]["pre_tool"]
                )
                * 1.0e6
            )

            ori_diff_deg = (
                rotation_difference_deg(
                    pre_a[key]["R"],
                    pre_b[key]["R"],
                )
            )

            max_pre_um = max(
                max_pre_um,
                pre_diff_um,
            )

            max_ori_deg = max(
                max_ori_deg,
                ori_diff_deg,
            )

        global_max_p0_um = max(
            global_max_p0_um,
            p0_diff_um,
        )

        global_max_pre_um = max(
            global_max_pre_um,
            max_pre_um,
        )

        global_max_ori_deg = max(
            global_max_ori_deg,
            max_ori_deg,
        )

        print(
            "{:<7d} {:<7d} {:>14.6f} "
            "{:>18.6f} {:>18.9f}".format(
                i,
                j,
                p0_diff_um,
                max_pre_um,
                max_ori_deg,
            )
        )

    print()
    print(
        "===== Maximum Difference ====="
    )
    print(
        "P0 position      : {:.6f} um".format(
            global_max_p0_um
        )
    )
    print(
        "PRE tool position: {:.6f} um".format(
            global_max_pre_um
        )
    )
    print(
        "PRE orientation  : {:.9f} deg".format(
            global_max_ori_deg
        )
    )


if __name__ == "__main__":
    main()
