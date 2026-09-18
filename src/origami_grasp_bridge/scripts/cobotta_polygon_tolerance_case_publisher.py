#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import math
import sys

import rospkg
import rospy
import yaml

from geometry_msgs.msg import Point32, PolygonStamped


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import cobotta_grasp_candidate_area as base


PAPER_TOPIC = "/origami/active_folding_paper_t0_ros"

SAFE_UPPER_TOPIC = (
    "/origami/diagnostic/tolerance_case/"
    "safe_grasp_area_upper"
)

SAFE_LOWER_TOPIC = (
    "/origami/diagnostic/tolerance_case/"
    "safe_grasp_area_lower"
)


def clip_polygon_raw(subject, clipper):
    """
    cobotta_grasp_candidate_area.py の clip_polygon と同じ処理。
    ただし最後の normalize_polygon() は行わない。
    """
    if len(subject) < 3 or len(clipper) < 3:
        return []

    output = list(subject)

    orientation = (
        1.0
        if base.polygon_signed_area(clipper) >= 0.0
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
            e_inside = base.inside(
                e, a, b, orientation
            )

            s_inside = base.inside(
                s, a, b, orientation
            )

            if e_inside:
                if not s_inside:
                    output.append(
                        base.line_intersection(
                            s, e, a, b
                        )
                    )

                output.append(e)

            elif s_inside:
                output.append(
                    base.line_intersection(
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


def distance(a, b):
    return math.hypot(
        a[0] - b[0],
        a[1] - b[1],
    )


def make_polygon_message(points_mm, stamp):
    msg = PolygonStamped()

    msg.header.frame_id = "paper_center"
    msg.header.stamp = stamp

    for x_mm, y_mm in points_mm:
        msg.polygon.points.append(
            Point32(
                x=x_mm / 1000.0,
                y=y_mm / 1000.0,
                z=0.0,
            )
        )

    return msg


class ToleranceCasePublisher:

    def __init__(self):
        self.relative_tolerance = float(
            rospy.get_param(
                "~relative_tolerance",
                2.0e-5,
            )
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

        self.paper_margin_mm = float(
            rospy.get_param(
                "~paper_margin_mm",
                0.0,
            )
        )

        self.access_margin_mm = float(
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

        self.access_zones = {}

        for name, zone in (
            config["access_zones"].items()
        ):
            converted = []

            for stand_x, stand_y in zone["vertices"]:
                converted.append((
                    center_x - float(stand_x),
                    center_y - float(stand_y),
                ))

            self.access_zones[name] = converted

        self.upper_pub = rospy.Publisher(
            SAFE_UPPER_TOPIC,
            PolygonStamped,
            queue_size=1,
            latch=True,
        )

        self.lower_pub = rospy.Publisher(
            SAFE_LOWER_TOPIC,
            PolygonStamped,
            queue_size=1,
            latch=True,
        )

        rospy.Subscriber(
            PAPER_TOPIC,
            PolygonStamped,
            self.paper_cb,
            queue_size=1,
        )

        rospy.loginfo(
            "Tolerance case publisher started: "
            "relative_tolerance=%.9e",
            self.relative_tolerance,
        )

    def paper_cb(self, msg):
        paper_mm = [
            (
                p.x * 1000.0,
                p.y * 1000.0,
            )
            for p in msg.polygon.points
        ]

        safe_paper = (
            base.shrink_convex_polygon(
                paper_mm,
                self.paper_margin_mm,
            )
        )

        safe_upper_access = (
            base.shrink_convex_polygon(
                self.access_zones["upper"],
                self.access_margin_mm,
            )
        )

        safe_lower_access = (
            base.shrink_convex_polygon(
                self.access_zones["lower"],
                self.access_margin_mm,
            )
        )

        raw_upper = clip_polygon_raw(
            safe_paper,
            safe_upper_access,
        )

        raw_lower = clip_polygon_raw(
            safe_paper,
            safe_lower_access,
        )

        safe_upper = base.normalize_polygon(
            raw_upper,
            relative_tolerance=(
                self.relative_tolerance
            ),
        )

        safe_lower = base.normalize_polygon(
            raw_lower,
            relative_tolerance=(
                self.relative_tolerance
            ),
        )

        self.upper_pub.publish(
            make_polygon_message(
                safe_upper,
                msg.header.stamp,
            )
        )

        self.lower_pub.publish(
            make_polygon_message(
                safe_lower,
                msg.header.stamp,
            )
        )

        lower_L = characteristic_length(
            raw_lower
        )

        if len(raw_lower) >= 2:
            lower_gap_um = (
                distance(
                    raw_lower[0],
                    raw_lower[-1],
                )
                * 1000.0
            )
        else:
            lower_gap_um = float("nan")

        epsilon_um = (
            lower_L
            * self.relative_tolerance
            * 1000.0
        )

        print()
        print(
            "===== Tolerance Case ====="
        )
        print(
            "relative tolerance : {:.9e}".format(
                self.relative_tolerance
            )
        )
        print(
            "lower L            : {:.9f} mm".format(
                lower_L
            )
        )
        print(
            "epsilon            : {:.6f} um".format(
                epsilon_um
            )
        )
        print(
            "first-last gap     : {:.6f} um".format(
                lower_gap_um
            )
        )
        print(
            "raw lower vertices : {}".format(
                len(raw_lower)
            )
        )
        print(
            "safe lower vertices: {}".format(
                len(safe_lower)
            )
        )


def main():
    rospy.init_node(
        "cobotta_polygon_tolerance_case_publisher"
    )

    ToleranceCasePublisher()

    rospy.spin()


if __name__ == "__main__":
    main()
