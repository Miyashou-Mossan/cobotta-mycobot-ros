#!/usr/bin/env python3

from pathlib import Path
import math

import rospkg
import rospy
import yaml
from geometry_msgs.msg import Point32, PolygonStamped


def polygon_signed_area(poly):
    return 0.5 * sum(
        poly[i][0] * poly[(i + 1) % len(poly)][1]
        - poly[(i + 1) % len(poly)][0] * poly[i][1]
        for i in range(len(poly))
    )


def polygon_area(poly):
    return abs(polygon_signed_area(poly))


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
    """
    凸多角形を指定margin [mm]だけ内側へ縮める。
    現在のT0紙面・ACCESS ZONE用。
    """
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
                a[1] + ny * margin
            ),
            (
                b[0] + nx * margin,
                b[1] + ny * margin
            )
        ))

    result = []

    for i in range(len(shifted_edges)):
        prev_edge = shifted_edges[i - 1]
        this_edge = shifted_edges[i]

        point = line_intersection(
            prev_edge[0],
            prev_edge[1],
            this_edge[0],
            this_edge[1]
        )

        if point is None:
            return []

        result.append(point)

    return result


def clip_polygon(subject, clipper):
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


class CobottaGraspCandidateArea:
    def __init__(self):
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
                str(default_config)
            )
        )

        with config_path.open() as f:
            self.config = yaml.safe_load(f)

        size_x = float(
            self.config["stand_top"]["size_x"]
        )

        size_y = float(
            self.config["stand_top"]["size_y"]
        )

        self.center_x = size_x / 2.0
        self.center_y = size_y / 2.0

        # 暫定安全マージン [mm]
        self.paper_margin_mm = float(
            rospy.get_param(
                "~paper_margin_mm",
                5.0
            )
        )

        self.access_margin_mm = float(
            rospy.get_param(
                "~access_margin_mm",
                10.0
            )
        )

        self.access_zones = {}

        for name, zone in (
            self.config["access_zones"].items()
        ):
            converted = []

            for stand_x, stand_y in zone["vertices"]:
                paper_x = (
                    self.center_x
                    - float(stand_x)
                )

                paper_y = (
                    self.center_y
                    - float(stand_y)
                )

                converted.append(
                    (paper_x, paper_y)
                )

            self.access_zones[name] = converted

        self.upper_pub = rospy.Publisher(
            "/origami/grasp_candidate_area_upper",
            PolygonStamped,
            queue_size=1,
            latch=True
        )

        self.lower_pub = rospy.Publisher(
            "/origami/grasp_candidate_area_lower",
            PolygonStamped,
            queue_size=1,
            latch=True
        )

        self.safe_upper_pub = rospy.Publisher(
            "/origami/safe_grasp_area_upper",
            PolygonStamped,
            queue_size=1,
            latch=True
        )

        self.safe_lower_pub = rospy.Publisher(
            "/origami/safe_grasp_area_lower",
            PolygonStamped,
            queue_size=1,
            latch=True
        )

        self.sub = rospy.Subscriber(
            "/origami/active_folding_paper_t0_ros",
            PolygonStamped,
            self.callback,
            queue_size=1
        )

        rospy.loginfo(
            "COBOTTA把持候補領域生成ノードを開始しました。"
        )
        rospy.loginfo(
            "ACCESS ZONE config: %s",
            str(config_path)
        )

        rospy.loginfo(
            "SAFE margin: paper=%.1f mm access=%.1f mm",
            self.paper_margin_mm,
            self.access_margin_mm
        )

    @staticmethod
    def make_polygon_message(points_mm):
        msg = PolygonStamped()

        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "paper_center"

        for x_mm, y_mm in points_mm:
            msg.polygon.points.append(
                Point32(
                    x=x_mm / 1000.0,
                    y=y_mm / 1000.0,
                    z=0.0
                )
            )

        return msg

    def callback(self, paper_msg):
        paper_mm = [
            (
                p.x * 1000.0,
                p.y * 1000.0
            )
            for p in paper_msg.polygon.points
        ]

        upper = clip_polygon(
            paper_mm,
            self.access_zones["upper"]
        )

        lower = clip_polygon(
            paper_mm,
            self.access_zones["lower"]
        )

        self.upper_pub.publish(
            self.make_polygon_message(upper)
        )

        self.lower_pub.publish(
            self.make_polygon_message(lower)
        )

        # 紙端とACCESS ZONE端から安全マージンを取る
        safe_paper = shrink_convex_polygon(
            paper_mm,
            self.paper_margin_mm
        )

        safe_upper_access = shrink_convex_polygon(
            self.access_zones["upper"],
            self.access_margin_mm
        )

        safe_lower_access = shrink_convex_polygon(
            self.access_zones["lower"],
            self.access_margin_mm
        )

        safe_upper = clip_polygon(
            safe_paper,
            safe_upper_access
        )

        safe_lower = clip_polygon(
            safe_paper,
            safe_lower_access
        )

        self.safe_upper_pub.publish(
            self.make_polygon_message(safe_upper)
        )

        self.safe_lower_pub.publish(
            self.make_polygon_message(safe_lower)
        )

        rospy.loginfo(
            "把持候補領域: "
            "upper=%.2f mm^2 lower=%.2f mm^2 | "
            "SAFE GRASP AREA: "
            "upper=%.2f mm^2 lower=%.2f mm^2",
            polygon_area(upper) if len(upper) >= 3 else 0.0,
            polygon_area(lower) if len(lower) >= 3 else 0.0,
            polygon_area(safe_upper) if len(safe_upper) >= 3 else 0.0,
            polygon_area(safe_lower) if len(safe_lower) >= 3 else 0.0
        )


def main():
    rospy.init_node(
        "cobotta_grasp_candidate_area"
    )

    CobottaGraspCandidateArea()

    rospy.spin()


if __name__ == "__main__":
    main()
