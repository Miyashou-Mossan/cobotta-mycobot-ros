#!/usr/bin/env python3

from pathlib import Path

import rospkg
import rospy
import yaml

from geometry_msgs.msg import Point
from geometry_msgs.msg import PolygonStamped
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


class SafeGraspAreaVisualizer:
    def __init__(self):
        self.z_offset = float(
            rospy.get_param(
                "~visualization_z_offset_m",
                0.001
            )
        )

        self.publisher = rospy.Publisher(
            "/origami/safe_grasp_area_markers",
            MarkerArray,
            queue_size=1,
            latch=True
        )

        self.t0_polygon = None
        self.upper_polygon = None
        self.lower_polygon = None

        # ACCESS ZONE YAMLを読む
        rospack = rospkg.RosPack()
        package_path = Path(
            rospack.get_path("origami_grasp_bridge")
        )

        config_path = package_path / (
            "config/cobotta_stand_access_zones_v1.yaml"
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

        self.access_upper = self.convert_access_zone(
            self.config["access_zones"]["upper"]["vertices"]
        )

        self.access_lower = self.convert_access_zone(
            self.config["access_zones"]["lower"]["vertices"]
        )

        rospy.Subscriber(
            "/origami/active_folding_paper_t0_ros",
            PolygonStamped,
            self.t0_callback,
            queue_size=1
        )

        rospy.Subscriber(
            "/origami/safe_grasp_area_upper",
            PolygonStamped,
            self.upper_callback,
            queue_size=1
        )

        rospy.Subscriber(
            "/origami/safe_grasp_area_lower",
            PolygonStamped,
            self.lower_callback,
            queue_size=1
        )

        rospy.loginfo(
            "T0 / ACCESS ZONE / SAFE GRASP AREA "
            "RViz diagnostic visualizer started."
        )

        self.publish_markers()

    def convert_access_zone(self, vertices):
        points = []

        for stand_x, stand_y in vertices:
            # cobotta_grasp_candidate_area.pyと
            # 同じ変換をそのまま使用
            paper_x_mm = self.center_x - float(stand_x)
            paper_y_mm = self.center_y - float(stand_y)

            p = Point()
            p.x = paper_x_mm / 1000.0
            p.y = paper_y_mm / 1000.0
            p.z = 0.0

            points.append(p)

        return points

    def t0_callback(self, msg):
        self.t0_polygon = msg
        self.publish_markers()

    def upper_callback(self, msg):
        self.upper_polygon = msg
        self.publish_markers()

    def lower_callback(self, msg):
        self.lower_polygon = msg
        self.publish_markers()

    def make_triangle_marker(
        self,
        points,
        marker_id,
        namespace,
        r,
        g,
        b,
        a,
        z_offset
    ):
        marker = Marker()

        marker.header.stamp = rospy.Time(0)
        marker.header.frame_id = "paper_center"

        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD

        marker.pose.orientation.w = 1.0

        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0

        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = a

        if len(points) >= 3:
            p0 = points[0]

            for i in range(1, len(points) - 1):
                for source in (
                    p0,
                    points[i],
                    points[i + 1]
                ):
                    p = Point()
                    p.x = source.x
                    p.y = source.y
                    p.z = source.z + z_offset

                    marker.points.append(p)

        return marker

    def publish_markers(self):
        marker_array = MarkerArray()

        # ACCESS ZONE upper：青
        marker_array.markers.append(
            self.make_triangle_marker(
                self.access_upper,
                10,
                "access_zone_upper",
                0.1,
                0.3,
                1.0,
                0.35,
                self.z_offset
            )
        )

        # ACCESS ZONE lower：橙
        marker_array.markers.append(
            self.make_triangle_marker(
                self.access_lower,
                11,
                "access_zone_lower",
                1.0,
                0.5,
                0.1,
                0.35,
                self.z_offset * 2.0
            )
        )

        # T0 ActiveFoldingPaper：水色
        if (
            self.t0_polygon is not None
            and len(self.t0_polygon.polygon.points) >= 3
        ):
            marker_array.markers.append(
                self.make_triangle_marker(
                    self.t0_polygon.polygon.points,
                    20,
                    "active_folding_paper_t0",
                    0.1,
                    0.9,
                    0.9,
                    0.25,
                    self.z_offset * 3.0
                )
            )

            # 診断用：
            # 現在のpaper_center座標をZ軸まわり180°回転した場合
            yaw180_points = []

            for source in self.t0_polygon.polygon.points:
                p = Point()
                p.x = -source.x
                p.y = -source.y
                p.z = source.z
                yaw180_points.append(p)

            marker_array.markers.append(
                self.make_triangle_marker(
                    yaw180_points,
                    21,
                    "active_folding_paper_t0_yaw180_test",
                    0.9,
                    0.1,
                    0.9,
                    0.40,
                    self.z_offset * 3.5
                )
            )

        # SAFE upper/lower：緑
        if (
            self.upper_polygon is not None
            and len(self.upper_polygon.polygon.points) >= 3
        ):
            marker_array.markers.append(
                self.make_triangle_marker(
                    self.upper_polygon.polygon.points,
                    30,
                    "safe_grasp_area_upper",
                    0.2,
                    0.9,
                    0.3,
                    0.8,
                    self.z_offset * 4.0
                )
            )

        if (
            self.lower_polygon is not None
            and len(self.lower_polygon.polygon.points) >= 3
        ):
            marker_array.markers.append(
                self.make_triangle_marker(
                    self.lower_polygon.polygon.points,
                    31,
                    "safe_grasp_area_lower",
                    0.2,
                    0.9,
                    0.3,
                    0.8,
                    self.z_offset * 4.0
                )
            )

        self.publisher.publish(marker_array)

        rospy.loginfo(
            "Diagnostic Marker update: count=%d",
            len(marker_array.markers)
        )


def main():
    rospy.init_node(
        "safe_grasp_area_visualizer"
    )

    SafeGraspAreaVisualizer()

    rospy.spin()


if __name__ == "__main__":
    main()
