#!/usr/bin/env python3

import rospy

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray


class OrigamiCoordinateVisualizer:
    """Publish paper-coordinate reference markers for RViz."""

    def __init__(self):
        self.frame_id = rospy.get_param("~frame_id", "paper_center")
        self.paper_size = float(rospy.get_param("~paper_size", 0.150))
        self.axis_length = float(rospy.get_param("~axis_length", 0.100))
        self.normal_length = float(rospy.get_param("~normal_length", 0.060))
        self.fold_line_axis = str(
            rospy.get_param("~fold_line_axis", "y")
        ).lower()
        self.fold_direction_sign = float(
            rospy.get_param("~fold_direction_sign", 1.0)
        )
        self.fold_line_offset = float(
            rospy.get_param("~fold_line_offset", 0.0)
        )

        if self.paper_size <= 0.0:
            raise ValueError("paper_size must be positive")

        if self.fold_line_axis not in ("x", "y"):
            raise ValueError("fold_line_axis must be 'x' or 'y'")

        if self.fold_direction_sign == 0.0:
            raise ValueError("fold_direction_sign must not be zero")

        self.corner_pub = rospy.Publisher(
            "/origami/debug/paper_corners",
            MarkerArray,
            queue_size=1,
            latch=True,
        )
        self.axes_pub = rospy.Publisher(
            "/origami/debug/paper_axes",
            MarkerArray,
            queue_size=1,
            latch=True,
        )
        self.fold_line_pub = rospy.Publisher(
            "/origami/debug/fold_line",
            Marker,
            queue_size=1,
            latch=True,
        )
        self.fold_direction_pub = rospy.Publisher(
            "/origami/debug/fold_direction",
            Marker,
            queue_size=1,
            latch=True,
        )
        self.normal_pub = rospy.Publisher(
            "/origami/debug/paper_normal",
            Marker,
            queue_size=1,
            latch=True,
        )

        rospy.loginfo("Origami coordinate visualizer started")
        rospy.loginfo("  frame_id: %s", self.frame_id)
        rospy.loginfo("  paper_size: %.3f m", self.paper_size)
        rospy.loginfo(
            "  provisional corner labels: "
            "P0=(-X,-Y), P1=(+X,-Y), "
            "P2=(+X,+Y), P3=(-X,+Y)"
        )
        rospy.logwarn(
            "P0-P3 are local coordinate labels only. "
            "Physical COBOTTA/MyCobot/front/back correspondence "
            "has not been confirmed yet."
        )

    @staticmethod
    def point(x, y, z):
        return Point(x=float(x), y=float(y), z=float(z))

    def base_marker(self, namespace, marker_id, marker_type):
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.lifetime = rospy.Duration(0)
        marker.frame_locked = True
        return marker

    @staticmethod
    def set_color(marker, red, green, blue, alpha=1.0):
        marker.color.r = red
        marker.color.g = green
        marker.color.b = blue
        marker.color.a = alpha

    def make_paper_corners(self):
        half = self.paper_size / 2.0
        z = 0.003

        corner_data = [
            ("P0", -half, -half),
            ("P1", half, -half),
            ("P2", half, half),
            ("P3", -half, half),
        ]

        markers = MarkerArray()

        outline = self.base_marker(
            "paper_outline",
            0,
            Marker.LINE_STRIP,
        )
        outline.scale.x = 0.003
        self.set_color(outline, 1.0, 1.0, 1.0)
        outline.points = [
            self.point(-half, -half, z),
            self.point(half, -half, z),
            self.point(half, half, z),
            self.point(-half, half, z),
            self.point(-half, -half, z),
        ]
        markers.markers.append(outline)

        corner_points = self.base_marker(
            "paper_corner_points",
            1,
            Marker.SPHERE_LIST,
        )
        corner_points.scale.x = 0.012
        corner_points.scale.y = 0.012
        corner_points.scale.z = 0.012
        self.set_color(corner_points, 1.0, 1.0, 0.0)
        corner_points.points = [
            self.point(x, y, z) for _, x, y in corner_data
        ]
        markers.markers.append(corner_points)

        for index, (name, x, y) in enumerate(corner_data):
            label = self.base_marker(
                "paper_corner_labels",
                10 + index,
                Marker.TEXT_VIEW_FACING,
            )
            label.pose.position = self.point(x, y, z + 0.020)
            label.scale.z = 0.020
            self.set_color(label, 1.0, 1.0, 0.0)
            label.text = "{} ({:+.3f}, {:+.3f})".format(
                name,
                x,
                y,
            )
            markers.markers.append(label)

        return markers

    def make_arrow(
        self,
        namespace,
        marker_id,
        start,
        end,
        color,
        shaft_width=0.006,
    ):
        arrow = self.base_marker(
            namespace,
            marker_id,
            Marker.ARROW,
        )
        arrow.points = [start, end]
        arrow.scale.x = shaft_width
        arrow.scale.y = shaft_width * 2.0
        arrow.scale.z = shaft_width * 3.0
        self.set_color(arrow, *color)
        return arrow

    def make_text(
        self,
        namespace,
        marker_id,
        text,
        position,
        color,
    ):
        label = self.base_marker(
            namespace,
            marker_id,
            Marker.TEXT_VIEW_FACING,
        )
        label.pose.position = position
        label.scale.z = 0.020
        self.set_color(label, *color)
        label.text = text
        return label

    def make_paper_axes(self):
        z = 0.008
        origin = self.point(0.0, 0.0, z)

        axes = MarkerArray()

        axes.markers.append(
            self.make_arrow(
                "paper_axes",
                0,
                origin,
                self.point(self.axis_length, 0.0, z),
                (1.0, 0.0, 0.0),
            )
        )
        axes.markers.append(
            self.make_arrow(
                "paper_axes",
                1,
                origin,
                self.point(0.0, self.axis_length, z),
                (0.0, 1.0, 0.0),
            )
        )
        axes.markers.append(
            self.make_arrow(
                "paper_axes",
                2,
                origin,
                self.point(0.0, 0.0, self.axis_length),
                (0.0, 0.3, 1.0),
            )
        )

        axes.markers.append(
            self.make_text(
                "paper_axis_labels",
                10,
                "+X",
                self.point(self.axis_length + 0.015, 0.0, z),
                (1.0, 0.0, 0.0),
            )
        )
        axes.markers.append(
            self.make_text(
                "paper_axis_labels",
                11,
                "+Y",
                self.point(0.0, self.axis_length + 0.015, z),
                (0.0, 1.0, 0.0),
            )
        )
        axes.markers.append(
            self.make_text(
                "paper_axis_labels",
                12,
                "+Z",
                self.point(0.0, 0.0, self.axis_length + 0.015),
                (0.0, 0.3, 1.0),
            )
        )

        return axes

    def make_fold_line(self):
        half = self.paper_size / 2.0
        z = 0.010

        line = self.base_marker(
            "fold_line",
            0,
            Marker.LINE_LIST,
        )
        line.scale.x = 0.006
        self.set_color(line, 1.0, 0.0, 1.0)

        if self.fold_line_axis == "y":
            line.points = [
                self.point(self.fold_line_offset, -half, z),
                self.point(self.fold_line_offset, half, z),
            ]
        else:
            line.points = [
                self.point(-half, self.fold_line_offset, z),
                self.point(half, self.fold_line_offset, z),
            ]

        return line

    def make_fold_direction(self):
        z = 0.018
        direction_length = self.paper_size * 0.35
        sign = 1.0 if self.fold_direction_sign > 0.0 else -1.0

        start = self.point(0.0, 0.0, z)

        if self.fold_line_axis == "y":
            end = self.point(sign * direction_length, 0.0, z)
        else:
            end = self.point(0.0, sign * direction_length, z)

        return self.make_arrow(
            "fold_direction",
            0,
            start,
            end,
            (1.0, 0.5, 0.0),
            shaft_width=0.008,
        )

    def make_paper_normal(self):
        return self.make_arrow(
            "paper_normal",
            0,
            self.point(0.0, 0.0, 0.012),
            self.point(0.0, 0.0, self.normal_length),
            (0.0, 1.0, 1.0),
            shaft_width=0.008,
        )

    def publish(self):
        self.corner_pub.publish(self.make_paper_corners())
        self.axes_pub.publish(self.make_paper_axes())
        self.fold_line_pub.publish(self.make_fold_line())
        self.fold_direction_pub.publish(
            self.make_fold_direction()
        )
        self.normal_pub.publish(self.make_paper_normal())

    def run(self):
        rate = rospy.Rate(1.0)

        while not rospy.is_shutdown():
            self.publish()
            rate.sleep()


def main():
    rospy.init_node("origami_coordinate_visualizer")

    try:
        visualizer = OrigamiCoordinateVisualizer()
        visualizer.run()
    except (ValueError, TypeError) as error:
        rospy.logfatal("Invalid visualizer configuration: %s", error)
        rospy.signal_shutdown(str(error))


if __name__ == "__main__":
    main()
