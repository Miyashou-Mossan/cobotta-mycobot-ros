#!/usr/bin/env python3

import math

import rospy

from geometry_msgs.msg import (
    Point,
    PointStamped,
    PolygonStamped,
    Vector3Stamped,
)
from visualization_msgs.msg import Marker, MarkerArray


class UnityCoordinateVisualizer:
    """Visualize Unity paper and fold data in paper_center."""

    def __init__(self):
        self.output_frame = rospy.get_param(
            "~output_frame",
            "paper_center",
        )

        # 表示専用。入力座標やロボット軌道には反映しない。
        self.visual_z_offset = float(
            rospy.get_param("~visual_z_offset", 0.010)
        )
        self.vector_length = float(
            rospy.get_param("~vector_length", 0.060)
        )

        # Unity入力
        self.all_vertices_topic = (
            "/origami/paper_vertices_unity"
        )
        self.selected_face_topic = (
            "/origami/selected_paper_vertices_unity"
        )
        self.grasp_vertex_topic = (
            "/origami/grasp_vertex_unity"
        )
        self.fold_line_topic = (
            "/origami/fold_line_unity"
        )
        self.fold_side_topic = (
            "/origami/fold_side_direction_unity"
        )
        self.folding_normal_topic = (
            "/origami/folding_paper_normal_unity"
        )
        self.fixed_normal_topic = (
            "/origami/fixed_paper_normal_unity"
        )

        # RViz出力
        self.all_vertices_pub = rospy.Publisher(
            "/origami/debug/unity_paper_vertices",
            MarkerArray,
            queue_size=1,
            latch=True,
        )
        self.selected_face_pub = rospy.Publisher(
            "/origami/debug/unity_selected_face",
            MarkerArray,
            queue_size=1,
            latch=True,
        )
        self.grasp_vertex_pub = rospy.Publisher(
            "/origami/debug/unity_grasp_vertex",
            MarkerArray,
            queue_size=1,
            latch=True,
        )
        self.fold_line_pub = rospy.Publisher(
            "/origami/debug/unity_fold_line",
            MarkerArray,
            queue_size=1,
            latch=True,
        )
        self.fold_side_pub = rospy.Publisher(
            "/origami/debug/unity_fold_side_direction",
            MarkerArray,
            queue_size=1,
            latch=True,
        )
        self.folding_normal_pub = rospy.Publisher(
            "/origami/debug/unity_folding_paper_normal",
            MarkerArray,
            queue_size=1,
            latch=True,
        )
        self.fixed_normal_pub = rospy.Publisher(
            "/origami/debug/unity_fixed_paper_normal",
            MarkerArray,
            queue_size=1,
            latch=True,
        )

        # 最新値を保持する。
        self.all_points = []
        self.selected_points = []
        self.fold_line_points = []

        self.fold_side_vector = None
        self.folding_normal_vector = None
        self.fixed_normal_vector = None

        rospy.Subscriber(
            self.all_vertices_topic,
            PolygonStamped,
            self.all_vertices_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            self.selected_face_topic,
            PolygonStamped,
            self.selected_face_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            self.grasp_vertex_topic,
            PointStamped,
            self.grasp_vertex_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            self.fold_line_topic,
            PolygonStamped,
            self.fold_line_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            self.fold_side_topic,
            Vector3Stamped,
            self.fold_side_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            self.folding_normal_topic,
            Vector3Stamped,
            self.folding_normal_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            self.fixed_normal_topic,
            Vector3Stamped,
            self.fixed_normal_callback,
            queue_size=1,
        )

        self.logged = set()

        rospy.loginfo("Unity coordinate visualizer started")
        rospy.loginfo("  output frame: %s", self.output_frame)
        rospy.loginfo(
            "  visual z offset: %.3f m",
            self.visual_z_offset,
        )

    def unity_point_to_ros(self, unity_point):
        """
        Unity point -> paper_center point

          x_ros = z_unity
          y_ros = -x_unity
          z_ros = y_unity

        visual_z_offsetはRViz表示だけに使用する。
        """
        point = Point()
        point.x = unity_point.z
        point.y = -unity_point.x
        point.z = (
            unity_point.y +
            self.visual_z_offset
        )
        return point

    @staticmethod
    def unity_vector_to_ros(unity_vector):
        """
        ベクトルには平行移動や表示オフセットを加えない。
        """
        vector = Point()
        vector.x = unity_vector.z
        vector.y = -unity_vector.x
        vector.z = unity_vector.y
        return vector

    def base_marker(
        self,
        namespace,
        marker_id,
        marker_type,
    ):
        marker = Marker()
        marker.header.frame_id = self.output_frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.frame_locked = True
        marker.lifetime = rospy.Duration(0)
        return marker

    @staticmethod
    def set_color(
        marker,
        red,
        green,
        blue,
        alpha=1.0,
    ):
        marker.color.r = red
        marker.color.g = green
        marker.color.b = blue
        marker.color.a = alpha

    @staticmethod
    def delete_all_marker():
        marker = Marker()
        marker.action = Marker.DELETEALL
        return marker

    @staticmethod
    def centroid(points):
        if not points:
            return None

        center = Point()

        for point in points:
            center.x += point.x
            center.y += point.y
            center.z += point.z

        count = float(len(points))
        center.x /= count
        center.y /= count
        center.z /= count
        return center

    @staticmethod
    def midpoint(point_a, point_b):
        point = Point()
        point.x = (point_a.x + point_b.x) * 0.5
        point.y = (point_a.y + point_b.y) * 0.5
        point.z = (point_a.z + point_b.z) * 0.5
        return point

    @staticmethod
    def distance(point_a, point_b):
        return math.sqrt(
            (point_a.x - point_b.x) ** 2 +
            (point_a.y - point_b.y) ** 2 +
            (point_a.z - point_b.z) ** 2
        )

    @staticmethod
    def normalized(vector):
        length = math.sqrt(
            vector.x ** 2 +
            vector.y ** 2 +
            vector.z ** 2
        )

        if length < 1.0e-9:
            return None

        result = Point()
        result.x = vector.x / length
        result.y = vector.y / length
        result.z = vector.z / length
        return result

    def make_text(
        self,
        namespace,
        marker_id,
        text,
        position,
        color,
    ):
        marker = self.base_marker(
            namespace,
            marker_id,
            Marker.TEXT_VIEW_FACING,
        )
        marker.pose.position = position
        marker.scale.z = 0.018
        self.set_color(marker, *color)
        marker.text = text
        return marker

    def make_arrow_array(
        self,
        namespace,
        label,
        origin,
        direction,
        color,
    ):
        markers = MarkerArray()
        markers.markers.append(self.delete_all_marker())

        direction = self.normalized(direction)

        if origin is None or direction is None:
            return markers

        end = Point()
        end.x = origin.x + direction.x * self.vector_length
        end.y = origin.y + direction.y * self.vector_length
        end.z = origin.z + direction.z * self.vector_length

        arrow = self.base_marker(
            namespace,
            0,
            Marker.ARROW,
        )
        arrow.points = [origin, end]
        arrow.scale.x = 0.006
        arrow.scale.y = 0.012
        arrow.scale.z = 0.018
        self.set_color(arrow, *color)
        markers.markers.append(arrow)

        label_position = Point()
        label_position.x = end.x
        label_position.y = end.y
        label_position.z = end.z + 0.015

        markers.markers.append(
            self.make_text(
                namespace + "_label",
                1,
                label,
                label_position,
                color,
            )
        )

        return markers

    def make_polygon_markers(
        self,
        points,
        namespace,
        label_prefix,
        color,
    ):
        markers = MarkerArray()
        markers.markers.append(self.delete_all_marker())

        if not points:
            return markers

        outline = self.base_marker(
            namespace + "_outline",
            0,
            Marker.LINE_STRIP,
        )
        outline.scale.x = 0.004
        self.set_color(outline, *color)
        outline.points = list(points)

        if len(points) >= 3:
            outline.points.append(points[0])

        markers.markers.append(outline)

        spheres = self.base_marker(
            namespace + "_points",
            1,
            Marker.SPHERE_LIST,
        )
        spheres.scale.x = 0.014
        spheres.scale.y = 0.014
        spheres.scale.z = 0.014
        self.set_color(spheres, *color)
        spheres.points = list(points)
        markers.markers.append(spheres)

        for index, point in enumerate(points):
            label_position = Point()
            label_position.x = point.x
            label_position.y = point.y
            label_position.z = point.z + 0.025

            text = (
                "{}{}\n"
                "({:+.3f}, {:+.3f}, {:+.3f})"
            ).format(
                label_prefix,
                index,
                point.x,
                point.y,
                point.z,
            )

            markers.markers.append(
                self.make_text(
                    namespace + "_labels",
                    10 + index,
                    text,
                    label_position,
                    color,
                )
            )

        return markers

    def all_vertices_callback(self, message):
        self.all_points = [
            self.unity_point_to_ros(point)
            for point in message.polygon.points
        ]

        self.all_vertices_pub.publish(
            self.make_polygon_markers(
                self.all_points,
                "unity_all_vertices",
                "V",
                (0.2, 1.0, 0.2, 1.0),
            )
        )

        self.publish_fixed_normal()

        if "all" not in self.logged:
            rospy.loginfo(
                "Received %d unique paper vertices",
                len(self.all_points),
            )
            self.logged.add("all")

    def selected_face_callback(self, message):
        self.selected_points = [
            self.unity_point_to_ros(point)
            for point in message.polygon.points
        ]

        self.selected_face_pub.publish(
            self.make_polygon_markers(
                self.selected_points,
                "unity_selected_face",
                "F",
                (1.0, 0.4, 1.0, 1.0),
            )
        )

        self.publish_folding_normal()
        self.publish_fixed_normal()

        if "selected" not in self.logged:
            rospy.loginfo(
                "Received %d selected-face vertices",
                len(self.selected_points),
            )
            self.logged.add("selected")

    def grasp_vertex_callback(self, message):
        point = self.unity_point_to_ros(message.point)

        markers = MarkerArray()
        markers.markers.append(self.delete_all_marker())

        sphere = self.base_marker(
            "unity_grasp_vertex",
            0,
            Marker.SPHERE,
        )
        sphere.pose.position = point
        sphere.scale.x = 0.024
        sphere.scale.y = 0.024
        sphere.scale.z = 0.024
        self.set_color(
            sphere,
            1.0,
            0.2,
            0.2,
            1.0,
        )
        markers.markers.append(sphere)

        label_position = Point()
        label_position.x = point.x
        label_position.y = point.y
        label_position.z = point.z + 0.035

        markers.markers.append(
            self.make_text(
                "unity_grasp_vertex_label",
                1,
                "Test9 grasp vertex",
                label_position,
                (1.0, 0.2, 0.2, 1.0),
            )
        )

        self.grasp_vertex_pub.publish(markers)

    def fold_line_callback(self, message):
        self.fold_line_points = [
            self.unity_point_to_ros(point)
            for point in message.polygon.points
        ]

        markers = MarkerArray()
        markers.markers.append(self.delete_all_marker())

        if len(self.fold_line_points) >= 2:
            line = self.base_marker(
                "unity_fold_line",
                0,
                Marker.LINE_LIST,
            )
            line.scale.x = 0.010
            self.set_color(
                line,
                1.0,
                0.6,
                0.0,
                1.0,
            )
            line.points = [
                self.fold_line_points[0],
                self.fold_line_points[1],
            ]
            markers.markers.append(line)

            center = self.midpoint(
                self.fold_line_points[0],
                self.fold_line_points[1],
            )
            center.z += 0.020

            markers.markers.append(
                self.make_text(
                    "unity_fold_line_label",
                    1,
                    "Actual fold line",
                    center,
                    (1.0, 0.6, 0.0, 1.0),
                )
            )

        self.fold_line_pub.publish(markers)

        self.publish_fold_side()
        self.publish_fixed_normal()

    def fold_side_callback(self, message):
        self.fold_side_vector = (
            self.unity_vector_to_ros(message.vector)
        )
        self.publish_fold_side()

    def folding_normal_callback(self, message):
        self.folding_normal_vector = (
            self.unity_vector_to_ros(message.vector)
        )
        self.publish_folding_normal()

    def fixed_normal_callback(self, message):
        self.fixed_normal_vector = (
            self.unity_vector_to_ros(message.vector)
        )
        self.publish_fixed_normal()

    def publish_fold_side(self):
        if (
            len(self.fold_line_points) < 2 or
            self.fold_side_vector is None
        ):
            return

        origin = self.midpoint(
            self.fold_line_points[0],
            self.fold_line_points[1],
        )

        self.fold_side_pub.publish(
            self.make_arrow_array(
                "unity_fold_side",
                "Moving side",
                origin,
                self.fold_side_vector,
                (1.0, 0.2, 1.0, 1.0),
            )
        )

    def publish_folding_normal(self):
        if (
            not self.selected_points or
            self.folding_normal_vector is None
        ):
            return

        origin = self.centroid(self.selected_points)

        self.folding_normal_pub.publish(
            self.make_arrow_array(
                "unity_folding_normal",
                "Moving normal",
                origin,
                self.folding_normal_vector,
                (0.0, 1.0, 1.0, 1.0),
            )
        )

    def find_fixed_face_center(self):
        if (
            not self.all_points or
            not self.selected_points or
            len(self.fold_line_points) < 2
        ):
            return None

        unmatched = []

        for all_point in self.all_points:
            minimum_distance = min(
                self.distance(all_point, selected_point)
                for selected_point in self.selected_points
            )

            if minimum_distance > 0.005:
                unmatched.append(all_point)

        if not unmatched:
            return None

        fixed_face_points = [
            self.fold_line_points[0],
            self.fold_line_points[1],
            unmatched[0],
        ]

        return self.centroid(fixed_face_points)

    def publish_fixed_normal(self):
        if self.fixed_normal_vector is None:
            return

        origin = self.find_fixed_face_center()

        if origin is None and len(self.fold_line_points) >= 2:
            origin = self.midpoint(
                self.fold_line_points[0],
                self.fold_line_points[1],
            )

        if origin is None:
            return

        self.fixed_normal_pub.publish(
            self.make_arrow_array(
                "unity_fixed_normal",
                "Fixed normal",
                origin,
                self.fixed_normal_vector,
                (1.0, 1.0, 0.0, 1.0),
            )
        )


def main():
    rospy.init_node(
        "origami_unity_coordinate_visualizer"
    )
    UnityCoordinateVisualizer()
    rospy.spin()


if __name__ == "__main__":
    main()
