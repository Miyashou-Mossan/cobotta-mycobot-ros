#!/usr/bin/env python3

import math
import rospy

from geometry_msgs.msg import Point, PoseArray, PolygonStamped
from visualization_msgs.msg import Marker, MarkerArray


class ActivePaperPoseHistoryVisualizer:
    def __init__(self):
        self.latest_t0_polygon = None

        self.publisher = rospy.Publisher(
            "/origami/active_paper_pose_history_markers",
            MarkerArray,
            queue_size=1,
            latch=True
        )

        rospy.Subscriber(
            "/origami/active_folding_paper_t0_ros",
            PolygonStamped,
            self.t0_callback,
            queue_size=1
        )

        rospy.Subscriber(
            "/origami/active_paper_pose_history_ros",
            PoseArray,
            self.pose_history_callback,
            queue_size=1
        )

        rospy.loginfo(
            "Active paper Pose history visualizer started."
        )

    @staticmethod
    def rotate_vector(qx, qy, qz, qw, v):
        vx, vy, vz = v

        tx = 2.0 * (qy * vz - qz * vy)
        ty = 2.0 * (qz * vx - qx * vz)
        tz = 2.0 * (qx * vy - qy * vx)

        rx = vx + qw * tx + (qy * tz - qz * ty)
        ry = vy + qw * ty + (qz * tx - qx * tz)
        rz = vz + qw * tz + (qx * ty - qy * tx)

        return rx, ry, rz

    def t0_callback(self, msg):
        if len(msg.polygon.points) < 3:
            return

        self.latest_t0_polygon = msg

        rospy.loginfo(
            "T0 polygon received: pointCount=%d",
            len(msg.polygon.points)
        )

    def pose_history_callback(self, msg):
        if self.latest_t0_polygon is None:
            rospy.logwarn(
                "Pose history received, but T0 polygon is not available."
            )
            return

        if len(msg.poses) == 0:
            return

        # --------------------------------------------
        # T0移動紙面の重心
        # --------------------------------------------
        points = self.latest_t0_polygon.polygon.points
        n = float(len(points))

        centroid = Point()
        centroid.x = sum(p.x for p in points) / n
        centroid.y = sum(p.y for p in points) / n
        centroid.z = sum(p.z for p in points) / n

        # --------------------------------------------
        # T0 Poseから紙ローカル座標へ逆変換
        # --------------------------------------------
        pose0 = msg.poses[0]
        q0 = pose0.orientation

        dx = centroid.x - pose0.position.x
        dy = centroid.y - pose0.position.y
        dz = centroid.z - pose0.position.z

        local_point = self.rotate_vector(
            -q0.x,
            -q0.y,
            -q0.z,
            q0.w,
            (dx, dy, dz)
        )

        # --------------------------------------------
        # 同じ紙ローカル点を62 Poseへ適用
        # --------------------------------------------
        trajectory_points = []

        for pose in msg.poses:
            q = pose.orientation

            rx, ry, rz = self.rotate_vector(
                q.x,
                q.y,
                q.z,
                q.w,
                local_point
            )

            p = Point()
            p.x = pose.position.x + rx
            p.y = pose.position.y + ry
            p.z = pose.position.z + rz

            trajectory_points.append(p)

        # T0再構成誤差
        start = trajectory_points[0]

        reconstruction_error_mm = 1000.0 * math.sqrt(
            (start.x - centroid.x) ** 2 +
            (start.y - centroid.y) ** 2 +
            (start.z - centroid.z) ** 2
        )

        z_values_mm = [
            p.z * 1000.0
            for p in trajectory_points
        ]

        rospy.loginfo(
            "T0 paper centroid local [mm] = "
            "(%.3f, %.3f, %.3f)",
            local_point[0] * 1000.0,
            local_point[1] * 1000.0,
            local_point[2] * 1000.0
        )

        rospy.loginfo(
            "T0 reconstruction error = %.9f mm",
            reconstruction_error_mm
        )

        rospy.loginfo(
            "Trajectory Z range = %.3f to %.3f mm",
            min(z_values_mm),
            max(z_values_mm)
        )

        # --------------------------------------------
        # RViz Marker
        # --------------------------------------------
        markers = MarkerArray()

        trajectory = Marker()
        trajectory.header.frame_id = (
            msg.header.frame_id or "paper_center"
        )
        trajectory.header.stamp = rospy.Time(0)

        trajectory.ns = "active_paper_real_point_trajectory"
        trajectory.id = 0
        trajectory.type = Marker.LINE_STRIP
        trajectory.action = Marker.ADD
        trajectory.pose.orientation.w = 1.0

        trajectory.scale.x = 0.002

        trajectory.color.r = 1.0
        trajectory.color.g = 0.1
        trajectory.color.b = 0.8
        trajectory.color.a = 1.0

        trajectory.points = trajectory_points
        markers.markers.append(trajectory)

        # START
        start_marker = Marker()
        start_marker.header = trajectory.header
        start_marker.ns = "active_paper_real_point_start"
        start_marker.id = 1
        start_marker.type = Marker.SPHERE
        start_marker.action = Marker.ADD
        start_marker.pose.position = trajectory_points[0]
        start_marker.pose.orientation.w = 1.0
        start_marker.scale.x = 0.008
        start_marker.scale.y = 0.008
        start_marker.scale.z = 0.008
        start_marker.color.r = 0.1
        start_marker.color.g = 1.0
        start_marker.color.b = 0.1
        start_marker.color.a = 1.0
        markers.markers.append(start_marker)

        # END
        end_marker = Marker()
        end_marker.header = trajectory.header
        end_marker.ns = "active_paper_real_point_end"
        end_marker.id = 2
        end_marker.type = Marker.SPHERE
        end_marker.action = Marker.ADD
        end_marker.pose.position = trajectory_points[-1]
        end_marker.pose.orientation.w = 1.0
        end_marker.scale.x = 0.008
        end_marker.scale.y = 0.008
        end_marker.scale.z = 0.008
        end_marker.color.r = 1.0
        end_marker.color.g = 0.1
        end_marker.color.b = 0.1
        end_marker.color.a = 1.0
        markers.markers.append(end_marker)

        self.publisher.publish(markers)

        rospy.loginfo(
            "Real-paper-point trajectory published: poseCount=%d",
            len(msg.poses)
        )


def main():
    rospy.init_node(
        "active_paper_pose_history_visualizer"
    )

    ActivePaperPoseHistoryVisualizer()
    rospy.spin()


if __name__ == "__main__":
    main()
