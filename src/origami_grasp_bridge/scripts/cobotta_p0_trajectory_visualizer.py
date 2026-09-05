#!/usr/bin/env python3

import math

import rospy

from geometry_msgs.msg import Point, PoseArray
from visualization_msgs.msg import Marker, MarkerArray


class CobottaP0TrajectoryVisualizer:
    def __init__(self):
        self.pose_history_msg = None
        self.p0_upper_msg = None
        self.p0_lower_msg = None

        self.last_processed_stamp = None

        self.marker_pub = rospy.Publisher(
            "/origami/cobotta_p0_trajectory_markers",
            MarkerArray,
            queue_size=1,
            latch=True,
        )

        rospy.Subscriber(
            "/origami/active_paper_pose_history_ros",
            PoseArray,
            self.pose_history_callback,
            queue_size=1,
        )

        rospy.Subscriber(
            "/origami/cobotta_p0_candidates_upper",
            PoseArray,
            self.p0_upper_callback,
            queue_size=1,
        )

        rospy.Subscriber(
            "/origami/cobotta_p0_candidates_lower",
            PoseArray,
            self.p0_lower_callback,
            queue_size=1,
        )

        rospy.loginfo(
            "COBOTTA P0 trajectory visualizer started."
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

    def pose_history_callback(self, msg):
        self.pose_history_msg = msg
        self.update()

    def p0_upper_callback(self, msg):
        self.p0_upper_msg = msg
        self.update()

    def p0_lower_callback(self, msg):
        self.p0_lower_msg = msg
        self.update()

    @staticmethod
    def stamp_nsec(msg):
        return msg.header.stamp.to_nsec()

    def make_trajectory(self, p0_pose, paper_poses):
        """
        P0をT0紙Poseに対するローカル点へ変換し、
        同じローカル点を全紙Poseへ適用する。
        """
        pose0 = paper_poses[0]
        q0 = pose0.orientation

        dx = p0_pose.position.x - pose0.position.x
        dy = p0_pose.position.y - pose0.position.y
        dz = p0_pose.position.z - pose0.position.z

        # T0 world -> paper local
        local_point = self.rotate_vector(
            -q0.x,
            -q0.y,
            -q0.z,
            q0.w,
            (dx, dy, dz),
        )

        trajectory = []

        # paper local -> each world pose
        for pose in paper_poses:
            q = pose.orientation

            rx, ry, rz = self.rotate_vector(
                q.x,
                q.y,
                q.z,
                q.w,
                local_point,
            )

            point = Point()
            point.x = pose.position.x + rx
            point.y = pose.position.y + ry
            point.z = pose.position.z + rz

            trajectory.append(point)

        return local_point, trajectory

    @staticmethod
    def distance(a, b):
        return math.sqrt(
            (a.x - b.x) ** 2
            + (a.y - b.y) ** 2
            + (a.z - b.z) ** 2
        )

    def add_trajectory_marker(
        self,
        marker_array,
        side,
        index,
        trajectory,
    ):
        marker = Marker()

        marker.header.frame_id = "paper_center"
        marker.header.stamp = rospy.Time(0)

        marker.ns = "p0_trajectory_" + side
        marker.id = index
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.0015

        if side == "lower":
            marker.color.r = 0.0
            marker.color.g = 0.8
            marker.color.b = 1.0
        else:
            marker.color.r = 1.0
            marker.color.g = 0.5
            marker.color.b = 0.0

        marker.color.a = 0.85
        marker.points = trajectory

        marker_array.markers.append(marker)

    def process_side(
        self,
        side,
        p0_msg,
        paper_poses,
        marker_array,
    ):
        if len(p0_msg.poses) == 0:
            return 0

        for index, p0_pose in enumerate(p0_msg.poses):
            local_point, trajectory = self.make_trajectory(
                p0_pose,
                paper_poses,
            )

            if not trajectory:
                continue

            reconstruction_error_mm = (
                self.distance(
                    trajectory[0],
                    p0_pose.position,
                )
                * 1000.0
            )

            z_values_mm = [
                point.z * 1000.0
                for point in trajectory
            ]

            max_step_mm = 0.0

            for i in range(len(trajectory) - 1):
                step_mm = (
                    self.distance(
                        trajectory[i],
                        trajectory[i + 1],
                    )
                    * 1000.0
                )

                max_step_mm = max(
                    max_step_mm,
                    step_mm,
                )

            start = trajectory[0]
            end = trajectory[-1]

            rospy.loginfo(
                "%s[%d]: poseCount=%d | "
                "local=(%.3f, %.3f, %.3f) mm | "
                "reconstruction=%.9f mm | "
                "START=(%.3f, %.3f, %.3f) mm | "
                "END=(%.3f, %.3f, %.3f) mm | "
                "Z=%.3f..%.3f mm | "
                "maxStep=%.3f mm",
                side,
                index,
                len(trajectory),
                local_point[0] * 1000.0,
                local_point[1] * 1000.0,
                local_point[2] * 1000.0,
                reconstruction_error_mm,
                start.x * 1000.0,
                start.y * 1000.0,
                start.z * 1000.0,
                end.x * 1000.0,
                end.y * 1000.0,
                end.z * 1000.0,
                min(z_values_mm),
                max(z_values_mm),
                max_step_mm,
            )

            self.add_trajectory_marker(
                marker_array,
                side,
                index,
                trajectory,
            )

        return len(p0_msg.poses)

    def update(self):
        if (
            self.pose_history_msg is None
            or self.p0_upper_msg is None
            or self.p0_lower_msg is None
        ):
            return

        pose_stamp = self.stamp_nsec(
            self.pose_history_msg
        )
        upper_stamp = self.stamp_nsec(
            self.p0_upper_msg
        )
        lower_stamp = self.stamp_nsec(
            self.p0_lower_msg
        )

        if not (
            pose_stamp == upper_stamp == lower_stamp
        ):
            return

        if self.last_processed_stamp == pose_stamp:
            return

        if len(self.pose_history_msg.poses) == 0:
            return

        marker_array = MarkerArray()

        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        upper_count = self.process_side(
            "upper",
            self.p0_upper_msg,
            self.pose_history_msg.poses,
            marker_array,
        )

        lower_count = self.process_side(
            "lower",
            self.p0_lower_msg,
            self.pose_history_msg.poses,
            marker_array,
        )

        self.marker_pub.publish(marker_array)

        self.last_processed_stamp = pose_stamp

        rospy.loginfo(
            "P0 trajectories published: "
            "poseCount=%d upper=%d lower=%d",
            len(self.pose_history_msg.poses),
            upper_count,
            lower_count,
        )


if __name__ == "__main__":
    rospy.init_node(
        "cobotta_p0_trajectory_visualizer"
    )

    CobottaP0TrajectoryVisualizer()
    rospy.spin()
