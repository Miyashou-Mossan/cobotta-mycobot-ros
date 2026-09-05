#!/usr/bin/env python3

import math

import rospy
from geometry_msgs.msg import PolygonStamped, Pose, PoseArray
from visualization_msgs.msg import Marker, MarkerArray


def polygon_centroid(points):
    """
    2次元凸多角形の面積重心を求める。
    points: [(x, y), ...] [m]
    """
    if len(points) < 3:
        return None

    cross_sum = 0.0
    cx_sum = 0.0
    cy_sum = 0.0

    for i in range(len(points)):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % len(points)]

        cross = x0 * y1 - x1 * y0
        cross_sum += cross
        cx_sum += (x0 + x1) * cross
        cy_sum += (y0 + y1) * cross

    if abs(cross_sum) < 1.0e-12:
        return (
            sum(p[0] for p in points) / len(points),
            sum(p[1] for p in points) / len(points),
        )

    return (
        cx_sum / (3.0 * cross_sum),
        cy_sum / (3.0 * cross_sum),
    )


def generate_inner_candidates(points):
    """
    SAFE Polygon内部に候補点を生成する。

    1. 重心
    2. 重心→各頂点の中間点
    3. 重心→各辺中央の中間点

    三角形なら 1 + 3 + 3 = 7点。
    """
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
        x1, y1 = points[(i + 1) % len(points)]

        edge_mid = (
            0.5 * (x0 + x1),
            0.5 * (y0 + y1),
        )

        candidates.append((
            0.5 * (center[0] + edge_mid[0]),
            0.5 * (center[1] + edge_mid[1]),
        ))

    return candidates


def find_plane(points):
    """
    T0 Polygonから紙面平面を求める。

    戻り値:
        (p0, normal)
    """
    if len(points) < 3:
        return None

    p0 = points[0]

    for i in range(1, len(points) - 1):
        p1 = points[i]

        for j in range(i + 1, len(points)):
            p2 = points[j]

            ax = p1[0] - p0[0]
            ay = p1[1] - p0[1]
            az = p1[2] - p0[2]

            bx = p2[0] - p0[0]
            by = p2[1] - p0[1]
            bz = p2[2] - p0[2]

            nx = ay * bz - az * by
            ny = az * bx - ax * bz
            nz = ax * by - ay * bx

            norm = math.sqrt(nx * nx + ny * ny + nz * nz)

            if norm > 1.0e-12:
                return (
                    p0,
                    (nx / norm, ny / norm, nz / norm),
                )

    return None


def solve_z_on_plane(x, y, plane):
    """
    平面上の(x, y)に対応するzを求める。
    """
    p0, normal = plane
    nx, ny, nz = normal

    if abs(nz) < 1.0e-9:
        return None

    return (
        p0[2]
        - (
            nx * (x - p0[0])
            + ny * (y - p0[1])
        ) / nz
    )


class CobottaP0Candidates:
    def __init__(self):
        self.t0_msg = None
        self.safe_upper_msg = None
        self.safe_lower_msg = None

        self.upper_pub = rospy.Publisher(
            "/origami/cobotta_p0_candidates_upper",
            PoseArray,
            queue_size=1,
            latch=True,
        )

        self.lower_pub = rospy.Publisher(
            "/origami/cobotta_p0_candidates_lower",
            PoseArray,
            queue_size=1,
            latch=True,
        )

        self.marker_pub = rospy.Publisher(
            "/origami/cobotta_p0_candidate_markers",
            MarkerArray,
            queue_size=1,
            latch=True,
        )

        rospy.Subscriber(
            "/origami/active_folding_paper_t0_ros",
            PolygonStamped,
            self.t0_callback,
            queue_size=1,
        )

        rospy.Subscriber(
            "/origami/safe_grasp_area_upper",
            PolygonStamped,
            self.safe_upper_callback,
            queue_size=1,
        )

        rospy.Subscriber(
            "/origami/safe_grasp_area_lower",
            PolygonStamped,
            self.safe_lower_callback,
            queue_size=1,
        )

        rospy.loginfo("COBOTTA P0 candidate generator started.")

    def t0_callback(self, msg):
        self.t0_msg = msg
        self.update()

    def safe_upper_callback(self, msg):
        self.safe_upper_msg = msg
        self.update()

    def safe_lower_callback(self, msg):
        self.safe_lower_msg = msg
        self.update()

    @staticmethod
    def safe_xy(msg):
        if msg is None:
            return []

        return [
            (p.x, p.y)
            for p in msg.polygon.points
        ]

    @staticmethod
    def make_pose_array(points_xyz, stamp):
        msg = PoseArray()
        msg.header.stamp = stamp
        msg.header.frame_id = "paper_center"

        for x, y, z in points_xyz:
            pose = Pose()

            pose.position.x = x
            pose.position.y = y
            pose.position.z = z

            # 現段階ではP0「位置」のみ。
            # 工具姿勢は後段のCOBOTTA評価で決める。
            pose.orientation.w = 1.0

            msg.poses.append(pose)

        return msg

    @staticmethod
    def add_marker(marker_array, marker_id, namespace, points_xyz,
                   r, g, b):
        marker = Marker()

        # latched/static表示なので最新時刻依存にしない
        marker.header.stamp = rospy.Time(0)
        marker.header.frame_id = "paper_center"

        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD

        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.004
        marker.scale.y = 0.004
        marker.scale.z = 0.004

        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = 1.0

        from geometry_msgs.msg import Point

        for x, y, z in points_xyz:
            point = Point()
            point.x = x
            point.y = y
            point.z = z
            marker.points.append(point)

        marker_array.markers.append(marker)

    def make_3d_candidates(self, safe_msg, plane):
        safe_points = self.safe_xy(safe_msg)

        if len(safe_points) < 3:
            return []

        candidates_2d = generate_inner_candidates(safe_points)

        candidates_3d = []

        for x, y in candidates_2d:
            z = solve_z_on_plane(x, y, plane)

            if z is None:
                rospy.logwarn(
                    "T0 paper plane is nearly vertical; "
                    "cannot reconstruct P0 z."
                )
                return []

            candidates_3d.append((x, y, z))

        return candidates_3d

    def update(self):
        if self.t0_msg is None:
            return

        if (
            self.safe_upper_msg is None
            or self.safe_lower_msg is None
        ):
            return

        t0_stamp = self.t0_msg.header.stamp.to_nsec()
        upper_stamp = self.safe_upper_msg.header.stamp.to_nsec()
        lower_stamp = self.safe_lower_msg.header.stamp.to_nsec()

        if not (
            t0_stamp == upper_stamp == lower_stamp
        ):
            rospy.logdebug(
                "Waiting for synchronized T0/SAFE data: "
                "T0=%d upper=%d lower=%d",
                t0_stamp,
                upper_stamp,
                lower_stamp,
            )
            return

        output_stamp = self.t0_msg.header.stamp

        t0_points = [
            (p.x, p.y, p.z)
            for p in self.t0_msg.polygon.points
        ]

        plane = find_plane(t0_points)

        if plane is None:
            rospy.logwarn("Could not determine T0 paper plane.")
            return

        upper_points = self.make_3d_candidates(
            self.safe_upper_msg,
            plane,
        )

        lower_points = self.make_3d_candidates(
            self.safe_lower_msg,
            plane,
        )

        self.upper_pub.publish(
            self.make_pose_array(
                upper_points,
                output_stamp,
            )
        )

        self.lower_pub.publish(
            self.make_pose_array(
                lower_points,
                output_stamp,
            )
        )

        marker_array = MarkerArray()

        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        if upper_points:
            self.add_marker(
                marker_array,
                0,
                "p0_upper",
                upper_points,
                1.0, 0.5, 0.0,
            )

        if lower_points:
            self.add_marker(
                marker_array,
                1,
                "p0_lower",
                lower_points,
                0.0, 0.8, 1.0,
            )

        self.marker_pub.publish(marker_array)

        if upper_points:
            rospy.loginfo(
                "P0 upper candidates: %d",
                len(upper_points)
            )

            for i, (x, y, z) in enumerate(upper_points):
                rospy.loginfo(
                    "  upper[%d] = "
                    "(%.3f, %.3f, %.3f) mm",
                    i,
                    x * 1000.0,
                    y * 1000.0,
                    z * 1000.0,
                )

        if lower_points:
            rospy.loginfo(
                "P0 lower candidates: %d",
                len(lower_points)
            )

            for i, (x, y, z) in enumerate(lower_points):
                rospy.loginfo(
                    "  lower[%d] = "
                    "(%.3f, %.3f, %.3f) mm",
                    i,
                    x * 1000.0,
                    y * 1000.0,
                    z * 1000.0,
                )


if __name__ == "__main__":
    rospy.init_node("cobotta_p0_candidates")
    CobottaP0Candidates()
    rospy.spin()
