#!/usr/bin/env python3

import math
import threading

import numpy as np
import rospy

from geometry_msgs.msg import Point
from geometry_msgs.msg import PolygonStamped
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import ColorRGBA
from tf.transformations import quaternion_from_matrix
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


class CobottaToolOrientationVisualizer:
    """
    Unityから受信した実折り筋・動く側方向・動く紙面法線から、
    COBOTTA工具姿勢の候補A/Bを生成してRVizへ表示する。

    工具ローカル軸:
      Z_tool: 工具長手方向
      Y_tool: グリッパ開閉方向
      X_tool: 残りの横方向

    候補A:
      X_tool = -paper_side
      Y_tool = +paper_normal
      Z_tool = +fold_axis

    候補B:
      X_tool = +paper_side
      Y_tool = -paper_normal
      Z_tool = +fold_axis
    """

    def __init__(self):
        self.output_frame = rospy.get_param(
            "~output_frame",
            "paper_center",
        )
        self.visual_z_offset = rospy.get_param(
            "~visual_z_offset",
            0.010,
        )
        self.axis_length = rospy.get_param(
            "~axis_length",
            0.060,
        )

        self.fold_line_points = None
        self.fold_side_vector = None
        self.folding_normal_vector = None

        # 3入力を同一組として処理するための更新フラグ。
        self.fold_line_updated = False
        self.fold_side_updated = False
        self.folding_normal_updated = False
        self.input_lock = threading.Lock()

        self.previous_fold_axis = None
        self.previous_quaternion_a = None
        self.previous_quaternion_b = None
        self.previous_quaternion_c = None
        self.previous_quaternion_target = None

        # 真の初期紙面に対するTest9成功姿勢の固定補正。
        # 候補Cの工具ローカルY軸まわりに-11度回転する。
        self.tool_pitch_offset_deg = rospy.get_param(
            "~tool_pitch_offset_deg",
            -11.0,
        )

        offset_angle = math.radians(
            self.tool_pitch_offset_deg
        )
        offset_cos = math.cos(offset_angle)
        offset_sin = math.sin(offset_angle)

        self.tool_offset_rotation = np.array(
            [
                [
                    offset_cos,
                    0.0,
                    offset_sin,
                ],
                [
                    0.0,
                    1.0,
                    0.0,
                ],
                [
                    -offset_sin,
                    0.0,
                    offset_cos,
                ],
            ],
            dtype=float,
        )

        self.pose_a_pub = rospy.Publisher(
            "/origami/debug/cobotta_tool_target_pose_a",
            PoseStamped,
            queue_size=1,
            latch=False,
        )
        self.pose_b_pub = rospy.Publisher(
            "/origami/debug/cobotta_tool_target_pose_b",
            PoseStamped,
            queue_size=1,
            latch=False,
        )
        self.axes_a_pub = rospy.Publisher(
            "/origami/debug/cobotta_tool_target_axes_a",
            MarkerArray,
            queue_size=1,
            latch=False,
        )
        self.axes_b_pub = rospy.Publisher(
            "/origami/debug/cobotta_tool_target_axes_b",
            MarkerArray,
            queue_size=1,
            latch=False,
        )
        self.pose_c_pub = rospy.Publisher(
            "/origami/debug/cobotta_tool_candidate_c_pose",
            PoseStamped,
            queue_size=1,
            latch=False,
        )
        self.axes_c_pub = rospy.Publisher(
            "/origami/debug/cobotta_tool_candidate_c_axes",
            MarkerArray,
            queue_size=1,
            latch=False,
        )
        self.target_pose_pub = rospy.Publisher(
            "/origami/debug/cobotta_tool_target_pose",
            PoseStamped,
            queue_size=1,
            latch=False,
        )
        self.target_axes_pub = rospy.Publisher(
            "/origami/debug/cobotta_tool_target_axes",
            MarkerArray,
            queue_size=1,
            latch=False,
        )

        rospy.Subscriber(
            "/origami/fold_line_unity",
            PolygonStamped,
            self.fold_line_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            "/origami/fold_side_direction_unity",
            Vector3Stamped,
            self.fold_side_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            "/origami/folding_paper_normal_unity",
            Vector3Stamped,
            self.folding_normal_callback,
            queue_size=1,
        )

        rospy.loginfo(
            "COBOTTA tool orientation visualizer started"
        )
        rospy.loginfo(
            "  output frame: %s",
            self.output_frame,
        )
        rospy.loginfo(
            "  visual z offset: %.3f m",
            self.visual_z_offset,
        )

    @staticmethod
    def unity_point_to_ros(point):
        """
        Unity点座標をpaper_center座標へ変換する。

          x_ros = z_unity
          y_ros = -x_unity
          z_ros = y_unity

        計算用座標なので表示オフセットは加えない。
        """
        return np.array(
            [
                point.z,
                -point.x,
                point.y,
            ],
            dtype=float,
        )

    @staticmethod
    def unity_vector_to_ros(vector):
        """
        Unity方向ベクトルをpaper_center座標へ変換する。
        """
        return np.array(
            [
                vector.z,
                -vector.x,
                vector.y,
            ],
            dtype=float,
        )

    @staticmethod
    def normalize(vector, name):
        vector = np.asarray(vector, dtype=float)

        if not np.all(np.isfinite(vector)):
            raise ValueError(
                "{} contains NaN or Inf".format(name)
            )

        length = np.linalg.norm(vector)

        if length < 1.0e-9:
            raise ValueError(
                "{} is too short".format(name)
            )

        return vector / length

    @staticmethod
    def numpy_to_point(vector):
        point = Point()
        point.x = float(vector[0])
        point.y = float(vector[1])
        point.z = float(vector[2])
        return point

    @staticmethod
    def make_color(red, green, blue, alpha=1.0):
        color = ColorRGBA()
        color.r = red
        color.g = green
        color.b = blue
        color.a = alpha
        return color

    def fold_line_callback(self, message):
        if len(message.polygon.points) < 2:
            rospy.logwarn_throttle(
                2.0,
                "Fold line contains fewer than two points",
            )
            return

        with self.input_lock:
            self.fold_line_points = [
                self.unity_point_to_ros(
                    message.polygon.points[0]
                ),
                self.unity_point_to_ros(
                    message.polygon.points[1]
                ),
            ]
            self.fold_line_updated = True
            self.try_update_orientation_locked()

    def fold_side_callback(self, message):
        with self.input_lock:
            self.fold_side_vector = (
                self.unity_vector_to_ros(
                    message.vector
                )
            )
            self.fold_side_updated = True
            self.try_update_orientation_locked()

    def folding_normal_callback(self, message):
        with self.input_lock:
            self.folding_normal_vector = (
                self.unity_vector_to_ros(
                    message.vector
                )
            )
            self.folding_normal_updated = True
            self.try_update_orientation_locked()

    def try_update_orientation_locked(self):
        """
        折り筋・動く側・紙面法線がすべて更新された場合だけ、
        3入力を1組として工具姿勢を生成する。

        この関数はself.input_lockを取得した状態で呼び出す。
        """
        if not (
            self.fold_line_updated
            and self.fold_side_updated
            and self.folding_normal_updated
        ):
            return

        # 次の入力組を取りこぼさないよう、計算前に解除する。
        self.fold_line_updated = False
        self.fold_side_updated = False
        self.folding_normal_updated = False

        self.update_orientation()

    def calculate_basis(self):
        if self.fold_line_points is None:
            return None

        if self.fold_side_vector is None:
            return None

        if self.folding_normal_vector is None:
            return None

        fold_start = self.fold_line_points[0]
        fold_end = self.fold_line_points[1]

        fold_axis = self.normalize(
            fold_end - fold_start,
            "fold_axis",
        )

        normal_raw = self.normalize(
            self.folding_normal_vector,
            "folding_normal_raw",
        )

        # 紙面法線から折り筋方向成分を除去する。
        normal_projected = (
            normal_raw
            - np.dot(normal_raw, fold_axis) * fold_axis
        )
        paper_normal = self.normalize(
            normal_projected,
            "paper_normal",
        )

        # 動く側方向も折り筋に垂直な成分へ投影する。
        fold_side_projected = (
            self.fold_side_vector
            - np.dot(
                self.fold_side_vector,
                fold_axis,
            ) * fold_axis
        )
        fold_side = self.normalize(
            fold_side_projected,
            "fold_side",
        )

        paper_side = self.normalize(
            np.cross(fold_axis, paper_normal),
            "paper_side",
        )

        # cross(fold_axis, paper_normal)が
        # Unityから得た動く側を向くように符号を決める。
        if np.dot(paper_side, fold_side) < 0.0:
            fold_axis = -fold_axis
            paper_side = -paper_side

        # 始点・終点の順序が変わっても、
        # 前回の折り筋方向との不連続がないことを確認する。
        if self.previous_fold_axis is not None:
            continuity = np.dot(
                self.previous_fold_axis,
                fold_axis,
            )

            if continuity < 0.0:
                rospy.logwarn_throttle(
                    1.0,
                    "Fold axis sign discontinuity detected",
                )

        self.previous_fold_axis = fold_axis.copy()

        midpoint = 0.5 * (fold_start + fold_end)

        return (
            midpoint,
            fold_axis,
            paper_normal,
            paper_side,
            fold_side,
        )

    def rotation_to_quaternion(
        self,
        rotation,
        previous_quaternion,
        candidate_name,
    ):
        orthogonality = np.matmul(
            rotation.T,
            rotation,
        )
        determinant = np.linalg.det(rotation)

        if not np.allclose(
            orthogonality,
            np.eye(3),
            atol=1.0e-5,
        ):
            raise ValueError(
                "{} rotation is not orthonormal".format(
                    candidate_name
                )
            )

        if not math.isclose(
            determinant,
            1.0,
            abs_tol=1.0e-5,
        ):
            raise ValueError(
                "{} determinant is {}".format(
                    candidate_name,
                    determinant,
                )
            )

        transform = np.eye(4)
        transform[:3, :3] = rotation

        quaternion = quaternion_from_matrix(
            transform
        )
        quaternion = self.normalize(
            quaternion,
            "{} quaternion".format(candidate_name),
        )

        # qと-qは同じ姿勢なので、前回値との内積が
        # 負の場合は符号を反転して連続性を維持する。
        if previous_quaternion is not None:
            if np.dot(
                previous_quaternion,
                quaternion,
            ) < 0.0:
                quaternion = -quaternion

        return quaternion

    def make_pose(
        self,
        stamp,
        position,
        quaternion,
    ):
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self.output_frame

        pose.pose.position = self.numpy_to_point(
            position
        )

        pose.pose.orientation.x = float(
            quaternion[0]
        )
        pose.pose.orientation.y = float(
            quaternion[1]
        )
        pose.pose.orientation.z = float(
            quaternion[2]
        )
        pose.pose.orientation.w = float(
            quaternion[3]
        )

        return pose

    def make_axis_marker(
        self,
        stamp,
        namespace,
        marker_id,
        origin,
        direction,
        color,
    ):
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = self.output_frame

        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        marker.points = [
            self.numpy_to_point(origin),
            self.numpy_to_point(
                origin + self.axis_length * direction
            ),
        ]

        marker.scale.x = 0.004
        marker.scale.y = 0.008
        marker.scale.z = 0.012

        marker.color = color
        marker.lifetime = rospy.Duration(0.0)

        return marker

    def make_axes(
        self,
        stamp,
        namespace,
        position,
        rotation,
    ):
        # Marker表示だけ10 mm持ち上げる。
        visual_origin = position.copy()
        visual_origin[2] += self.visual_z_offset

        markers = MarkerArray()

        markers.markers.append(
            self.make_axis_marker(
                stamp,
                namespace,
                0,
                visual_origin,
                rotation[:, 0],
                self.make_color(1.0, 0.0, 0.0),
            )
        )
        markers.markers.append(
            self.make_axis_marker(
                stamp,
                namespace,
                1,
                visual_origin,
                rotation[:, 1],
                self.make_color(0.0, 1.0, 0.0),
            )
        )
        markers.markers.append(
            self.make_axis_marker(
                stamp,
                namespace,
                2,
                visual_origin,
                rotation[:, 2],
                self.make_color(0.0, 0.0, 1.0),
            )
        )

        return markers

    def update_orientation(self):
        try:
            result = self.calculate_basis()

            if result is None:
                return

            (
                midpoint,
                fold_axis,
                paper_normal,
                paper_side,
                fold_side,
            ) = result

            # 回転行列の列は、
            # paper_center上で見たX_tool, Y_tool, Z_tool。
            rotation_a = np.column_stack(
                (
                    -paper_side,
                    paper_normal,
                    fold_axis,
                )
            )

            rotation_b = np.column_stack(
                (
                    paper_side,
                    -paper_normal,
                    fold_axis,
                )
            )

            # Test9工具軸と紙面基底の対応。
            #
            #   X_tool = -paper_normal
            #   Y_tool = -paper_side
            #   Z_tool =  fold_axis
            rotation_c = np.column_stack(
                (
                    -paper_normal,
                    -paper_side,
                    fold_axis,
                )
            )

            # 初期状態で成功済みTest9姿勢と一致し、
            # その固定傾斜を保持したまま紙面へ追従する。
            rotation_target = np.matmul(
                rotation_c,
                self.tool_offset_rotation,
            )

            quaternion_a = self.rotation_to_quaternion(
                rotation_a,
                self.previous_quaternion_a,
                "candidate A",
            )
            quaternion_b = self.rotation_to_quaternion(
                rotation_b,
                self.previous_quaternion_b,
                "candidate B",
            )
            quaternion_c = self.rotation_to_quaternion(
                rotation_c,
                self.previous_quaternion_c,
                "candidate C",
            )
            quaternion_target = self.rotation_to_quaternion(
                rotation_target,
                self.previous_quaternion_target,
                "tracked target",
            )

            self.previous_quaternion_a = (
                quaternion_a.copy()
            )
            self.previous_quaternion_b = (
                quaternion_b.copy()
            )
            self.previous_quaternion_c = (
                quaternion_c.copy()
            )
            self.previous_quaternion_target = (
                quaternion_target.copy()
            )

            stamp = rospy.Time.now()

            self.pose_a_pub.publish(
                self.make_pose(
                    stamp,
                    midpoint,
                    quaternion_a,
                )
            )
            self.pose_b_pub.publish(
                self.make_pose(
                    stamp,
                    midpoint,
                    quaternion_b,
                )
            )
            self.pose_c_pub.publish(
                self.make_pose(
                    stamp,
                    midpoint,
                    quaternion_c,
                )
            )
            self.target_pose_pub.publish(
                self.make_pose(
                    stamp,
                    midpoint,
                    quaternion_target,
                )
            )

            self.axes_a_pub.publish(
                self.make_axes(
                    stamp,
                    "cobotta_tool_candidate_a",
                    midpoint,
                    rotation_a,
                )
            )
            self.axes_b_pub.publish(
                self.make_axes(
                    stamp,
                    "cobotta_tool_candidate_b",
                    midpoint,
                    rotation_b,
                )
            )
            self.axes_c_pub.publish(
                self.make_axes(
                    stamp,
                    "cobotta_tool_candidate_c",
                    midpoint,
                    rotation_c,
                )
            )
            self.target_axes_pub.publish(
                self.make_axes(
                    stamp,
                    "cobotta_tool_tracked_target",
                    midpoint,
                    rotation_target,
                )
            )

            rospy.loginfo_throttle(
                1.0,
                (
                    "Tool orientation valid: "
                    "side_alignment=%.6f, "
                    "detA=%.6f, detB=%.6f, "
                    "qA_norm=%.6f, qB_norm=%.6f"
                ),
                float(np.dot(paper_side, fold_side)),
                float(np.linalg.det(rotation_a)),
                float(np.linalg.det(rotation_b)),
                float(np.linalg.norm(quaternion_a)),
                float(np.linalg.norm(quaternion_b)),
            )

            rospy.loginfo_throttle(
                1.0,
                (
                    "Tracked target valid: "
                    "detC=%.6f, detTarget=%.6f, "
                    "qC_norm=%.6f, qTarget_norm=%.6f, "
                    "offset_angle=%.6f deg"
                ),
                float(np.linalg.det(rotation_c)),
                float(np.linalg.det(rotation_target)),
                float(np.linalg.norm(quaternion_c)),
                float(np.linalg.norm(quaternion_target)),
                abs(float(self.tool_pitch_offset_deg)),
            )

        except ValueError as error:
            rospy.logwarn_throttle(
                1.0,
                "Tool orientation calculation skipped: %s",
                str(error),
            )


def main():
    rospy.init_node(
        "cobotta_tool_orientation_visualizer"
    )

    CobottaToolOrientationVisualizer()

    rospy.spin()


if __name__ == "__main__":
    main()
