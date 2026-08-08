#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import math
import os
import sys

import moveit_commander
import rospy
import yaml

from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory


class UnityTrajectoryRobotPlanner:
    """Unity軌道を計画し、1本のRobotTrajectoryへ統合する。"""

    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)

        self.trajectory_file = os.path.expanduser(
            rospy.get_param(
                "~trajectory_file",
                "~/grasp_trajectory_auto_fold.yaml",
            )
        )

        self.output_file = os.path.expanduser(
            rospy.get_param(
                "~output_file",
                "~/combined_robot_trajectory_0_19.yaml",
            )
        )

        self.start_index = rospy.get_param("~start_index", 0)
        self.end_index = rospy.get_param("~end_index", 19)

        self.reference_frame = rospy.get_param(
            "~reference_frame",
            "paper_center",
        )

        self.fixed_z = rospy.get_param("~fixed_z", 0.065)

        # unity_fixed_z:
        #   従来どおりUnity座標を変換し、Zはfixed_zを使用
        # ros_xyz:
        #   入力YAMLのtranslationをpaper_center基準ROS座標として直接使用
        self.input_mode = rospy.get_param(
            "~input_mode",
            "unity_fixed_z",
        )

        if self.input_mode not in (
            "unity_fixed_z",
            "ros_xyz",
        ):
            raise rospy.ROSInitException(
                "未対応のinput_modeです: {}".format(
                    self.input_mode
                )
            )

        self.orientation = [
            rospy.get_param("~orientation_x", 0.232963),
            rospy.get_param("~orientation_y", 0.562422),
            rospy.get_param("~orientation_z", -0.303603),
            rospy.get_param("~orientation_w", 0.732963),
        ]

        # fixed:
        #   従来どおり全点で固定Quaternionを使用
        # from_input:
        #   入力MultiDOFJointTrajectoryのrotationを各点で使用
        self.orientation_mode = rospy.get_param(
            "~orientation_mode",
            "fixed",
        )

        if self.orientation_mode not in (
            "fixed",
            "from_input",
        ):
            raise rospy.ROSInitException(
                "未対応のorientation_modeです: {}".format(
                    self.orientation_mode
                )
            )

        self.group_name = rospy.get_param(
            "~group_name",
            "cobotta_arm",
        )

        self.end_effector_link = rospy.get_param(
            "~end_effector_link",
            "cobotta_tool_link",
        )

        # 実機の現在関節角をMoveIt計画開始状態として指定できるようにする
        self.start_joint_positions = rospy.get_param(
            "~start_joint_positions",
            [],
        )

        self.planning_time = rospy.get_param(
            "~planning_time",
            5.0,
        )

        self.planning_attempts = rospy.get_param(
            "~planning_attempts",
            10,
        )

        self.velocity_scaling = rospy.get_param(
            "~velocity_scaling",
            0.10,
        )

        self.acceleration_scaling = rospy.get_param(
            "~acceleration_scaling",
            0.10,
        )

        self.duplicate_tolerance = rospy.get_param(
            "~duplicate_tolerance",
            1.0e-6,
        )

        self.display_wait = rospy.get_param(
            "~display_wait",
            5.0,
        )

        self._normalize_orientation()

        self.move_group = moveit_commander.MoveGroupCommander(
            self.group_name,
            wait_for_servers=20.0,
        )

        self.move_group.set_end_effector_link(
            self.end_effector_link
        )
        self.move_group.set_planning_time(
            self.planning_time
        )
        self.move_group.set_num_planning_attempts(
            self.planning_attempts
        )
        self.move_group.set_max_velocity_scaling_factor(
            self.velocity_scaling
        )
        self.move_group.set_max_acceleration_scaling_factor(
            self.acceleration_scaling
        )

        self.display_publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            DisplayTrajectory,
            queue_size=1,
            latch=True,
        )

        rospy.loginfo(
            "Planning Group: %s",
            self.group_name,
        )
        rospy.loginfo(
            "End Effector Link: %s",
            self.end_effector_link,
        )
        rospy.loginfo(
            "対象index: %d～%d",
            self.start_index,
            self.end_index,
        )
        rospy.loginfo(
            "固定高さ: %.6f m",
            self.fixed_z,
        )
        rospy.loginfo(
            "姿勢入力モード: %s",
            self.orientation_mode,
        )
        rospy.loginfo(
            "速度スケーリング: %.3f",
            self.velocity_scaling,
        )
        rospy.loginfo(
            "加速度スケーリング: %.3f",
            self.acceleration_scaling,
        )
        rospy.logwarn(
            "Plan only: 実機Executeは行いません。"
        )

    def _normalize_orientation(self):
        norm = math.sqrt(
            sum(value * value for value in self.orientation)
        )

        if norm <= 1.0e-12:
            raise ValueError(
                "姿勢Quaternionのノルムが0です。"
            )

        self.orientation = [
            value / norm
            for value in self.orientation
        ]

    def load_unity_points(self):
        if not os.path.isfile(self.trajectory_file):
            raise FileNotFoundError(
                "軌道ファイルがありません: {}".format(
                    self.trajectory_file
                )
            )

        with open(
            self.trajectory_file,
            "r",
            encoding="utf-8",
        ) as yaml_file:
            documents = [
                document
                for document in yaml.safe_load_all(yaml_file)
                if document is not None
            ]

        if not documents:
            raise ValueError(
                "YAML内に有効な文書がありません。"
            )

        data = documents[0]
        points = data.get("points")

        if not isinstance(points, list) or not points:
            raise ValueError(
                "YAML内に有効なpoints配列がありません。"
            )

        if self.start_index < 0:
            raise ValueError(
                "start_indexは0以上にしてください。"
            )

        if self.end_index < self.start_index:
            raise ValueError(
                "end_indexはstart_index以上にしてください。"
            )

        if self.end_index >= len(points):
            raise IndexError(
                "end_index={}ですが、軌道点数は{}点です。".format(
                    self.end_index,
                    len(points),
                )
            )

        rospy.loginfo(
            "Unity軌道点数: %d",
            len(points),
        )

        return points

    def make_target_pose(self, unity_point, point_index):
        try:
            translation = (
                unity_point["transforms"][0]["translation"]
            )

            unity_x = float(translation["x"])
            unity_y = float(translation["y"])
            unity_z = float(translation["z"])

        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
        ) as error:
            raise ValueError(
                "index {} のUnity座標を取得できません: {}".format(
                    point_index,
                    error,
                )
            )

        if self.input_mode == "ros_xyz":
            # 入力値は既にpaper_center基準のROS座標
            ros_x = unity_x
            ros_y = unity_y
            ros_z = unity_z
        else:
            # 従来のUnity → paper_center変換
            ros_x = unity_z
            ros_y = -unity_x
            ros_z = self.fixed_z

        pose = PoseStamped()

        # 最新のTFを使用する
        pose.header.stamp = rospy.Time(0)
        pose.header.frame_id = self.reference_frame

        pose.pose.position.x = ros_x
        pose.pose.position.y = ros_y
        pose.pose.position.z = ros_z

        if self.orientation_mode == "from_input":
            try:
                rotation = (
                    unity_point["transforms"][0]["rotation"]
                )

                orientation = [
                    float(rotation["x"]),
                    float(rotation["y"]),
                    float(rotation["z"]),
                    float(rotation["w"]),
                ]

            except (
                KeyError,
                IndexError,
                TypeError,
                ValueError,
            ) as error:
                raise ValueError(
                    "index {} の入力Quaternionを取得できません: "
                    "{}".format(
                        point_index,
                        error,
                    )
                )

            norm = math.sqrt(
                sum(value * value for value in orientation)
            )

            if norm <= 1.0e-12:
                raise ValueError(
                    "index {} の入力Quaternionノルムが0です。"
                    .format(point_index)
                )

            orientation = [
                value / norm
                for value in orientation
            ]

        else:
            orientation = self.orientation

        pose.pose.orientation.x = orientation[0]
        pose.pose.orientation.y = orientation[1]
        pose.pose.orientation.z = orientation[2]
        pose.pose.orientation.w = orientation[3]

        rospy.loginfo(
            "[index %d] orientation: "
            "x=%.9f, y=%.9f, z=%.9f, w=%.9f",
            point_index,
            orientation[0],
            orientation[1],
            orientation[2],
            orientation[3],
        )

        return pose

    @staticmethod
    def unpack_plan_result(result):
        if isinstance(result, tuple):
            success = bool(result[0])
            trajectory = result[1]
            planning_time = float(result[2])
        else:
            trajectory = result
            success = bool(
                trajectory.joint_trajectory.points
            )
            planning_time = 0.0

        return success, trajectory, planning_time

    @staticmethod
    def make_final_state(start_state, trajectory):
        joint_names = list(
            trajectory.joint_trajectory.joint_names
        )

        final_positions = list(
            trajectory.joint_trajectory.points[-1].positions
        )

        final_state = copy.deepcopy(start_state)

        state_names = list(
            final_state.joint_state.name
        )
        state_positions = list(
            final_state.joint_state.position
        )

        state_index = {
            name: index
            for index, name in enumerate(state_names)
        }

        for joint_name, joint_position in zip(
            joint_names,
            final_positions,
        ):
            if joint_name not in state_index:
                raise RuntimeError(
                    "RobotStateに関節がありません: {}".format(
                        joint_name
                    )
                )

            state_positions[state_index[joint_name]] = (
                joint_position
            )

        final_state.joint_state.position = state_positions
        final_state.joint_state.header.stamp = rospy.Time.now()
        final_state.is_diff = False

        return final_state

    def plan_segments(self, unity_points):
        initial_state = self.move_group.get_current_state()

        if self.start_joint_positions:
            active_joints = list(
                self.move_group.get_active_joints()
            )

            if len(self.start_joint_positions) != len(active_joints):
                raise ValueError(
                    "start_joint_positionsは{}要素必要ですが、"
                    "{}要素です。".format(
                        len(active_joints),
                        len(self.start_joint_positions),
                    )
                )

            state_names = list(initial_state.joint_state.name)
            state_positions = list(
                initial_state.joint_state.position
            )
            state_index = {
                name: index
                for index, name in enumerate(state_names)
            }

            for joint_name, joint_position in zip(
                active_joints,
                self.start_joint_positions,
            ):
                if joint_name not in state_index:
                    raise RuntimeError(
                        "RobotStateに関節がありません: {}".format(
                            joint_name
                        )
                    )

                state_positions[state_index[joint_name]] = float(
                    joint_position
                )

            initial_state.joint_state.position = state_positions
            initial_state.joint_state.header.stamp = rospy.Time(0)
            initial_state.is_diff = False

            rospy.loginfo(
                "計画開始状態に実機関節角を使用: %s",
                self.start_joint_positions,
            )
        else:
            rospy.loginfo(
                "計画開始状態にMoveIt現在状態を使用"
            )

        start_state = copy.deepcopy(initial_state)

        segments = []
        total_planning_time = 0.0

        for point_index in range(
            self.start_index,
            self.end_index + 1,
        ):
            if rospy.is_shutdown():
                raise rospy.ROSInterruptException()

            target_pose = self.make_target_pose(
                unity_points[point_index],
                point_index,
            )

            rospy.loginfo(
                "[index %d] target: "
                "frame=%s, x=%.6f, y=%.6f, z=%.6f",
                point_index,
                target_pose.header.frame_id,
                target_pose.pose.position.x,
                target_pose.pose.position.y,
                target_pose.pose.position.z,
            )

            try:
                self.move_group.set_start_state(start_state)
                self.move_group.set_pose_target(
                    target_pose,
                    self.end_effector_link,
                )

                result = self.move_group.plan()

            finally:
                self.move_group.clear_pose_targets()

            (
                success,
                trajectory,
                planning_time,
            ) = self.unpack_plan_result(result)

            if (
                not success
                or not trajectory.joint_trajectory.points
            ):
                raise RuntimeError(
                    "index {} の軌道計画に失敗しました。".format(
                        point_index
                    )
                )

            point_count = len(
                trajectory.joint_trajectory.points
            )

            rospy.loginfo(
                "[index %d] 計画成功: "
                "軌道点数=%d, planning_time=%.3f s",
                point_index,
                point_count,
                planning_time,
            )

            segments.append(
                {
                    "index": point_index,
                    "trajectory": trajectory,
                    "planning_time": planning_time,
                }
            )

            total_planning_time += planning_time

            start_state = self.make_final_state(
                start_state,
                trajectory,
            )

        rospy.loginfo(
            "個別計画合計時間: %.3f s",
            total_planning_time,
        )

        return initial_state, segments

    def positions_are_equal(
        self,
        first_positions,
        second_positions,
    ):
        if len(first_positions) != len(second_positions):
            return False

        return all(
            abs(first - second)
            <= self.duplicate_tolerance
            for first, second in zip(
                first_positions,
                second_positions,
            )
        )

    def combine_segments(self, segments):
        if not segments:
            raise ValueError(
                "結合対象の軌道がありません。"
            )

        combined = RobotTrajectory()

        first_trajectory = segments[0]["trajectory"]
        expected_joint_names = list(
            first_trajectory.joint_trajectory.joint_names
        )

        combined.joint_trajectory.header = copy.deepcopy(
            first_trajectory.joint_trajectory.header
        )

        combined.joint_trajectory.joint_names = (
            expected_joint_names
        )

        source_point_count = 0
        duplicate_count = 0

        last_positions = None

        for segment in segments:
            trajectory = segment["trajectory"]

            joint_names = list(
                trajectory.joint_trajectory.joint_names
            )

            if joint_names != expected_joint_names:
                raise RuntimeError(
                    "区間ごとの関節名または順序が一致しません。"
                )

            for source_point in (
                trajectory.joint_trajectory.points
            ):
                source_point_count += 1

                if (
                    last_positions is not None
                    and self.positions_are_equal(
                        last_positions,
                        source_point.positions,
                    )
                ):
                    duplicate_count += 1
                    continue

                point = copy.deepcopy(source_point)

                # 全区間を結合後に再度時間付与するため初期化
                point.time_from_start = rospy.Duration(0)
                point.velocities = []
                point.accelerations = []
                point.effort = []

                combined.joint_trajectory.points.append(
                    point
                )

                last_positions = list(
                    source_point.positions
                )

        if len(combined.joint_trajectory.points) < 2:
            raise RuntimeError(
                "重複除去後の軌道点が2点未満です。"
            )

        rospy.loginfo(
            "個別軌道内の合計点数: %d",
            source_point_count,
        )
        rospy.loginfo(
            "除去した連続重複点数: %d",
            duplicate_count,
        )
        rospy.loginfo(
            "結合後の軌道点数: %d",
            len(combined.joint_trajectory.points),
        )

        return combined

    def retime_combined_trajectory(
        self,
        initial_state,
        combined,
    ):
        retimed = self.move_group.retime_trajectory(
            initial_state,
            combined,
            velocity_scaling_factor=self.velocity_scaling,
            acceleration_scaling_factor=(
                self.acceleration_scaling
            ),
            algorithm="iterative_time_parameterization",
        )

        if (
            retimed is None
            or not retimed.joint_trajectory.points
        ):
            raise RuntimeError(
                "時間パラメータ付与に失敗しました。"
            )

        return retimed

    @staticmethod
    def validate_trajectory(trajectory):
        points = trajectory.joint_trajectory.points
        joint_count = len(
            trajectory.joint_trajectory.joint_names
        )

        times = []

        for point_index, point in enumerate(points):
            if len(point.positions) != joint_count:
                raise RuntimeError(
                    "軌道点{}の関節角数が一致しません。".format(
                        point_index
                    )
                )

            if not all(
                math.isfinite(value)
                for value in point.positions
            ):
                raise RuntimeError(
                    "軌道点{}に非有限の関節角があります。".format(
                        point_index
                    )
                )

            times.append(
                point.time_from_start.to_sec()
            )

        non_increasing = [
            index
            for index in range(1, len(times))
            if times[index] <= times[index - 1]
        ]

        if non_increasing:
            raise RuntimeError(
                "time_from_startが単調増加していません。"
                " 該当点={}".format(non_increasing)
            )

        minimum_interval = min(
            times[index] - times[index - 1]
            for index in range(1, len(times))
        )

        return {
            "point_count": len(points),
            "total_duration": times[-1],
            "minimum_interval": minimum_interval,
        }

    def save_trajectory(self, trajectory):
        output_directory = os.path.dirname(
            self.output_file
        )

        if output_directory:
            os.makedirs(
                output_directory,
                exist_ok=True,
            )

        with open(
            self.output_file,
            "w",
            encoding="utf-8",
        ) as output_file:
            output_file.write(str(trajectory))
            output_file.write("\n")

    def publish_trajectory(
        self,
        initial_state,
        trajectory,
    ):
        display = DisplayTrajectory()
        display.trajectory_start = initial_state
        display.trajectory.append(trajectory)

        self.display_publisher.publish(display)

        rospy.loginfo(
            "統合RobotTrajectoryをRVizへPublishしました。"
        )

    def run(self):
        unity_points = self.load_unity_points()

        initial_state, segments = self.plan_segments(
            unity_points
        )

        combined = self.combine_segments(segments)

        retimed = self.retime_combined_trajectory(
            initial_state,
            combined,
        )

        validation = self.validate_trajectory(
            retimed
        )

        self.save_trajectory(retimed)

        self.publish_trajectory(
            initial_state,
            retimed,
        )

        rospy.loginfo(
            "============ 統合軌道結果 ============"
        )
        rospy.loginfo(
            "入力目標点数: %d",
            self.end_index - self.start_index + 1,
        )
        rospy.loginfo(
            "統合後の関節軌道点数: %d",
            validation["point_count"],
        )
        rospy.loginfo(
            "総軌道時間: %.6f s",
            validation["total_duration"],
        )
        rospy.loginfo(
            "最小時間間隔: %.9f s",
            validation["minimum_interval"],
        )
        rospy.loginfo(
            "time_from_start: 単調増加"
        )
        rospy.loginfo(
            "保存先: %s",
            self.output_file,
        )
        rospy.loginfo(
            "実機Execute: 実施していません"
        )
        rospy.loginfo(
            "======================================"
        )

        rospy.sleep(self.display_wait)


def main():
    rospy.init_node(
        "plan_unity_trajectory_to_robot_trajectory"
    )

    try:
        planner = UnityTrajectoryRobotPlanner()
        planner.run()

    except (
        FileNotFoundError,
        ValueError,
        IndexError,
        RuntimeError,
        rospy.ROSException,
        rospy.ROSInterruptException,
    ) as error:
        rospy.logfatal("%s", error)
        sys.exit(1)

    finally:
        moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    main()
