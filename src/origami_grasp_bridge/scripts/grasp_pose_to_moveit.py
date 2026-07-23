#!/usr/bin/env python3

import copy
import sys

import moveit_commander
import rospy
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import DisplayTrajectory
from moveit_msgs.srv import GetPositionFK, GetPositionFKRequest
from std_srvs.srv import Trigger, TriggerResponse


class GraspPosePlanner:
    """最新の把持Poseを保存し、明示的な指令を受けたときだけ計画する。"""

    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)

        self.latest_pose = None
        self.last_planned_state = None

        self.group_name = rospy.get_param("~group_name", "cobotta_arm")
        self.end_effector_link = rospy.get_param(
            "~end_effector_link",
            "cobotta_tool_link",
        )

        # position_only：位置だけを指定
        # pose：位置と姿勢の両方を指定
        self.target_mode = rospy.get_param(
            "~target_mode",
            "position_only",
        )

        # Trueの場合、前回の計画終点を次回計画の開始状態にする
        self.chain_plans = rospy.get_param(
            "~chain_plans",
            False,
        )

        if self.target_mode not in ("position_only", "pose"):
            raise rospy.ROSInitException(
                "未対応のtarget_modeです: {}".format(
                    self.target_mode
                )
            )

        # 紙面に直接接触させず、まず紙面上方の安全な接近点を計画する
        self.approach_offset_z = rospy.get_param(
            "~approach_offset_z",
            0.100,
        )

        self.move_group = moveit_commander.MoveGroupCommander(
            self.group_name,
            wait_for_servers=20.0,
        )

        self.move_group.set_end_effector_link(self.end_effector_link)
        self.move_group.set_planning_time(5.0)
        self.move_group.set_num_planning_attempts(10)
        self.move_group.set_max_velocity_scaling_factor(0.10)
        self.move_group.set_max_acceleration_scaling_factor(0.10)

        # 計画終点の関節角からTCP姿勢を求めるためのFKサービス
        rospy.wait_for_service("/compute_fk", timeout=20.0)
        self.compute_fk = rospy.ServiceProxy(
            "/compute_fk",
            GetPositionFK,
        )

        self.display_publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            DisplayTrajectory,
            queue_size=1,
            latch=True,
        )

        self.pose_subscriber = rospy.Subscriber(
            "/origami/grasp_pose",
            PoseStamped,
            self.pose_callback,
            queue_size=1,
        )

        self.plan_service = rospy.Service(
            "/origami/plan_grasp_pose",
            Trigger,
            self.plan_callback,
        )

        rospy.loginfo("grasp_pose_to_moveit node started")
        rospy.loginfo("Planning Group: %s", self.group_name)
        rospy.loginfo("End Effector Link: %s", self.end_effector_link)
        rospy.loginfo("Target mode: %s", self.target_mode)
        rospy.loginfo("Chain planned states: %s", self.chain_plans)
        rospy.loginfo(
            "Approach offset: %.3f m",
            self.approach_offset_z,
        )
        rospy.logwarn("Plan only: 実機Executeは行いません。")

    def pose_callback(self, message):
        """受信した最新Poseを保存する。受信時には計画しない。"""
        self.latest_pose = copy.deepcopy(message)

        rospy.loginfo_throttle(
            1.0,
            "Latest Pose: frame=%s, x=%.4f, y=%.4f, z=%.4f",
            message.header.frame_id,
            message.pose.position.x,
            message.pose.position.y,
            message.pose.position.z,
        )

    def plan_callback(self, _request):
        """サービス指令を受けたときだけ、安全な接近点への計画を行う。"""
        if self.latest_pose is None:
            return TriggerResponse(
                success=False,
                message="/origami/grasp_poseをまだ受信していません。",
            )

        target_pose = copy.deepcopy(self.latest_pose)

        # 受信Poseの基準座標系におけるZ方向へ安全距離だけ退避
        target_pose.pose.position.z += self.approach_offset_z

        rospy.loginfo(
            "Planning target: frame=%s, x=%.4f, y=%.4f, z=%.4f",
            target_pose.header.frame_id,
            target_pose.pose.position.x,
            target_pose.pose.position.y,
            target_pose.pose.position.z,
        )

        try:
            if (
                self.chain_plans
                and self.last_planned_state is not None
            ):
                self.move_group.set_start_state(
                    self.last_planned_state
                )
                rospy.loginfo(
                    "Start state: previous planned endpoint"
                )
            else:
                self.move_group.set_start_state_to_current_state()
                rospy.loginfo("Start state: current state")

            if self.target_mode == "position_only":
                # テスト5で確認した方式。受信Poseの姿勢は使用しない
                self.move_group.set_pose_reference_frame(
                    target_pose.header.frame_id
                )
                self.move_group.set_position_target(
                    [
                        target_pose.pose.position.x,
                        target_pose.pose.position.y,
                        target_pose.pose.position.z,
                    ],
                    self.end_effector_link,
                )
            else:
                # 修正版テスト6以降で、検証済みの工具姿勢を使用する
                self.move_group.set_pose_target(
                    target_pose,
                    self.end_effector_link,
                )

            result = self.move_group.plan()

            # ROS Noeticでは通常、plan()は4要素のタプルを返す
            if isinstance(result, tuple):
                success = result[0]
                trajectory = result[1]
                planning_time = result[2]
            else:
                trajectory = result
                success = bool(trajectory.joint_trajectory.points)
                planning_time = 0.0

            if not success or not trajectory.joint_trajectory.points:
                return TriggerResponse(
                    success=False,
                    message="軌道計画に失敗しました。実行はしていません。",
                )

            # 計画軌道の終点関節角を取得
            joint_names = list(
                trajectory.joint_trajectory.joint_names
            )
            final_positions = list(
                trajectory.joint_trajectory.points[-1].positions
            )

            rospy.loginfo("=== Planned final joint positions ===")
            for joint_name, joint_position in zip(
                joint_names,
                final_positions,
            ):
                rospy.loginfo(
                    "%s: %.6f rad",
                    joint_name,
                    joint_position,
                )

            # 現在のRobotStateへ計画終点の関節角を反映
            final_state = self.move_group.get_current_state()
            state_names = list(final_state.joint_state.name)
            state_positions = list(final_state.joint_state.position)
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
            final_state.is_diff = False

            if self.chain_plans:
                self.last_planned_state = copy.deepcopy(
                    final_state
                )

            # 計画終点におけるTCP姿勢を順運動学で計算
            fk_request = GetPositionFKRequest()
            fk_request.header.frame_id = (
                self.move_group.get_planning_frame()
            )
            fk_request.fk_link_names = [
                self.end_effector_link
            ]
            fk_request.robot_state = final_state

            fk_response = self.compute_fk(fk_request)

            if (
                fk_response.error_code.val == 1
                and fk_response.pose_stamped
            ):
                final_pose = fk_response.pose_stamped[0]
                rospy.loginfo(
                    "Final TCP position: "
                    "frame=%s, x=%.6f, y=%.6f, z=%.6f",
                    final_pose.header.frame_id,
                    final_pose.pose.position.x,
                    final_pose.pose.position.y,
                    final_pose.pose.position.z,
                )
                rospy.loginfo(
                    "Final TCP orientation: "
                    "x=%.6f, y=%.6f, z=%.6f, w=%.6f",
                    final_pose.pose.orientation.x,
                    final_pose.pose.orientation.y,
                    final_pose.pose.orientation.z,
                    final_pose.pose.orientation.w,
                )
            else:
                rospy.logwarn(
                    "計画終点のFK計算に失敗しました。"
                    " error_code=%d",
                    fk_response.error_code.val,
                )

            display_trajectory = DisplayTrajectory()
            display_trajectory.trajectory_start = (
                self.move_group.get_current_state()
            )
            display_trajectory.trajectory.append(trajectory)
            self.display_publisher.publish(display_trajectory)

            return TriggerResponse(
                success=True,
                message=(
                    "軌道計画に成功しました。"
                    " planning_time={:.3f}s。"
                    "RViz表示のみで、Executeはしていません。"
                ).format(planning_time),
            )

        except Exception as error:
            rospy.logerr("Planning error: %s", error)
            return TriggerResponse(
                success=False,
                message="例外により計画できませんでした: {}".format(error),
            )

        finally:
            self.move_group.clear_pose_targets()


def main():
    rospy.init_node("grasp_pose_to_moveit")

    GraspPosePlanner()
    rospy.spin()

    moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    main()
