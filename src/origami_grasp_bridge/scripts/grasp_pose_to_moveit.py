#!/usr/bin/env python3

import copy
import sys

import moveit_commander
import rospy
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import DisplayTrajectory
from std_srvs.srv import Trigger, TriggerResponse


class GraspPosePlanner:
    """最新の把持Poseを保存し、明示的な指令を受けたときだけ計画する。"""

    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)

        self.latest_pose = None

        self.group_name = rospy.get_param("~group_name", "mycobot_arm")
        self.end_effector_link = rospy.get_param(
            "~end_effector_link",
            "mycobot_link6",
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

        # mycobot_baseの鉛直上方向へ100 mm退避
        target_pose.pose.position.z += self.approach_offset_z

        rospy.loginfo(
            "Planning target: frame=%s, x=%.4f, y=%.4f, z=%.4f",
            target_pose.header.frame_id,
            target_pose.pose.position.x,
            target_pose.pose.position.y,
            target_pose.pose.position.z,
        )

        try:
            self.move_group.set_start_state_to_current_state()
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
