#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import os
import sys
from datetime import datetime

import rospy
import yaml

from geometry_msgs.msg import PointStamped
from std_srvs.srv import Trigger


class SavedUnityTrajectoryPlanner:
    """保存済みUnity軌道を1点ずつMoveItへ入力する試験ノード。"""

    def __init__(self):
        rospy.init_node("plan_saved_unity_trajectory")

        self.trajectory_file = os.path.expanduser(
            rospy.get_param(
                "~trajectory_file",
                "~/grasp_trajectory_auto_fold.yaml",
            )
        )

        self.start_index = rospy.get_param("~start_index", 9)
        self.end_index = rospy.get_param("~end_index", 16)

        self.fixed_z = rospy.get_param("~fixed_z", 0.065)
        self.publish_wait = rospy.get_param("~publish_wait", 0.5)
        self.connection_timeout = rospy.get_param(
            "~connection_timeout",
            5.0,
        )

        self.continue_on_failure = rospy.get_param(
            "~continue_on_failure",
            False,
        )

        default_result_file = (
            "~/unity_trajectory_plan_results_"
            f"{self.start_index}_{self.end_index}.csv"
        )

        self.result_file = os.path.expanduser(
            rospy.get_param(
                "~result_file",
                default_result_file,
            )
        )

        self.point_topic = rospy.get_param(
            "~point_topic",
            "/origami/grasp_point_ros",
        )

        self.plan_service_name = rospy.get_param(
            "~plan_service",
            "/origami/plan_grasp_pose",
        )

        self.publisher = rospy.Publisher(
            self.point_topic,
            PointStamped,
            queue_size=1,
        )

        self.plan_service = None

    def load_points(self):
        """MultiDOFJointTrajectory形式のYAMLからUnity座標を読む。"""

        if not os.path.isfile(self.trajectory_file):
            raise FileNotFoundError(
                f"軌道ファイルがありません: {self.trajectory_file}"
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

        if len(documents) > 1:
            rospy.logwarn(
                "YAML内に複数の有効な文書があります。"
                "最初の文書のみ使用します。文書数=%d",
                len(documents),
            )

        if not isinstance(data, dict):
            raise ValueError("YAMLの最上位が辞書形式ではありません。")

        points = data.get("points")

        if not isinstance(points, list):
            raise ValueError(
                "YAML内にpoints配列がありません。"
            )

        if not points:
            raise ValueError("軌道点が0点です。")

        return points

    @staticmethod
    def get_unity_translation(point, index):
        """1軌道点からUnity座標のtranslationを取り出す。"""

        try:
            transforms = point["transforms"]
            translation = transforms[0]["translation"]

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
                f"index {index} の座標を読み取れません: {error}"
            )

        return unity_x, unity_y, unity_z

    def wait_for_subscriber(self):
        """PointStampedの購読ノードが接続されるまで待つ。"""

        start_time = rospy.Time.now()

        while not rospy.is_shutdown():
            if self.publisher.get_num_connections() > 0:
                rospy.loginfo(
                    "点送信先へ接続しました。connections=%d",
                    self.publisher.get_num_connections(),
                )
                return

            elapsed = (
                rospy.Time.now() - start_time
            ).to_sec()

            if elapsed >= self.connection_timeout:
                raise RuntimeError(
                    f"{self.point_topic} のSubscriberが"
                    f"{self.connection_timeout:.1f}秒以内に"
                    "接続されませんでした。"
                )

            rospy.sleep(0.1)

    def prepare_service(self):
        """計画サービスへ接続する。"""

        rospy.loginfo(
            "サービス待機中: %s",
            self.plan_service_name,
        )

        rospy.wait_for_service(
            self.plan_service_name,
            timeout=10.0,
        )

        self.plan_service = rospy.ServiceProxy(
            self.plan_service_name,
            Trigger,
        )

        rospy.loginfo(
            "計画サービスへ接続しました。"
        )

    def publish_point(
        self,
        point_index,
        ros_x,
        ros_y,
        ros_z,
    ):
        """paper_center基準の目標点をPublishする。"""

        message = PointStamped()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = "paper_center"

        message.point.x = ros_x
        message.point.y = ros_y
        message.point.z = ros_z

        self.publisher.publish(message)

        rospy.loginfo(
            "[index %d] Publish: "
            "paper_center=(%.6f, %.6f, %.6f)",
            point_index,
            ros_x,
            ros_y,
            ros_z,
        )

        # grasp_point_to_pose.pyへ反映されるまで待つ
        rospy.sleep(self.publish_wait)

    def run(self):
        points = self.load_points()
        point_count = len(points)

        rospy.loginfo(
            "軌道ファイル: %s",
            self.trajectory_file,
        )
        rospy.loginfo(
            "軌道点数: %d",
            point_count,
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
            "失敗後の継続: %s",
            self.continue_on_failure,
        )

        if self.start_index < 0:
            raise ValueError("start_indexは0以上にしてください。")

        if self.end_index < self.start_index:
            raise ValueError(
                "end_indexはstart_index以上にしてください。"
            )

        if self.end_index >= point_count:
            raise IndexError(
                f"end_index={self.end_index}ですが、"
                f"軌道点数は{point_count}点です。"
            )

        self.wait_for_subscriber()
        self.prepare_service()

        result_directory = os.path.dirname(
            self.result_file
        )

        if result_directory:
            os.makedirs(
                result_directory,
                exist_ok=True,
            )

        success_count = 0
        failure_count = 0

        with open(
            self.result_file,
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            writer = csv.writer(csv_file)

            writer.writerow(
                [
                    "recorded_at",
                    "index",
                    "unity_x",
                    "unity_y",
                    "unity_z",
                    "paper_center_x",
                    "paper_center_y",
                    "paper_center_z",
                    "plan_success",
                    "service_message",
                ]
            )

            for point_index in range(
                self.start_index,
                self.end_index + 1,
            ):
                if rospy.is_shutdown():
                    break

                unity_x, unity_y, unity_z = (
                    self.get_unity_translation(
                        points[point_index],
                        point_index,
                    )
                )

                # Unity → paper_center
                ros_x = unity_z
                ros_y = -unity_x
                ros_z = self.fixed_z

                self.publish_point(
                    point_index,
                    ros_x,
                    ros_y,
                    ros_z,
                )

                try:
                    response = self.plan_service()
                    plan_success = bool(response.success)
                    service_message = response.message

                except rospy.ServiceException as error:
                    plan_success = False
                    service_message = (
                        f"ServiceException: {error}"
                    )

                writer.writerow(
                    [
                        datetime.now().isoformat(
                            timespec="seconds"
                        ),
                        point_index,
                        f"{unity_x:.12f}",
                        f"{unity_y:.12f}",
                        f"{unity_z:.12f}",
                        f"{ros_x:.12f}",
                        f"{ros_y:.12f}",
                        f"{ros_z:.12f}",
                        plan_success,
                        service_message,
                    ]
                )
                csv_file.flush()

                if plan_success:
                    success_count += 1

                    rospy.loginfo(
                        "[index %d] 計画成功: %s",
                        point_index,
                        service_message,
                    )

                else:
                    failure_count += 1

                    rospy.logerr(
                        "[index %d] 計画失敗: %s",
                        point_index,
                        service_message,
                    )

                    if not self.continue_on_failure:
                        rospy.logerr(
                            "失敗点で処理を停止します。"
                        )
                        break

        rospy.loginfo("============ 試験結果 ============")
        rospy.loginfo(
            "成功: %d点",
            success_count,
        )
        rospy.loginfo(
            "失敗: %d点",
            failure_count,
        )
        rospy.loginfo(
            "CSV保存先: %s",
            self.result_file,
        )
        rospy.loginfo("==================================")


if __name__ == "__main__":
    try:
        planner = SavedUnityTrajectoryPlanner()
        planner.run()

    except (
        FileNotFoundError,
        ValueError,
        IndexError,
        RuntimeError,
        rospy.ROSException,
    ) as error:
        rospy.logfatal("%s", error)
        sys.exit(1)

    except KeyboardInterrupt:
        pass
