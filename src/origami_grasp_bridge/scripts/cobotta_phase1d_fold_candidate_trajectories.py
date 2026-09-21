#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math

import numpy as np
import rospy

from geometry_msgs.msg import Pose, PoseArray
from std_msgs.msg import String

from tf.transformations import (
    quaternion_inverse,
    quaternion_matrix,
    quaternion_multiply,
)


PAPER_HISTORY_TOPIC = (
    "/origami/active_paper_pose_history_ros"
)

INPUT_RESULT_TOPIC = (
    "/origami/debug/"
    "cobotta_pregrasp_feasibility_results"
)

METADATA_TOPIC = (
    "/origami/debug/"
    "cobotta_phase1d_fold_candidate_metadata"
)

# cobotta_tool_link -> actual_grasp_point
R_GRASP = np.array(
    [0.002000, 0.000000, -0.004894],
    dtype=float,
)


def q_array(q):
    return np.array(
        [q.x, q.y, q.z, q.w],
        dtype=float,
    )


def normalize_q(q):
    norm = np.linalg.norm(q)

    if norm < 1.0e-12:
        raise RuntimeError(
            "Quaternion norm is zero."
        )

    return q / norm


def rotate_vector(q, v):
    return quaternion_matrix(q)[:3, :3].dot(v)


def quaternion_angle_deg(q0, q1):
    q0 = normalize_q(q0)
    q1 = normalize_q(q1)

    dot = abs(float(np.dot(q0, q1)))
    dot = max(-1.0, min(1.0, dot))

    return math.degrees(
        2.0 * math.acos(dot)
    )


def dict_to_q(pose_dict):
    ori = pose_dict["orientation"]

    return normalize_q(
        np.array([
            ori["x"],
            ori["y"],
            ori["z"],
            ori["w"],
        ], dtype=float)
    )


def dict_to_position(pose_dict):
    pos = pose_dict["position"]

    return np.array([
        pos["x"],
        pos["y"],
        pos["z"],
    ], dtype=float)


def candidate_topic(
    p0_index,
    candidate_index,
    normal_sign,
):
    sign_text = (
        "nplus"
        if normal_sign == "N+"
        else "nminus"
    )

    return (
        "/origami/"
        "cobotta_phase1d_fold_candidate_trajectory/"
        "p0_{}_c{}_{}"
        .format(
            p0_index,
            candidate_index,
            sign_text,
        )
    )


def build_candidate_trajectory(
    paper_poses,
    grasp_tool_pose,
):
    """
    折り開始時T0を基準に、

      1. actual_grasp_pointの紙ローカル位置
      2. 工具の紙に対する相対姿勢

    を求め、それを全紙Pose履歴へ適用する。
    """

    if len(paper_poses) == 0:
        raise RuntimeError(
            "Paper pose history is empty."
        )

    # ==========================================
    # T0の紙Pose
    # ==========================================
    paper0 = paper_poses[0]

    q_paper0 = normalize_q(
        q_array(
            paper0.orientation
        )
    )

    p_paper0 = np.array([
        paper0.position.x,
        paper0.position.y,
        paper0.position.z,
    ], dtype=float)

    # ==========================================
    # P0到着時のtool_link Pose
    # ==========================================
    q_tool0 = dict_to_q(
        grasp_tool_pose
    )

    p_tool0 = dict_to_position(
        grasp_tool_pose
    )

    # ==========================================
    # tool_link Poseからactual_grasp_pointを復元
    # ==========================================
    grasp_point0 = (
        p_tool0
        + rotate_vector(
            q_tool0,
            R_GRASP,
        )
    )

    # ==========================================
    # actual_grasp_pointを
    # T0紙ローカル座標へ変換
    # ==========================================
    local_p0 = rotate_vector(
        quaternion_inverse(
            q_paper0
        ),
        grasp_point0 - p_paper0,
    )

    # ==========================================
    # 工具姿勢を
    # T0紙に対する相対姿勢へ変換
    # ==========================================
    q_relative = normalize_q(
        quaternion_multiply(
            quaternion_inverse(
                q_paper0
            ),
            q_tool0,
        )
    )

    output_poses = []

    previous_q_tool = None

    for paper_pose in paper_poses:

        q_paper = normalize_q(
            q_array(
                paper_pose.orientation
            )
        )

        p_paper = np.array([
            paper_pose.position.x,
            paper_pose.position.y,
            paper_pose.position.z,
        ], dtype=float)

        # 紙と一緒にactual_grasp_pointを移動
        grasp_point = (
            p_paper
            + rotate_vector(
                q_paper,
                local_p0,
            )
        )

        # 紙に対する工具姿勢を維持
        q_tool = normalize_q(
            quaternion_multiply(
                q_paper,
                q_relative,
            )
        )

        # Quaternion q/-q の符号を連続化
        if previous_q_tool is not None:
            if np.dot(
                previous_q_tool,
                q_tool,
            ) < 0.0:
                q_tool = -q_tool

        # actual_grasp_pointから
        # cobotta_tool_link位置を逆算
        p_tool = (
            grasp_point
            - rotate_vector(
                q_tool,
                R_GRASP,
            )
        )

        pose = Pose()

        pose.position.x = float(
            p_tool[0]
        )
        pose.position.y = float(
            p_tool[1]
        )
        pose.position.z = float(
            p_tool[2]
        )

        pose.orientation.x = float(
            q_tool[0]
        )
        pose.orientation.y = float(
            q_tool[1]
        )
        pose.orientation.z = float(
            q_tool[2]
        )
        pose.orientation.w = float(
            q_tool[3]
        )

        output_poses.append(
            pose
        )

        previous_q_tool = q_tool

    # ==========================================
    # 生成軌道STARTが入力GRASP Poseと一致するか確認
    # ==========================================
    start = output_poses[0]

    start_position = np.array([
        start.position.x,
        start.position.y,
        start.position.z,
    ], dtype=float)

    start_q = normalize_q(
        q_array(
            start.orientation
        )
    )

    start_position_error = (
        np.linalg.norm(
            start_position - p_tool0
        )
    )

    start_orientation_error = (
        quaternion_angle_deg(
            start_q,
            q_tool0,
        )
    )

    return (
        local_p0,
        q_relative,
        output_poses,
        start_position_error,
        start_orientation_error,
    )


def main():
    rospy.init_node(
        "cobotta_phase1d_fold_candidate_trajectories"
    )

    print(
        "===== PHASE1-D FOLD CANDIDATE "
        "TRAJECTORY GENERATOR ====="
    )

    print(
        "Waiting for paper pose history..."
    )

    paper_msg = rospy.wait_for_message(
        PAPER_HISTORY_TOPIC,
        PoseArray,
        timeout=15.0,
    )

    print(
        "paper poses = {}".format(
            len(paper_msg.poses)
        )
    )

    print(
        "Waiting for PRE-GRASP "
        "structured results..."
    )

    result_msg = rospy.wait_for_message(
        INPUT_RESULT_TOPIC,
        String,
        timeout=15.0,
    )

    data = json.loads(
        result_msg.data
    )

    candidates = []

    for p0 in data["p0_results"]:
        for candidate in p0[
            "candidates"
        ]:

            if (
                candidate.get("result")
                != "FEASIBLE_FOUND"
            ):
                continue

            grasp_joints = (
                candidate.get(
                    "grasp_joints_rad"
                )
            )

            grasp_tool_pose = (
                candidate.get(
                    "grasp_tool_pose"
                )
            )

            if (
                grasp_joints is None
                or len(grasp_joints) != 6
            ):
                raise RuntimeError(
                    "FEASIBLE candidate has no "
                    "valid grasp_joints_rad."
                )

            if grasp_tool_pose is None:
                raise RuntimeError(
                    "FEASIBLE candidate has no "
                    "grasp_tool_pose."
                )

            candidates.append({
                "p0_index":
                    int(
                        p0["p0_index"]
                    ),
                "candidate_index":
                    int(
                        candidate[
                            "candidate_index"
                        ]
                    ),
                "edge":
                    int(
                        candidate["edge"]
                    ),
                "normal_sign":
                    str(
                        candidate[
                            "normal_sign"
                        ]
                    ),
                "grasp_joints_rad": [
                    float(v)
                    for v in grasp_joints
                ],
                "grasp_tool_pose":
                    grasp_tool_pose,
            })

    print(
        "input fold candidates = {}".format(
            len(candidates)
        )
    )

    publishers = []
    metadata = []

    for candidate in candidates:

        topic = candidate_topic(
            candidate["p0_index"],
            candidate["candidate_index"],
            candidate["normal_sign"],
        )

        pub = rospy.Publisher(
            topic,
            PoseArray,
            queue_size=1,
            latch=True,
        )

        publishers.append(pub)

        (
            local_p0,
            q_relative,
            trajectory,
            start_position_error,
            start_orientation_error,
        ) = build_candidate_trajectory(
            paper_msg.poses,
            candidate[
                "grasp_tool_pose"
            ],
        )

        if start_position_error > 1.0e-6:
            raise RuntimeError(
                "START position reconstruction "
                "error too large: {} m"
                .format(
                    start_position_error
                )
            )

        if start_orientation_error > 1.0e-5:
            raise RuntimeError(
                "START orientation reconstruction "
                "error too large: {} deg"
                .format(
                    start_orientation_error
                )
            )

        output = PoseArray()

        output.header.stamp = (
            paper_msg.header.stamp
        )

        output.header.frame_id = (
            paper_msg.header.frame_id
            if paper_msg.header.frame_id
            else "paper_center"
        )

        output.poses = trajectory

        pub.publish(
            output
        )

        print()
        print(
            "P0[{}] candidate {} edge{} {}"
            .format(
                candidate["p0_index"],
                candidate[
                    "candidate_index"
                ],
                candidate["edge"],
                candidate[
                    "normal_sign"
                ],
            )
        )

        print(
            "  topic       = {}".format(
                topic
            )
        )

        print(
            "  poses       = {}".format(
                len(trajectory)
            )
        )

        print(
            "  local P0 mm = "
            "[{:+.3f}, {:+.3f}, {:+.3f}]"
            .format(
                local_p0[0] * 1000.0,
                local_p0[1] * 1000.0,
                local_p0[2] * 1000.0,
            )
        )

        print(
            "  START pos err = "
            "{:.9f} mm"
            .format(
                start_position_error
                * 1000.0
            )
        )

        print(
            "  START ori err = "
            "{:.9f} deg"
            .format(
                start_orientation_error
            )
        )

        metadata.append({
            "p0_index":
                candidate["p0_index"],
            "candidate_index":
                candidate[
                    "candidate_index"
                ],
            "edge":
                candidate["edge"],
            "normal_sign":
                candidate[
                    "normal_sign"
                ],
            "trajectory_topic":
                topic,
            "pose_count":
                len(trajectory),
            "grasp_joints_rad":
                candidate[
                    "grasp_joints_rad"
                ],
            "local_p0_m": [
                float(v)
                for v in local_p0
            ],
            "q_tool_relative_to_paper":
                [
                    float(v)
                    for v in q_relative
                ],
            "start_position_error_m":
                float(
                    start_position_error
                ),
            "start_orientation_error_deg":
                float(
                    start_orientation_error
                ),
        })

    meta_pub = rospy.Publisher(
        METADATA_TOPIC,
        String,
        queue_size=1,
        latch=True,
    )

    rospy.sleep(0.5)

    meta_msg = String()

    meta_msg.data = json.dumps(
        {
            "schema_version": 1,
            "candidate_count":
                len(metadata),
            "paper_pose_count":
                len(
                    paper_msg.poses
                ),
            "candidates":
                metadata,
        },
        ensure_ascii=False,
        sort_keys=True,
    )

    meta_pub.publish(
        meta_msg
    )

    print()
    print(
        "===== GENERATION COMPLETE ====="
    )

    print(
        "candidate count = {}".format(
            len(metadata)
        )
    )

    print(
        "paper pose count = {}".format(
            len(paper_msg.poses)
        )
    )

    print(
        "metadata topic:"
    )

    print(
        "  {}".format(
            METADATA_TOPIC
        )
    )

    rospy.spin()


if __name__ == "__main__":
    main()
