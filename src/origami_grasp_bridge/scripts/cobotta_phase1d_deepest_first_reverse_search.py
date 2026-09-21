#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import json
import math
import sys
import time

import numpy as np
import rospy
import moveit_commander

from geometry_msgs.msg import (
    Pose,
    PoseArray,
    PoseStamped,
)
from std_msgs.msg import String
from urdf_parser_py.urdf import URDF

from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import (
    GetPositionIK,
    GetPositionIKRequest,
    GetStateValidity,
    GetStateValidityRequest,
)

from tf.transformations import (
    quaternion_inverse,
    quaternion_matrix,
    quaternion_multiply,
)


GROUP = "cobotta_arm"
TIP = "cobotta_tool_link"

FOLD_META_TOPIC = (
    "/origami/debug/"
    "cobotta_phase1d_exhaustive_fold_candidate_metadata"
)

PRE_RESULT_TOPIC = (
    "/origami/debug/"
    "cobotta_pregrasp_exhaustive_feasibility_results"
)

PAPER_HISTORY_TOPIC = (
    "/origami/active_paper_pose_history_ros"
)

OUTPUT_TOPIC = (
    "/origami/debug/"
    "cobotta_phase1d_deepest_first_reverse_search_results"
)

COBOTTA_JOINTS = [
    "cobotta_joint_1",
    "cobotta_joint_2",
    "cobotta_joint_3",
    "cobotta_joint_4",
    "cobotta_joint_5",
    "cobotta_joint_6",
]

R_GRASP = np.array(
    [0.002000, 0.000000, -0.004894],
    dtype=float,
)

GRIPPER_CLOSED = 0.0

IK_TIMEOUT = 0.10

GLOBAL_SEED_COUNT = 32

DISTINCT_TOL_DEG = 1.0
DISTINCT_TOL_RAD = math.radians(
    DISTINCT_TOL_DEG
)

MAX_JUMP_DEG = 10.0
MAX_JUMP_RAD = math.radians(
    MAX_JUMP_DEG
)

PRE_STEPS = 30


def set_arm_joints(
    state,
    joints,
):
    state = copy.deepcopy(state)

    names = list(
        state.joint_state.name
    )

    positions = list(
        state.joint_state.position
    )

    lookup = {
        name: i
        for i, name in enumerate(names)
    }

    for name, value in zip(
        COBOTTA_JOINTS,
        joints,
    ):
        if name not in lookup:
            raise RuntimeError(
                "Missing joint: " + name
            )

        positions[
            lookup[name]
        ] = float(value)

    state.joint_state.position = (
        positions
    )

    return state


def set_gripper_closed(state):
    state = copy.deepcopy(state)

    names = list(
        state.joint_state.name
    )

    positions = list(
        state.joint_state.position
    )

    lookup = {
        name: i
        for i, name in enumerate(names)
    }

    if "cobotta_joint_gripper" in lookup:
        positions[
            lookup[
                "cobotta_joint_gripper"
            ]
        ] = GRIPPER_CLOSED

    if (
        "cobotta_joint_gripper_mimic"
        in lookup
    ):
        positions[
            lookup[
                "cobotta_joint_gripper_mimic"
            ]
        ] = -GRIPPER_CLOSED

    state.joint_state.position = (
        positions
    )

    return state


def extract_arm_joints(state):
    q_map = dict(zip(
        state.joint_state.name,
        state.joint_state.position,
    ))

    return [
        float(q_map[name])
        for name in COBOTTA_JOINTS
    ]


def joint_distance_max(q1, q2):
    return max(
        abs(a - b)
        for a, b in zip(
            q1,
            q2,
        )
    )


def halton(index, base):
    result = 0.0
    fraction = 1.0 / base
    i = index

    while i > 0:
        result += (
            fraction
            * (i % base)
        )

        i //= base
        fraction /= base

    return result


def get_joint_limits():
    urdf = URDF.from_xml_string(
        rospy.get_param(
            "/robot_description"
        )
    )

    limits = []

    for name in COBOTTA_JOINTS:
        joint = urdf.joint_map[name]

        if joint.limit is None:
            raise RuntimeError(
                "Joint has no limits: "
                + name
            )

        limits.append(
            (
                float(
                    joint.limit.lower
                ),
                float(
                    joint.limit.upper
                ),
            )
        )

    return limits


def make_global_seeds(
    joint_limits,
    count,
):
    bases = [
        2, 3, 5, 7, 11, 13
    ]

    seeds = []

    for i in range(
        1,
        count + 1,
    ):
        q = []

        for j, (
            lower,
            upper,
        ) in enumerate(
            joint_limits
        ):
            f = halton(
                i,
                bases[j],
            )

            q.append(
                lower
                + f
                * (
                    upper
                    - lower
                )
            )

        seeds.append(q)

    return seeds


def append_unique_seed(
    seeds,
    candidate,
    tol=1.0e-8,
):
    for existing in seeds:
        if (
            joint_distance_max(
                existing,
                candidate,
            )
            <= tol
        ):
            return

    seeds.append(
        list(candidate)
    )


def add_distinct_solution(
    solutions,
    state,
):
    joints = extract_arm_joints(
        state
    )

    for existing in solutions:
        if (
            joint_distance_max(
                existing["joints"],
                joints,
            )
            <= DISTINCT_TOL_RAD
        ):
            return False

    solutions.append({
        "joints":
            joints,
        "state":
            copy.deepcopy(state),
    })

    return True


def solve_ik(
    compute_ik,
    target,
    seed_state,
):
    req = GetPositionIKRequest()

    req.ik_request.group_name = GROUP
    req.ik_request.ik_link_name = TIP

    req.ik_request.pose_stamped = (
        target
    )

    req.ik_request.robot_state = (
        copy.deepcopy(
            seed_state
        )
    )

    req.ik_request.avoid_collisions = (
        False
    )

    req.ik_request.timeout = (
        rospy.Duration(
            IK_TIMEOUT
        )
    )

    res = compute_ik(req)

    if (
        res.error_code.val
        != MoveItErrorCodes.SUCCESS
    ):
        return None

    return set_gripper_closed(
        res.solution
    )


def collision_check(
    check_validity,
    state,
):
    req = GetStateValidityRequest()

    req.robot_state = state
    req.group_name = GROUP

    res = check_validity(req)

    pairs = sorted(set(
        tuple(sorted([
            c.contact_body_1,
            c.contact_body_2,
        ]))
        for c in res.contacts
    ))

    return bool(
        res.valid
    ), pairs


def make_target(
    trajectory,
    index,
):
    target = PoseStamped()

    target.header.stamp = (
        rospy.Time(0)
    )

    target.header.frame_id = (
        trajectory.header.frame_id
        if trajectory.header.frame_id
        else "paper_center"
    )

    target.pose = (
        trajectory.poses[index]
    )

    return target


def q_array(q):
    return np.array(
        [
            q.x,
            q.y,
            q.z,
            q.w,
        ],
        dtype=float,
    )


def normalize_q(q):
    n = np.linalg.norm(q)

    if n < 1.0e-12:
        raise RuntimeError(
            "Quaternion norm zero."
        )

    return q / n


def fold_angle_deg(
    q0,
    qi,
):
    q0 = normalize_q(q0)
    qi = normalize_q(qi)

    q_rel = quaternion_multiply(
        quaternion_inverse(q0),
        qi,
    )

    q_rel = normalize_q(
        q_rel
    )

    w = abs(
        float(q_rel[3])
    )

    w = max(
        -1.0,
        min(1.0, w)
    )

    return math.degrees(
        2.0
        * math.acos(w)
    )


def dict_pose_to_pose(
    pose_dict,
):
    pose = Pose()

    pose.position.x = float(
        pose_dict[
            "position"
        ]["x"]
    )

    pose.position.y = float(
        pose_dict[
            "position"
        ]["y"]
    )

    pose.position.z = float(
        pose_dict[
            "position"
        ]["z"]
    )

    pose.orientation.x = float(
        pose_dict[
            "orientation"
        ]["x"]
    )

    pose.orientation.y = float(
        pose_dict[
            "orientation"
        ]["y"]
    )

    pose.orientation.z = float(
        pose_dict[
            "orientation"
        ]["z"]
    )

    pose.orientation.w = float(
        pose_dict[
            "orientation"
        ]["w"]
    )

    return pose


def reconstruct_pre_pose(
    pre_record,
):
    grasp_pose = dict_pose_to_pose(
        pre_record[
            "grasp_tool_pose"
        ]
    )

    q = normalize_q(
        q_array(
            grasp_pose.orientation
        )
    )

    R = quaternion_matrix(
        q
    )[:3, :3]

    pre_grasp_point = np.array(
        pre_record[
            "pre_grasp_point_m"
        ],
        dtype=float,
    )

    pre_tool = (
        pre_grasp_point
        - R.dot(R_GRASP)
    )

    pre_pose = copy.deepcopy(
        grasp_pose
    )

    pre_pose.position.x = float(
        pre_tool[0]
    )
    pre_pose.position.y = float(
        pre_tool[1]
    )
    pre_pose.position.z = float(
        pre_tool[2]
    )

    return pre_pose


def interpolate_pose(
    grasp,
    pre,
    alpha,
    frame_id,
):
    # alpha=0 : grasp(P0)
    # alpha=1 : PRE

    target = PoseStamped()

    target.header.stamp = (
        rospy.Time(0)
    )

    target.header.frame_id = (
        frame_id
    )

    target.pose.position.x = (
        (1.0 - alpha)
        * grasp.position.x
        + alpha
        * pre.position.x
    )

    target.pose.position.y = (
        (1.0 - alpha)
        * grasp.position.y
        + alpha
        * pre.position.y
    )

    target.pose.position.z = (
        (1.0 - alpha)
        * grasp.position.z
        + alpha
        * pre.position.z
    )

    # PRE-P0は同一工具姿勢
    target.pose.orientation = (
        copy.deepcopy(
            grasp.orientation
        )
    )

    return target


def reverse_fold_track(
    start_index,
    finish_state,
    trajectory,
    compute_ik,
    check_validity,
):
    seed = copy.deepcopy(
        finish_state
    )

    previous_joints = (
        extract_arm_joints(
            seed
        )
    )

    # finish point自体はすでに
    # Collision-free確認済み。
    for index in range(
        start_index - 1,
        -1,
        -1,
    ):
        target = make_target(
            trajectory,
            index,
        )

        solution = solve_ik(
            compute_ik,
            target,
            seed,
        )

        if solution is None:
            return (
                False,
                None,
                {
                    "index": index,
                    "reason": "IK_FAILED",
                },
            )

        current_joints = (
            extract_arm_joints(
                solution
            )
        )

        jump = joint_distance_max(
            previous_joints,
            current_joints,
        )

        if jump > MAX_JUMP_RAD:
            return (
                False,
                None,
                {
                    "index": index,
                    "reason":
                        "JOINT_JUMP",
                    "jump_deg":
                        math.degrees(
                            jump
                        ),
                },
            )

        valid, pairs = (
            collision_check(
                check_validity,
                solution,
            )
        )

        if not valid:
            return (
                False,
                None,
                {
                    "index": index,
                    "reason":
                        "COLLISION",
                    "pairs": [
                        "{}<->{}".format(
                            a,
                            b,
                        )
                        for a, b in pairs
                    ],
                },
            )

        seed = copy.deepcopy(
            solution
        )

        previous_joints = (
            current_joints
        )

    return (
        True,
        seed,
        None,
    )


def reverse_p0_to_pre(
    p0_state,
    pre_record,
    compute_ik,
    check_validity,
):
    grasp_pose = dict_pose_to_pose(
        pre_record[
            "grasp_tool_pose"
        ]
    )

    pre_pose = reconstruct_pre_pose(
        pre_record
    )

    frame_id = str(
        pre_record[
            "grasp_tool_pose"
        ].get(
            "frame_id",
            "paper_center",
        )
    )

    seed = copy.deepcopy(
        p0_state
    )

    previous_joints = (
        extract_arm_joints(
            seed
        )
    )

    # step0=P0はfold reverseで
    # 既に評価済みなのでstep1からPREへ。
    for step in range(
        1,
        PRE_STEPS + 1,
    ):
        alpha = (
            float(step)
            / float(PRE_STEPS)
        )

        target = interpolate_pose(
            grasp_pose,
            pre_pose,
            alpha,
            frame_id,
        )

        solution = solve_ik(
            compute_ik,
            target,
            seed,
        )

        if solution is None:
            return (
                False,
                None,
                {
                    "step": step,
                    "reason":
                        "IK_FAILED",
                },
            )

        current_joints = (
            extract_arm_joints(
                solution
            )
        )

        jump = joint_distance_max(
            previous_joints,
            current_joints,
        )

        if jump > MAX_JUMP_RAD:
            return (
                False,
                None,
                {
                    "step": step,
                    "reason":
                        "JOINT_JUMP",
                    "jump_deg":
                        math.degrees(
                            jump
                        ),
                },
            )

        valid, pairs = (
            collision_check(
                check_validity,
                solution,
            )
        )

        if not valid:
            return (
                False,
                None,
                {
                    "step": step,
                    "reason":
                        "COLLISION",
                    "pairs": [
                        "{}<->{}".format(
                            a,
                            b,
                        )
                        for a, b in pairs
                    ],
                },
            )

        seed = copy.deepcopy(
            solution
        )

        previous_joints = (
            current_joints
        )

    return (
        True,
        seed,
        None,
    )


def main():
    rospy.init_node(
        "cobotta_phase1d_"
        "deepest_first_reverse_search"
    )

    moveit_commander.roscpp_initialize(
        sys.argv
    )

    global_seed_count = int(
        rospy.get_param(
            "~global_seed_count",
            GLOBAL_SEED_COUNT,
        )
    )

    print(
        "===== PHASE1-D DEEPEST-FIRST "
        "REVERSE SEARCH ====="
    )

    print(
        "strategy          : "
        "deepest index -> reverse to P0 -> PRE"
    )

    print(
        "FINISH seeds      : "
        "grasp + midpoint + {} Halton"
        .format(
            global_seed_count
        )
    )

    print(
        "forward baseline  : NOT USED"
    )

    print(
        "max joint jump    : "
        "{:.1f} deg"
        .format(
            MAX_JUMP_DEG
        )
    )

    print(
        "gripper fold      : CLOSED 0 mm"
    )

    robot = (
        moveit_commander.RobotCommander()
    )

    rospy.wait_for_service(
        "/compute_ik",
        timeout=30.0,
    )

    rospy.wait_for_service(
        "/check_state_validity",
        timeout=30.0,
    )

    compute_ik = rospy.ServiceProxy(
        "/compute_ik",
        GetPositionIK,
        persistent=True,
    )

    check_validity = rospy.ServiceProxy(
        "/check_state_validity",
        GetStateValidity,
        persistent=True,
    )

    fold_meta_msg = (
        rospy.wait_for_message(
            FOLD_META_TOPIC,
            String,
            timeout=15.0,
        )
    )

    pre_result_msg = (
        rospy.wait_for_message(
            PRE_RESULT_TOPIC,
            String,
            timeout=15.0,
        )
    )

    paper_msg = (
        rospy.wait_for_message(
            PAPER_HISTORY_TOPIC,
            PoseArray,
            timeout=15.0,
        )
    )

    fold_meta = json.loads(
        fold_meta_msg.data
    )

    pre_data = json.loads(
        pre_result_msg.data
    )

    candidates = (
        fold_meta["candidates"]
    )

    if not candidates:
        raise RuntimeError(
            "No fold candidates."
        )

    print()
    print(
        "candidate count = {}"
        .format(
            len(candidates)
        )
    )

    print(
        "paper poses     = {}"
        .format(
            len(
                paper_msg.poses
            )
        )
    )

    # PRE-resultをcandidate_indexで引けるようにする
    pre_lookup = {}

    for record in pre_data[
        "candidates"
    ]:
        if (
            record.get("result")
            == "FEASIBLE_FOUND"
        ):
            pre_lookup[
                int(
                    record[
                        "candidate_index"
                    ]
                )
            ] = record

    # trajectoryを先に取得
    trajectories = {}

    for candidate in candidates:
        cid = int(
            candidate[
                "candidate_index"
            ]
        )

        trajectories[cid] = (
            rospy.wait_for_message(
                candidate[
                    "trajectory_topic"
                ],
                PoseArray,
                timeout=15.0,
            )
        )

    pose_count = len(
        paper_msg.poses
    )

    for cid, traj in trajectories.items():
        if len(traj.poses) != pose_count:
            raise RuntimeError(
                "trajectory pose count mismatch "
                "candidate {}".format(cid)
            )

    joint_limits = (
        get_joint_limits()
    )

    midpoint_seed = [
        0.5 * (lo + hi)
        for lo, hi
        in joint_limits
    ]

    global_seeds = (
        make_global_seeds(
            joint_limits,
            global_seed_count,
        )
    )

    robot_state = (
        robot.get_current_state()
    )

    q_paper0 = q_array(
        paper_msg.poses[
            0
        ].orientation
    )

    fold_angles = [
        fold_angle_deg(
            q_paper0,
            q_array(
                p.orientation
            ),
        )
        for p in paper_msg.poses
    ]

    total_start = (
        time.perf_counter()
    )

    total_ik_calls = 0

    search_log = []

    found = None

    # ==========================================
    # 最深点から順番に探索
    # ==========================================
    for finish_index in range(
        pose_count - 1,
        -1,
        -1,
    ):
        angle_deg = (
            fold_angles[
                finish_index
            ]
        )

        print()
        print(
            "================================================"
        )

        print(
            "SEARCH index {} / angle {:.3f} deg"
            .format(
                finish_index,
                angle_deg,
            )
        )

        print(
            "================================================"
        )

        index_summary = {
            "finish_index":
                int(finish_index),
            "fold_angle_deg":
                float(angle_deg),
            "candidates_tested":
                0,
            "finish_ik_solutions":
                0,
            "finish_collision_free_solutions":
                0,
            "reverse_fold_success":
                0,
            "pre_success":
                0,
        }

        for candidate in candidates:
            cid = int(
                candidate[
                    "candidate_index"
                ]
            )

            if cid not in pre_lookup:
                continue

            index_summary[
                "candidates_tested"
            ] += 1

            trajectory = (
                trajectories[cid]
            )

            target = make_target(
                trajectory,
                finish_index,
            )

            # ------------------------------
            # FINISH単点 seed bank
            # ------------------------------
            seed_bank = []

            grasp_joints = [
                float(v)
                for v in candidate[
                    "grasp_joints_rad"
                ]
            ]

            append_unique_seed(
                seed_bank,
                grasp_joints,
            )

            append_unique_seed(
                seed_bank,
                midpoint_seed,
            )

            for seed in global_seeds:
                append_unique_seed(
                    seed_bank,
                    seed,
                )

            finish_solutions = []

            for seed_joints in seed_bank:
                seed_state = (
                    set_arm_joints(
                        robot_state,
                        seed_joints,
                    )
                )

                seed_state = (
                    set_gripper_closed(
                        seed_state
                    )
                )

                total_ik_calls += 1

                solution = solve_ik(
                    compute_ik,
                    target,
                    seed_state,
                )

                if solution is None:
                    continue

                index_summary[
                    "finish_ik_solutions"
                ] += 1

                valid, pairs = (
                    collision_check(
                        check_validity,
                        solution,
                    )
                )

                if not valid:
                    continue

                if add_distinct_solution(
                    finish_solutions,
                    solution,
                ):
                    index_summary[
                        "finish_collision_free_solutions"
                    ] += 1

            if not finish_solutions:
                continue

            print(
                "candidate {} dir{} {:.1f}deg {} : "
                "{} free FINISH branches"
                .format(
                    cid,
                    candidate[
                        "direction_index"
                    ],
                    candidate[
                        "approach_angle_deg"
                    ],
                    candidate[
                        "normal_sign"
                    ],
                    len(
                        finish_solutions
                    ),
                )
            )

            # ------------------------------
            # 各FINISH IK枝から逆追跡
            # ------------------------------
            for branch_index, branch in enumerate(
                finish_solutions
            ):
                (
                    reverse_ok,
                    p0_state,
                    reverse_failure,
                ) = reverse_fold_track(
                    finish_index,
                    branch["state"],
                    trajectory,
                    compute_ik,
                    check_validity,
                )

                # reverse_fold_track内部のIK数
                # 厳密カウントは後で改善可能。
                # v1ではFINISH seed試行数を主計測。
                if not reverse_ok:
                    continue

                index_summary[
                    "reverse_fold_success"
                ] += 1

                (
                    pre_ok,
                    pre_state,
                    pre_failure,
                ) = reverse_p0_to_pre(
                    p0_state,
                    pre_lookup[cid],
                    compute_ik,
                    check_validity,
                )

                if not pre_ok:
                    continue

                index_summary[
                    "pre_success"
                ] += 1

                found = {
                    "finish_index":
                        int(
                            finish_index
                        ),
                    "fold_angle_deg":
                        float(
                            angle_deg
                        ),
                    "candidate_index":
                        int(cid),
                    "p0_index":
                        int(
                            candidate[
                                "p0_index"
                            ]
                        ),
                    "direction_index":
                        int(
                            candidate[
                                "direction_index"
                            ]
                        ),
                    "approach_angle_deg":
                        float(
                            candidate[
                                "approach_angle_deg"
                            ]
                        ),
                    "entry_edge":
                        int(
                            candidate[
                                "entry_edge"
                            ]
                        ),
                    "normal_sign":
                        str(
                            candidate[
                                "normal_sign"
                            ]
                        ),
                    "finish_branch_index":
                        int(
                            branch_index
                        ),
                    "finish_joints_rad":
                        [
                            float(v)
                            for v in branch[
                                "joints"
                            ]
                        ],
                    "p0_joints_rad":
                        extract_arm_joints(
                            p0_state
                        ),
                    "pre_joints_rad":
                        extract_arm_joints(
                            pre_state
                        ),
                }

                break

            if found is not None:
                break

        search_log.append(
            index_summary
        )

        print(
            "index summary:"
        )

        print(
            "  candidates tested       = {}"
            .format(
                index_summary[
                    "candidates_tested"
                ]
            )
        )

        print(
            "  FINISH IK hits          = {}"
            .format(
                index_summary[
                    "finish_ik_solutions"
                ]
            )
        )

        print(
            "  distinct free branches  = {}"
            .format(
                index_summary[
                    "finish_collision_free_solutions"
                ]
            )
        )

        print(
            "  reverse fold success    = {}"
            .format(
                index_summary[
                    "reverse_fold_success"
                ]
            )
        )

        print(
            "  PRE connection success  = {}"
            .format(
                index_summary[
                    "pre_success"
                ]
            )
        )

        if found is not None:
            break

    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    output = {
        "schema_version": 1,
        "strategy":
            "deepest_first_reverse_search",
        "global_seed_count":
            int(global_seed_count),
        "finish_seed_rule":
            "grasp + midpoint + Halton",
        "forward_baseline_seed":
            False,
        "max_joint_jump_deg":
            float(
                MAX_JUMP_DEG
            ),
        "pre_steps":
            int(
                PRE_STEPS
            ),
        "candidate_count":
            int(
                len(candidates)
            ),
        "paper_pose_count":
            int(
                pose_count
            ),
        "total_time_sec":
            float(
                total_elapsed
            ),
        "finish_ik_call_count":
            int(
                total_ik_calls
            ),
        "found":
            found,
        "search_log":
            search_log,
    }

    pub = rospy.Publisher(
        OUTPUT_TOPIC,
        String,
        queue_size=1,
        latch=True,
    )

    rospy.sleep(0.5)

    msg = String()

    msg.data = json.dumps(
        output,
        ensure_ascii=False,
        sort_keys=True,
    )

    pub.publish(
        msg
    )

    print()
    print(
        "================================================"
    )

    print(
        " DEEPEST-FIRST FINAL RESULT"
    )

    print(
        "================================================"
    )

    if found is None:
        print(
            "NO COMPLETE REVERSE CONNECTION FOUND"
        )
    else:
        print(
            "deepest connected index : {}"
            .format(
                found[
                    "finish_index"
                ]
            )
        )

        print(
            "fold angle              : "
            "{:.3f} deg"
            .format(
                found[
                    "fold_angle_deg"
                ]
            )
        )

        print(
            "candidate               : {}"
            .format(
                found[
                    "candidate_index"
                ]
            )
        )

        print(
            "approach angle          : "
            "{:.1f} deg"
            .format(
                found[
                    "approach_angle_deg"
                ]
            )
        )

        print(
            "normal sign             : {}"
            .format(
                found[
                    "normal_sign"
                ]
            )
        )

        print(
            "finish branch           : {}"
            .format(
                found[
                    "finish_branch_index"
                ]
            )
        )

    print(
        "total search time        : "
        "{:.3f} s"
        .format(
            total_elapsed
        )
    )

    print(
        "FINISH IK calls          : {}"
        .format(
            total_ik_calls
        )
    )

    print(
        "result topic:"
    )

    print(
        "  {}".format(
            OUTPUT_TOPIC
        )
    )

    rospy.spin()


if __name__ == "__main__":
    main()
