#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import json
import math
import sys
import time

import moveit_commander
import rospy

from geometry_msgs.msg import PoseArray, PoseStamped
from std_msgs.msg import String
from urdf_parser_py.urdf import URDF

from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import (
    GetPositionIK,
    GetPositionIKRequest,
    GetStateValidity,
    GetStateValidityRequest,
)


GROUP = "cobotta_arm"
TIP = "cobotta_tool_link"

METADATA_TOPIC = (
    "/origami/debug/"
    "cobotta_phase1d_fold_candidate_metadata"
)

COBOTTA_JOINTS = [
    "cobotta_joint_1",
    "cobotta_joint_2",
    "cobotta_joint_3",
    "cobotta_joint_4",
    "cobotta_joint_5",
    "cobotta_joint_6",
]

GRIPPER_CLOSED = 0.0
IK_TIMEOUT = 0.10

START_INDEX = 24
END_INDEX = 40

# URDFの関節範囲全体に配置する決定論的seed数
GLOBAL_SEED_COUNT = 32

# 同じIK解とみなすための重複除去幅
# これは安全判定値ではなく、診断上のクラスタリング専用。
DISTINCT_TOL_DEG = 1.0
DISTINCT_TOL_RAD = math.radians(DISTINCT_TOL_DEG)


def set_arm_joints(state, joints):
    state = copy.deepcopy(state)

    names = list(state.joint_state.name)
    positions = list(state.joint_state.position)

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

        positions[lookup[name]] = float(value)

    state.joint_state.position = positions

    return state


def set_gripper_closed(state):
    state = copy.deepcopy(state)

    names = list(state.joint_state.name)
    positions = list(state.joint_state.position)

    lookup = {
        name: i
        for i, name in enumerate(names)
    }

    if "cobotta_joint_gripper" in lookup:
        positions[
            lookup["cobotta_joint_gripper"]
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

    state.joint_state.position = positions

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


def solve_ik(
    compute_ik,
    target,
    seed_state,
):
    req = GetPositionIKRequest()

    req.ik_request.group_name = GROUP
    req.ik_request.ik_link_name = TIP
    req.ik_request.pose_stamped = target

    req.ik_request.robot_state = (
        copy.deepcopy(seed_state)
    )

    # IK探索とCollision評価は分離する。
    req.ik_request.avoid_collisions = False

    req.ik_request.timeout = rospy.Duration(
        IK_TIMEOUT
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


def check_collision(
    check_validity,
    state,
):
    req = GetStateValidityRequest()

    req.robot_state = state
    req.group_name = GROUP

    res = check_validity(req)

    return bool(res.valid)


def halton(index, base):
    result = 0.0
    fraction = 1.0 / base
    i = index

    while i > 0:
        result += fraction * (i % base)
        i //= base
        fraction /= base

    return result


def get_joint_limits():
    xml = rospy.get_param("/robot_description")

    urdf = URDF.from_xml_string(xml)

    limits = []

    for name in COBOTTA_JOINTS:
        joint = urdf.joint_map[name]

        if joint.limit is None:
            raise RuntimeError(
                "Joint has no limits: "
                + name
            )

        lower = float(joint.limit.lower)
        upper = float(joint.limit.upper)

        limits.append(
            (lower, upper)
        )

    return limits


def make_global_seeds(
    joint_limits,
    count,
):
    # 6次元用の互いに異なる素数。
    bases = [
        2, 3, 5, 7, 11, 13
    ]

    seeds = []

    for i in range(1, count + 1):
        q = []

        for joint_index, (
            lower,
            upper,
        ) in enumerate(joint_limits):

            fraction = halton(
                i,
                bases[joint_index],
            )

            value = (
                lower
                + fraction
                * (upper - lower)
            )

            q.append(value)

        seeds.append(q)

    return seeds


def joint_distance_max(q1, q2):
    # COBOTTA各関節は有限範囲を持つため、
    # 今回は直接の関節角差で比較する。
    return max(
        abs(a - b)
        for a, b in zip(q1, q2)
    )


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
    joints,
):
    for solution in solutions:
        if (
            joint_distance_max(
                solution["joints"],
                joints,
            )
            <= DISTINCT_TOL_RAD
        ):
            return False

    solutions.append({
        "joints": list(joints),
        "collision_free": None,
    })

    return True


def make_target(
    trajectory,
    index,
):
    target = PoseStamped()

    target.header.stamp = rospy.Time(0)

    target.header.frame_id = (
        trajectory.header.frame_id
        if trajectory.header.frame_id
        else "paper_center"
    )

    target.pose = trajectory.poses[index]

    return target


def main():
    rospy.init_node(
        "cobotta_phase1d_fold_"
        "multiseed_ik_diagnostic"
    )

    moveit_commander.roscpp_initialize(
        sys.argv
    )

    p0_index = int(
        rospy.get_param(
            "~p0_index",
            0,
        )
    )

    normal_sign = str(
        rospy.get_param(
            "~normal_sign",
            "N+",
        )
    )

    global_seed_count = int(
        rospy.get_param(
            "~global_seed_count",
            GLOBAL_SEED_COUNT,
        )
    )

    print(
        "===== PHASE1-D MULTI-SEED "
        "IK DIAGNOSTIC ====="
    )

    print(
        "candidate         : P0[{}] {}"
        .format(
            p0_index,
            normal_sign,
        )
    )

    print(
        "diagnostic range  : {}-{}"
        .format(
            START_INDEX,
            END_INDEX,
        )
    )

    print(
        "global seed count : {}"
        .format(
            global_seed_count
        )
    )

    print(
        "IK timeout        : {:.3f} s"
        .format(
            IK_TIMEOUT
        )
    )

    print(
        "distinct tol      : {:.3f} deg"
        .format(
            DISTINCT_TOL_DEG
        )
    )

    print(
        "gripper           : CLOSED 0 mm"
    )

    robot = moveit_commander.RobotCommander()

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

    meta_msg = rospy.wait_for_message(
        METADATA_TOPIC,
        String,
        timeout=15.0,
    )

    metadata = json.loads(meta_msg.data)

    matches = [
        c
        for c in metadata["candidates"]
        if (
            int(c["p0_index"])
            == p0_index
            and str(c["normal_sign"])
            == normal_sign
        )
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one candidate, "
            "found {}".format(
                len(matches)
            )
        )

    candidate = matches[0]

    trajectory = rospy.wait_for_message(
        candidate["trajectory_topic"],
        PoseArray,
        timeout=15.0,
    )

    grasp_joints = [
        float(v)
        for v in candidate[
            "grasp_joints_rad"
        ]
    ]

    robot_state = robot.get_current_state()

    joint_limits = get_joint_limits()

    print()
    print("Joint limits:")

    for name, (
        lower,
        upper,
    ) in zip(
        COBOTTA_JOINTS,
        joint_limits,
    ):
        print(
            "  {} : {:+.3f} .. {:+.3f} deg"
            .format(
                name,
                math.degrees(lower),
                math.degrees(upper),
            )
        )

    global_seeds = make_global_seeds(
        joint_limits,
        global_seed_count,
    )

    midpoint_seed = [
        0.5 * (lower + upper)
        for lower, upper
        in joint_limits
    ]

    # ---------------------------------------
    # まず従来方式を0から再現し、
    # 各index直前で使用していたseedを保存。
    # ---------------------------------------
    baseline_seed_state = (
        set_arm_joints(
            robot_state,
            grasp_joints,
        )
    )

    baseline_seed_state = (
        set_gripper_closed(
            baseline_seed_state
        )
    )

    baseline_seed_before = {}

    for index in range(
        0,
        END_INDEX + 1,
    ):
        baseline_seed_before[index] = (
            extract_arm_joints(
                baseline_seed_state
            )
        )

        target = make_target(
            trajectory,
            index,
        )

        solution = solve_ik(
            compute_ik,
            target,
            baseline_seed_state,
        )

        if solution is not None:
            baseline_seed_state = (
                copy.deepcopy(
                    solution
                )
            )

    # ---------------------------------------
    # Multi-seed診断
    # ---------------------------------------
    total_start = time.perf_counter()

    summary = []

    print()
    print(
        "============================================================"
    )
    print(
        " MULTI-SEED RESULTS"
    )
    print(
        "============================================================"
    )

    for index in range(
        START_INDEX,
        END_INDEX + 1,
    ):
        seed_bank = []

        # 候補のP0到着状態
        append_unique_seed(
            seed_bank,
            grasp_joints,
        )

        # 従来の連続IKがこのindex直前で
        # 使用していたseed
        append_unique_seed(
            seed_bank,
            baseline_seed_before[index],
        )

        # 関節可動域の中央
        append_unique_seed(
            seed_bank,
            midpoint_seed,
        )

        # 関節範囲全体の決定論的seed
        for seed in global_seeds:
            append_unique_seed(
                seed_bank,
                seed,
            )

        target = make_target(
            trajectory,
            index,
        )

        raw_ik_success = 0
        distinct_solutions = []

        start_time = time.perf_counter()

        for seed_joints in seed_bank:
            seed_state = set_arm_joints(
                robot_state,
                seed_joints,
            )

            seed_state = (
                set_gripper_closed(
                    seed_state
                )
            )

            solution = solve_ik(
                compute_ik,
                target,
                seed_state,
            )

            if solution is None:
                continue

            raw_ik_success += 1

            joints = extract_arm_joints(
                solution
            )

            add_distinct_solution(
                distinct_solutions,
                joints,
            )

        # 各distinct IK解をCollision確認
        collision_free_count = 0

        for solution in distinct_solutions:
            state = set_arm_joints(
                robot_state,
                solution["joints"],
            )

            state = set_gripper_closed(
                state
            )

            collision_free = check_collision(
                check_validity,
                state,
            )

            solution[
                "collision_free"
            ] = collision_free

            if collision_free:
                collision_free_count += 1

        elapsed = (
            time.perf_counter()
            - start_time
        )

        summary.append({
            "index": index,
            "seed_count":
                len(seed_bank),
            "raw_ik_success":
                raw_ik_success,
            "distinct":
                len(distinct_solutions),
            "collision_free":
                collision_free_count,
            "time_sec":
                elapsed,
        })

        print()
        print(
            "index {:3d} | "
            "seeds={:2d} | "
            "IKsuccess={:2d} | "
            "distinct={:2d} | "
            "collision-free={:2d} | "
            "time={:.3f}s"
            .format(
                index,
                len(seed_bank),
                raw_ik_success,
                len(distinct_solutions),
                collision_free_count,
                elapsed,
            )
        )

        for solution_index, solution in enumerate(
            distinct_solutions
        ):
            q_deg = [
                math.degrees(q)
                for q in solution["joints"]
            ]

            print(
                "  sol{:02d} {} : {}"
                .format(
                    solution_index,
                    (
                        "FREE"
                        if solution[
                            "collision_free"
                        ]
                        else "COLLISION"
                    ),
                    " ".join(
                        "{:+8.3f}".format(v)
                        for v in q_deg
                    ),
                )
            )

    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    print()
    print(
        "============================================================"
    )
    print(
        " SUMMARY"
    )
    print(
        "============================================================"
    )

    print(
        "idx | seeds | IK success | "
        "distinct | collision-free | time[s]"
    )

    for row in summary:
        print(
            "{:3d} | {:5d} | {:10d} | "
            "{:8d} | {:14d} | {:.3f}"
            .format(
                row["index"],
                row["seed_count"],
                row["raw_ik_success"],
                row["distinct"],
                row["collision_free"],
                row["time_sec"],
            )
        )

    print()
    print(
        "total diagnostic time = {:.3f} s"
        .format(
            total_elapsed
        )
    )

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "0 distinct solutions means that no IK solution "
        "was found from the tested seeds."
    )
    print(
        "It does NOT mathematically prove that "
        "no IK solution exists."
    )


if __name__ == "__main__":
    main()
