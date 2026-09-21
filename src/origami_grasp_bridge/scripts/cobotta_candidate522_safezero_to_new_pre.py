#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import rospy
import moveit_commander

from sensor_msgs.msg import JointState
from moveit_msgs.msg import Constraints
from moveit_msgs.srv import (
    GetStateValidity,
)


SAFEZERO_BASE_FILE = Path(
    "/home/maeda/catkin_ws/src/"
    "origami_grasp_bridge/scripts/"
    "cobotta_safezero_to_pregrasp_feasibility_diagnostic.py"
)

OPEN_PRE_RESULT_FILE = Path(
    "/home/maeda/catkin_ws/results/phase1d/"
    "candidate522_index73_open_pre_connection.json"
)

OUTPUT_FILE = Path(
    "/home/maeda/catkin_ws/results/phase1d/"
    "candidate522_safezero_to_new_pre.json"
)


def load_safezero_module():
    spec = importlib.util.spec_from_file_location(
        "safezero_base",
        str(SAFEZERO_BASE_FILE),
    )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def main():
    rospy.init_node(
        "cobotta_candidate522_"
        "safezero_to_new_pre"
    )

    moveit_commander.roscpp_initialize(
        sys.argv
    )

    base = load_safezero_module()

    # --------------------------------------
    # 前回保存したOPEN PRE結果を読む
    # --------------------------------------
    if not OPEN_PRE_RESULT_FILE.exists():
        raise RuntimeError(
            "Result file not found: "
            + str(OPEN_PRE_RESULT_FILE)
        )

    previous = json.loads(
        OPEN_PRE_RESULT_FILE.read_text()
    )

    usable = [
        r
        for r in previous["results"]
        if (
            r.get("fold_reverse")
            and
            r.get("open_pre_connection")
        )
    ]

    if not usable:
        raise RuntimeError(
            "No successful OPEN PRE branch "
            "in previous result."
        )

    # 今回は成功した枝が1本。
    selected = usable[0]

    target = [
        float(v)
        for v in selected[
            "new_pre_joints_rad"
        ]
    ]

    branch_index = int(
        selected["branch_index"]
    )

    print(
        "===== CANDIDATE 522 "
        "SAFE-ZERO -> NEW PRE ====="
    )

    print(
        "source branch : {}".format(
            branch_index
        )
    )

    print(
        "safe-zero     : {}".format(
            base.SAFE_ZERO
        )
    )

    print(
        "gripper       : OPEN {:.1f} mm"
        .format(
            base.GRIPPER_OPEN
            * 1000.0
        )
    )

    print(
        "planning      : {:.1f} s x {} attempts"
        .format(
            base.PLANNING_TIME,
            base.PLANNING_ATTEMPTS,
        )
    )

    print(
        "dense step    : <= {:.4f} rad"
        .format(
            base.MAX_INTERPOLATION_STEP
        )
    )

    print()
    print(
        "new PRE joints [deg]:"
    )

    print(
        "  "
        + " ".join(
            "{:+.3f}".format(
                math.degrees(v)
            )
            for v in target
        )
    )

    # --------------------------------------
    # 現在joint_states取得
    # --------------------------------------
    current = rospy.wait_for_message(
        "/joint_states",
        JointState,
        timeout=5.0,
    )

    rospy.wait_for_service(
        "/check_state_validity",
        timeout=10.0,
    )

    check = rospy.ServiceProxy(
        "/check_state_validity",
        GetStateValidity,
    )

    group = (
        moveit_commander.MoveGroupCommander(
            base.GROUP
        )
    )

    group.set_planning_time(
        base.PLANNING_TIME
    )

    group.set_num_planning_attempts(
        base.PLANNING_ATTEMPTS
    )

    start_time = time.perf_counter()

    # --------------------------------------
    # Safe-Zero状態確認
    # --------------------------------------
    (
        safezero_valid,
        safezero_contacts,
    ) = base.check_state(
        check,
        current,
        base.SAFE_ZERO,
    )

    print()
    print(
        "Safe-Zero state valid : {}"
        .format(
            safezero_valid
        )
    )

    if not safezero_valid:
        print(
            "Safe-Zero contacts:"
        )

        for c in safezero_contacts:
            print(
                "  {} <-> {} depth={}"
                .format(
                    c["body_1"],
                    c["body_2"],
                    c["depth"],
                )
            )

        result = {
            "result":
                "START_STATE_INVALID",
            "candidate_index": 522,
            "branch_index":
                branch_index,
            "safezero_joints_rad":
                list(base.SAFE_ZERO),
            "new_pre_joints_rad":
                target,
            "contacts":
                safezero_contacts,
        }

        OUTPUT_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        OUTPUT_FILE.write_text(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            )
        )

        return

    # --------------------------------------
    # new PRE状態確認
    # --------------------------------------
    (
        target_valid,
        target_contacts,
    ) = base.check_state(
        check,
        current,
        target,
    )

    print(
        "new PRE state valid   : {}"
        .format(
            target_valid
        )
    )

    if not target_valid:
        print(
            "new PRE contacts:"
        )

        for c in target_contacts:
            print(
                "  {} <-> {} depth={}"
                .format(
                    c["body_1"],
                    c["body_2"],
                    c["depth"],
                )
            )

        result = {
            "result":
                "TARGET_STATE_INVALID",
            "candidate_index": 522,
            "branch_index":
                branch_index,
            "safezero_joints_rad":
                list(base.SAFE_ZERO),
            "new_pre_joints_rad":
                target,
            "contacts":
                target_contacts,
        }

        OUTPUT_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        OUTPUT_FILE.write_text(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            )
        )

        return

    # --------------------------------------
    # Safe-Zero -> new PRE planning
    # --------------------------------------
    start_state = base.make_full_state(
        current,
        base.SAFE_ZERO,
    )

    group.stop()
    group.clear_pose_targets()

    group.set_start_state(
        start_state
    )

    target_map = {
        name: value
        for name, value in zip(
            base.COBOTTA_JOINTS,
            target,
        )
    }

    group.set_joint_value_target(
        target_map
    )

    plan_result = group.plan()

    if isinstance(
        plan_result,
        tuple,
    ):
        success = bool(
            plan_result[0]
        )

        plan = plan_result[1]

    else:
        plan = plan_result

        success = (
            len(
                plan.joint_trajectory.points
            )
            > 0
        )

    points = (
        plan.joint_trajectory.points
    )

    trajectory_joint_names = list(
        plan.joint_trajectory.joint_names
    )

    print()
    print(
        "plan success : {}".format(
            success
        )
    )

    print(
        "plan points  : {}".format(
            len(points)
        )
    )

    if not success or not points:
        result_name = "PLAN_FAILED"

        duration_sec = None
        max_end_error = None
        dense_state_count = 0
        invalid_dense_index = None
        invalid_contacts = []

    else:
        final_q = (
            base.trajectory_point_to_cobotta(
                trajectory_joint_names,
                points[-1].positions,
            )
        )

        max_end_error = max(
            abs(a - b)
            for a, b in zip(
                final_q,
                target,
            )
        )

        duration_sec = (
            points[-1]
            .time_from_start
            .to_sec()
        )

        print(
            "duration     : {:.6f} s"
            .format(
                duration_sec
            )
        )

        print(
            "max end error: {:.9f} rad"
            .format(
                max_end_error
            )
        )

        if (
            max_end_error
            >= base.MAX_END_ERROR
        ):
            result_name = "END_ERROR"

            dense_state_count = 0
            invalid_dense_index = None
            invalid_contacts = []

        else:
            # ----------------------------------
            # Dense Collision Validation
            # ----------------------------------
            previous_q = list(
                base.SAFE_ZERO
            )

            dense_state_count = 1

            invalid_found = False
            invalid_dense_index = None
            invalid_contacts = []

            for point in points:

                current_q = (
                    base.trajectory_point_to_cobotta(
                        trajectory_joint_names,
                        point.positions,
                    )
                )

                max_delta = max(
                    abs(b - a)
                    for a, b in zip(
                        previous_q,
                        current_q,
                    )
                )

                subdivisions = max(
                    1,
                    int(
                        math.ceil(
                            max_delta
                            / base.MAX_INTERPOLATION_STEP
                        )
                    ),
                )

                for k in range(
                    1,
                    subdivisions + 1,
                ):
                    t = (
                        float(k)
                        / float(subdivisions)
                    )

                    q = [
                        a + t * (b - a)
                        for a, b in zip(
                            previous_q,
                            current_q,
                        )
                    ]

                    valid, contacts = (
                        base.check_state(
                            check,
                            current,
                            q,
                        )
                    )

                    if not valid:
                        invalid_found = True

                        invalid_dense_index = (
                            dense_state_count
                        )

                        invalid_contacts = contacts

                        break

                    dense_state_count += 1

                if invalid_found:
                    break

                previous_q = current_q

            print(
                "dense states : {}"
                .format(
                    dense_state_count
                )
            )

            if invalid_found:
                result_name = (
                    "DENSE_COLLISION"
                )

                print(
                    "invalid dense index : {}"
                    .format(
                        invalid_dense_index
                    )
                )

                for c in invalid_contacts:
                    print(
                        "  CONTACT: {} <-> {} "
                        "depth={}"
                        .format(
                            c["body_1"],
                            c["body_2"],
                            c["depth"],
                        )
                    )

            else:
                result_name = (
                    "SAFEZERO_TO_NEW_PRE_FEASIBLE"
                )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    result = {
        "candidate_index": 522,
        "source_finish_index": 73,
        "branch_index":
            branch_index,

        "safezero_joints_rad":
            list(base.SAFE_ZERO),

        "new_pre_joints_rad":
            target,

        "gripper_open_m":
            float(
                base.GRIPPER_OPEN
            ),

        "planning_time_sec":
            float(
                base.PLANNING_TIME
            ),

        "planning_attempts":
            int(
                base.PLANNING_ATTEMPTS
            ),

        "max_interpolation_step_rad":
            float(
                base.MAX_INTERPOLATION_STEP
            ),

        "result":
            result_name,

        "plan_points":
            int(
                len(points)
            )
            if success
            else 0,

        "duration_sec":
            (
                float(duration_sec)
                if duration_sec
                is not None
                else None
            ),

        "max_end_error_rad":
            (
                float(max_end_error)
                if max_end_error
                is not None
                else None
            ),

        "dense_state_count":
            int(
                dense_state_count
            ),

        "invalid_dense_index":
            (
                int(
                    invalid_dense_index
                )
                if invalid_dense_index
                is not None
                else None
            ),

        "contacts":
            invalid_contacts,

        "diagnostic_elapsed_sec":
            float(
                elapsed
            ),
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FILE.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print(
        "========================================"
    )

    print(
        " FINAL SAFE-ZERO -> NEW PRE RESULT"
    )

    print(
        "========================================"
    )

    print(
        "RESULT       : {}".format(
            result_name
        )
    )

    print(
        "plan points  : {}".format(
            result["plan_points"]
        )
    )

    print(
        "dense states : {}".format(
            result[
                "dense_state_count"
            ]
        )
    )

    print(
        "elapsed      : {:.3f} s"
        .format(
            elapsed
        )
    )

    print()
    print(
        "saved JSON:"
    )

    print(
        "  {}".format(
            OUTPUT_FILE
        )
    )


if __name__ == "__main__":
    main()
