#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import json
import math
from pathlib import Path

import rospy

from moveit_msgs.msg import PlanningSceneComponents
from moveit_msgs.srv import (
    GetPlanningScene,
    GetPlanningSceneRequest,
    GetPositionFK,
    GetPositionIK,
    GetStateValidity,
)

from cobotta_generic_gripper import set_gripper

from cobotta_generic_release_clearance_search import (
    load_arm_posture,
    get_scene_state,
    set_named_joints,
    joint_values,
    check_state,
    pair_text,
    get_fk_pose,
    shifted_pose,
    solve_ik,
    gripper_sweep,
    make_distances,
)


def load_clearance_result(path_text):
    path = Path(path_text).expanduser()

    if not path.is_absolute():
        path = Path.cwd() / path

    if not path.exists():
        raise RuntimeError(
            "clearance JSON not found: {}".format(path)
        )

    with path.open() as f:
        data = json.load(f)

    required = [
        "input_json",
        "segment_key",
        "point_index",
        "frame_id",
        "tip_link",
        "retreat_direction",
        "retreat_distance_m",
        "search_resolution_m",
        "gripper_from_m",
        "release_opening_m",
        "arm_joint_names",
        "release_ready_joints_rad",
    ]

    for key in required:
        if key not in data:
            raise RuntimeError(
                "clearance JSON missing key: {}".format(
                    key
                )
            )

    return str(path), data


def max_joint_delta(
    joints_a,
    joints_b,
):
    return max(
        abs(a - b)
        for a, b in zip(
            joints_a,
            joints_b,
        )
    )


def main():
    rospy.init_node(
        "cobotta_generic_release_retreat_path"
    )

    clearance_json = rospy.get_param(
        "~clearance_json"
    )

    output_json = rospy.get_param(
        "~output_json"
    )

    group = rospy.get_param(
        "~group",
        "cobotta_arm",
    )

    ik_timeout = float(
        rospy.get_param(
            "~ik_timeout",
            0.10,
        )
    )

    # 既に合意済みのgripper sweep分解能
    gripper_step_m = float(
        rospy.get_param(
            "~gripper_step_m",
            0.00001,
        )
    )

    (
        resolved_clearance_json,
        clearance,
    ) = load_clearance_result(
        clearance_json
    )

    input_json = clearance["input_json"]
    segment_key = clearance["segment_key"]
    point_index = int(
        clearance["point_index"]
    )

    frame = clearance["frame_id"]
    tip = clearance["tip_link"]

    direction = [
        float(v)
        for v in clearance[
            "retreat_direction"
        ]
    ]

    retreat_distance_m = float(
        clearance[
            "retreat_distance_m"
        ]
    )

    path_step_m = float(
        clearance[
            "search_resolution_m"
        ]
    )

    gripper_from_m = float(
        clearance[
            "gripper_from_m"
        ]
    )

    release_opening_m = float(
        clearance[
            "release_opening_m"
        ]
    )

    reference_release_joints = [
        float(q)
        for q in clearance[
            "release_ready_joints_rad"
        ]
    ]

    if path_step_m <= 0.0:
        raise RuntimeError(
            "search_resolution_m must be > 0"
        )

    if retreat_distance_m < 0.0:
        raise RuntimeError(
            "retreat_distance_m must be >= 0"
        )

    if gripper_step_m <= 0.0:
        raise RuntimeError(
            "gripper_step_m must be > 0"
        )

    norm = math.sqrt(
        sum(v * v for v in direction)
    )

    if norm <= 1.0e-12:
        raise RuntimeError(
            "retreat direction is zero"
        )

    direction = [
        v / norm
        for v in direction
    ]

    (
        resolved_input_json,
        arm_joint_names,
        start_joints,
        resolved_index,
        segment_count,
    ) = load_arm_posture(
        input_json,
        segment_key,
        point_index,
    )

    if (
        list(arm_joint_names)
        != list(
            clearance["arm_joint_names"]
        )
    ):
        raise RuntimeError(
            "arm joint names differ between "
            "input and clearance result"
        )

    for service in [
        "/get_planning_scene",
        "/compute_fk",
        "/compute_ik",
        "/check_state_validity",
    ]:
        rospy.wait_for_service(
            service,
            timeout=20.0,
        )

    get_scene = rospy.ServiceProxy(
        "/get_planning_scene",
        GetPlanningScene,
    )

    compute_fk = rospy.ServiceProxy(
        "/compute_fk",
        GetPositionFK,
    )

    compute_ik = rospy.ServiceProxy(
        "/compute_ik",
        GetPositionIK,
    )

    check_validity = rospy.ServiceProxy(
        "/check_state_validity",
        GetStateValidity,
    )

    scene_state = get_scene_state(
        get_scene
    )

    start_state = set_named_joints(
        scene_state,
        arm_joint_names,
        start_joints,
    )

    start_state = set_gripper(
        start_state,
        gripper_from_m,
    )

    start_pose = get_fk_pose(
        compute_fk,
        start_state,
        frame,
        tip,
    )

    distances = make_distances(
        0.0,
        retreat_distance_m,
        path_step_m,
    )

    print()
    print(
        "===== Generic Release Retreat Path ====="
    )
    print(
        "clearance JSON : {}".format(
            resolved_clearance_json
        )
    )
    print(
        "source JSON    : {}".format(
            resolved_input_json
        )
    )
    print(
        "source point   : {} / {}".format(
            resolved_index,
            segment_count - 1,
        )
    )
    print(
        "frame          : {}".format(
            frame
        )
    )
    print(
        "tip            : {}".format(
            tip
        )
    )
    print(
        "retreat dir    : "
        "({:.6f}, {:.6f}, {:.6f})".format(
            direction[0],
            direction[1],
            direction[2],
        )
    )
    print(
        "distance       : {:.3f} mm".format(
            retreat_distance_m * 1000.0
        )
    )
    print(
        "path step      : {:.3f} mm".format(
            path_step_m * 1000.0
        )
    )
    print(
        "point count    : {}".format(
            len(distances)
        )
    )
    print(
        "gripper CLOSED : {:.3f} mm/side".format(
            gripper_from_m * 1000.0
        )
    )

    print()
    print("----- RETREAT PATH -----")

    path = []

    previous_state = copy.deepcopy(
        start_state
    )

    previous_joints = None
    max_adjacent_delta_rad = 0.0
    max_adjacent_index = None

    for i, distance_m in enumerate(
        distances
    ):
        if i == 0:
            state = copy.deepcopy(
                start_state
            )

        else:
            target_pose = shifted_pose(
                start_pose,
                direction,
                distance_m,
            )

            state, ik_error = solve_ik(
                compute_ik,
                target_pose,
                previous_state,
                group,
                tip,
                ik_timeout,
            )

            if state is None:
                raise RuntimeError(
                    "IK failed at index {} "
                    "distance {:.6f} mm "
                    "error={}".format(
                        i,
                        distance_m * 1000.0,
                        ik_error,
                    )
                )

            state = set_gripper(
                state,
                gripper_from_m,
            )

        state.joint_state.header.stamp = (
            rospy.Time(0)
        )

        valid, pairs = check_state(
            check_validity,
            state,
            group,
        )

        if not valid:
            raise RuntimeError(
                "Collision at index {} "
                "distance {:.6f} mm : {}".format(
                    i,
                    distance_m * 1000.0,
                    pair_text(pairs),
                )
            )

        joints = joint_values(
            state,
            arm_joint_names,
        )

        if previous_joints is None:
            adjacent_delta_rad = 0.0
        else:
            adjacent_delta_rad = (
                max_joint_delta(
                    previous_joints,
                    joints,
                )
            )

            if (
                adjacent_delta_rad
                > max_adjacent_delta_rad
            ):
                max_adjacent_delta_rad = (
                    adjacent_delta_rad
                )
                max_adjacent_index = (
                    i - 1,
                    i,
                )

        path.append({
            "index": i,
            "distance_m": float(
                distance_m
            ),
            "joints_rad": [
                float(q)
                for q in joints
            ],
            "gripper_m": float(
                gripper_from_m
            ),
        })

        print(
            "[{:02d}/{:02d}] "
            "distance={:.3f} mm | "
            "VALID | "
            "max_joint_step={:.6f} deg".format(
                i,
                len(distances) - 1,
                distance_m * 1000.0,
                math.degrees(
                    adjacent_delta_rad
                ),
            )
        )

        previous_state = copy.deepcopy(
            state
        )
        previous_joints = list(joints)

    final_state = copy.deepcopy(
        previous_state
    )

    final_joints = joint_values(
        final_state,
        arm_joint_names,
    )

    final_pose = get_fk_pose(
        compute_fk,
        final_state,
        frame,
        tip,
    )

    reference_joint_diff_rad = (
        max_joint_delta(
            final_joints,
            reference_release_joints,
        )
    )

    print()
    print(
        "----- FINAL RELEASE SWEEP CHECK -----"
    )

    (
        release_valid,
        first_bad_opening,
        release_pairs,
        sweep_samples,
    ) = gripper_sweep(
        check_validity,
        final_state,
        group,
        gripper_from_m,
        release_opening_m,
        gripper_step_m,
    )

    if not release_valid:
        raise RuntimeError(
            "Final release sweep collided "
            "at q={:.6f} mm/side : {}".format(
                first_bad_opening * 1000.0,
                pair_text(release_pairs),
            )
        )

    print(
        "q={:.3f} -> {:.3f} mm/side : "
        "FULL SWEEP VALID".format(
            gripper_from_m * 1000.0,
            release_opening_m * 1000.0,
        )
    )

    print(
        "sweep samples : {}".format(
            sweep_samples
        )
    )

    print()
    print(
        "===== RETREAT PATH RESULT ====="
    )
    print("RESULT                  : VALID")
    print(
        "path point count        : {}".format(
            len(path)
        )
    )
    print(
        "retreat distance        : {:.3f} mm".format(
            retreat_distance_m * 1000.0
        )
    )
    print(
        "max adjacent joint step : {:.9f} deg".format(
            math.degrees(
                max_adjacent_delta_rad
            )
        )
    )

    if max_adjacent_index is None:
        print(
            "max step location       : -"
        )
    else:
        print(
            "max step location       : "
            "{} -> {}".format(
                max_adjacent_index[0],
                max_adjacent_index[1],
            )
        )

    print(
        "final vs search result   : "
        "{:.9f} deg max joint diff".format(
            math.degrees(
                reference_joint_diff_rad
            )
        )
    )

    output_path = Path(
        output_json
    ).expanduser()

    if not output_path.is_absolute():
        output_path = (
            Path.cwd()
            / output_path
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "schema_version": 1,
        "type":
            "generic_release_retreat_path",
        "clearance_result_json":
            resolved_clearance_json,
        "source_input_json":
            resolved_input_json,
        "source_segment_key":
            segment_key,
        "source_point_index":
            resolved_index,
        "frame_id":
            frame,
        "tip_link":
            tip,
        "arm_joint_names":
            arm_joint_names,
        "retreat_direction":
            direction,
        "retreat_distance_m":
            retreat_distance_m,
        "path_resolution_m":
            path_step_m,
        "gripper_closed_m":
            gripper_from_m,
        "release_opening_m":
            release_opening_m,
        "gripper_sweep_resolution_m":
            gripper_step_m,
        "path_point_count":
            len(path),
        "max_adjacent_joint_delta_rad":
            max_adjacent_delta_rad,
        "reference_release_ready_max_joint_diff_rad":
            reference_joint_diff_rad,
        "retreat_path":
            path,
        "release_event": {
            "from_m":
                gripper_from_m,
            "to_m":
                release_opening_m,
            "collision_check":
                "FULL_SWEEP_VALID",
            "sample_count":
                sweep_samples,
        },
    }

    with output_path.open("w") as f:
        json.dump(
            payload,
            f,
            indent=2,
        )

    print()
    print(
        "saved : {}".format(
            output_path
        )
    )


if __name__ == "__main__":
    main()
