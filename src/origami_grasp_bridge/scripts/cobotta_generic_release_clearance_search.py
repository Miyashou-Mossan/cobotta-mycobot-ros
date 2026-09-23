#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import json
import math
from pathlib import Path

import rospy

from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import (
    MoveItErrorCodes,
    PlanningSceneComponents,
)
from moveit_msgs.srv import (
    GetPlanningScene,
    GetPlanningSceneRequest,
    GetPositionFK,
    GetPositionFKRequest,
    GetPositionIK,
    GetPositionIKRequest,
    GetStateValidity,
    GetStateValidityRequest,
)

from cobotta_generic_gripper import (
    set_gripper,
    interpolate_openings,
)


def load_arm_posture(
    input_json,
    segment_key,
    point_index,
):
    path = Path(input_json).expanduser()

    if not path.is_absolute():
        path = Path.cwd() / path

    if not path.exists():
        raise RuntimeError(
            "input JSON not found: {}".format(path)
        )

    with path.open() as f:
        data = json.load(f)

    names = list(data["joint_names"])
    segment = data[segment_key]

    if not segment:
        raise RuntimeError(
            "segment is empty: {}".format(segment_key)
        )

    index = int(point_index)

    if index < 0:
        index = len(segment) + index

    if index < 0 or index >= len(segment):
        raise RuntimeError(
            "point_index out of range"
        )

    joints = [
        float(q)
        for q in segment[index]["joints_rad"]
    ]

    return (
        str(path),
        names,
        joints,
        index,
        len(segment),
    )


def get_scene_state(get_scene):
    req = GetPlanningSceneRequest()

    req.components.components = (
        PlanningSceneComponents.ROBOT_STATE
    )

    res = get_scene(req)

    state = copy.deepcopy(
        res.scene.robot_state
    )

    if not state.joint_state.name:
        raise RuntimeError(
            "Planning Scene RobotState is empty"
        )

    return state


def set_named_joints(
    state,
    joint_names,
    values,
):
    result = copy.deepcopy(state)

    names = list(
        result.joint_state.name
    )
    positions = list(
        result.joint_state.position
    )

    lookup = {
        name: i
        for i, name in enumerate(names)
    }

    for name, value in zip(
        joint_names,
        values,
    ):
        if name not in lookup:
            raise RuntimeError(
                "RobotState missing joint: {}".format(
                    name
                )
            )

        positions[
            lookup[name]
        ] = float(value)

    result.joint_state.position = positions
    result.joint_state.header.stamp = rospy.Time(0)

    return result


def joint_values(
    state,
    joint_names,
):
    lookup = dict(zip(
        state.joint_state.name,
        state.joint_state.position,
    ))

    return [
        float(lookup[name])
        for name in joint_names
    ]


def max_joint_diff(
    state_a,
    state_b,
    joint_names,
):
    a = joint_values(
        state_a,
        joint_names,
    )
    b = joint_values(
        state_b,
        joint_names,
    )

    return max(
        abs(x - y)
        for x, y in zip(a, b)
    )


def check_state(
    check_validity,
    state,
    group,
):
    req = GetStateValidityRequest()

    req.robot_state = copy.deepcopy(state)
    req.group_name = group

    res = check_validity(req)

    pairs = []

    for contact in res.contacts:
        pair = tuple(sorted([
            contact.contact_body_1,
            contact.contact_body_2,
        ]))

        if pair not in pairs:
            pairs.append(pair)

    return bool(res.valid), pairs


def pair_text(pairs):
    if not pairs:
        return "-"

    return "; ".join(
        "{}<->{}".format(a, b)
        for a, b in pairs
    )


def get_fk_pose(
    compute_fk,
    state,
    frame,
    tip,
):
    req = GetPositionFKRequest()

    req.header.frame_id = frame
    req.header.stamp = rospy.Time(0)

    req.fk_link_names = [tip]
    req.robot_state = copy.deepcopy(state)

    res = compute_fk(req)

    if (
        res.error_code.val
        != MoveItErrorCodes.SUCCESS
    ):
        raise RuntimeError(
            "FK failed: {}".format(
                res.error_code.val
            )
        )

    if not res.pose_stamped:
        raise RuntimeError(
            "FK returned no pose"
        )

    return copy.deepcopy(
        res.pose_stamped[0]
    )


def shifted_pose(
    base_pose,
    direction,
    distance_m,
):
    pose = copy.deepcopy(base_pose)

    pose.header.stamp = rospy.Time(0)

    pose.pose.position.x += (
        direction[0] * distance_m
    )
    pose.pose.position.y += (
        direction[1] * distance_m
    )
    pose.pose.position.z += (
        direction[2] * distance_m
    )

    return pose


def solve_ik(
    compute_ik,
    pose,
    seed_state,
    group,
    tip,
    timeout_sec,
):
    req = GetPositionIKRequest()

    req.ik_request.group_name = group
    req.ik_request.ik_link_name = tip

    req.ik_request.pose_stamped = copy.deepcopy(
        pose
    )

    req.ik_request.robot_state = copy.deepcopy(
        seed_state
    )

    # IKとCollision判定は分離する。
    req.ik_request.avoid_collisions = False

    req.ik_request.timeout = rospy.Duration(
        timeout_sec
    )

    res = compute_ik(req)

    if (
        res.error_code.val
        != MoveItErrorCodes.SUCCESS
    ):
        return None, res.error_code.val

    return copy.deepcopy(
        res.solution
    ), res.error_code.val


def gripper_sweep(
    check_validity,
    arm_state,
    group,
    from_m,
    to_m,
    step_m,
):
    distance = abs(to_m - from_m)

    if distance == 0.0:
        openings = [from_m]
    else:
        steps = int(
            math.ceil(
                distance / step_m
            )
        )

        openings = interpolate_openings(
            from_m,
            to_m,
            steps,
        )

    for opening in openings:
        state = set_gripper(
            arm_state,
            opening,
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
            return (
                False,
                opening,
                pairs,
                len(openings),
            )

    return (
        True,
        None,
        [],
        len(openings),
    )


def make_distances(
    start_m,
    end_m,
    step_m,
):
    if end_m < start_m:
        raise ValueError(
            "end_m must be >= start_m"
        )

    if step_m <= 0.0:
        raise ValueError(
            "step_m must be > 0"
        )

    distance = end_m - start_m

    if distance <= 1.0e-15:
        return [start_m]

    steps = int(
        math.ceil(
            distance / step_m
        )
    )

    return [
        start_m
        + min(
            float(i) * step_m,
            distance,
        )
        for i in range(steps + 1)
    ]


def evaluate_distance(
    distance_m,
    base_pose,
    direction,
    seed_state,
    base_state,
    arm_joint_names,
    compute_ik,
    check_validity,
    group,
    tip,
    ik_timeout,
    gripper_from_m,
    release_opening_m,
    gripper_step_m,
):
    if abs(distance_m) <= 1.0e-15:
        arm_state = copy.deepcopy(
            base_state
        )

        ik_error = MoveItErrorCodes.SUCCESS
        joint_diff = 0.0

    else:
        target_pose = shifted_pose(
            base_pose,
            direction,
            distance_m,
        )

        arm_state, ik_error = solve_ik(
            compute_ik,
            target_pose,
            seed_state,
            group,
            tip,
            ik_timeout,
        )

        if arm_state is None:
            return {
                "distance_m": distance_m,
                "ik_ok": False,
                "ik_error": ik_error,
            }

        arm_state = set_gripper(
            arm_state,
            gripper_from_m,
        )

        joint_diff = max_joint_diff(
            seed_state,
            arm_state,
            arm_joint_names,
        )

    arm_state = set_gripper(
        arm_state,
        gripper_from_m,
    )

    closed_valid, closed_pairs = (
        check_state(
            check_validity,
            arm_state,
            group,
        )
    )

    if not closed_valid:
        return {
            "distance_m": distance_m,
            "ik_ok": True,
            "arm_state": arm_state,
            "joint_diff_rad": joint_diff,
            "closed_valid": False,
            "closed_pairs": closed_pairs,
        }

    (
        release_valid,
        first_bad_opening,
        release_pairs,
        sweep_samples,
    ) = gripper_sweep(
        check_validity,
        arm_state,
        group,
        gripper_from_m,
        release_opening_m,
        gripper_step_m,
    )

    return {
        "distance_m": distance_m,
        "ik_ok": True,
        "arm_state": arm_state,
        "joint_diff_rad": joint_diff,
        "closed_valid": True,
        "closed_pairs": [],
        "release_valid": release_valid,
        "first_bad_opening_m":
            first_bad_opening,
        "release_pairs": release_pairs,
        "sweep_samples": sweep_samples,
    }


def pose_to_dict(pose_stamped):
    p = pose_stamped.pose.position
    q = pose_stamped.pose.orientation

    return {
        "frame_id":
            pose_stamped.header.frame_id,
        "position_m": [
            float(p.x),
            float(p.y),
            float(p.z),
        ],
        "orientation_xyzw": [
            float(q.x),
            float(q.y),
            float(q.z),
            float(q.w),
        ],
    }


def main():
    rospy.init_node(
        "cobotta_generic_release_clearance_search"
    )

    input_json = rospy.get_param(
        "~input_json"
    )

    segment_key = rospy.get_param(
        "~segment_key",
        "fold_p0_to_finish",
    )

    point_index = int(
        rospy.get_param(
            "~point_index",
            -1,
        )
    )

    frame = rospy.get_param(
        "~frame",
        "paper_center",
    )

    group = rospy.get_param(
        "~group",
        "cobotta_arm",
    )

    tip = rospy.get_param(
        "~tip",
        "cobotta_tool_link",
    )

    gripper_from_m = float(
        rospy.get_param(
            "~gripper_from_m",
            0.0,
        )
    )

    release_opening_m = float(
        rospy.get_param(
            "~release_opening_m",
            0.00024,
        )
    )

    gripper_step_m = float(
        rospy.get_param(
            "~gripper_step_m",
            0.00001,
        )
    )

    max_retreat_m = float(
        rospy.get_param(
            "~max_retreat_m",
            0.005,
        )
    )

    coarse_step_m = float(
        rospy.get_param(
            "~coarse_step_m",
            0.0001,
        )
    )

    fine_step_m = float(
        rospy.get_param(
            "~fine_step_m",
            0.00001,
        )
    )

    retreat_dx = float(
        rospy.get_param("~retreat_dx", 0.0)
    )
    retreat_dy = float(
        rospy.get_param("~retreat_dy", 0.0)
    )
    retreat_dz = float(
        rospy.get_param("~retreat_dz", 1.0)
    )

    ik_timeout = float(
        rospy.get_param(
            "~ik_timeout",
            0.10,
        )
    )

    output_json = rospy.get_param(
        "~output_json",
        "",
    )

    if max_retreat_m < 0.0:
        raise ValueError(
            "max_retreat_m must be >= 0"
        )

    if coarse_step_m <= 0.0:
        raise ValueError(
            "coarse_step_m must be > 0"
        )

    if fine_step_m <= 0.0:
        raise ValueError(
            "fine_step_m must be > 0"
        )

    if fine_step_m > coarse_step_m:
        raise ValueError(
            "fine_step_m must be <= coarse_step_m"
        )

    norm = math.sqrt(
        retreat_dx * retreat_dx
        + retreat_dy * retreat_dy
        + retreat_dz * retreat_dz
    )

    if norm <= 1.0e-12:
        raise ValueError(
            "retreat direction is zero"
        )

    direction = (
        retreat_dx / norm,
        retreat_dy / norm,
        retreat_dz / norm,
    )

    (
        resolved_input,
        arm_joint_names,
        arm_joints,
        resolved_index,
        segment_count,
    ) = load_arm_posture(
        input_json,
        segment_key,
        point_index,
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

    base_state = set_named_joints(
        scene_state,
        arm_joint_names,
        arm_joints,
    )

    base_state = set_gripper(
        base_state,
        gripper_from_m,
    )

    base_pose = get_fk_pose(
        compute_fk,
        base_state,
        frame,
        tip,
    )

    print()
    print(
        "===== Generic Release Clearance Search ====="
    )
    print(
        "input_json      : {}".format(
            resolved_input
        )
    )
    print(
        "segment         : {}".format(
            segment_key
        )
    )
    print(
        "point           : {} / {}".format(
            resolved_index,
            segment_count - 1,
        )
    )
    print(
        "frame           : {}".format(
            frame
        )
    )
    print(
        "tip             : {}".format(
            tip
        )
    )
    print(
        "retreat dir     : "
        "({:.6f}, {:.6f}, {:.6f})".format(
            *direction
        )
    )
    print(
        "release opening : {:.6f} mm/side".format(
            release_opening_m * 1000.0
        )
    )
    print(
        "relative opening: {:.6f} mm".format(
            2.0
            * release_opening_m
            * 1000.0
        )
    )
    print(
        "max retreat     : {:.3f} mm".format(
            max_retreat_m * 1000.0
        )
    )
    print(
        "coarse step     : {:.3f} mm".format(
            coarse_step_m * 1000.0
        )
    )
    print(
        "fine step       : {:.3f} mm".format(
            fine_step_m * 1000.0
        )
    )

    p = base_pose.pose.position

    print(
        "start tool pos  : "
        "({:.3f}, {:.3f}, {:.3f}) mm".format(
            p.x * 1000.0,
            p.y * 1000.0,
            p.z * 1000.0,
        )
    )

    # -------------------------------------------------
    # COARSE SEARCH
    # -------------------------------------------------
    print()
    print("----- COARSE SEARCH -----")

    distances = make_distances(
        0.0,
        max_retreat_m,
        coarse_step_m,
    )

    previous_state = copy.deepcopy(
        base_state
    )

    coarse_records = []
    coarse_solution_index = None

    for i, distance_m in enumerate(distances):
        result = evaluate_distance(
            distance_m,
            base_pose,
            direction,
            previous_state,
            base_state,
            arm_joint_names,
            compute_ik,
            check_validity,
            group,
            tip,
            ik_timeout,
            gripper_from_m,
            release_opening_m,
            gripper_step_m,
        )

        coarse_records.append(result)

        if not result["ik_ok"]:
            print(
                "[C {:02d}] z={:.3f} mm | "
                "IK_FAIL code={}".format(
                    i,
                    distance_m * 1000.0,
                    result["ik_error"],
                )
            )

            print(
                "Search stopped because "
                "continuous retreat path was lost."
            )
            break

        previous_state = copy.deepcopy(
            result["arm_state"]
        )

        if not result["closed_valid"]:
            print(
                "[C {:02d}] z={:.3f} mm | "
                "CLOSED_COLLISION | {}".format(
                    i,
                    distance_m * 1000.0,
                    pair_text(
                        result["closed_pairs"]
                    ),
                )
            )

            print(
                "Search stopped because "
                "CLOSED retreat path is not valid."
            )
            break

        if result["release_valid"]:
            print(
                "[C {:02d}] z={:.3f} mm | "
                "RELEASE_SWEEP_VALID | "
                "max_joint_step={:.6f} deg".format(
                    i,
                    distance_m * 1000.0,
                    math.degrees(
                        result["joint_diff_rad"]
                    ),
                )
            )

            coarse_solution_index = i
            break

        print(
            "[C {:02d}] z={:.3f} mm | "
            "RELEASE_COLLISION at "
            "q={:.3f} mm/side | {} | "
            "max_joint_step={:.6f} deg".format(
                i,
                distance_m * 1000.0,
                result[
                    "first_bad_opening_m"
                ] * 1000.0,
                pair_text(
                    result["release_pairs"]
                ),
                math.degrees(
                    result["joint_diff_rad"]
                ),
            )
        )

    if coarse_solution_index is None:
        print()
        print(
            "RESULT : NO RELEASE-CLEAR "
            "STATE FOUND"
        )
        return

    coarse_solution = coarse_records[
        coarse_solution_index
    ]

    # 0 mmで成功していればfine探索不要
    if coarse_solution_index == 0:
        final_result = coarse_solution

    else:
        # ---------------------------------------------
        # FINE SEARCH
        # ---------------------------------------------
        low_result = coarse_records[
            coarse_solution_index - 1
        ]

        low_distance = low_result[
            "distance_m"
        ]

        high_distance = coarse_solution[
            "distance_m"
        ]

        print()
        print("----- FINE SEARCH -----")
        print(
            "range : {:.3f} .. {:.3f} mm".format(
                low_distance * 1000.0,
                high_distance * 1000.0,
            )
        )

        fine_distances = make_distances(
            low_distance,
            high_distance,
            fine_step_m,
        )

        # low_distance自体はcoarseで評価済み。
        fine_distances = [
            d for d in fine_distances
            if d > low_distance + 1.0e-12
        ]

        previous_state = copy.deepcopy(
            low_result["arm_state"]
        )

        final_result = None

        for i, distance_m in enumerate(
            fine_distances
        ):
            result = evaluate_distance(
                distance_m,
                base_pose,
                direction,
                previous_state,
                base_state,
                arm_joint_names,
                compute_ik,
                check_validity,
                group,
                tip,
                ik_timeout,
                gripper_from_m,
                release_opening_m,
                gripper_step_m,
            )

            if not result["ik_ok"]:
                print(
                    "[F {:02d}] z={:.3f} mm | "
                    "IK_FAIL code={}".format(
                        i,
                        distance_m * 1000.0,
                        result["ik_error"],
                    )
                )
                break

            previous_state = copy.deepcopy(
                result["arm_state"]
            )

            if not result["closed_valid"]:
                print(
                    "[F {:02d}] z={:.3f} mm | "
                    "CLOSED_COLLISION | {}".format(
                        i,
                        distance_m * 1000.0,
                        pair_text(
                            result["closed_pairs"]
                        ),
                    )
                )
                break

            if result["release_valid"]:
                print(
                    "[F {:02d}] z={:.3f} mm | "
                    "RELEASE_SWEEP_VALID".format(
                        i,
                        distance_m * 1000.0,
                    )
                )

                final_result = result
                break

            print(
                "[F {:02d}] z={:.3f} mm | "
                "RELEASE_COLLISION at "
                "q={:.3f} mm/side | {}".format(
                    i,
                    distance_m * 1000.0,
                    result[
                        "first_bad_opening_m"
                    ] * 1000.0,
                    pair_text(
                        result["release_pairs"]
                    ),
                )
            )

        if final_result is None:
            # coarseの成功点は既に確認済み。
            final_result = coarse_solution

    final_distance = final_result[
        "distance_m"
    ]

    final_state = final_result[
        "arm_state"
    ]

    final_pose = get_fk_pose(
        compute_fk,
        final_state,
        frame,
        tip,
    )

    final_arm_joints = joint_values(
        final_state,
        arm_joint_names,
    )

    print()
    print("===== SEARCH RESULT =====")
    print("RESULT : FOUND")
    print(
        "minimum sampled retreat : "
        "{:.3f} mm".format(
            final_distance * 1000.0
        )
    )
    print(
        "search resolution        : "
        "{:.3f} mm".format(
            fine_step_m * 1000.0
        )
    )
    print(
        "release opening          : "
        "{:.3f} mm/side".format(
            release_opening_m * 1000.0
        )
    )
    print(
        "relative finger opening  : "
        "{:.3f} mm".format(
            2.0
            * release_opening_m
            * 1000.0
        )
    )

    print()
    print("--- Release-ready arm joints ---")

    for name, q in zip(
        arm_joint_names,
        final_arm_joints,
    ):
        print(
            "{} = {:.12f}".format(
                name,
                q,
            )
        )

    if output_json:
        out = Path(
            output_json
        ).expanduser()

        if not out.is_absolute():
            out = Path.cwd() / out

        out.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "schema_version": 1,
            "input_json": resolved_input,
            "segment_key": segment_key,
            "point_index": resolved_index,
            "frame_id": frame,
            "tip_link": tip,
            "retreat_direction": list(
                direction
            ),
            "retreat_distance_m":
                final_distance,
            "search_resolution_m":
                fine_step_m,
            "gripper_from_m":
                gripper_from_m,
            "release_opening_m":
                release_opening_m,
            "relative_finger_opening_m":
                2.0 * release_opening_m,
            "arm_joint_names":
                arm_joint_names,
            "release_ready_joints_rad":
                final_arm_joints,
            "start_tool_pose":
                pose_to_dict(base_pose),
            "release_ready_tool_pose":
                pose_to_dict(final_pose),
        }

        with out.open("w") as f:
            json.dump(
                payload,
                f,
                indent=2,
            )

        print()
        print(
            "saved : {}".format(out)
        )


if __name__ == "__main__":
    main()
