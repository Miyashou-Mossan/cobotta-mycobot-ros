#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import json
import math
from pathlib import Path

import numpy as np
import rospy

from tf.transformations import quaternion_matrix

from moveit_msgs.msg import (
    MoveItErrorCodes,
    PlanningSceneComponents,
)
from moveit_msgs.srv import (
    GetPlanningScene,
    GetPlanningSceneRequest,
    GetPositionFK,
    GetPositionIK,
    GetStateValidity,
)

from cobotta_generic_gripper import set_gripper

from cobotta_generic_release_clearance_search import (
    set_named_joints,
    joint_values,
    check_state,
    pair_text,
    get_fk_pose,
    shifted_pose,
    solve_ik,
    gripper_sweep,
)


def load_json(path_text):
    path = Path(path_text).expanduser()

    if not path.is_absolute():
        path = Path.cwd() / path

    if not path.exists():
        raise RuntimeError(
            "JSON not found: {}".format(path)
        )

    with path.open() as f:
        data = json.load(f)

    return str(path), data


def get_scene(get_scene_srv):
    req = GetPlanningSceneRequest()

    req.components.components = (
        PlanningSceneComponents.ROBOT_STATE
        | PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
    )

    return get_scene_srv(req).scene


def mesh_aabb_center(obj):
    points = []

    for mesh, pose in zip(
        obj.meshes,
        obj.mesh_poses,
    ):
        T = quaternion_matrix([
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ])

        T[0, 3] = pose.position.x
        T[1, 3] = pose.position.y
        T[2, 3] = pose.position.z

        for v in mesh.vertices:
            p = np.array([
                v.x,
                v.y,
                v.z,
                1.0,
            ])

            points.append(
                T.dot(p)[:3]
            )

    if points:
        points = np.asarray(points)

        vmin = points.min(axis=0)
        vmax = points.max(axis=0)

        return 0.5 * (
            vmin + vmax
        )

    # primitiveの場合は、
    # 探索順を決めるための代表点としてpose中心を使う。
    if obj.primitive_poses:
        xyz = np.asarray([
            [
                p.position.x,
                p.position.y,
                p.position.z,
            ]
            for p in obj.primitive_poses
        ])

        return xyz.mean(axis=0)

    raise RuntimeError(
        "Object has no usable geometry: {}".format(
            obj.id
        )
    )


def find_object(scene, object_id):
    for obj in scene.world.collision_objects:
        if obj.id == object_id:
            return obj

    raise RuntimeError(
        "CollisionObject not found: {}".format(
            object_id
        )
    )


def rotate_xy(vector, angle_deg):
    a = math.radians(
        angle_deg
    )

    c = math.cos(a)
    s = math.sin(a)

    x = vector[0]
    y = vector[1]

    result = np.array([
        c * x - s * y,
        s * x + c * y,
        0.0,
    ])

    n = np.linalg.norm(result)

    if n < 1.0e-12:
        raise RuntimeError(
            "Could not define horizontal direction"
        )

    return result / n


def coarse_offsets(step_deg):
    offsets = [0.0]

    k = 1

    while k * step_deg < 180.0 - 1.0e-9:
        value = k * step_deg

        offsets.append(
            value
        )
        offsets.append(
            -value
        )

        k += 1

    offsets.append(180.0)

    return offsets


def fine_offsets(
    coarse_step_deg,
    fine_offset_deg,
):
    offsets = []

    value = fine_offset_deg

    while value < 180.0 - 1.0e-9:
        offsets.append(
            value
        )
        offsets.append(
            -value
        )

        value += coarse_step_deg

    return offsets


def checkpoint_distances(
    step_m,
    max_m,
):
    if step_m <= 0.0:
        raise ValueError(
            "distance_step_m must be > 0"
        )

    if max_m <= 0.0:
        raise ValueError(
            "max_distance_m must be > 0"
        )

    values = []

    d = step_m

    while d < max_m - 1.0e-12:
        values.append(d)
        d += step_m

    if (
        not values
        or abs(values[-1] - max_m)
        > 1.0e-12
    ):
        values.append(max_m)

    return values


def interpolate_distances(
    from_m,
    to_m,
    check_step_m,
):
    if to_m <= from_m:
        return []

    length = to_m - from_m

    steps = int(
        math.ceil(
            length / check_step_m
        )
    )

    return [
        from_m
        + (
            float(i)
            / float(steps)
        ) * length
        for i in range(1, steps + 1)
    ]


def state_joint_step(
    a,
    b,
    joint_names,
):
    qa = joint_values(
        a,
        joint_names,
    )

    qb = joint_values(
        b,
        joint_names,
    )

    return max(
        abs(x - y)
        for x, y in zip(
            qa,
            qb,
        )
    )


def main():
    rospy.init_node(
        "cobotta_generic_post_release_retreat_search"
    )

    clearance_json = rospy.get_param(
        "~clearance_json"
    )

    sequence_json = rospy.get_param(
        "~sequence_json"
    )

    output_json = rospy.get_param(
        "~output_json"
    )

    obstacle_id = rospy.get_param(
        "~reference_object",
        "paper_stand",
    )

    group = rospy.get_param(
        "~group",
        "cobotta_arm",
    )

    full_open_m = float(
        rospy.get_param(
            "~full_open_m",
            0.015,
        )
    )

    coarse_angle_step_deg = float(
        rospy.get_param(
            "~coarse_angle_step_deg",
            10.0,
        )
    )

    fine_angle_offset_deg = float(
        rospy.get_param(
            "~fine_angle_offset_deg",
            5.0,
        )
    )

    distance_step_m = float(
        rospy.get_param(
            "~distance_step_m",
            0.003,
        )
    )

    max_distance_m = float(
        rospy.get_param(
            "~max_distance_m",
            0.020,
        )
    )

    path_check_step_m = float(
        rospy.get_param(
            "~path_check_step_m",
            0.001,
        )
    )

    gripper_sweep_step_m = float(
        rospy.get_param(
            "~gripper_sweep_step_m",
            0.0001,
        )
    )

    ik_timeout = float(
        rospy.get_param(
            "~ik_timeout",
            0.10,
        )
    )

    if coarse_angle_step_deg <= 0.0:
        raise ValueError(
            "coarse_angle_step_deg must be > 0"
        )

    if fine_angle_offset_deg <= 0.0:
        raise ValueError(
            "fine_angle_offset_deg must be > 0"
        )

    if path_check_step_m <= 0.0:
        raise ValueError(
            "path_check_step_m must be > 0"
        )

    (
        resolved_clearance,
        clearance,
    ) = load_json(
        clearance_json
    )

    (
        resolved_sequence,
        sequence,
    ) = load_json(
        sequence_json
    )

    arm_joint_names = list(
        clearance["arm_joint_names"]
    )

    if (
        list(sequence["joint_names"])
        != arm_joint_names
    ):
        raise RuntimeError(
            "sequence/clearance joint_names mismatch"
        )

    if not sequence.get("actions"):
        raise RuntimeError(
            "sequence actions are empty"
        )

    last_action = sequence["actions"][-1]

    if last_action.get("type") != "GRIPPER_EVENT":
        raise RuntimeError(
            "sequence last action must be GRIPPER_EVENT"
        )

    release_ready_joints = [
        float(q)
        for q in last_action[
            "arm_joints_rad"
        ]
    ]

    release_opening_m = float(
        last_action["to_m"]
    )

    clearance_release_opening_m = float(
        clearance["release_opening_m"]
    )

    if abs(
        release_opening_m
        - clearance_release_opening_m
    ) > 1.0e-12:
        raise RuntimeError(
            "sequence/clearance release opening mismatch"
        )

    frame = "world"

    tip = clearance.get(
        "tip_link",
        "cobotta_tool_link",
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

    get_scene_srv = rospy.ServiceProxy(
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

    scene = get_scene(
        get_scene_srv
    )

    reference_object = find_object(
        scene,
        obstacle_id,
    )

    if reference_object.header.frame_id != frame:
        raise RuntimeError(
            "{} frame is '{}', expected '{}'".format(
                obstacle_id,
                reference_object.header.frame_id,
                frame,
            )
        )

    obstacle_center = mesh_aabb_center(
        reference_object
    )

    start_state = set_named_joints(
        scene.robot_state,
        arm_joint_names,
        release_ready_joints,
    )

    start_state = set_gripper(
        start_state,
        release_opening_m,
    )

    start_state.joint_state.header.stamp = (
        rospy.Time(0)
    )

    valid, pairs = check_state(
        check_validity,
        start_state,
        group,
    )

    if not valid:
        raise RuntimeError(
            "Release-ready start state is in collision: "
            + pair_text(pairs)
        )

    start_pose = get_fk_pose(
        compute_fk,
        start_state,
        frame,
        tip,
    )

    p = start_pose.pose.position

    tool_position = np.array([
        p.x,
        p.y,
        p.z,
    ])

    base_direction = (
        tool_position
        - obstacle_center
    )

    # world XY水平面
    base_direction[2] = 0.0

    norm = np.linalg.norm(
        base_direction
    )

    if norm < 1.0e-12:
        raise RuntimeError(
            "Could not define outward direction"
        )

    base_direction /= norm

    base_angle_deg = math.degrees(
        math.atan2(
            base_direction[1],
            base_direction[0],
        )
    )

    checkpoints = checkpoint_distances(
        distance_step_m,
        max_distance_m,
    )

    coarse = coarse_offsets(
        coarse_angle_step_deg
    )

    fine = fine_offsets(
        coarse_angle_step_deg,
        fine_angle_offset_deg,
    )

    passes = [
        ("COARSE_10_DEG", coarse),
        ("FINE_5_DEG", fine),
    ]

    print()
    print(
        "===== Generic Post-Release "
        "Horizontal Retreat Search ====="
    )
    print(
        "clearance input     : {}".format(
            resolved_clearance
        )
    )
    print(
        "sequence input      : {}".format(
            resolved_sequence
        )
    )
    print(
        "reference object    : {}".format(
            obstacle_id
        )
    )
    print(
        "object center [m]   : "
        "({:.6f}, {:.6f}, {:.6f})".format(
            *obstacle_center
        )
    )
    print(
        "start tool [m]      : "
        "({:.6f}, {:.6f}, {:.6f})".format(
            *tool_position
        )
    )
    print(
        "base outward angle  : {:.3f} deg".format(
            base_angle_deg
        )
    )
    print(
        "release opening     : {:.3f} mm/side".format(
            release_opening_m * 1000.0
        )
    )
    print(
        "FULL OPEN           : {:.3f} mm/side".format(
            full_open_m * 1000.0
        )
    )
    print(
        "distance checkpoints: {}".format(
            ", ".join(
                "{:.0f}".format(
                    d * 1000.0
                )
                for d in checkpoints
            )
        )
        + " mm"
    )
    print(
        "path check step     : {:.3f} mm".format(
            path_check_step_m * 1000.0
        )
    )

    selected = None

    attempted_directions = 0

    for pass_name, offsets in passes:
        print()
        print(
            "----- {} -----".format(
                pass_name
            )
        )

        for offset_deg in offsets:
            attempted_directions += 1

            direction = rotate_xy(
                base_direction,
                offset_deg,
            )

            world_angle_deg = math.degrees(
                math.atan2(
                    direction[1],
                    direction[0],
                )
            )

            print()
            print(
                "[DIR {:02d}] "
                "offset={:+.1f} deg "
                "world={:.3f} deg "
                "dir=({:.4f}, {:.4f})".format(
                    attempted_directions,
                    offset_deg,
                    world_angle_deg,
                    direction[0],
                    direction[1],
                )
            )

            current_state = copy.deepcopy(
                start_state
            )

            current_distance = 0.0

            path_records = [{
                "index": 0,
                "distance_m": 0.0,
                "joints_rad": [
                    float(q)
                    for q in joint_values(
                        start_state,
                        arm_joint_names,
                    )
                ],
                "gripper_m":
                    release_opening_m,
            }]

            direction_failed = False

            max_adjacent_joint_delta = 0.0

            for checkpoint in checkpoints:
                intermediate_distances = (
                    interpolate_distances(
                        current_distance,
                        checkpoint,
                        path_check_step_m,
                    )
                )

                for distance_m in (
                    intermediate_distances
                ):
                    target_pose = shifted_pose(
                        start_pose,
                        direction,
                        distance_m,
                    )

                    next_state, ik_error = solve_ik(
                        compute_ik,
                        target_pose,
                        current_state,
                        group,
                        tip,
                        ik_timeout,
                    )

                    if next_state is None:
                        print(
                            "  distance={:.1f} mm "
                            "-> IK_FAIL code={}".format(
                                distance_m
                                * 1000.0,
                                ik_error,
                            )
                        )

                        direction_failed = True
                        break

                    next_state = set_gripper(
                        next_state,
                        release_opening_m,
                    )

                    next_state.joint_state.header.stamp = (
                        rospy.Time(0)
                    )

                    valid, pairs = check_state(
                        check_validity,
                        next_state,
                        group,
                    )

                    if not valid:
                        print(
                            "  distance={:.1f} mm "
                            "-> PATH_COLLISION | {}".format(
                                distance_m
                                * 1000.0,
                                pair_text(pairs),
                            )
                        )

                        direction_failed = True
                        break

                    joint_step = state_joint_step(
                        current_state,
                        next_state,
                        arm_joint_names,
                    )

                    max_adjacent_joint_delta = max(
                        max_adjacent_joint_delta,
                        joint_step,
                    )

                    path_records.append({
                        "index":
                            len(path_records),
                        "distance_m":
                            float(distance_m),
                        "joints_rad": [
                            float(q)
                            for q in joint_values(
                                next_state,
                                arm_joint_names,
                            )
                        ],
                        "gripper_m":
                            release_opening_m,
                    })

                    current_state = copy.deepcopy(
                        next_state
                    )

                if direction_failed:
                    break

                current_distance = checkpoint

                # ----------------------------------
                # まずFULL OPEN終点だけ確認
                # ----------------------------------
                full_state = set_gripper(
                    current_state,
                    full_open_m,
                )

                full_valid, full_pairs = check_state(
                    check_validity,
                    full_state,
                    group,
                )

                if not full_valid:
                    print(
                        "  checkpoint={:.1f} mm "
                        "-> FULL_ENDPOINT_COLLISION | {}".format(
                            checkpoint * 1000.0,
                            pair_text(full_pairs),
                        )
                    )
                    continue

                # ----------------------------------
                # 終点VALIDならOPEN途中を確認
                # ----------------------------------
                (
                    sweep_valid,
                    first_bad_opening,
                    sweep_pairs,
                    sweep_samples,
                ) = gripper_sweep(
                    check_validity,
                    current_state,
                    group,
                    release_opening_m,
                    full_open_m,
                    gripper_sweep_step_m,
                )

                if not sweep_valid:
                    print(
                        "  checkpoint={:.1f} mm "
                        "-> OPEN_SWEEP_COLLISION "
                        "q={:.3f} mm/side | {}".format(
                            checkpoint * 1000.0,
                            first_bad_opening
                            * 1000.0,
                            pair_text(
                                sweep_pairs
                            ),
                        )
                    )
                    continue

                print(
                    "  checkpoint={:.1f} mm "
                    "-> FULL OPEN SWEEP VALID".format(
                        checkpoint * 1000.0
                    )
                )

                selected = {
                    "search_pass":
                        pass_name,
                    "angle_offset_deg":
                        float(offset_deg),
                    "world_angle_deg":
                        float(world_angle_deg),
                    "direction": [
                        float(direction[0]),
                        float(direction[1]),
                        0.0,
                    ],
                    "retreat_distance_m":
                        float(checkpoint),
                    "path":
                        path_records,
                    "max_adjacent_joint_delta_rad":
                        float(
                            max_adjacent_joint_delta
                        ),
                    "full_open_sweep_samples":
                        int(sweep_samples),
                    "final_state":
                        copy.deepcopy(
                            current_state
                        ),
                }

                # Feasible-first:
                # 最初の成立解で全探索終了
                break

            if selected is not None:
                break

        if selected is not None:
            break

    print()
    print(
        "===== POST-RELEASE SEARCH RESULT ====="
    )

    if selected is None:
        print(
            "RESULT : NO FEASIBLE RETREAT FOUND"
        )
        print(
            "attempted directions : {}".format(
                attempted_directions
            )
        )
        return

    final_joints = joint_values(
        selected["final_state"],
        arm_joint_names,
    )

    print(
        "RESULT                : FOUND"
    )
    print(
        "search pass           : {}".format(
            selected["search_pass"]
        )
    )
    print(
        "attempted directions  : {}".format(
            attempted_directions
        )
    )
    print(
        "angle offset          : {:+.1f} deg".format(
            selected["angle_offset_deg"]
        )
    )
    print(
        "world XY angle        : {:.3f} deg".format(
            selected["world_angle_deg"]
        )
    )
    print(
        "retreat distance      : {:.3f} mm".format(
            selected[
                "retreat_distance_m"
            ] * 1000.0
        )
    )
    print(
        "path point count      : {}".format(
            len(selected["path"])
        )
    )
    print(
        "max joint step        : {:.6f} deg".format(
            math.degrees(
                selected[
                    "max_adjacent_joint_delta_rad"
                ]
            )
        )
    )
    print(
        "FULL OPEN             : {:.3f} mm/side".format(
            full_open_m * 1000.0
        )
    )

    payload = {
        "schema_version": 1,
        "type":
            "generic_post_release_horizontal_retreat",
        "clearance_source_json":
            resolved_clearance,
        "sequence_source_json":
            resolved_sequence,
        "reference_object_id":
            obstacle_id,
        "reference_object_aabb_center_m": [
            float(v)
            for v in obstacle_center
        ],
        "frame_id":
            frame,
        "tip_link":
            tip,
        "arm_joint_names":
            arm_joint_names,
        "release_opening_m":
            release_opening_m,
        "full_open_m":
            full_open_m,
        "search_settings": {
            "coarse_angle_step_deg":
                coarse_angle_step_deg,
            "fine_angle_offset_deg":
                fine_angle_offset_deg,
            "distance_step_m":
                distance_step_m,
            "max_distance_m":
                max_distance_m,
            "path_check_step_m":
                path_check_step_m,
            "gripper_sweep_step_m":
                gripper_sweep_step_m,
            "strategy":
                "FEASIBLE_FIRST",
        },
        "base_outward_direction": [
            float(v)
            for v in base_direction
        ],
        "base_world_angle_deg":
            float(base_angle_deg),
        "selected": {
            "search_pass":
                selected["search_pass"],
            "attempted_direction_count":
                attempted_directions,
            "angle_offset_deg":
                selected[
                    "angle_offset_deg"
                ],
            "world_angle_deg":
                selected[
                    "world_angle_deg"
                ],
            "direction":
                selected["direction"],
            "retreat_distance_m":
                selected[
                    "retreat_distance_m"
                ],
            "path_point_count":
                len(selected["path"]),
            "max_adjacent_joint_delta_rad":
                selected[
                    "max_adjacent_joint_delta_rad"
                ],
            "final_joints_rad": [
                float(q)
                for q in final_joints
            ],
        },
        "retreat_path":
            selected["path"],
        "full_open_event": {
            "from_m":
                release_opening_m,
            "to_m":
                full_open_m,
            "collision_check":
                "FULL_SWEEP_VALID",
            "sample_count":
                selected[
                    "full_open_sweep_samples"
                ],
        },
    }

    out = Path(
        output_json
    ).expanduser()

    if not out.is_absolute():
        out = (
            Path.cwd()
            / out
        )

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
