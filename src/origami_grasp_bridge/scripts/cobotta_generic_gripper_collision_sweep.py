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
    GetStateValidity,
    GetStateValidityRequest,
)

from cobotta_generic_gripper import set_gripper


DEFAULT_GROUP = "cobotta_arm"


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

    if "joint_names" not in data:
        raise RuntimeError(
            "joint_names not found in input JSON"
        )

    if segment_key not in data:
        raise RuntimeError(
            "segment '{}' not found".format(
                segment_key
            )
        )

    segment = data[segment_key]

    if not segment:
        raise RuntimeError(
            "segment '{}' is empty".format(
                segment_key
            )
        )

    index = int(point_index)

    if index < 0:
        index = len(segment) + index

    if index < 0 or index >= len(segment):
        raise RuntimeError(
            "point_index out of range: {}".format(
                point_index
            )
        )

    point = segment[index]

    if "joints_rad" not in point:
        raise RuntimeError(
            "joints_rad not found at selected point"
        )

    names = list(data["joint_names"])
    joints = [
        float(q)
        for q in point["joints_rad"]
    ]

    if len(names) != len(joints):
        raise RuntimeError(
            "joint_names/joints_rad length mismatch"
        )

    return (
        str(path),
        names,
        joints,
        index,
        len(segment),
    )


def make_openings(
    from_m,
    to_m,
    step_m,
):
    from_m = float(from_m)
    to_m = float(to_m)
    step_m = float(step_m)

    for name, value in [
        ("from_m", from_m),
        ("to_m", to_m),
        ("step_m", step_m),
    ]:
        if not math.isfinite(value):
            raise ValueError(
                "{} must be finite".format(name)
            )

    if step_m <= 0.0:
        raise ValueError(
            "step_m must be > 0"
        )

    distance = abs(to_m - from_m)

    if distance == 0.0:
        return [from_m]

    steps = int(
        math.ceil(
            distance / step_m
        )
    )

    # endpointを必ず含める。
    # 割り切れない場合は、実際の刻みをstep_m以下にする。
    return [
        from_m
        + (
            float(i)
            / float(steps)
        )
        * (to_m - from_m)
        for i in range(steps + 1)
    ]


def get_current_robot_state(
    get_scene,
):
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


def set_arm_joints(
    state,
    joint_names,
    joints_rad,
):
    result = copy.deepcopy(state)

    names = list(
        result.joint_state.name
    )

    positions = list(
        result.joint_state.position
    )

    if len(names) != len(positions):
        raise RuntimeError(
            "RobotState name/position length mismatch"
        )

    lookup = {
        name: i
        for i, name in enumerate(names)
    }

    for name, q in zip(
        joint_names,
        joints_rad,
    ):
        if name not in lookup:
            raise RuntimeError(
                "RobotState does not contain {}".format(
                    name
                )
            )

        positions[
            lookup[name]
        ] = float(q)

    result.joint_state.position = positions
    result.joint_state.header.stamp = rospy.Time(0)

    return result


def check_state(
    check_validity,
    state,
    group_name,
):
    req = GetStateValidityRequest()

    req.robot_state = copy.deepcopy(state)
    req.group_name = group_name

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


def main():
    rospy.init_node(
        "cobotta_generic_gripper_collision_sweep"
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

    from_m = float(
        rospy.get_param(
            "~from_m",
            0.0,
        )
    )

    to_m = float(
        rospy.get_param(
            "~to_m",
            0.00024,
        )
    )

    step_m = float(
        rospy.get_param(
            "~step_m",
            0.00001,
        )
    )

    group_name = rospy.get_param(
        "~group",
        DEFAULT_GROUP,
    )

    (
        resolved_path,
        arm_joint_names,
        arm_joints,
        resolved_index,
        segment_count,
    ) = load_arm_posture(
        input_json,
        segment_key,
        point_index,
    )

    openings = make_openings(
        from_m,
        to_m,
        step_m,
    )

    rospy.wait_for_service(
        "/get_planning_scene",
        timeout=20.0,
    )

    rospy.wait_for_service(
        "/check_state_validity",
        timeout=20.0,
    )

    get_scene = rospy.ServiceProxy(
        "/get_planning_scene",
        GetPlanningScene,
    )

    check_validity = rospy.ServiceProxy(
        "/check_state_validity",
        GetStateValidity,
    )

    current_state = get_current_robot_state(
        get_scene
    )

    fixed_arm_state = set_arm_joints(
        current_state,
        arm_joint_names,
        arm_joints,
    )

    print()
    print(
        "===== Generic Gripper Collision Sweep ====="
    )
    print(
        "input_json   : {}".format(
            resolved_path
        )
    )
    print(
        "segment      : {}".format(
            segment_key
        )
    )
    print(
        "point        : {} / {}".format(
            resolved_index,
            segment_count - 1,
        )
    )
    print(
        "group        : {}".format(
            group_name
        )
    )
    print(
        "from         : {:.6f} mm / side".format(
            from_m * 1000.0
        )
    )
    print(
        "to           : {:.6f} mm / side".format(
            to_m * 1000.0
        )
    )
    print(
        "samples      : {}".format(
            len(openings)
        )
    )

    if len(openings) >= 2:
        actual_step = abs(
            openings[1] - openings[0]
        )

        print(
            "actual step  : {:.6f} mm / side".format(
                actual_step * 1000.0
            )
        )

    print()
    print("--- Fixed arm joints ---")

    for name, q in zip(
        arm_joint_names,
        arm_joints,
    ):
        print(
            "{} = {:.12f} rad".format(
                name,
                q,
            )
        )

    print()
    print("--- Sweep ---")

    collision_count = 0
    first_collision = None
    all_pairs = []

    for sample_index, opening in enumerate(
        openings
    ):
        state = set_gripper(
            fixed_arm_state,
            opening,
        )

        state.joint_state.header.stamp = (
            rospy.Time(0)
        )

        valid, pairs = check_state(
            check_validity,
            state,
            group_name,
        )

        result = (
            "VALID"
            if valid
            else "COLLISION"
        )

        print(
            "[{:02d}/{:02d}] "
            "q={:.6f} mm/side | "
            "relative_open={:.6f} mm | "
            "{} | {}".format(
                sample_index,
                len(openings) - 1,
                opening * 1000.0,
                2.0 * opening * 1000.0,
                result,
                pair_text(pairs),
            )
        )

        if not valid:
            collision_count += 1

            if first_collision is None:
                first_collision = (
                    sample_index,
                    opening,
                    pairs,
                )

        for pair in pairs:
            if pair not in all_pairs:
                all_pairs.append(pair)

    print()
    print("===== Sweep Summary =====")
    print(
        "sample_count    : {}".format(
            len(openings)
        )
    )
    print(
        "valid_count     : {}".format(
            len(openings)
            - collision_count
        )
    )
    print(
        "collision_count : {}".format(
            collision_count
        )
    )

    if first_collision is None:
        print(
            "first_collision : NONE"
        )
        print(
            "RESULT          : FULL SWEEP VALID"
        )
    else:
        index, opening, pairs = (
            first_collision
        )

        print(
            "first_collision : "
            "sample={} "
            "q={:.6f} mm/side "
            "relative_open={:.6f} mm".format(
                index,
                opening * 1000.0,
                2.0 * opening * 1000.0,
            )
        )

        print(
            "collision_pairs : {}".format(
                pair_text(all_pairs)
            )
        )

        print(
            "RESULT          : COLLISION EXISTS"
        )


if __name__ == "__main__":
    main()
