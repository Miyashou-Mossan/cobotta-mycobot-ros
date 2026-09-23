#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import math


GRIPPER_JOINT = "cobotta_joint_gripper"
GRIPPER_MIMIC_JOINT = "cobotta_joint_gripper_mimic"


def set_gripper(
    state,
    opening_m,
):
    """
    RobotStateの腕姿勢を変えず、
    COBOTTA gripper openingだけを変更する。
    """

    opening_m = float(opening_m)

    if not math.isfinite(opening_m):
        raise ValueError(
            "opening_m must be finite"
        )

    result = copy.deepcopy(state)

    names = list(
        result.joint_state.name
    )

    positions = list(
        result.joint_state.position
    )

    if len(names) != len(positions):
        raise RuntimeError(
            "RobotState joint name/position length mismatch"
        )

    lookup = {
        name: i
        for i, name in enumerate(names)
    }

    if GRIPPER_JOINT not in lookup:
        raise RuntimeError(
            "RobotState does not contain {}".format(
                GRIPPER_JOINT
            )
        )

    positions[
        lookup[GRIPPER_JOINT]
    ] = opening_m

    # mimic jointがRobotStateに存在する場合だけ更新
    if GRIPPER_MIMIC_JOINT in lookup:
        positions[
            lookup[GRIPPER_MIMIC_JOINT]
        ] = -opening_m

    result.joint_state.position = positions

    return result


def interpolate_openings(
    from_m,
    to_m,
    steps,
):
    """
    from_m → to_mを線形補間する。
    開閉方向には依存しないため、
    CLOSE / OPENの両方に使用できる。
    """

    from_m = float(from_m)
    to_m = float(to_m)
    steps = int(steps)

    if not math.isfinite(from_m):
        raise ValueError(
            "from_m must be finite"
        )

    if not math.isfinite(to_m):
        raise ValueError(
            "to_m must be finite"
        )

    if steps < 1:
        raise ValueError(
            "steps must be >= 1"
        )

    return [
        from_m
        + (
            float(i)
            / float(steps)
        )
        * (to_m - from_m)
        for i in range(steps + 1)
    ]


def build_gripper_frames(
    state,
    from_m,
    to_m,
    steps,
):
    """
    同じ腕姿勢のまま、
    gripperだけをfrom_m → to_mへ変化させる。
    """

    openings = interpolate_openings(
        from_m,
        to_m,
        steps,
    )

    return [
        set_gripper(
            state,
            opening,
        )
        for opening in openings
    ]
