#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy

from moveit_msgs.msg import (
    PlanningScene,
    AllowedCollisionEntry,
    PlanningSceneComponents,
)

from moveit_msgs.srv import (
    GetPlanningScene,
    GetPlanningSceneRequest,
)


OBJECT_NAME = "paper_stand"

LINK_NAMES = [
    "cobotta_gripper_base",
    "cobotta_left_finger",
    "cobotta_right_finger",
]


def ensure_name(acm, name):
    if name in acm.entry_names:
        return

    old_size = len(acm.entry_names)

    acm.entry_names.append(name)

    for entry in acm.entry_values:
        entry.enabled.append(False)

    new_entry = AllowedCollisionEntry()
    new_entry.enabled = [False] * (old_size + 1)

    acm.entry_values.append(new_entry)


def set_allowed(acm, name1, name2, allowed):
    ensure_name(acm, name1)
    ensure_name(acm, name2)

    i = acm.entry_names.index(name1)
    j = acm.entry_names.index(name2)

    acm.entry_values[i].enabled[j] = allowed
    acm.entry_values[j].enabled[i] = allowed


def main():
    rospy.init_node(
        "cobotta_ignore_gripper_stand_collision"
    )

    allow = bool(
        rospy.get_param("~allow", True)
    )

    rospy.wait_for_service(
        "/get_planning_scene",
        timeout=20.0,
    )

    get_scene = rospy.ServiceProxy(
        "/get_planning_scene",
        GetPlanningScene,
    )

    req = GetPlanningSceneRequest()
    req.components.components = (
        PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
    )

    res = get_scene(req)
    acm = res.scene.allowed_collision_matrix

    for link_name in LINK_NAMES:
        set_allowed(
            acm,
            link_name,
            OBJECT_NAME,
            allow,
        )

    scene_pub = rospy.Publisher(
        "/planning_scene",
        PlanningScene,
        queue_size=1,
        latch=True,
    )

    rospy.sleep(1.0)

    scene = PlanningScene()
    scene.is_diff = True
    scene.allowed_collision_matrix = acm

    scene_pub.publish(scene)

    state = "ALLOWED" if allow else "FORBIDDEN"

    for link_name in LINK_NAMES:
        rospy.loginfo(
            "%s: %s <-> %s",
            state,
            link_name,
            OBJECT_NAME,
        )

    rospy.sleep(1.0)


if __name__ == "__main__":
    main()
