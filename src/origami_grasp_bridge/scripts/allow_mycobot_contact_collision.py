#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy

from moveit_msgs.msg import (
    PlanningScene,
    AllowedCollisionEntry,
)

from moveit_msgs.srv import (
    GetPlanningScene,
    GetPlanningSceneRequest,
)

from moveit_msgs.msg import PlanningSceneComponents


LINK_NAME = "mycobot_allowed_contact_link"
OBJECT_NAME = "fourfold_face_collision"


def ensure_name(acm, name):
    if name in acm.entry_names:
        return

    old_size = len(acm.entry_names)

    acm.entry_names.append(name)

    # 既存行に新しい列を追加
    for entry in acm.entry_values:
        entry.enabled.append(False)

    # 新しい行を追加
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
        "allow_mycobot_contact_collision"
    )

    rospy.wait_for_service(
        "/get_planning_scene"
    )

    get_scene = rospy.ServiceProxy(
        "/get_planning_scene",
        GetPlanningScene
    )

    req = GetPlanningSceneRequest()

    req.components.components = (
        PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
    )

    res = get_scene(req)

    acm = res.scene.allowed_collision_matrix

    set_allowed(
        acm,
        LINK_NAME,
        OBJECT_NAME,
        True
    )

    set_allowed(
        acm,
        LINK_NAME,
        "paper_stand",
        True
    )

    set_allowed(
        acm,
        LINK_NAME,
        "mycobot_tool_link",
        True
    )

    scene_pub = rospy.Publisher(
        "/planning_scene",
        PlanningScene,
        queue_size=1,
        latch=True
    )

    rospy.sleep(1.0)

    scene = PlanningScene()
    scene.is_diff = True
    scene.allowed_collision_matrix = acm

    scene_pub.publish(scene)

    rospy.loginfo(
        "Allowed collision: %s <-> %s",
        LINK_NAME,
        OBJECT_NAME
    )

    rospy.loginfo(
        "Allowed collision: %s <-> paper_stand",
        LINK_NAME
    )

    rospy.loginfo(
        "Allowed collision: %s <-> mycobot_tool_link",
        LINK_NAME
    )

    rospy.loginfo(
        "mycobot_tool_link remains forbidden "
        "against paper and paper_stand"
    )

    rospy.sleep(1.0)


if __name__ == "__main__":
    main()
