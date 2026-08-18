#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import os
import sys

import moveit_commander
import rospy

from geometry_msgs.msg import Point
from moveit_msgs.msg import (
    DisplayRobotState,
    ObjectColor,
)
from moveit_msgs.srv import (
    GetStateValidity,
    GetStateValidityRequest,
)
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


DEFAULT_CSV = (
    "/home/maeda/"
    "directionA_reverse_local_z_m25_collision_scan.csv"
)

GROUP_NAME = "cobotta_arm"

STATE_TOPIC = (
    "/origami/debug/cobotta_collision_state"
)

CONTACT_TOPIC = (
    "/origami/debug/cobotta_collision_contacts"
)


def load_row(csv_path, point_index):
    with open(
        csv_path,
        newline="",
        encoding="utf-8-sig",
    ) as f:
        rows = list(csv.DictReader(f))

    if point_index < 0 or point_index >= len(rows):
        raise IndexError(
            "point_index={} is outside 0..{}".format(
                point_index,
                len(rows) - 1,
            )
        )

    return rows[point_index]


def make_color(r, g, b, a=1.0):
    c = ColorRGBA()
    c.r = r
    c.g = g
    c.b = b
    c.a = a
    return c


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node(
        "cobotta_collision_contact_visualizer"
    )

    csv_path = os.path.expanduser(
        rospy.get_param(
            "~csv",
            DEFAULT_CSV,
        )
    )

    point_index = int(
        rospy.get_param(
            "~point_index",
            309,
        )
    )

    marker_size = float(
        rospy.get_param(
            "~marker_size",
            0.010,
        )
    )

    group = moveit_commander.MoveGroupCommander(
        GROUP_NAME,
        wait_for_servers=20.0,
    )

    active_joints = list(
        group.get_active_joints()
    )

    planning_frame = group.get_planning_frame()

    row = load_row(
        csv_path,
        point_index,
    )

    state = group.get_current_state()

    names = list(
        state.joint_state.name
    )
    positions = list(
        state.joint_state.position
    )

    name_to_index = {
        name: i
        for i, name in enumerate(names)
    }

    print()
    print("===== Collision state =====")
    print("CSV        :", csv_path)
    print("index      :", point_index)
    print("result     :", row.get("result", ""))
    print(
        "xyz [mm]   : ({:+.3f}, {:+.3f}, {:+.3f})".format(
            float(row["x"]) * 1000.0,
            float(row["y"]) * 1000.0,
            float(row["z"]) * 1000.0,
        )
    )

    for joint_name in active_joints:
        value_text = row.get(
            joint_name,
            "",
        )

        if value_text == "":
            raise RuntimeError(
                "CSV has no joint value for {}".format(
                    joint_name
                )
            )

        if joint_name not in name_to_index:
            raise RuntimeError(
                "RobotState has no joint {}".format(
                    joint_name
                )
            )

        value = float(value_text)

        positions[
            name_to_index[joint_name]
        ] = value

        print(
            "{:16s}: {:+.9f} rad".format(
                joint_name,
                value,
            )
        )

    state.joint_state.position = positions
    state.joint_state.header.stamp = rospy.Time(0)
    state.is_diff = False

    rospy.wait_for_service(
        "/check_state_validity",
        timeout=20.0,
    )

    check_validity = rospy.ServiceProxy(
        "/check_state_validity",
        GetStateValidity,
    )

    req = GetStateValidityRequest()
    req.robot_state = state
    req.group_name = GROUP_NAME

    res = check_validity(req)

    print()
    print("state valid :", res.valid)
    print("contacts    :", len(res.contacts))

    state_msg = DisplayRobotState()
    state_msg.state = state

    highlighted = set()

    for contact in res.contacts:
        for body in [
            contact.contact_body_1,
            contact.contact_body_2,
        ]:
            if body.startswith("cobotta_"):
                highlighted.add(body)

    for link_name in sorted(highlighted):
        item = ObjectColor()
        item.id = link_name
        item.color = make_color(
            1.0,
            0.0,
            0.0,
            1.0,
        )
        state_msg.highlight_links.append(item)

    markers = MarkerArray()

    if not res.contacts:
        print(
            "No contacts returned by "
            "/check_state_validity."
        )

    for i, contact in enumerate(res.contacts):
        frame_id = (
            contact.header.frame_id
            if contact.header.frame_id
            else planning_frame
        )

        sphere = Marker()
        sphere.header.frame_id = frame_id
        sphere.header.stamp = rospy.Time(0)
        sphere.ns = "cobotta_collision_contact"
        sphere.id = i
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD

        sphere.pose.position = contact.position
        sphere.pose.orientation.w = 1.0

        sphere.scale.x = marker_size
        sphere.scale.y = marker_size
        sphere.scale.z = marker_size

        sphere.color = make_color(
            1.0,
            0.0,
            0.0,
            1.0,
        )

        sphere.lifetime = rospy.Duration(0)
        markers.markers.append(sphere)

        text = Marker()
        text.header.frame_id = frame_id
        text.header.stamp = rospy.Time(0)
        text.ns = "cobotta_collision_text"
        text.id = 1000 + i
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD

        text.pose.position.x = (
            contact.position.x
        )
        text.pose.position.y = (
            contact.position.y
        )
        text.pose.position.z = (
            contact.position.z + 0.015
        )
        text.pose.orientation.w = 1.0

        text.scale.z = 0.010

        text.color = make_color(
            1.0,
            1.0,
            1.0,
            1.0,
        )

        text.text = (
            "{} <-> {}\n"
            "depth={:.3f} mm"
        ).format(
            contact.contact_body_1,
            contact.contact_body_2,
            contact.depth * 1000.0,
        )

        text.lifetime = rospy.Duration(0)
        markers.markers.append(text)

        print(
            "[{}] {} <-> {}".format(
                i,
                contact.contact_body_1,
                contact.contact_body_2,
            )
        )
        print(
            "    frame : {}".format(
                frame_id
            )
        )
        print(
            "    pos   : "
            "({:+.6f}, {:+.6f}, {:+.6f}) m".format(
                contact.position.x,
                contact.position.y,
                contact.position.z,
            )
        )
        print(
            "    depth : {:.3f} mm".format(
                contact.depth * 1000.0
            )
        )

    state_pub = rospy.Publisher(
        STATE_TOPIC,
        DisplayRobotState,
        queue_size=1,
        latch=True,
    )

    contact_pub = rospy.Publisher(
        CONTACT_TOPIC,
        MarkerArray,
        queue_size=1,
        latch=True,
    )

    rospy.sleep(1.0)

    state_pub.publish(state_msg)
    contact_pub.publish(markers)

    print()
    print("===== RViz topics =====")
    print("robot state :", STATE_TOPIC)
    print("contacts    :", CONTACT_TOPIC)
    print()
    print(
        "Keep this node running while "
        "checking RViz."
    )
    print(
        "Press Ctrl+C when finished."
    )

    rospy.spin()


if __name__ == "__main__":
    main()
