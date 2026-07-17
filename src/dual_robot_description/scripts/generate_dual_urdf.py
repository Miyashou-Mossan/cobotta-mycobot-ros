#!/usr/bin/env python3

import copy
import xml.etree.ElementTree as ET
from pathlib import Path


def indent(element, level=0):
    space = "\n" + level * "  "
    if len(element):
        if not element.text or not element.text.strip():
            element.text = space + "  "
        for child in element:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = space
    if level and (not element.tail or not element.tail.strip()):
        element.tail = space


def add_prefix(robot, prefix, keep_world=False):
    link_names = {}

    for link in robot.findall("link"):
        old_name = link.get("name")
        if keep_world and old_name == "world":
            new_name = "world"
        else:
            new_name = prefix + old_name

        link_names[old_name] = new_name
        link.set("name", new_name)

    joint_names = {}

    for joint in robot.findall("joint"):
        old_name = joint.get("name")
        new_name = prefix + old_name
        joint_names[old_name] = new_name
        joint.set("name", new_name)

        parent = joint.find("parent")
        child = joint.find("child")

        if parent is not None:
            old_parent = parent.get("link")
            parent.set("link", link_names.get(old_parent, prefix + old_parent))

        if child is not None:
            old_child = child.get("link")
            child.set("link", link_names.get(old_child, prefix + old_child))

    for mimic in robot.findall(".//mimic"):
        old_joint = mimic.get("joint")
        if old_joint in joint_names:
            mimic.set("joint", joint_names[old_joint])

    for transmission in robot.findall("transmission"):
        if transmission.get("name"):
            transmission.set("name", prefix + transmission.get("name"))

        for joint in transmission.findall("joint"):
            old_joint = joint.get("name")
            if old_joint in joint_names:
                joint.set("name", joint_names[old_joint])

        for actuator in transmission.findall("actuator"):
            if actuator.get("name"):
                actuator.set("name", prefix + actuator.get("name"))

    for gazebo in robot.findall("gazebo"):
        reference = gazebo.get("reference")
        if reference in link_names:
            gazebo.set("reference", link_names[reference])

    return link_names


home = Path.home()

cobotta_file = (
    home
    / "cobotta_src/cobotta/denso_cobotta_descriptions"
    / "cobotta_description/cobotta.urdf"
)

mycobot_file = (
    home
    / "catkin_ws/src/mycobot_280_description_ros1"
    / "urdf/mycobot_280.urdf"
)

output_file = (
    home
    / "catkin_ws/src/dual_robot_description"
    / "urdf/dual_robot.urdf"
)

cobotta_robot = ET.parse(cobotta_file).getroot()
mycobot_robot = ET.parse(mycobot_file).getroot()

add_prefix(cobotta_robot, "cobotta_", keep_world=True)
mycobot_links = add_prefix(mycobot_robot, "mycobot_")

dual_robot = ET.Element("robot", {"name": "cobotta_mycobot_dual_robot"})
ET.SubElement(dual_robot, "link", {"name": "world"})

# COBOTTAを追加する。元のworldリンクだけは共通worldと重複するため除外。
for element in cobotta_robot:
    if element.tag == "link" and element.get("name") == "world":
        continue
    dual_robot.append(copy.deepcopy(element))

# myCobotを追加する。
for element in mycobot_robot:
    dual_robot.append(copy.deepcopy(element))

# 仮の配置：COBOTTAからX方向に0.6 m離してmyCobotを設置。
fixed_joint = ET.SubElement(
    dual_robot,
    "joint",
    {
        "name": "world_to_mycobot_base",
        "type": "fixed",
    },
)

ET.SubElement(fixed_joint, "parent", {"link": "world"})
ET.SubElement(
    fixed_joint,
    "child",
    {"link": mycobot_links["base"]},
)
ET.SubElement(
    fixed_joint,
    "origin",
    {
        "xyz": "0.6 0 0",
        "rpy": "0 0 0",
    },
)

indent(dual_robot)

tree = ET.ElementTree(dual_robot)
tree.write(output_file, encoding="utf-8", xml_declaration=True)

print(f"作成しました: {output_file}")
