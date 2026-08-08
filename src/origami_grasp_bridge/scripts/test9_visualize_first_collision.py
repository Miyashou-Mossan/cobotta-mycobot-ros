#!/usr/bin/env python3

import csv
import importlib.util
import sys

import moveit_commander
import rospy

from moveit_msgs.msg import (
    DisplayRobotState,
    ObjectColor,
)
from moveit_msgs.srv import (
    GetStateValidity,
    GetStateValidityRequest,
)
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import (
    Marker,
    MarkerArray,
)


BOUNDARY_MODULE_PATH = (
    "/home/maeda/catkin_ws/src/"
    "origami_grasp_bridge/scripts/"
    "test9_boundary_refine_scan.py"
)

FULL_SCAN_MODULE_PATH = (
    "/home/maeda/catkin_ws/src/"
    "origami_grasp_bridge/scripts/"
    "test9_full_local_y_correction_scan.py"
)

ORIGINAL_CSV = (
    "/home/maeda/test9_pf_ik_collision_scan.csv"
)

CORRECTED_CSV = (
    "/home/maeda/test9_local_y_5p0deg_forward.csv"
)

POINT_INDEX = 0
CORRECTION_DEG = 5.0


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    boundary = load_module(
        BOUNDARY_MODULE_PATH,
        "boundary_module",
    )

    full_scan = load_module(
        FULL_SCAN_MODULE_PATH,
        "full_scan_module",
    )

    moveit_commander.roscpp_initialize(sys.argv)

    rospy.init_node(
        "test9_visualize_first_collision",
        anonymous=True,
    )

    drop_mm = float(
        rospy.get_param("~drop_mm", 3.0)
    )

    with open(ORIGINAL_CSV, newline="") as f:
        original_rows = list(csv.DictReader(f))

    with open(CORRECTED_CSV, newline="") as f:
        corrected_rows = list(csv.DictReader(f))

    original_by_index = {
        int(row["index"]): row
        for row in original_rows
    }

    corrected_by_index = {
        int(row["index"]): row
        for row in corrected_rows
    }

    original_row = original_by_index[POINT_INDEX]
    corrected_row = corrected_by_index[POINT_INDEX]

    rospy.wait_for_service(
        "/compute_ik",
        timeout=20.0,
    )

    rospy.wait_for_service(
        "/check_state_validity",
        timeout=20.0,
    )

    compute_ik = rospy.ServiceProxy(
        "/compute_ik",
        boundary.GetPositionIK,
    )

    check_validity = rospy.ServiceProxy(
        "/check_state_validity",
        GetStateValidity,
    )

    robot = moveit_commander.RobotCommander()

    seed_state = boundary.seed_from_row(
        robot,
        corrected_row,
    )

    pose = full_scan.make_corrected_pose(
        boundary,
        original_row,
        CORRECTION_DEG,
    )

    pose.pose.position.z -= drop_mm / 1000.0

    ik_result = full_scan.call_ik(
        boundary,
        compute_ik,
        check_validity,
        pose,
        seed_state,
        avoid_collisions=False,
    )

    print("\n===== Test9 collision-state visualization =====")
    print("point index       :", POINT_INDEX)
    print("local_y correction:", CORRECTION_DEG, "deg")
    print("drop              :", drop_mm, "mm")
    print(
        "target Z          : {:.6f} mm".format(
            pose.pose.position.z * 1000.0
        )
    )

    if not ik_result["success"]:
        print("IK                : FAILED")
        print("error code        :", ik_result["code"])
        return

    print("IK                : SUCCESS")

    validity_request = GetStateValidityRequest()
    validity_request.robot_state = ik_result["state"]
    validity_request.group_name = boundary.GROUP_NAME

    validity_response = check_validity(
        validity_request
    )

    print("state valid       :", validity_response.valid)
    print(
        "contact count     :",
        len(validity_response.contacts),
    )

    colliding_robot_links = set()

    for index, contact in enumerate(
        validity_response.contacts
    ):
        print("\ncontact", index)
        print(
            "  pair     : {} <-> {}".format(
                contact.contact_body_1,
                contact.contact_body_2,
            )
        )
        print(
            "  frame    :",
            contact.header.frame_id or "world",
        )
        print(
            "  position : "
            "({:.9f}, {:.9f}, {:.9f})".format(
                contact.position.x,
                contact.position.y,
                contact.position.z,
            )
        )
        print(
            "  normal   : "
            "({:.9f}, {:.9f}, {:.9f})".format(
                contact.normal.x,
                contact.normal.y,
                contact.normal.z,
            )
        )
        print(
            "  depth    : {:.9f} m "
            "({:.6f} mm)".format(
                contact.depth,
                contact.depth * 1000.0,
            )
        )

        for body_name in (
            contact.contact_body_1,
            contact.contact_body_2,
        ):
            if body_name.startswith("cobotta_"):
                colliding_robot_links.add(
                    body_name
                )

    display_publisher = rospy.Publisher(
        "/display_robot_state",
        DisplayRobotState,
        queue_size=1,
        latch=True,
    )

    marker_publisher = rospy.Publisher(
        "/test9_collision_contacts",
        MarkerArray,
        queue_size=1,
        latch=True,
    )

    display_message = DisplayRobotState()
    display_message.state = ik_result["state"]

    for link_name in sorted(
        colliding_robot_links
    ):
        object_color = ObjectColor()
        object_color.id = link_name
        object_color.color = ColorRGBA(
            r=1.0,
            g=0.1,
            b=0.1,
            a=1.0,
        )
        display_message.highlight_links.append(
            object_color
        )

    marker_array = MarkerArray()

    for index, contact in enumerate(
        validity_response.contacts
    ):
        marker = Marker()
        marker.header.frame_id = (
            contact.header.frame_id or "world"
        )
        marker.header.stamp = rospy.Time.now()
        marker.ns = "test9_collision_contacts"
        marker.id = index
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = contact.position.x
        marker.pose.position.y = contact.position.y
        marker.pose.position.z = contact.position.z
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.008
        marker.scale.y = 0.008
        marker.scale.z = 0.008

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker_array.markers.append(marker)

    rospy.sleep(1.0)

    print("\nPublishing:")
    print("  /display_robot_state")
    print("  /test9_collision_contacts")
    print("RViz確認後、Ctrl+Cで終了してください。")

    rate = rospy.Rate(2.0)

    while not rospy.is_shutdown():
        display_message.state.joint_state.header.stamp = (
            rospy.Time.now()
        )

        display_publisher.publish(
            display_message
        )
        marker_publisher.publish(
            marker_array
        )

        rate.sleep()


if __name__ == "__main__":
    main()
