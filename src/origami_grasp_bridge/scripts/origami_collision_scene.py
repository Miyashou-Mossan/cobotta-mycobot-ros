#!/usr/bin/env python3

import sys
import rospy
import moveit_commander
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import PlanningScene, ObjectColor

def set_object_color(scene, object_id, red, green, blue, alpha=1.0):
    planning_scene = PlanningScene()
    planning_scene.is_diff = True

    object_color = ObjectColor()
    object_color.id = object_id
    object_color.color.r = red
    object_color.color.g = green
    object_color.color.b = blue
    object_color.color.a = alpha

    planning_scene.object_colors.append(object_color)
    scene.apply_planning_scene(planning_scene)

def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("origami_collision_scene")

    scene = moveit_commander.PlanningSceneInterface(
        synchronous=True
    )

    rospy.sleep(1.0)

    ground_pose = PoseStamped()
    ground_pose.header.frame_id = "world"

    # 床上面を z=0.0 に置くため、
    # 厚さ0.02 mの箱の中心を z=-0.01 m に設定
    ground_pose.pose.position.x = 0.0
    ground_pose.pose.position.y = 0.0
    ground_pose.pose.position.z = -0.01

    ground_pose.pose.orientation.x = 0.0
    ground_pose.pose.orientation.y = 0.0
    ground_pose.pose.orientation.z = 0.0
    ground_pose.pose.orientation.w = 1.0

    scene.remove_world_object("ground")
    rospy.sleep(0.5)

    scene.add_box(
        "ground",
        ground_pose,
        size=(2.0, 2.0, 0.02),
    )

    # COBOTTA固定台
    cobotta_mount_pose = PoseStamped()
    cobotta_mount_pose.header.frame_id = "world"
    cobotta_mount_pose.pose.orientation.w = 1.0
    cobotta_mount_pose.pose.position.x = 0.010
    cobotta_mount_pose.pose.position.y = 0.000
    cobotta_mount_pose.pose.position.z = 0.011

    scene.remove_world_object("cobotta_mount")
    rospy.sleep(0.5)

    scene.add_box(
        "cobotta_mount",
        cobotta_mount_pose,
        size=(0.280, 0.260, 0.022),
    )

    # MyCobot固定台
    mycobot_mount_pose = PoseStamped()
    mycobot_mount_pose.header.frame_id = "world"
    mycobot_mount_pose.pose.orientation.w = 1.0
    mycobot_mount_pose.pose.position.x = 0.446
    mycobot_mount_pose.pose.position.y = -0.0715
    mycobot_mount_pose.pose.position.z = 0.040

    scene.remove_world_object("mycobot_mount")
    rospy.sleep(0.5)

    scene.add_box(
        "mycobot_mount",
        mycobot_mount_pose,
        size=(0.062, 0.130, 0.080),
    )

    table_pose = PoseStamped()
    table_pose.header.frame_id = "world"

    # 仮の作業机
    # 天板上面を z = 0.75 m に設定
    # 厚さ0.03 mなので中心は z = 0.735 m
    table_pose.pose.position.x = 0.260
    table_pose.pose.position.y = -0.070
    table_pose.pose.position.z = 0.1045

    table_pose.pose.orientation.x = 0.0
    table_pose.pose.orientation.y = 0.0
    table_pose.pose.orientation.z = 0.382683
    table_pose.pose.orientation.w = 0.923880

    scene.remove_world_object("table")
    scene.remove_world_object("paper_stand")
    rospy.sleep(0.5)

    scene.add_box(
    	"paper_stand",
    	table_pose,
    	size=(0.150, 0.150, 0.005),
    )

    rospy.sleep(1.0)

    # Planning Scene内の物体を見分けやすくする
    set_object_color(
    	scene,
    	"ground",
    	0.45, 0.45, 0.45, 1.0
    )

    set_object_color(
    	scene,
    	"paper_stand",
    	0.10, 0.30, 0.90, 1.0
    )

    rospy.sleep(0.5)

    known_objects = scene.get_known_object_names()

    if "ground" in known_objects:
        rospy.loginfo("Ground added to Planning Scene.")
    else:
        rospy.logerr("Failed to add ground.")

    if "paper_stand" in known_objects:
        rospy.loginfo("paper_stand added to Planning Scene.")
        rospy.loginfo("Size: 0.150 x 0.150 x 0.005 m")
        rospy.loginfo("Top surface: z = 0.107 m")
    else:
        rospy.logerr("Failed to add table.")

    rospy.spin()


if __name__ == "__main__":
    main()
