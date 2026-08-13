#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import rospy
import numpy as np

from geometry_msgs.msg import Point, Pose
from shape_msgs.msg import Mesh, MeshTriangle
from moveit_msgs.msg import CollisionObject


CSV_PATH = "/home/maeda/20230214OrigamiSim/Assets/folding_face_90.csv"
OBJECT_NAME = "folding_face_90_collision"


def unity_to_ros(x_u, y_u, z_u):
    """
    Unity -> ROS
    paper_center basis

    x_ros =  z_unity
    y_ros = -x_unity
    z_ros =  y_unity
    """
    return np.array(
        [
            z_u,
            -x_u,
            y_u
        ],
        dtype=float
    )


def load_points(path):
    rows = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            rows.append(line)

    reader = csv.DictReader(rows)

    points = []

    for row in reader:
        p = unity_to_ros(
            float(row["x_m"]),
            float(row["y_m"]),
            float(row["z_m"])
        )

        points.append(p)

    if len(points) < 3:
        raise RuntimeError(
            "Need at least 3 vertices, got {}".format(
                len(points)
            )
        )

    return points


def build_mesh(points):
    mesh = Mesh()

    # 頂点
    for p in points:
        vertex = Point()

        vertex.x = float(p[0])
        vertex.y = float(p[1])
        vertex.z = float(p[2])

        mesh.vertices.append(vertex)

    # 現在はPaperObject.getAllPoints()の
    # 周回順に並んだ凸ポリゴンを前提として
    # fan triangulationする。
    #
    # 3頂点:
    #   0-1-2
    #
    # 4頂点:
    #   0-1-2
    #   0-2-3
    #
    # 5頂点:
    #   0-1-2
    #   0-2-3
    #   0-3-4
    for i in range(
        1,
        len(points) - 1
    ):
        tri = MeshTriangle()

        tri.vertex_indices = [
            0,
            i,
            i + 1
        ]

        mesh.triangles.append(tri)

    return mesh


def main():
    rospy.init_node(
        "folding_face_90_collision"
    )

    pub = rospy.Publisher(
        "/collision_object",
        CollisionObject,
        queue_size=10,
        latch=True
    )

    rospy.sleep(1.0)

    points = load_points(
        CSV_PATH
    )

    rospy.loginfo(
        "Loaded %d Unity folding-face vertices",
        len(points)
    )

    for i, p in enumerate(points):
        rospy.loginfo(
            "P%d = (%.6f, %.6f, %.6f)",
            i,
            p[0],
            p[1],
            p[2]
        )

    mesh = build_mesh(
        points
    )

    rospy.loginfo(
        "Generated %d mesh triangles",
        len(mesh.triangles)
    )

    collision = CollisionObject()

    collision.header.frame_id = "paper_center"
    collision.header.stamp = rospy.Time.now()

    collision.id = OBJECT_NAME

    collision.meshes.append(
        mesh
    )

    # 頂点がすでにpaper_center座標なので
    # CollisionObject側のposeはidentity
    mesh_pose = Pose()
    mesh_pose.orientation.w = 1.0

    collision.mesh_poses.append(
        mesh_pose
    )

    collision.operation = CollisionObject.ADD

    pub.publish(
        collision
    )

    rospy.loginfo(
        "Published CollisionObject: %s",
        OBJECT_NAME
    )

    rospy.loginfo(
        "topic: /collision_object"
    )

    rospy.spin()


if __name__ == "__main__":
    main()
