#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import rospy
import tf
import numpy as np

from geometry_msgs.msg import Point, Pose
from shape_msgs.msg import Mesh, MeshTriangle
from moveit_msgs.msg import CollisionObject


CSV_PATH = (
    "/home/maeda/20230214OrigamiSim/Assets/"
    "folding_face_90_base_triangle.csv"
)

OBJECT_NAME = "fourfold_face_collision"


def unity_to_ros(x_u, y_u, z_u):
    return np.array([
        z_u,
        -x_u,
        y_u
    ], dtype=float)


def load_base_triangle():
    rows = []

    with open(CSV_PATH, "r") as f:
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
        points.append(
            unity_to_ros(
                float(row["x_m"]),
                float(row["y_m"]),
                float(row["z_m"])
            )
        )

    if len(points) != 3:
        raise RuntimeError(
            "Expected 3 base points, got {}".format(
                len(points)
            )
        )

    return points


def make_collision_mesh(points):
    mesh = Mesh()

    for p in points:
        q = Point()
        q.x = float(p[0])
        q.y = float(p[1])
        q.z = float(p[2])
        mesh.vertices.append(q)

    tri = MeshTriangle()
    tri.vertex_indices = [0, 1, 2]

    mesh.triangles.append(tri)

    return mesh


def main():
    rospy.init_node(
        "fourfold_half_face_collision"
    )

    listener = tf.TransformListener()

    pub = rospy.Publisher(
        "/collision_object",
        CollisionObject,
        queue_size=10,
        latch=True
    )

    rospy.sleep(1.0)

    p0, p1, p2 = load_base_triangle()

    midpoint = (p0 + p1) * 0.5

    listener.waitForTransform(
        "paper_center",
        "cobotta_base_link",
        rospy.Time(0),
        rospy.Duration(3.0)
    )

    cobotta, _ = listener.lookupTransform(
        "paper_center",
        "cobotta_base_link",
        rospy.Time(0)
    )

    cobotta = np.array(
        cobotta,
        dtype=float
    )

    d0 = np.linalg.norm(
        p0[:2] - cobotta[:2]
    )

    d1 = np.linalg.norm(
        p1[:2] - cobotta[:2]
    )

    if d0 <= d1:
        keep = p0
        keep_name = "P0"
    else:
        keep = p1
        keep_name = "P1"

    half_points = [
        keep,
        midpoint,
        p2
    ]

    mesh = make_collision_mesh(
        half_points
    )

    collision = CollisionObject()

    collision.header.frame_id = "paper_center"
    collision.header.stamp = rospy.Time.now()

    collision.id = OBJECT_NAME

    collision.meshes.append(
        mesh
    )

    pose = Pose()
    pose.orientation.w = 1.0

    collision.mesh_poses.append(
        pose
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
        "COBOTTA-side endpoint = %s",
        keep_name
    )

    for i, p in enumerate(half_points):
        rospy.loginfo(
            "P%d = (%.6f, %.6f, %.6f)",
            i,
            p[0],
            p[1],
            p[2]
        )

    rospy.spin()


if __name__ == "__main__":
    main()
