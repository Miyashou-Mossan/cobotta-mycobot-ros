#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy

from geometry_msgs.msg import Point, PolygonStamped, Pose
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import Mesh, MeshTriangle


PAPER_TOPIC = (
    "/origami/active_folding_paper_t0_ros"
)

OBJECT_NAME = (
    "cobotta_t0_full_paper_collision"
)


class T0FullPaperCollision:

    def __init__(self):
        self.published = False
        self.frame_id = "paper_center"

        self.pub = rospy.Publisher(
            "/collision_object",
            CollisionObject,
            queue_size=10,
            latch=True,
        )

        rospy.Subscriber(
            PAPER_TOPIC,
            PolygonStamped,
            self.paper_callback,
            queue_size=1,
        )

        rospy.on_shutdown(
            self.remove_object
        )

        rospy.loginfo(
            "Waiting for T0 paper: %s",
            PAPER_TOPIC,
        )

    @staticmethod
    def make_point(p):
        point = Point()

        point.x = float(p.x)
        point.y = float(p.y)
        point.z = float(p.z)

        return point

    def build_mesh(self, points):
        mesh = Mesh()

        # PolygonStampedの頂点をそのまま使用
        for p in points:
            mesh.vertices.append(
                self.make_point(p)
            )

        # 周回順に並んだ凸Polygonを前提に
        # fan triangulation
        for i in range(
            1,
            len(points) - 1,
        ):
            tri = MeshTriangle()

            tri.vertex_indices = [
                0,
                i,
                i + 1,
            ]

            mesh.triangles.append(
                tri
            )

        return mesh

    def paper_callback(self, msg):
        if self.published:
            return

        points = msg.polygon.points

        if len(points) < 3:
            rospy.logerr(
                "T0 paper needs at least "
                "3 vertices, got %d",
                len(points),
            )
            return

        self.frame_id = (
            msg.header.frame_id
            if msg.header.frame_id
            else "paper_center"
        )

        mesh = self.build_mesh(
            points
        )

        collision = CollisionObject()

        collision.header.frame_id = (
            self.frame_id
        )

        collision.header.stamp = (
            rospy.Time.now()
        )

        collision.id = (
            OBJECT_NAME
        )

        collision.meshes.append(
            mesh
        )

        # 頂点自体がpaper_center座標なので
        # CollisionObject側はidentity
        mesh_pose = Pose()
        mesh_pose.orientation.w = 1.0

        collision.mesh_poses.append(
            mesh_pose
        )

        collision.operation = (
            CollisionObject.ADD
        )

        self.pub.publish(
            collision
        )

        self.published = True

        rospy.loginfo(
            "Published T0 full-paper "
            "CollisionObject: %s",
            OBJECT_NAME,
        )

        rospy.loginfo(
            "frame=%s vertices=%d triangles=%d",
            self.frame_id,
            len(mesh.vertices),
            len(mesh.triangles),
        )

        for i, p in enumerate(points):
            rospy.loginfo(
                "P%d = "
                "(%.6f, %.6f, %.6f)",
                i,
                p.x,
                p.y,
                p.z,
            )

    def remove_object(self):
        if not self.published:
            return

        collision = CollisionObject()

        collision.header.frame_id = (
            self.frame_id
        )

        collision.header.stamp = (
            rospy.Time.now()
        )

        collision.id = (
            OBJECT_NAME
        )

        collision.operation = (
            CollisionObject.REMOVE
        )

        self.pub.publish(
            collision
        )

        rospy.sleep(0.2)


def main():
    rospy.init_node(
        "cobotta_t0_full_paper_collision"
    )

    T0FullPaperCollision()

    rospy.spin()


if __name__ == "__main__":
    main()
