#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os

import numpy as np
import pyassimp
import rospy
import rospkg

from urdf_parser_py.urdf import URDF


TOOL_LINK = "cobotta_tool_link"

TARGET_LINKS = [
    "cobotta_gripper_base",
    "cobotta_left_finger",
    "cobotta_right_finger",
]

# cobotta_tool_link -> actual_grasp_point [m]
R_GRASP = np.array(
    [0.002000, 0.000000, -0.004894],
    dtype=float,
)

# 最大OPEN
GRIPPER_OPEN = 0.015


def origin_matrix(origin):
    T = np.eye(4)

    if origin is None:
        return T

    xyz = (
        np.asarray(origin.xyz, dtype=float)
        if origin.xyz is not None
        else np.zeros(3)
    )

    rpy = (
        np.asarray(origin.rpy, dtype=float)
        if origin.rpy is not None
        else np.zeros(3)
    )

    r, p, y = rpy

    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)

    Rx = np.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr, cr],
    ])

    Ry = np.array([
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp],
    ])

    Rz = np.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1],
    ])

    T[:3, :3] = Rz @ Ry @ Rx
    T[:3, 3] = xyz

    return T


def axis_rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)

    x, y, z = axis

    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c

    R = np.array([
        [
            c + x*x*C,
            x*y*C - z*s,
            x*z*C + y*s,
        ],
        [
            y*x*C + z*s,
            c + y*y*C,
            y*z*C - x*s,
        ],
        [
            z*x*C - y*s,
            z*y*C + x*s,
            c + z*z*C,
        ],
    ])

    T = np.eye(4)
    T[:3, :3] = R

    return T


def joint_value(joint, q_values):
    if joint.mimic is not None:
        source = joint.mimic.joint

        q = q_values.get(
            source,
            0.0,
        )

        multiplier = (
            joint.mimic.multiplier
            if joint.mimic.multiplier is not None
            else 1.0
        )

        offset = (
            joint.mimic.offset
            if joint.mimic.offset is not None
            else 0.0
        )

        return multiplier * q + offset

    return q_values.get(
        joint.name,
        0.0,
    )


def joint_motion_matrix(joint, q):
    T = np.eye(4)

    if joint.type in [
        "revolute",
        "continuous",
    ]:
        return axis_rotation(
            joint.axis,
            q,
        )

    if joint.type == "prismatic":
        axis = np.asarray(
            joint.axis,
            dtype=float,
        )

        axis /= np.linalg.norm(axis)

        T[:3, 3] = axis * q

    return T


def build_transforms(robot, q_values):
    children = {}

    for joint in robot.joints:
        children.setdefault(
            joint.parent,
            []
        ).append(joint)

    root = robot.get_root()

    transforms = {
        root: np.eye(4)
    }

    queue = [root]

    while queue:
        parent = queue.pop(0)

        for joint in children.get(
            parent,
            [],
        ):
            q = joint_value(
                joint,
                q_values,
            )

            T = (
                transforms[parent]
                @ origin_matrix(joint.origin)
                @ joint_motion_matrix(joint, q)
            )

            transforms[joint.child] = T
            queue.append(joint.child)

    return transforms


def resolve_mesh_path(filename):
    if filename.startswith(
        "package://"
    ):
        rel = filename[
            len("package://"):
        ]

        package, rest = rel.split(
            "/",
            1,
        )

        package_dir = (
            rospkg.RosPack()
            .get_path(package)
        )

        return os.path.join(
            package_dir,
            rest,
        )

    if filename.startswith(
        "file://"
    ):
        return filename[
            len("file://"):
        ]

    return filename


def load_vertices(path):
    processing = 0

    try:
        processing |= (
            pyassimp.postprocess
            .aiProcess_PreTransformVertices
        )
    except Exception:
        pass

    loaded = pyassimp.load(
        path,
        processing=processing,
    )

    # pyassimpのバージョン差を吸収
    if hasattr(
        loaded,
        "__enter__",
    ):
        with loaded as scene:
            vertices = [
                np.asarray(
                    mesh.vertices,
                    dtype=float,
                )
                for mesh in scene.meshes
            ]

            return np.vstack(
                vertices
            )

    scene = loaded

    try:
        vertices = [
            np.asarray(
                mesh.vertices,
                dtype=float,
            )
            for mesh in scene.meshes
        ]

        return np.vstack(
            vertices
        )

    finally:
        try:
            pyassimp.release(scene)
        except Exception:
            pass


def transform_vertices(
    vertices,
    T,
    scale,
):
    scale = np.asarray(
        scale,
        dtype=float,
    )

    v = vertices * scale

    vh = np.hstack([
        v,
        np.ones(
            (len(v), 1),
            dtype=float,
        ),
    ])

    return (
        T @ vh.T
    ).T[:, :3]


def main():
    rospy.init_node(
        "cobotta_open_gripper_front_extent"
    )

    robot = URDF.from_xml_string(
        rospy.get_param(
            "/robot_description"
        )
    )

    # 最大OPEN。
    # mimic jointはURDFのmimic定義から自動計算する。
    q_values = {
        "cobotta_joint_gripper":
            GRIPPER_OPEN,
    }

    transforms = build_transforms(
        robot,
        q_values,
    )

    if TOOL_LINK not in transforms:
        raise RuntimeError(
            "Tool link not found: "
            + TOOL_LINK
        )

    T_root_tool = transforms[
        TOOL_LINK
    ]

    T_tool_root = np.linalg.inv(
        T_root_tool
    )

    overall_min_z = None
    overall_max_z = None

    print()
    print(
        "===== OPEN Gripper Front Extent ====="
    )
    print(
        "gripper opening : "
        "{:.3f} mm".format(
            GRIPPER_OPEN * 1000.0
        )
    )
    print(
        "reference       : actual_grasp_point = P0"
    )
    print(
        "+Z_tool         : paper edge -> P0"
    )
    print()

    for link_name in TARGET_LINKS:
        link = robot.link_map[
            link_name
        ]

        if link_name not in transforms:
            raise RuntimeError(
                "Transform missing: "
                + link_name
            )

        link_points = []

        for collision in link.collisions:
            geometry = collision.geometry

            if not hasattr(
                geometry,
                "filename",
            ):
                continue

            path = resolve_mesh_path(
                geometry.filename
            )

            scale = (
                geometry.scale
                if geometry.scale is not None
                else [1.0, 1.0, 1.0]
            )

            vertices = load_vertices(
                path
            )

            # tool <- root <- link <- collision mesh
            T_tool_mesh = (
                T_tool_root
                @ transforms[link_name]
                @ origin_matrix(
                    collision.origin
                )
            )

            points_tool = (
                transform_vertices(
                    vertices,
                    T_tool_mesh,
                    scale,
                )
            )

            # P0 = actual_grasp_point を原点にする
            points_p0 = (
                points_tool
                - R_GRASP
            )

            link_points.append(
                points_p0
            )

        if not link_points:
            print(
                "{} : no mesh collision".format(
                    link_name
                )
            )
            continue

        points = np.vstack(
            link_points
        )

        min_z = np.min(
            points[:, 2]
        )

        max_z = np.max(
            points[:, 2]
        )

        print(
            "{}".format(
                link_name
            )
        )
        print(
            "  Z relative to P0 : "
            "{:+.3f} .. {:+.3f} mm".format(
                min_z * 1000.0,
                max_z * 1000.0,
            )
        )

        if (
            overall_min_z is None
            or min_z < overall_min_z
        ):
            overall_min_z = min_z

        if (
            overall_max_z is None
            or max_z > overall_max_z
        ):
            overall_max_z = max_z

    print()
    print(
        "===== Combined OPEN Gripper ====="
    )

    print(
        "Z relative to P0 : "
        "{:+.3f} .. {:+.3f} mm".format(
            overall_min_z * 1000.0,
            overall_max_z * 1000.0,
        )
    )

    print()
    print(
        "L_front = {:.3f} mm".format(
            overall_max_z * 1000.0
        )
    )

    print()
    print(
        "PRE-GRASP retreat for each edge:"
    )
    print(
        "  d_pre = |E - P0| + L_front"
    )


if __name__ == "__main__":
    main()
