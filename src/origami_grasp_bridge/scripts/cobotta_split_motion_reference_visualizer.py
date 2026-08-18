#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import os
import yaml
import rospy

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray


DEFAULT_YAML = os.path.expanduser(
    "~/directionA_reverse_local_z_m25_multidof.yaml"
)

DEFAULT_SUMMARY_CSV = os.path.expanduser(
    "~/cobotta_split_motion_diagnostic_summary.csv"
)

FRAME = "paper_center"

TOPIC = (
    "/origami/debug/"
    "cobotta_split_motion_reference"
)

START_INDEX = 0
EXPECTED_SPLIT_COUNT = 8


def load_position(points, index):
    if index < 0 or index >= len(points):
        raise RuntimeError(
            "paper index {} exceeds trajectory range 0..{}"
            .format(
                index,
                len(points) - 1,
            )
        )

    tf = points[index]["transforms"][0]
    t = tf["translation"]

    return [
        float(t["x"]),
        float(t["y"]),
        float(t["z"]),
    ]


def make_sphere(
    marker_id,
    namespace,
    position,
    scale,
    r,
    g,
    b,
):
    marker = Marker()

    marker.header.frame_id = FRAME
    marker.header.stamp = rospy.Time(0)

    marker.ns = namespace
    marker.id = marker_id
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD

    marker.pose.position.x = position[0]
    marker.pose.position.y = position[1]
    marker.pose.position.z = position[2]
    marker.pose.orientation.w = 1.0

    marker.scale.x = scale
    marker.scale.y = scale
    marker.scale.z = scale

    marker.color.r = r
    marker.color.g = g
    marker.color.b = b
    marker.color.a = 1.0

    marker.lifetime = rospy.Duration(0)

    return marker


def make_text(
    marker_id,
    namespace,
    position,
    text,
    r,
    g,
    b,
):
    marker = Marker()

    marker.header.frame_id = FRAME
    marker.header.stamp = rospy.Time(0)

    marker.ns = namespace
    marker.id = marker_id
    marker.type = Marker.TEXT_VIEW_FACING
    marker.action = Marker.ADD

    marker.pose.position.x = position[0]
    marker.pose.position.y = position[1]
    marker.pose.position.z = position[2] + 0.012
    marker.pose.orientation.w = 1.0

    marker.scale.z = 0.006

    marker.color.r = r
    marker.color.g = g
    marker.color.b = b
    marker.color.a = 1.0

    marker.text = text
    marker.lifetime = rospy.Duration(0)

    return marker



def make_leader_line(
    marker_id,
    start_position,
    end_position,
):
    marker = Marker()

    marker.header.frame_id = FRAME
    marker.header.stamp = rospy.Time(0)

    marker.ns = "split_motion_leader_lines"
    marker.id = marker_id
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD

    marker.scale.x = 0.001

    marker.color.r = 1.0
    marker.color.g = 0.4
    marker.color.b = 0.9
    marker.color.a = 0.85

    start = Point()
    start.x = start_position[0]
    start.y = start_position[1]
    start.z = start_position[2]

    end = Point()
    end.x = end_position[0]
    end.y = end_position[1]
    end.z = end_position[2]

    marker.points = [
        start,
        end,
    ]

    marker.lifetime = rospy.Duration(0)

    return marker



def main():
    rospy.init_node(
        "cobotta_split_motion_reference_visualizer"
    )

    yaml_path = rospy.get_param(
        "~yaml",
        DEFAULT_YAML,
    )

    summary_csv = rospy.get_param(
        "~summary_csv",
        DEFAULT_SUMMARY_CSV,
    )

    with open(
        yaml_path,
        "r",
        encoding="utf-8",
    ) as f:
        doc = yaml.safe_load(f)

    points = doc["points"]

    with open(
        summary_csv,
        newline="",
        encoding="utf-8-sig",
    ) as f:
        rows = list(csv.DictReader(f))

    pass_rows = [
        row
        for row in rows
        if row.get("status", "").strip() == "PASS"
    ]

    if len(pass_rows) != EXPECTED_SPLIT_COUNT:
        raise RuntimeError(
            "expected {} PASS split rows, found {}"
            .format(
                EXPECTED_SPLIT_COUNT,
                len(pass_rows),
            )
        )

    start_position = load_position(
        points,
        START_INDEX,
    )

    pub = rospy.Publisher(
        TOPIC,
        MarkerArray,
        queue_size=1,
        latch=True,
    )

    rospy.sleep(1.0)

    msg = MarkerArray()

    # START: green
    msg.markers.append(
        make_sphere(
            0,
            "split_motion_start",
            start_position,
            0.008,
            0.1,
            1.0,
            0.2,
        )
    )

    msg.markers.append(
        make_text(
            100,
            "split_motion_text",
            start_position,
            "START",
            0.1,
            1.0,
            0.2,
        )
    )

    split_info = []

    # Split地点そのものは動かさない。
    # ラベルだけを軌道の左右へ交互に逃がして、
    # 密集するS1〜S8を読みやすくする。
    #
    # paper軌道は概ね x=-y 方向へ進むため、
    # (+x,+y) / (-x,-y) を交互に使うと
    # 軌道に対して横方向へラベルを逃がせる。
    label_offsets = [
        (+0.018, +0.018, +0.010),
        (-0.018, -0.018, +0.008),
        (+0.020, +0.020, +0.008),
        (-0.020, -0.020, +0.006),
        (+0.022, +0.022, +0.008),
        (-0.022, -0.022, +0.006),
        (+0.024, +0.024, +0.008),
        (-0.024, -0.024, +0.006),
    ]

    for split_number, row in enumerate(
        pass_rows,
        start=1,
    ):
        paper_from = int(row["paper_from"])
        paper_to = int(row["paper_to"])

        position = load_position(
            points,
            paper_from,
        )

        split_info.append(
            (
                split_number,
                row,
                position,
            )
        )

        # Split-Motion point: magenta
        msg.markers.append(
            make_sphere(
                split_number,
                "split_motion_points",
                position,
                0.005,
                1.0,
                0.2,
                0.8,
            )
        )

        offset = label_offsets[
            split_number - 1
        ]

        label_position = [
            position[0] + offset[0],
            position[1] + offset[1],
            position[2] + offset[2],
        ]

        msg.markers.append(
            make_text(
                100 + split_number,
                "split_motion_text",
                label_position,
                "S{}".format(
                    split_number,
                ),
                1.0,
                0.2,
                0.8,
            )
        )

        msg.markers.append(
            make_leader_line(
                200 + split_number,
                position,
                label_position,
            )
        )

    pub.publish(msg)

    print(
        "===== Split-Motion RViz reference ====="
    )
    print("frame       :", FRAME)
    print("yaml        :", yaml_path)
    print("summary csv :", summary_csv)
    print()

    print(
        "START : paper index {} "
        "({:+.3f}, {:+.3f}, {:+.3f}) mm"
        .format(
            START_INDEX,
            start_position[0] * 1000.0,
            start_position[1] * 1000.0,
            start_position[2] * 1000.0,
        )
    )

    print()

    for split_number, row, position in split_info:
        print(
            "S{} : path {}->{} / paper {}->{} / "
            "local ({:+.1f},{:+.1f})"
            " -> ({:+.1f},{:+.1f}) deg"
            .format(
                split_number,
                row["path_from"],
                row["path_to"],
                row["paper_from"],
                row["paper_to"],
                float(row["x_from"]),
                float(row["y_from"]),
                float(row["x_to"]),
                float(row["y_to"]),
            )
        )

        print(
            "     position "
            "({:+.3f}, {:+.3f}, {:+.3f}) mm"
            .format(
                position[0] * 1000.0,
                position[1] * 1000.0,
                position[2] * 1000.0,
            )
        )

    print()
    print("split count :", len(split_info))
    print("topic       :", TOPIC)

    rospy.spin()


if __name__ == "__main__":
    main()
