#!/usr/bin/env python3

import bisect
import csv
import glob
import math
import os

import rosbag
import yaml


CSV_GLOB = os.path.expanduser(
    "~/20230214OrigamiSim/TrajectoryOutput/"
    "vertex_2_grasp_20mm_20260817_160945.csv"
)

BAG_PATH = os.path.expanduser(
    "~/test9_full_fold_pose_directionA_20260817.bag"
)

TOPIC = "/origami/debug/cobotta_tool_target_pose"

OUTPUT_YAML = os.path.expanduser(
    "~/directionA_full_fold_current_pose_20260817_160945_multidof.yaml"
)

OUTPUT_CSV = os.path.expanduser(
    "~/directionA_full_fold_current_pose_20260817_160945.csv"
)


def normalize(q):
    n = math.sqrt(sum(v * v for v in q))
    if n <= 1.0e-12:
        raise RuntimeError("Quaternion norm is zero.")
    return [v / n for v in q]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def slerp(q0, q1, t):
    q0 = normalize(q0)
    q1 = normalize(q1)

    d = dot(q0, q1)

    if d < 0.0:
        q1 = [-v for v in q1]
        d = -d

    d = max(-1.0, min(1.0, d))

    if d > 0.9995:
        q = [
            (1.0 - t) * a + t * b
            for a, b in zip(q0, q1)
        ]
        return normalize(q)

    theta = math.acos(d)
    s = math.sin(theta)

    a = math.sin((1.0 - t) * theta) / s
    b = math.sin(t * theta) / s

    return [
        a * x + b * y
        for x, y in zip(q0, q1)
    ]


def quaternion_step(q0, q1):
    d = abs(dot(normalize(q0), normalize(q1)))
    d = max(-1.0, min(1.0, d))
    return 2.0 * math.acos(d)


def cumulative_position_progress(points):
    cumulative = [0.0]

    for p0, p1 in zip(points[:-1], points[1:]):
        d = math.sqrt(
            sum(
                (b - a) ** 2
                for a, b in zip(p0, p1)
            )
        )
        cumulative.append(cumulative[-1] + d)

    total = cumulative[-1]

    if total <= 1.0e-9:
        raise RuntimeError("Position trajectory has zero length.")

    return [v / total for v in cumulative], total


def cumulative_orientation_progress(quaternions):
    continuous = [normalize(quaternions[0])]

    for q in quaternions[1:]:
        q = normalize(q)

        if dot(continuous[-1], q) < 0.0:
            q = [-v for v in q]

        continuous.append(q)

    cumulative = [0.0]

    for q0, q1 in zip(continuous[:-1], continuous[1:]):
        cumulative.append(
            cumulative[-1] + quaternion_step(q0, q1)
        )

    total = cumulative[-1]

    if total <= math.radians(1.0):
        raise RuntimeError(
            "Orientation trajectory hardly changed."
        )

    progress = [v / total for v in cumulative]

    return continuous, progress, total


def interpolate_orientation(
    target_progress,
    orientation_progress,
    quaternions,
):
    if target_progress <= orientation_progress[0]:
        return quaternions[0]

    if target_progress >= orientation_progress[-1]:
        return quaternions[-1]

    i = bisect.bisect_right(
        orientation_progress,
        target_progress,
    ) - 1

    i = max(0, min(i, len(quaternions) - 2))

    p0 = orientation_progress[i]
    p1 = orientation_progress[i + 1]

    if p1 - p0 <= 1.0e-12:
        return quaternions[i]

    local_t = (
        (target_progress - p0)
        / (p1 - p0)
    )

    return slerp(
        quaternions[i],
        quaternions[i + 1],
        local_t,
    )


csv_files = sorted(
    glob.glob(CSV_GLOB),
    key=os.path.getmtime,
)

if not csv_files:
    raise RuntimeError("Trajectory CSV was not found.")

csv_path = csv_files[-1]

with open(
    csv_path,
    newline="",
    encoding="utf-8-sig",
) as f:
    rows = list(csv.DictReader(f))

if not rows:
    raise RuntimeError("Trajectory CSV is empty.")

times = []
positions_ros = []

for row in rows:
    t = float(row["time_s"])

    ux = float(row["grasp_meter_x"])
    uy = float(row["grasp_meter_y"])
    uz = float(row["grasp_meter_z"])

    # Unity -> ROS / paper_center
    rx = uz
    ry = -ux
    rz = uy

    times.append(t)
    positions_ros.append([rx, ry, rz])

bag_quaternions = []

with rosbag.Bag(BAG_PATH, "r") as bag:
    for _, msg, _ in bag.read_messages(
        topics=[TOPIC]
    ):
        q = msg.pose.orientation

        bag_quaternions.append([
            float(q.x),
            float(q.y),
            float(q.z),
            float(q.w),
        ])

if len(bag_quaternions) < 2:
    raise RuntimeError(
        "Not enough orientation messages in bag."
    )

position_progress, path_length = (
    cumulative_position_progress(positions_ros)
)

(
    bag_quaternions,
    orientation_progress,
    orientation_total,
) = cumulative_orientation_progress(
    bag_quaternions
)

output_points = []
output_rows = []

t0 = times[0]

for index, (
    t,
    position,
    progress,
) in enumerate(
    zip(
        times,
        positions_ros,
        position_progress,
    )
):
    q = interpolate_orientation(
        progress,
        orientation_progress,
        bag_quaternions,
    )

    relative_time = max(0.0, t - t0)

    secs = int(relative_time)
    nsecs = int(
        round(
            (relative_time - secs)
            * 1.0e9
        )
    )

    if nsecs >= 1000000000:
        secs += 1
        nsecs -= 1000000000

    output_points.append({
        "transforms": [{
            "translation": {
                "x": float(position[0]),
                "y": float(position[1]),
                "z": float(position[2]),
            },
            "rotation": {
                "x": float(q[0]),
                "y": float(q[1]),
                "z": float(q[2]),
                "w": float(q[3]),
            },
        }],
        "time_from_start": {
            "secs": secs,
            "nsecs": nsecs,
        },
    })

    output_rows.append({
        "index": index,
        "time_s": relative_time,
        "x": position[0],
        "y": position[1],
        "z": position[2],
        "qx": q[0],
        "qy": q[1],
        "qz": q[2],
        "qw": q[3],
        "progress": progress,
    })


with open(OUTPUT_YAML, "w") as f:
    yaml.safe_dump(
        {
            "frame_id": "paper_center",
            "points": output_points,
        },
        f,
        sort_keys=False,
    )


with open(
    OUTPUT_CSV,
    "w",
    newline="",
) as f:
    fieldnames = [
        "index",
        "time_s",
        "x",
        "y",
        "z",
        "qx",
        "qy",
        "qz",
        "qw",
        "progress",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(output_rows)


print("===== Full fold Pose generated =====")
print("input CSV             :", csv_path)
print("input position points :", len(rows))
print("input orientation msgs:", len(bag_quaternions))
print(
    "position path length  : {:.3f} mm".format(
        path_length * 1000.0
    )
)
print(
    "orientation travel    : {:.3f} deg".format(
        math.degrees(orientation_total)
    )
)
print("output points         :", len(output_points))
print("output YAML           :", OUTPUT_YAML)
print("output CSV            :", OUTPUT_CSV)
print(
    "Z range               : {:.3f} to {:.3f} mm".format(
        min(p[2] for p in positions_ros) * 1000.0,
        max(p[2] for p in positions_ros) * 1000.0,
    )
)
