#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import rospy

from geometry_msgs.msg import PoseArray
from tf.transformations import (
    quaternion_inverse,
    quaternion_multiply,
)

TOPIC = "/origami/active_paper_pose_history_ros"


def q_array(q):
    return np.array(
        [q.x, q.y, q.z, q.w],
        dtype=float,
    )


def normalize(q):
    n = np.linalg.norm(q)

    if n < 1.0e-12:
        raise RuntimeError(
            "Zero quaternion."
        )

    return q / n


def relative_angle_deg(q0, qi):
    q0 = normalize(q0)
    qi = normalize(qi)

    q_rel = quaternion_multiply(
        quaternion_inverse(q0),
        qi,
    )

    q_rel = normalize(q_rel)

    # q と -q は同じ姿勢
    w = abs(float(q_rel[3]))
    w = max(-1.0, min(1.0, w))

    angle = (
        2.0
        * math.acos(w)
    )

    return math.degrees(angle)


def main():
    rospy.init_node(
        "cobotta_fold_angle_history_diagnostic"
    )

    msg = rospy.wait_for_message(
        TOPIC,
        PoseArray,
        timeout=15.0,
    )

    if not msg.poses:
        raise RuntimeError(
            "Paper pose history is empty."
        )

    q0 = q_array(
        msg.poses[0].orientation
    )

    print(
        "===== PAPER FOLD ANGLE HISTORY ====="
    )

    print(
        "pose count = {}".format(
            len(msg.poses)
        )
    )

    print()
    print(
        "index | relative angle [deg]"
    )
    print(
        "------+---------------------"
    )

    angles = []

    for i, pose in enumerate(
        msg.poses
    ):
        angle = relative_angle_deg(
            q0,
            q_array(
                pose.orientation
            ),
        )

        angles.append(angle)

        print(
            "{:5d} | {:10.3f}"
            .format(
                i,
                angle,
            )
        )

    print()
    print(
        "===== IMPORTANT INDICES ====="
    )

    for i in [
        28,
        34,
        45,
        53,
        60,
        69,
        73,
        81,
    ]:
        if i < len(angles):
            print(
                "index {:2d} = {:8.3f} deg"
                .format(
                    i,
                    angles[i],
                )
            )

    print()
    print(
        "final angle = {:.3f} deg"
        .format(
            angles[-1]
        )
    )


if __name__ == "__main__":
    main()
