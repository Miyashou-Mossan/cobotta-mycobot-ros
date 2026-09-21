#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import rospy

from std_msgs.msg import String


RESULT_TOPIC = (
    "/origami/debug/"
    "cobotta_pregrasp_exhaustive_feasibility_results"
)


def main():
    rospy.init_node(
        "cobotta_pregrasp_exhaustive_angle_map"
    )

    print(
        "===== EXHAUSTIVE APPROACH ANGLE MAP ====="
    )

    msg = rospy.wait_for_message(
        RESULT_TOPIC,
        String,
        timeout=10.0,
    )

    data = json.loads(
        msg.data
    )

    candidates = data[
        "candidates"
    ]

    # direction_index単位でN+/N-をまとめる
    direction_map = {}

    for c in candidates:
        direction_index = int(
            c["direction_index"]
        )

        if direction_index not in direction_map:
            direction_map[
                direction_index
            ] = {
                "angle":
                    float(
                        c[
                            "approach_angle_deg"
                        ]
                    ),
                "edge":
                    int(
                        c[
                            "entry_edge"
                        ]
                    ),
                "N+":
                    None,
                "N-":
                    None,
            }

        sign = str(
            c["normal_sign"]
        )

        direction_map[
            direction_index
        ][sign] = c

    print()
    print(
        "P0 index = {}".format(
            data["p0_index"]
        )
    )

    print(
        "direction count = {}".format(
            len(direction_map)
        )
    )

    print()
    print(
        "angle | edge | N+ result / reason "
        "| N- result / reason"
    )

    print(
        "------------------------------------------------------------"
    )

    for direction_index in sorted(
        direction_map.keys()
    ):
        d = direction_map[
            direction_index
        ]

        plus = d["N+"]
        minus = d["N-"]

        plus_text = (
            "{} / {}".format(
                plus["result"],
                plus["reason"],
            )
            if plus is not None
            else "MISSING"
        )

        minus_text = (
            "{} / {}".format(
                minus["result"],
                minus["reason"],
            )
            if minus is not None
            else "MISSING"
        )

        print(
            "{:6.1f} | {:4d} | {:35s} | {}"
            .format(
                d["angle"],
                d["edge"],
                plus_text,
                minus_text,
            )
        )

    # ------------------------------------------------
    # 角度ごとの状態を短い記号でも表示
    #
    # F : N+ / N-ともFEASIBLE
    # P : N+のみFEASIBLE
    # M : N-のみFEASIBLE
    # I : IK failureあり
    # S : stand collisionあり
    # X : その他
    # ------------------------------------------------

    print()
    print(
        "===== COMPACT MAP ====="
    )

    print(
        "F=both feasible, "
        "P=N+ only, M=N- only, "
        "I=IK fail, S=stand collision, X=other"
    )

    symbols = []

    for direction_index in sorted(
        direction_map.keys()
    ):
        d = direction_map[
            direction_index
        ]

        plus = d["N+"]
        minus = d["N-"]

        plus_ok = (
            plus is not None
            and plus["result"]
            == "FEASIBLE_FOUND"
        )

        minus_ok = (
            minus is not None
            and minus["result"]
            == "FEASIBLE_FOUND"
        )

        if plus_ok and minus_ok:
            symbol = "F"

        elif plus_ok:
            symbol = "P"

        elif minus_ok:
            symbol = "M"

        else:
            reasons = []

            if plus is not None:
                reasons.append(
                    plus["reason"]
                )

            if minus is not None:
                reasons.append(
                    minus["reason"]
                )

            if "STAND_COLLISION" in reasons:
                symbol = "S"

            elif "IK_FAILED" in reasons:
                symbol = "I"

            else:
                symbol = "X"

        symbols.append(
            (
                d["angle"],
                symbol,
                d["edge"],
            )
        )

    # 10°ごとにまとめて見やすく出す
    for start in range(
        0,
        360,
        10,
    ):
        chunk = [
            x
            for x in symbols
            if (
                start
                <= x[0]
                < start + 10
            )
        ]

        text = "".join(
            x[1]
            for x in chunk
        )

        edges = "".join(
            str(x[2])
            for x in chunk
        )

        print(
            "{:3d}-{:3d} deg : {}   edges:{}"
            .format(
                start,
                start + 9,
                text,
                edges,
            )
        )

    # ------------------------------------------------
    # FEASIBLE連続区間
    # ------------------------------------------------

    feasible_angles = [
        angle
        for angle, symbol, _
        in symbols
        if symbol == "F"
    ]

    ranges = []

    if feasible_angles:
        start = feasible_angles[0]
        previous = feasible_angles[0]

        for angle in feasible_angles[1:]:
            if abs(
                angle - previous - 1.0
            ) < 1.0e-6:
                previous = angle
                continue

            ranges.append(
                (start, previous)
            )

            start = angle
            previous = angle

        ranges.append(
            (start, previous)
        )

    print()
    print(
        "===== BOTH-FEASIBLE RANGES ====="
    )

    if not ranges:
        print(
            "NONE"
        )
    else:
        for start, end in ranges:
            print(
                "{:.1f} -> {:.1f} deg"
                .format(
                    start,
                    end,
                )
            )


if __name__ == "__main__":
    main()
