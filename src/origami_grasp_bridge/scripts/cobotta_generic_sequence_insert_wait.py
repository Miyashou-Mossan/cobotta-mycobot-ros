#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--after-action-index",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--duration-sec",
        type=float,
        required=True,
    )

    args = parser.parse_args()

    if (
        not math.isfinite(args.duration_sec)
        or args.duration_sec <= 0.0
    ):
        raise ValueError(
            "duration-sec must be finite and > 0"
        )

    input_path = Path(
        args.input
    ).expanduser().resolve()

    output_path = Path(
        args.output
    ).expanduser().resolve()

    with input_path.open() as f:
        data = json.load(f)

    if "actions" not in data:
        raise RuntimeError(
            "Missing actions"
        )

    actions = data["actions"]

    target_matches = [
        i
        for i, action in enumerate(actions)
        if (
            action.get("action_index")
            == args.after_action_index
        )
    ]

    if len(target_matches) != 1:
        raise RuntimeError(
            "Expected exactly one action_index {}, found {}".format(
                args.after_action_index,
                len(target_matches),
            )
        )

    insert_position = (
        target_matches[0] + 1
    )

    new_actions = []

    for i, action in enumerate(actions):

        new_actions.append(
            copy.deepcopy(action)
        )

        if i == target_matches[0]:
            new_actions.append({
                "type": "WAIT",
                "duration_sec":
                    float(args.duration_sec),
                "reason":
                    "user_defined_fixed_wait",
            })

    # action_indexを先頭から振り直す
    for new_index, action in enumerate(
        new_actions
    ):
        action["action_index"] = new_index

    output_data = copy.deepcopy(data)
    output_data["actions"] = new_actions

    if "summary" in output_data:
        output_data["summary"][
            "action_count"
        ] = len(new_actions)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w") as f:
        json.dump(
            output_data,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "===== WAIT INSERT COMPLETE ====="
    )
    print("input :", input_path)
    print("output:", output_path)
    print(
        "after original action:",
        args.after_action_index,
    )
    print(
        "duration_sec:",
        args.duration_sec,
    )

    print()
    print("===== ACTIONS =====")

    for action in new_actions:
        print(
            "[{}] {}".format(
                action["action_index"],
                action["type"],
            )
        )

        if action["type"] == "WAIT":
            print(
                "    duration_sec:",
                action["duration_sec"],
            )


if __name__ == "__main__":
    main()
