#!/usr/bin/env python3

import math
import yaml


INPUT_FILE = "/home/maeda/mycobot_foldline_trajectory.yaml"

EXPECTED_JOINT_COUNT = 6


def duration_to_sec(time_dict):
    secs = float(time_dict.get("secs", 0))
    nsecs = float(time_dict.get("nsecs", 0))

    return secs + nsecs * 1e-9


def main():

    print("")
    print("===== Saved MyCobot trajectory validation =====")
    print("file :", INPUT_FILE)

    # ========================================
    # Load YAML
    # ========================================

    try:
        with open(INPUT_FILE, "r") as f:
            data = yaml.safe_load(f)

    except Exception as e:
        print("")
        print("YAML load : FAIL")
        print(str(e))
        return

    print("YAML load : SUCCESS")

    # ========================================
    # Basic structure
    # ========================================

    if "joint_trajectory" not in data:
        print("joint_trajectory : MISSING")
        return

    trajectory = data["joint_trajectory"]

    joint_names = trajectory.get(
        "joint_names",
        []
    )

    points = trajectory.get(
        "points",
        []
    )

    print("")
    print("===== Basic information =====")

    print(
        "joint count  : %d"
        % len(joint_names)
    )

    print(
        "point count  : %d"
        % len(points)
    )

    print(
        "joint names  : %s"
        % ", ".join(joint_names)
    )

    # ========================================
    # Joint count
    # ========================================

    joint_count_ok = (
        len(joint_names)
        == EXPECTED_JOINT_COUNT
    )

    # ========================================
    # Point validation
    # ========================================

    array_length_ok = True
    finite_ok = True
    time_monotonic_ok = True

    previous_time = None
    previous_positions = None

    max_adjacent_delta_rad = 0.0
    max_adjacent_point = None
    max_adjacent_joint = None

    first_time = None
    final_time = None

    for i, point in enumerate(points):

        positions = point.get(
            "positions",
            []
        )

        velocities = point.get(
            "velocities",
            []
        )

        accelerations = point.get(
            "accelerations",
            []
        )

        effort = point.get(
            "effort",
            []
        )

        time_dict = point.get(
            "time_from_start",
            {}
        )

        # ----------------------------
        # Array lengths
        # ----------------------------

        if len(positions) != len(joint_names):
            print(
                "point %d: positions length = %d"
                % (
                    i,
                    len(positions)
                )
            )

            array_length_ok = False

        if (
            len(velocities) != 0
            and len(velocities) != len(joint_names)
        ):
            print(
                "point %d: velocities length = %d"
                % (
                    i,
                    len(velocities)
                )
            )

            array_length_ok = False

        if (
            len(accelerations) != 0
            and len(accelerations) != len(joint_names)
        ):
            print(
                "point %d: accelerations length = %d"
                % (
                    i,
                    len(accelerations)
                )
            )

            array_length_ok = False

        # ----------------------------
        # NaN / Inf
        # ----------------------------

        values = []

        values.extend(positions)
        values.extend(velocities)
        values.extend(accelerations)
        values.extend(effort)

        for value in values:

            if not math.isfinite(float(value)):

                print(
                    "point %d: NaN / Inf detected"
                    % i
                )

                finite_ok = False

        # ----------------------------
        # Time
        # ----------------------------

        current_time = duration_to_sec(
            time_dict
        )

        if first_time is None:
            first_time = current_time

        final_time = current_time

        if previous_time is not None:

            if current_time <= previous_time:

                print(
                    "point %d: non-monotonic time "
                    "(%.9f <= %.9f)"
                    % (
                        i,
                        current_time,
                        previous_time
                    )
                )

                time_monotonic_ok = False

        previous_time = current_time

        # ----------------------------
        # Adjacent joint delta
        # ----------------------------

        if previous_positions is not None:

            for j, (
                current,
                previous
            ) in enumerate(
                zip(
                    positions,
                    previous_positions
                )
            ):

                delta = abs(
                    float(current)
                    - float(previous)
                )

                if delta > max_adjacent_delta_rad:

                    max_adjacent_delta_rad = delta
                    max_adjacent_point = i
                    max_adjacent_joint = (
                        joint_names[j]
                    )

        previous_positions = positions

    # ========================================
    # Final statistics
    # ========================================

    max_adjacent_delta_deg = math.degrees(
        max_adjacent_delta_rad
    )

    if (
        first_time is not None
        and final_time is not None
    ):
        total_duration = (
            final_time - first_time
        )
    else:
        total_duration = 0.0

    print("")
    print("===== Validation result =====")

    print(
        "joint count              : %s"
        % (
            "PASS"
            if joint_count_ok
            else "FAIL"
        )
    )

    print(
        "array lengths            : %s"
        % (
            "PASS"
            if array_length_ok
            else "FAIL"
        )
    )

    print(
        "NaN / Inf                : %s"
        % (
            "NONE"
            if finite_ok
            else "FOUND"
        )
    )

    print(
        "time monotonic           : %s"
        % (
            "PASS"
            if time_monotonic_ok
            else "FAIL"
        )
    )

    print(
        "total duration           : %.6f s"
        % total_duration
    )

    print(
        "max adjacent joint delta : %.6f deg"
        % max_adjacent_delta_deg
    )

    if (
        max_adjacent_point is not None
        and max_adjacent_joint is not None
    ):

        print(
            "max delta location       : "
            "point %d / %s"
            % (
                max_adjacent_point,
                max_adjacent_joint
            )
        )

    # ========================================
    # Metadata cross-check
    # ========================================

    metadata = data.get(
        "metadata",
        {}
    )

    if metadata:

        print("")
        print("===== Metadata =====")

        print(
            "robot group              : %s"
            % metadata.get(
                "robot_group",
                "UNKNOWN"
            )
        )

        print(
            "trajectory type          : %s"
            % metadata.get(
                "trajectory_type",
                "UNKNOWN"
            )
        )

        print(
            "saved point count        : %s"
            % metadata.get(
                "trajectory_point_count",
                "UNKNOWN"
            )
        )

        print(
            "saved duration           : %s"
            % metadata.get(
                "total_duration_sec",
                "UNKNOWN"
            )
        )

    # ========================================
    # Overall judgment
    # ========================================

    overall_pass = (
        joint_count_ok
        and array_length_ok
        and finite_ok
        and time_monotonic_ok
        and len(points) > 0
    )

    print("")
    print("===== Overall judgment =====")

    if overall_pass:
        print("judgment : PASS")
    else:
        print("judgment : FAIL")


if __name__ == "__main__":
    main()
