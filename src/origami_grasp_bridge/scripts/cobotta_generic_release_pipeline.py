#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print()
    print("=" * 80)
    print("RUN")
    print("=" * 80)
    print(" ".join(cmd))
    print()

    subprocess.run(
        cmd,
        check=True,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sequence-json",
        required=True,
        help="Generic Sequence before RELEASE",
    )

    parser.add_argument(
        "--source-path-json",
        required=True,
        help="Source fold path JSON",
    )

    parser.add_argument(
        "--segment-key",
        default="fold_p0_to_finish",
    )

    parser.add_argument(
        "--point-index",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--layer-count",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--paper-thickness-m",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--release-margin-m",
        type=float,
        default=0.00050,
    )

    parser.add_argument(
        "--retreat-dx",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--retreat-dy",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--retreat-dz",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--gripper-step-m",
        type=float,
        default=0.00001,
    )

    parser.add_argument(
        "--max-retreat-m",
        type=float,
        default=0.005,
    )

    parser.add_argument(
        "--coarse-step-m",
        type=float,
        default=0.0001,
    )

    parser.add_argument(
        "--fine-step-m",
        type=float,
        default=0.00001,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    args = parser.parse_args()

    if args.layer_count < 1:
        raise ValueError(
            "layer-count must be >= 1"
        )

    for name, value in [
        (
            "paper-thickness-m",
            args.paper_thickness_m,
        ),
        (
            "release-margin-m",
            args.release_margin_m,
        ),
        (
            "gripper-step-m",
            args.gripper_step_m,
        ),
    ]:
        if (
            not math.isfinite(value)
            or value < 0.0
        ):
            raise ValueError(
                "{} must be finite and >= 0".format(
                    name
                )
            )

    bundle_thickness_m = (
        args.layer_count
        * args.paper_thickness_m
    )

    relative_release_opening_m = (
        bundle_thickness_m
        + args.release_margin_m
    )

    release_opening_m = (
        relative_release_opening_m
        / 2.0
    )

    output_dir = Path(
        args.output_dir
    ).expanduser().resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    clearance_json = (
        output_dir
        / "release_clearance.json"
    )

    retreat_json = (
        output_dir
        / "release_retreat_path.json"
    )

    sequence_release_json = (
        output_dir
        / "generic_sequence_with_release.json"
    )

    timed_dir = (
        output_dir
        / "timed"
    )

    summary_json = (
        output_dir
        / "release_pipeline_summary.json"
    )

    print()
    print(
        "===== Generic Release Pipeline ====="
    )
    print(
        "layer count              : {}".format(
            args.layer_count
        )
    )
    print(
        "paper thickness          : {:.6f} mm".format(
            args.paper_thickness_m * 1000.0
        )
    )
    print(
        "bundle thickness         : {:.6f} mm".format(
            bundle_thickness_m * 1000.0
        )
    )
    print(
        "release margin           : {:.6f} mm".format(
            args.release_margin_m * 1000.0
        )
    )
    print(
        "relative release opening : {:.6f} mm".format(
            relative_release_opening_m * 1000.0
        )
    )
    print(
        "q_release                : {:.6f} mm/side".format(
            release_opening_m * 1000.0
        )
    )

    # -------------------------------------------------
    # 1. RELEASE CLEARANCE SEARCH
    # -------------------------------------------------
    run([
        "rosrun",
        "origami_grasp_bridge",
        "cobotta_generic_release_clearance_search.py",

        "_input_json:={}".format(
            str(
                Path(
                    args.source_path_json
                ).expanduser().resolve()
            )
        ),

        "_segment_key:={}".format(
            args.segment_key
        ),

        "_point_index:={}".format(
            args.point_index
        ),

        "_frame:=paper_center",

        "_retreat_dx:={}".format(
            args.retreat_dx
        ),
        "_retreat_dy:={}".format(
            args.retreat_dy
        ),
        "_retreat_dz:={}".format(
            args.retreat_dz
        ),

        "_gripper_from_m:=0.0",

        "_release_opening_m:={:.12f}".format(
            release_opening_m
        ),

        "_gripper_step_m:={:.12f}".format(
            args.gripper_step_m
        ),

        "_max_retreat_m:={:.12f}".format(
            args.max_retreat_m
        ),

        "_coarse_step_m:={:.12f}".format(
            args.coarse_step_m
        ),

        "_fine_step_m:={:.12f}".format(
            args.fine_step_m
        ),

        "_output_json:={}".format(
            clearance_json
        ),
    ])

    # -------------------------------------------------
    # 2. RELEASE RETREAT PATH
    # -------------------------------------------------
    run([
        "rosrun",
        "origami_grasp_bridge",
        "cobotta_generic_release_retreat_path.py",

        "_clearance_json:={}".format(
            clearance_json
        ),

        "_gripper_step_m:={:.12f}".format(
            args.gripper_step_m
        ),

        "_output_json:={}".format(
            retreat_json
        ),
    ])

    # -------------------------------------------------
    # 3. APPEND TO GENERIC SEQUENCE
    # -------------------------------------------------
    run([
        "rosrun",
        "origami_grasp_bridge",
        "cobotta_generic_sequence_append_release.py",

        "_sequence_json:={}".format(
            str(
                Path(
                    args.sequence_json
                ).expanduser().resolve()
            )
        ),

        "_retreat_json:={}".format(
            retreat_json
        ),

        "_output_json:={}".format(
            sequence_release_json
        ),
    ])

    # -------------------------------------------------
    # 4. TIME PARAMETERIZATION
    # -------------------------------------------------
    run([
        "rosrun",
        "origami_grasp_bridge",
        "cobotta_generic_arm_motion_time_parameterizer.py",

        "--input",
        str(sequence_release_json),

        "--output-dir",
        str(timed_dir),
    ])

    summary = {
        "schema_version": 1,
        "pipeline_type":
            "generic_release_pipeline",

        "inputs": {
            "sequence_json":
                str(
                    Path(
                        args.sequence_json
                    ).expanduser().resolve()
                ),
            "source_path_json":
                str(
                    Path(
                        args.source_path_json
                    ).expanduser().resolve()
                ),
            "segment_key":
                args.segment_key,
            "point_index":
                args.point_index,
        },

        "release_model": {
            "layer_count":
                args.layer_count,
            "paper_thickness_m":
                args.paper_thickness_m,
            "bundle_thickness_m":
                bundle_thickness_m,
            "release_margin_m":
                args.release_margin_m,
            "relative_release_opening_m":
                relative_release_opening_m,
            "release_opening_m":
                release_opening_m,
        },

        "retreat_direction": [
            args.retreat_dx,
            args.retreat_dy,
            args.retreat_dz,
        ],

        "outputs": {
            "clearance_json":
                str(clearance_json),
            "retreat_json":
                str(retreat_json),
            "sequence_with_release_json":
                str(sequence_release_json),
            "timed_sequence_json":
                str(
                    timed_dir
                    / "generic_sequence_timed.json"
                ),
        },
    }

    with summary_json.open("w") as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("=" * 80)
    print(
        "GENERIC RELEASE PIPELINE COMPLETE"
    )
    print("=" * 80)
    print(
        "summary : {}".format(
            summary_json
        )
    )
    print(
        "timed sequence : {}".format(
            timed_dir
            / "generic_sequence_timed.json"
        )
    )


if __name__ == "__main__":
    main()
