#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import json
import sys
import time

import numpy as np
import rospy
import moveit_commander

from geometry_msgs.msg import PoseArray, PoseStamped
from std_msgs.msg import String

from moveit_msgs.msg import (
    MoveItErrorCodes,
    PlanningSceneComponents,
)
from moveit_msgs.srv import (
    GetPositionIK,
    GetPositionIKRequest,
    GetStateValidity,
    GetStateValidityRequest,
    GetPlanningScene,
    GetPlanningSceneRequest,
)
from urdf_parser_py.urdf import URDF


GROUP = "cobotta_arm"
TIP = "cobotta_tool_link"

COBOTTA_JOINTS = [
    "cobotta_joint_1",
    "cobotta_joint_2",
    "cobotta_joint_3",
    "cobotta_joint_4",
    "cobotta_joint_5",
    "cobotta_joint_6",
]

PRE_TOPIC = (
    "/origami/debug/"
    "cobotta_pregrasp_exhaustive_candidate_poses"
)

GRASP_TOPIC = (
    "/origami/debug/"
    "cobotta_grasp_exhaustive_candidate_poses"
)

METADATA_TOPIC = (
    "/origami/debug/"
    "cobotta_pregrasp_exhaustive_metadata"
)

RESULT_TOPIC = (
    "/origami/debug/"
    "cobotta_pregrasp_exhaustive_feasibility_results"
)

LOCAL_PAPER_OBJECT = (
    "cobotta_local_grasp_paper_collision"
)

GRIPPER_OPEN = 0.015


class ExhaustiveDiagnostic:

    def __init__(self):
        moveit_commander.roscpp_initialize(
            sys.argv
        )

        self.steps = int(
            rospy.get_param(
                "~steps",
                30,
            )
        )

        self.robot = (
            moveit_commander.RobotCommander()
        )

        self.urdf = URDF.from_xml_string(
            rospy.get_param(
                "/robot_description"
            )
        )

        rospy.wait_for_service(
            "/compute_ik",
            timeout=30.0,
        )

        rospy.wait_for_service(
            "/check_state_validity",
            timeout=30.0,
        )

        rospy.wait_for_service(
            "/get_planning_scene",
            timeout=30.0,
        )

        self.compute_ik = rospy.ServiceProxy(
            "/compute_ik",
            GetPositionIK,
            persistent=True,
        )

        self.check_validity = rospy.ServiceProxy(
            "/check_state_validity",
            GetStateValidity,
            persistent=True,
        )

        self.get_scene = rospy.ServiceProxy(
            "/get_planning_scene",
            GetPlanningScene,
            persistent=True,
        )

        self.result_pub = rospy.Publisher(
            RESULT_TOPIC,
            String,
            queue_size=1,
            latch=True,
        )

    def local_paper_exists(self):
        req = GetPlanningSceneRequest()

        req.components.components = (
            PlanningSceneComponents
            .WORLD_OBJECT_NAMES
        )

        res = self.get_scene(req)

        return any(
            obj.id == LOCAL_PAPER_OBJECT
            for obj in
            res.scene.world.collision_objects
        )

    @staticmethod
    def interpolate_pose(
        pre,
        grasp,
        alpha,
        frame_id,
    ):
        target = PoseStamped()

        target.header.frame_id = frame_id
        target.header.stamp = rospy.Time(0)

        # position
        target.pose.position.x = (
            (1.0 - alpha) * pre.position.x
            + alpha * grasp.position.x
        )

        target.pose.position.y = (
            (1.0 - alpha) * pre.position.y
            + alpha * grasp.position.y
        )

        target.pose.position.z = (
            (1.0 - alpha) * pre.position.z
            + alpha * grasp.position.z
        )

        # PREとP0では同一姿勢なので、
        # quaternionはPRE側をそのまま使用。
        target.pose.orientation = copy.deepcopy(
            pre.orientation
        )

        return target

    def solve_ik(
        self,
        target,
        seed_state,
    ):
        req = GetPositionIKRequest()

        req.ik_request.group_name = GROUP
        req.ik_request.ik_link_name = TIP

        req.ik_request.pose_stamped = (
            target
        )

        req.ik_request.robot_state = (
            copy.deepcopy(seed_state)
        )

        # Collisionは別途check_state_validityで評価
        req.ik_request.avoid_collisions = False

        req.ik_request.timeout = (
            rospy.Duration(0.10)
        )

        res = self.compute_ik(req)

        if (
            res.error_code.val
            != MoveItErrorCodes.SUCCESS
        ):
            return None

        return res.solution

    @staticmethod
    def set_gripper_open(state):
        state = copy.deepcopy(state)

        names = list(
            state.joint_state.name
        )
        positions = list(
            state.joint_state.position
        )

        lookup = {
            name: i
            for i, name in enumerate(names)
        }

        if "cobotta_joint_gripper" in lookup:
            positions[
                lookup[
                    "cobotta_joint_gripper"
                ]
            ] = GRIPPER_OPEN

        if (
            "cobotta_joint_gripper_mimic"
            in lookup
        ):
            positions[
                lookup[
                    "cobotta_joint_gripper_mimic"
                ]
            ] = -GRIPPER_OPEN

        state.joint_state.position = positions

        return state

    @staticmethod
    def extract_cobotta_joints(state):
        lookup = dict(zip(
            state.joint_state.name,
            state.joint_state.position,
        ))

        return [
            float(lookup[name])
            for name in COBOTTA_JOINTS
        ]

    def joint_limit_check(self, state):
        q_map = dict(zip(
            state.joint_state.name,
            state.joint_state.position,
        ))

        violations = []

        for name in COBOTTA_JOINTS:
            if name not in q_map:
                violations.append(
                    "{} missing".format(name)
                )
                continue

            joint = self.urdf.joint_map.get(
                name
            )

            if (
                joint is None
                or joint.limit is None
            ):
                continue

            q = q_map[name]

            lower = joint.limit.lower
            upper = joint.limit.upper

            tol = 1.0e-8

            if (
                lower is not None
                and q < lower - tol
            ):
                violations.append(
                    "{} below lower limit"
                    .format(name)
                )

            if (
                upper is not None
                and q > upper + tol
            ):
                violations.append(
                    "{} above upper limit"
                    .format(name)
                )

        return violations

    def collision_check(self, state):
        req = GetStateValidityRequest()

        req.robot_state = state
        req.group_name = GROUP

        res = self.check_validity(req)

        pairs = []

        if not res.valid:
            for contact in res.contacts:
                pairs.append(
                    tuple(sorted([
                        contact.contact_body_1,
                        contact.contact_body_2,
                    ]))
                )

        return bool(res.valid), sorted(
            set(pairs)
        )

    @staticmethod
    def collision_type(pairs):
        if any(
            "paper_stand" in (a, b)
            for a, b in pairs
        ):
            return "STAND_COLLISION"

        if any(
            a.startswith("cobotta_")
            and b.startswith("cobotta_")
            for a, b in pairs
        ):
            return "SELF_COLLISION"

        return "OTHER_COLLISION"

    def build_ik_states(
        self,
        pre,
        grasp,
        frame_id,
    ):
        seed = (
            self.robot.get_current_state()
        )

        states = []

        for step in range(
            self.steps + 1
        ):
            alpha = (
                float(step)
                / float(self.steps)
            )

            target = self.interpolate_pose(
                pre,
                grasp,
                alpha,
                frame_id,
            )

            solution = self.solve_ik(
                target,
                seed,
            )

            if solution is None:
                return None, {
                    "result":
                        "NO_PATH_FOUND_CURRENT_SEED",
                    "reason":
                        "IK_FAILED",
                    "step":
                        int(step),
                }

            solution = (
                self.set_gripper_open(
                    solution
                )
            )

            states.append(
                solution
            )

            # 前点IKを次点seedにする
            seed = solution

        return states, None

    def evaluate_states(
        self,
        states,
    ):
        for step, state in enumerate(
            states
        ):
            violations = (
                self.joint_limit_check(
                    state
                )
            )

            if violations:
                return {
                    "result":
                        "NO_PATH_FOUND_CURRENT_SEED",
                    "reason":
                        "JOINT_LIMIT",
                    "step":
                        int(step),
                    "collision_pairs":
                        [],
                }

            valid, pairs = (
                self.collision_check(
                    state
                )
            )

            if not valid:
                return {
                    "result":
                        "NO_PATH_FOUND_CURRENT_SEED",
                    "reason":
                        self.collision_type(
                            pairs
                        ),
                    "step":
                        int(step),
                    "collision_pairs":
                        [
                            "{}<->{}".format(
                                a,
                                b,
                            )
                            for a, b in pairs
                        ],
                }

        return {
            "result":
                "FEASIBLE_FOUND",
            "reason":
                "-",
            "step":
                int(self.steps),
            "collision_pairs":
                [],

            "pregrasp_joint_names":
                list(COBOTTA_JOINTS),
            "pregrasp_joints_rad":
                self.extract_cobotta_joints(
                    states[0]
                ),

            "grasp_joint_names":
                list(COBOTTA_JOINTS),
            "grasp_joints_rad":
                self.extract_cobotta_joints(
                    states[-1]
                ),
        }

    def build_j6_flipped_states(
        self,
        plus_states,
    ):
        joint_name = (
            "cobotta_joint_6"
        )

        joint_model = (
            self.urdf.joint_map.get(
                joint_name
            )
        )

        if (
            joint_model is None
            or joint_model.limit is None
        ):
            return (
                None,
                None,
                "J6_MODEL_NOT_FOUND",
            )

        lower = (
            joint_model.limit.lower
        )
        upper = (
            joint_model.limit.upper
        )

        tol = 1.0e-8

        for offset in [
            np.pi,
            -np.pi,
        ]:
            flipped_states = []
            valid_offset = True

            for state in plus_states:
                new_state = copy.deepcopy(
                    state
                )

                names = list(
                    new_state.joint_state.name
                )

                positions = list(
                    new_state.joint_state.position
                )

                if joint_name not in names:
                    return (
                        None,
                        None,
                        "J6_NOT_IN_STATE",
                    )

                j6_index = names.index(
                    joint_name
                )

                q_new = (
                    positions[j6_index]
                    + offset
                )

                if (
                    (
                        lower is not None
                        and q_new
                        < lower - tol
                    )
                    or
                    (
                        upper is not None
                        and q_new
                        > upper + tol
                    )
                ):
                    valid_offset = False
                    break

                positions[
                    j6_index
                ] = q_new

                new_state.joint_state.position = (
                    positions
                )

                new_state.joint_state.header.stamp = (
                    rospy.Time(0)
                )

                flipped_states.append(
                    new_state
                )

            if valid_offset:
                return (
                    flipped_states,
                    float(offset),
                    None,
                )

        return (
            None,
            None,
            "J6_FLIP_JOINT_LIMIT",
        )

    @staticmethod
    def grasp_pose_record(
        pose,
        frame_id,
    ):
        return {
            "frame_id":
                str(frame_id),
            "position": {
                "x":
                    float(pose.position.x),
                "y":
                    float(pose.position.y),
                "z":
                    float(pose.position.z),
            },
            "orientation": {
                "x":
                    float(
                        pose.orientation.x
                    ),
                "y":
                    float(
                        pose.orientation.y
                    ),
                "z":
                    float(
                        pose.orientation.z
                    ),
                "w":
                    float(
                        pose.orientation.w
                    ),
            },
        }

    def run(self):
        print(
            "===== EXHAUSTIVE PRE-GRASP "
            "FEASIBILITY DIAGNOSTIC ====="
        )

        print(
            "Waiting for exhaustive inputs..."
        )

        pre_msg = rospy.wait_for_message(
            PRE_TOPIC,
            PoseArray,
            timeout=15.0,
        )

        grasp_msg = rospy.wait_for_message(
            GRASP_TOPIC,
            PoseArray,
            timeout=15.0,
        )

        meta_msg = rospy.wait_for_message(
            METADATA_TOPIC,
            String,
            timeout=15.0,
        )

        metadata = json.loads(
            meta_msg.data
        )

        candidates = metadata[
            "candidates"
        ]

        count = len(
            candidates
        )

        if (
            len(pre_msg.poses) != count
            or len(grasp_msg.poses)
            != count
        ):
            raise RuntimeError(
                "Candidate count mismatch: "
                "meta={} pre={} grasp={}"
                .format(
                    count,
                    len(pre_msg.poses),
                    len(grasp_msg.poses),
                )
            )

        if count % 2 != 0:
            raise RuntimeError(
                "Candidate count must be even."
            )

        # metadataの並びを厳密確認
        for i, c in enumerate(
            candidates
        ):
            if int(
                c["candidate_index"]
            ) != i:
                raise RuntimeError(
                    "candidate_index mismatch "
                    "at {}".format(i)
                )

        for i in range(
            0,
            count,
            2,
        ):
            plus = candidates[i]
            minus = candidates[i + 1]

            if (
                plus["normal_sign"] != "N+"
                or minus["normal_sign"] != "N-"
                or plus["direction_index"]
                != minus["direction_index"]
                or plus["approach_angle_deg"]
                != minus["approach_angle_deg"]
                or plus["entry_edge"]
                != minus["entry_edge"]
            ):
                raise RuntimeError(
                    "Broken N+/N- pair at "
                    "candidate {}".format(i)
                )

        print()
        print(
            "P0 index         : {}".format(
                metadata["p0_index"]
            )
        )

        print(
            "direction count  : {}".format(
                metadata[
                    "direction_count"
                ]
            )
        )

        print(
            "candidate count  : {}".format(
                count
            )
        )

        print(
            "steps/candidate  : {}".format(
                self.steps
            )
        )

        print(
            "gripper          : OPEN 15 mm"
        )

        print(
            "collision        : current Planning Scene"
        )

        print(
            "IK strategy      : "
            "N+ direct -> N- J6 flip"
        )

        print(
            "N+ IK fallback   : "
            "N- direct IK"
        )

        if self.local_paper_exists():
            raise RuntimeError(
                "Local Paper Collision "
                "is still in the Planning Scene."
            )

        results = []

        total_start = (
            time.perf_counter()
        )

        pair_total = count // 2

        for pair_index, plus_index in enumerate(
            range(0, count, 2)
        ):
            minus_index = (
                plus_index + 1
            )

            plus_meta = candidates[
                plus_index
            ]
            minus_meta = candidates[
                minus_index
            ]

            angle = float(
                plus_meta[
                    "approach_angle_deg"
                ]
            )

            edge = int(
                plus_meta[
                    "entry_edge"
                ]
            )

            print()
            print(
                "========================================"
            )

            print(
                "PAIR {}/{} | "
                "direction={} "
                "angle={:.1f}deg "
                "edge={}"
                .format(
                    pair_index + 1,
                    pair_total,
                    plus_meta[
                        "direction_index"
                    ],
                    angle,
                    edge,
                )
            )

            print(
                "========================================"
            )

            pair_start = (
                time.perf_counter()
            )

            plus_states, plus_failure = (
                self.build_ik_states(
                    pre_msg.poses[
                        plus_index
                    ],
                    grasp_msg.poses[
                        plus_index
                    ],
                    pre_msg.header.frame_id,
                )
            )

            if plus_states is None:
                plus_result = (
                    plus_failure
                )

                # N+が作れないときは
                # N-を直接IKで試す
                minus_states, minus_failure = (
                    self.build_ik_states(
                        pre_msg.poses[
                            minus_index
                        ],
                        grasp_msg.poses[
                            minus_index
                        ],
                        pre_msg.header.frame_id,
                    )
                )

                if minus_states is None:
                    minus_result = (
                        minus_failure
                    )
                else:
                    minus_result = (
                        self.evaluate_states(
                            minus_states
                        )
                    )

                    minus_result[
                        "ik_source"
                    ] = (
                        "DIRECT_IK_FALLBACK"
                    )

            else:
                plus_result = (
                    self.evaluate_states(
                        plus_states
                    )
                )

                plus_result[
                    "ik_source"
                ] = "DIRECT_IK"

                (
                    minus_states,
                    j6_offset,
                    flip_error,
                ) = (
                    self.build_j6_flipped_states(
                        plus_states
                    )
                )

                if minus_states is None:
                    minus_result = {
                        "result":
                            "NO_PATH_FOUND_J6_FLIP",
                        "reason":
                            str(flip_error),
                        "step":
                            0,
                        "collision_pairs":
                            [],
                        "ik_source":
                            "J6_FROM_NPLUS",
                    }

                else:
                    minus_result = (
                        self.evaluate_states(
                            minus_states
                        )
                    )

                    minus_result[
                        "ik_source"
                    ] = "J6_FROM_NPLUS"

                    minus_result[
                        "j6_offset_rad"
                    ] = float(
                        j6_offset
                    )

            if "ik_source" not in plus_result:
                plus_result[
                    "ik_source"
                ] = "DIRECT_IK_FAILED"

            pair_elapsed = (
                time.perf_counter()
                - pair_start
            )

            # metadata + evaluationを統合
            for (
                candidate_index,
                meta,
                result,
            ) in [
                (
                    plus_index,
                    plus_meta,
                    plus_result,
                ),
                (
                    minus_index,
                    minus_meta,
                    minus_result,
                ),
            ]:
                record = copy.deepcopy(
                    meta
                )

                record.update({
                    "result":
                        str(
                            result["result"]
                        ),
                    "reason":
                        str(
                            result["reason"]
                        ),
                    "step":
                        int(
                            result["step"]
                        ),
                    "ik_source":
                        str(
                            result.get(
                                "ik_source",
                                "-"
                            )
                        ),
                    "collision_pairs":
                        list(
                            result.get(
                                "collision_pairs",
                                []
                            )
                        ),
                })

                if (
                    result["result"]
                    == "FEASIBLE_FOUND"
                    and
                    "pregrasp_joints_rad"
                    in result
                ):
                    record[
                        "pregrasp_joint_names"
                    ] = list(
                        result[
                            "pregrasp_joint_names"
                        ]
                    )

                    record[
                        "pregrasp_joints_rad"
                    ] = [
                        float(v)
                        for v in result[
                            "pregrasp_joints_rad"
                        ]
                    ]

                    record[
                        "grasp_joint_names"
                    ] = list(
                        result[
                            "grasp_joint_names"
                        ]
                    )

                    record[
                        "grasp_joints_rad"
                    ] = [
                        float(v)
                        for v in result[
                            "grasp_joints_rad"
                        ]
                    ]

                    record[
                        "grasp_tool_pose"
                    ] = (
                        self.grasp_pose_record(
                            grasp_msg.poses[
                                candidate_index
                            ],
                            grasp_msg.header.frame_id,
                        )
                    )

                results.append(
                    record
                )

            print(
                "  N+ : {} / {}"
                .format(
                    plus_result[
                        "result"
                    ],
                    plus_result[
                        "reason"
                    ],
                )
            )

            print(
                "  N- : {} / {}"
                .format(
                    minus_result[
                        "result"
                    ],
                    minus_result[
                        "reason"
                    ],
                )
            )

            print(
                "  pair time = {:.3f} s"
                .format(
                    pair_elapsed
                )
            )

        total_elapsed = (
            time.perf_counter()
            - total_start
        )

        success = [
            r for r in results
            if r["result"]
            == "FEASIBLE_FOUND"
        ]

        reason_counts = {}

        for r in results:
            reason = r["reason"]

            reason_counts[
                reason
            ] = (
                reason_counts.get(
                    reason,
                    0
                )
                + 1
            )

        output = {
            "schema_version": 1,
            "p0_index":
                int(
                    metadata[
                        "p0_index"
                    ]
                ),
            "angle_step_deg":
                float(
                    metadata[
                        "angle_step_deg"
                    ]
                ),
            "direction_count":
                int(
                    metadata[
                        "direction_count"
                    ]
                ),
            "candidate_count":
                int(count),
            "steps_per_candidate":
                int(self.steps),
            "gripper_open_m":
                float(
                    GRIPPER_OPEN
                ),
            "feasible_count":
                int(
                    len(success)
                ),
            "total_time_sec":
                float(
                    total_elapsed
                ),
            "reason_counts":
                reason_counts,
            "candidates":
                results,
        }

        msg = String()

        msg.data = json.dumps(
            output,
            ensure_ascii=False,
            sort_keys=True,
        )

        self.result_pub.publish(
            msg
        )

        print()
        print(
            "========================================"
        )

        print(
            " EXHAUSTIVE FEASIBILITY SUMMARY"
        )

        print(
            "========================================"
        )

        print(
            "directions       : {}".format(
                metadata[
                    "direction_count"
                ]
            )
        )

        print(
            "candidates       : {}".format(
                count
            )
        )

        print(
            "FEASIBLE_FOUND   : {}/{}"
            .format(
                len(success),
                count,
            )
        )

        print(
            "total time       : {:.3f} s"
            .format(
                total_elapsed
            )
        )

        print()
        print(
            "failure reasons:"
        )

        for reason, n in sorted(
            reason_counts.items(),
            key=lambda x: (
                -x[1],
                x[0],
            )
        ):
            print(
                "  {:30s} : {}"
                .format(
                    reason,
                    n,
                )
            )

        # 角度範囲を一覧表示
        feasible_angles = sorted(
            set(
                float(r[
                    "approach_angle_deg"
                ])
                for r in success
            )
        )

        print()
        print(
            "feasible approach angles:"
        )

        if feasible_angles:
            print(
                "  count = {}".format(
                    len(
                        feasible_angles
                    )
                )
            )

            print(
                "  min   = {:.1f} deg"
                .format(
                    min(
                        feasible_angles
                    )
                )
            )

            print(
                "  max   = {:.1f} deg"
                .format(
                    max(
                        feasible_angles
                    )
                )
            )
        else:
            print(
                "  NONE"
            )

        print()
        print(
            "Published structured result:"
        )

        print(
            "  {}".format(
                RESULT_TOPIC
            )
        )

        rospy.spin()


def main():
    rospy.init_node(
        "cobotta_pregrasp_exhaustive_feasibility_diagnostic"
    )

    diagnostic = (
        ExhaustiveDiagnostic()
    )

    diagnostic.run()


if __name__ == "__main__":
    main()
