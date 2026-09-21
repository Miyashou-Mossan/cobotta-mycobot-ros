#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import json
import sys

import numpy as np
import rospy
import moveit_commander

from geometry_msgs.msg import PoseArray, PoseStamped
from std_msgs.msg import String, UInt64MultiArray
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
    "cobotta_pregrasp_candidate_poses"
)

GRASP_TOPIC = (
    "/origami/debug/"
    "cobotta_grasp_candidate_poses"
)

BATCH_TOPIC = (
    "/origami/debug/"
    "cobotta_pregrasp_batch"
)

DONE_TOPIC = (
    "/origami/debug/"
    "cobotta_pregrasp_feasibility_done"
)

RESULT_TOPIC = (
    "/origami/debug/"
    "cobotta_pregrasp_feasibility_results"
)

LOCAL_PAPER_OBJECT = (
    "cobotta_local_grasp_paper_collision"
)


class Diagnostic:

    def __init__(self):
        moveit_commander.roscpp_initialize(
            sys.argv
        )

        # これまでの診断と同じ30分割。
        # 最終的な安全分解能ではない。
        self.steps = int(
            rospy.get_param(
                "~steps",
                30,
            )
        )

        self.pre_msg = None
        self.grasp_msg = None
        self.batch_msg = None

        self.processing = False
        self.last_processed_key = None

        # 現在の全P0評価結果を保持する。
        # p0_index=0を受けた時点で新しい一連の評価として
        # リセットする。
        self.result_history = {}

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

        self.done_pub = rospy.Publisher(
            DONE_TOPIC,
            UInt64MultiArray,
            queue_size=1,
            latch=False,
        )

        # 後段処理が利用する構造化評価結果。
        # 最新の集約結果を後から確認できるようlatchする。
        self.result_pub = rospy.Publisher(
            RESULT_TOPIC,
            String,
            queue_size=1,
            latch=True,
        )

        rospy.Subscriber(
            BATCH_TOPIC,
            UInt64MultiArray,
            self.batch_cb,
            queue_size=1,
        )

        rospy.Subscriber(
            PRE_TOPIC,
            PoseArray,
            self.pre_cb,
            queue_size=1,
        )

        rospy.Subscriber(
            GRASP_TOPIC,
            PoseArray,
            self.grasp_cb,
            queue_size=1,
        )

        rospy.loginfo(
            "Waiting for PRE-GRASP candidates..."
        )

    def batch_cb(self, msg):
        self.batch_msg = msg
        self.try_run()

    def pre_cb(self, msg):
        self.pre_msg = msg
        self.try_run()

    def grasp_cb(self, msg):
        self.grasp_msg = msg
        self.try_run()

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
    def label(index):
        edge = index // 2

        sign = (
            "N+"
            if index % 2 == 0
            else "N-"
        )

        return "edge {} {}".format(
            edge,
            sign,
        )

    def interpolate_pose(
        self,
        pre,
        grasp,
        alpha,
        frame_id,
    ):
        p0 = np.array([
            pre.position.x,
            pre.position.y,
            pre.position.z,
        ])

        p1 = np.array([
            grasp.position.x,
            grasp.position.y,
            grasp.position.z,
        ])

        p = (
            (1.0 - alpha) * p0
            + alpha * p1
        )

        target = PoseStamped()

        target.header.frame_id = frame_id
        target.header.stamp = rospy.Time(0)

        target.pose.position.x = float(
            p[0]
        )
        target.pose.position.y = float(
            p[1]
        )
        target.pose.position.z = float(
            p[2]
        )

        # PREとP0で工具姿勢は固定
        target.pose.orientation = (
            copy.deepcopy(
                pre.orientation
            )
        )

        return target

    def solve_ik(
        self,
        target,
        seed,
    ):
        req = GetPositionIKRequest()

        req.ik_request.group_name = GROUP
        req.ik_request.ik_link_name = TIP

        req.ik_request.pose_stamped = (
            target
        )

        req.ik_request.robot_state = (
            copy.deepcopy(seed)
        )

        # IKとCollision判定を分離
        req.ik_request.avoid_collisions = (
            False
        )

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

        if (
            "cobotta_joint_gripper"
            in lookup
        ):
            positions[
                lookup[
                    "cobotta_joint_gripper"
                ]
            ] = 0.015

        if (
            "cobotta_joint_gripper_mimic"
            in lookup
        ):
            positions[
                lookup[
                    "cobotta_joint_gripper_mimic"
                ]
            ] = -0.015

        state.joint_state.position = (
            positions
        )

        state.joint_state.header.stamp = (
            rospy.Time(0)
        )

        return state

    def extract_cobotta_joints(self, state):
        """
        RobotStateからCOBOTTA J1～J6を
        明示的な関節名に基づいて取り出す。
        """
        q_map = dict(zip(
            state.joint_state.name,
            state.joint_state.position,
        ))

        missing = [
            name
            for name in COBOTTA_JOINTS
            if name not in q_map
        ]

        if missing:
            raise RuntimeError(
                "Missing COBOTTA joints in RobotState: "
                + ", ".join(missing)
            )

        return [
            float(q_map[name])
            for name in COBOTTA_JOINTS
        ]


    def joint_limit_check(self, state):
        q_map = dict(zip(
            state.joint_state.name,
            state.joint_state.position,
        ))

        violations = []

        for joint in self.urdf.joints:
            if joint.name not in q_map:
                continue

            if joint.type in [
                "fixed",
                "continuous",
            ]:
                continue

            if joint.limit is None:
                continue

            q = q_map[joint.name]

            lower = joint.limit.lower
            upper = joint.limit.upper

            tol = 1.0e-8

            if (
                lower is not None
                and q < lower - tol
            ):
                violations.append(
                    "{} below lower".format(
                        joint.name
                    )
                )

            if (
                upper is not None
                and q > upper + tol
            ):
                violations.append(
                    "{} above upper".format(
                        joint.name
                    )
                )

        return violations

    def collision_check(self, state):
        req = GetStateValidityRequest()

        req.robot_state = (
            copy.deepcopy(state)
        )

        req.group_name = GROUP

        res = self.check_validity(req)

        pairs = []

        for contact in res.contacts:
            pair = tuple(sorted([
                contact.contact_body_1,
                contact.contact_body_2,
            ]))

            if pair not in pairs:
                pairs.append(pair)

        return bool(res.valid), pairs

    @staticmethod
    def collision_type(pairs):
        if any(
            "paper_stand" in pair
            for pair in pairs
        ):
            return "STAND_COLLISION"

        if any(
            a.startswith("cobotta_")
            and b.startswith("cobotta_")
            for a, b in pairs
        ):
            return "SELF_COLLISION"

        return "OTHER_COLLISION"

    def test_candidate(
        self,
        index,
        pre,
        grasp,
        frame_id,
    ):
        name = self.label(index)

        pre_xyz = np.array([
            pre.position.x,
            pre.position.y,
            pre.position.z,
        ])

        grasp_xyz = np.array([
            grasp.position.x,
            grasp.position.y,
            grasp.position.z,
        ])

        distance = np.linalg.norm(
            grasp_xyz - pre_xyz
        )

        print()
        print(
            "----- candidate {} : {} -----".format(
                index,
                name,
            )
        )

        print(
            "approach distance : "
            "{:.3f} mm".format(
                distance * 1000.0
            )
        )

        # 候補ごとに同じ現在状態から開始
        seed = self.robot.get_current_state()

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
                print(
                    "FAIL: IK"
                )
                print(
                    "  step     : {}/{}".format(
                        step,
                        self.steps,
                    )
                )
                print(
                    "  progress : {:.3f}".format(
                        alpha
                    )
                )

                return {
                    "label": name,
                    "result":
                        "NO_PATH_FOUND_CURRENT_SEED",
                    "reason": "IK_FAILED",
                    "step": step,
                }

            solution = self.set_gripper_open(
                solution
            )

            violations = (
                self.joint_limit_check(
                    solution
                )
            )

            if violations:
                print(
                    "FAIL: JOINT_LIMIT"
                )

                for v in violations:
                    print(
                        "  " + v
                    )

                return {
                    "label": name,
                    "result":
                        "NO_PATH_FOUND_CURRENT_SEED",
                    "reason": "JOINT_LIMIT",
                    "step": step,
                }

            valid, pairs = (
                self.collision_check(
                    solution
                )
            )

            if not valid:
                kind = self.collision_type(
                    pairs
                )

                print(
                    "FAIL: {}".format(
                        kind
                    )
                )
                print(
                    "  step     : {}/{}".format(
                        step,
                        self.steps,
                    )
                )
                print(
                    "  progress : {:.3f}".format(
                        alpha
                    )
                )

                for a, b in pairs:
                    print(
                        "  {} <-> {}".format(
                            a,
                            b,
                        )
                    )

                return {
                    "label": name,
                    "result":
                        "NO_PATH_FOUND_CURRENT_SEED",
                    "reason": kind,
                    "step": step,
                }

            # 次点は前点のIK解をseedにする
            seed = solution

        print(
            "RESULT: FEASIBLE_FOUND"
        )

        return {
            "label": name,
            "result": "FEASIBLE_FOUND",
            "reason": "-",
            "step": self.steps,
        }

    def build_ik_states(
        self,
        index,
        pre,
        grasp,
        frame_id,
    ):
        """
        PRE -> P0 のIK軌道だけを生成する。
        Collision評価はここでは行わない。
        """
        name = self.label(index)

        pre_xyz = np.array([
            pre.position.x,
            pre.position.y,
            pre.position.z,
        ])

        grasp_xyz = np.array([
            grasp.position.x,
            grasp.position.y,
            grasp.position.z,
        ])

        distance = np.linalg.norm(
            grasp_xyz - pre_xyz
        )

        print()
        print(
            "----- candidate {} : {} [IK BUILD] -----".format(
                index,
                name,
            )
        )

        print(
            "approach distance : "
            "{:.3f} mm".format(
                distance * 1000.0
            )
        )

        seed = self.robot.get_current_state()

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
                print(
                    "FAIL: IK"
                )
                print(
                    "  step     : {}/{}".format(
                        step,
                        self.steps,
                    )
                )
                print(
                    "  progress : {:.3f}".format(
                        alpha
                    )
                )

                return None, {
                    "label": name,
                    "result":
                        "NO_PATH_FOUND_CURRENT_SEED",
                    "reason": "IK_FAILED",
                    "step": step,
                }

            solution = self.set_gripper_open(
                solution
            )

            states.append(solution)

            # 前点IK解を次点seedにする
            seed = solution

        print(
            "IK PATH BUILD: SUCCESS"
        )

        return states, None

    def evaluate_states(
        self,
        index,
        states,
        source,
    ):
        """
        生成済み関節軌道について
        関節限界とCollisionだけを評価する。
        """
        name = self.label(index)

        print()
        print(
            "----- candidate {} : {} [EVALUATE] -----".format(
                index,
                name,
            )
        )

        print(
            "source : {}".format(
                source
            )
        )

        for step, state in enumerate(states):
            alpha = (
                float(step)
                / float(self.steps)
            )

            violations = (
                self.joint_limit_check(
                    state
                )
            )

            if violations:
                print(
                    "FAIL: JOINT_LIMIT"
                )

                print(
                    "  step     : {}/{}".format(
                        step,
                        self.steps,
                    )
                )

                for v in violations:
                    print(
                        "  " + v
                    )

                return {
                    "label": name,
                    "result":
                        "NO_PATH_FOUND_CURRENT_SEED",
                    "reason": "JOINT_LIMIT",
                    "step": step,
                }

            valid, pairs = (
                self.collision_check(
                    state
                )
            )

            if not valid:
                kind = self.collision_type(
                    pairs
                )

                print(
                    "FAIL: {}".format(
                        kind
                    )
                )

                print(
                    "  step     : {}/{}".format(
                        step,
                        self.steps,
                    )
                )

                print(
                    "  progress : {:.3f}".format(
                        alpha
                    )
                )

                for a, b in pairs:
                    print(
                        "  {} <-> {}".format(
                            a,
                            b,
                        )
                    )

                return {
                    "label": name,
                    "result":
                        "NO_PATH_FOUND_CURRENT_SEED",
                    "reason": kind,
                    "step": step,
                }

        print(
            "RESULT: FEASIBLE_FOUND"
        )

        return {
            "label": name,
            "result": "FEASIBLE_FOUND",
            "reason": "-",
            "step": self.steps,

            # PRE-GRASP:
            # 紙へ横から進入する直前の関節姿勢
            "pregrasp_joint_names":
                list(COBOTTA_JOINTS),
            "pregrasp_joints_rad":
                self.extract_cobotta_joints(
                    states[0]
                ),

            # GRASP:
            # P0へ到着した瞬間の関節姿勢。
            # この状態を次の折り軌道の初期seedとして使う。
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
        """
        N+軌道のJ1～J5を維持し、
        J6だけを全step同じ方向へ
        +180 deg または -180 deg する。

        全stepがJ6関節限界内に入る
        offsetだけを採用する。
        """
        joint_name = "cobotta_joint_6"

        joint_model = None

        for joint in self.urdf.joints:
            if joint.name == joint_name:
                joint_model = joint
                break

        if (
            joint_model is None
            or joint_model.limit is None
        ):
            return (
                None,
                None,
                "J6_MODEL_NOT_FOUND",
            )

        lower = joint_model.limit.lower
        upper = joint_model.limit.upper

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
                        and q_new < lower - tol
                    )
                    or
                    (
                        upper is not None
                        and q_new > upper + tol
                    )
                ):
                    valid_offset = False
                    break

                positions[j6_index] = q_new

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
                    offset,
                    None,
                )

        return (
            None,
            None,
            "J6_FLIP_JOINT_LIMIT",
        )

    def try_run(self):
        if self.processing:
            return

        if (
            self.pre_msg is None
            or self.grasp_msg is None
            or self.batch_msg is None
        ):
            return

        if len(self.batch_msg.data) != 3:
            return

        # PREとGRASPは同じtimestampでなければならない
        if (
            self.pre_msg.header.stamp
            != self.grasp_msg.header.stamp
        ):
            return

        p0_index = int(
            self.batch_msg.data[0]
        )

        batch_secs = int(
            self.batch_msg.data[1]
        )

        batch_nsecs = int(
            self.batch_msg.data[2]
        )

        # Batch metadataとPoseArrayのtimestampを照合
        if (
            self.pre_msg.header.stamp.secs
            != batch_secs
            or
            self.pre_msg.header.stamp.nsecs
            != batch_nsecs
        ):
            return

        batch_key = (
            p0_index,
            batch_secs,
            batch_nsecs,
        )

        if (
            batch_key
            == self.last_processed_key
        ):
            return

        self.processing = True

        print()
        print(
            "P0 index           : {}".format(
                p0_index
            )
        )

        if self.local_paper_exists():
            print()
            print(
                "ERROR:"
            )
            print(
                "Local Paper Collision is still "
                "in the Planning Scene."
            )
            print(
                "Remove it before this test."
            )
            self.processing = False
            return

        if (
            len(self.pre_msg.poses)
            != len(self.grasp_msg.poses)
        ):
            print(
                "ERROR: candidate counts differ."
            )
            self.processing = False
            return

        count = len(
            self.pre_msg.poses
        )

        frame_id = (
            self.pre_msg.header.frame_id
        )

        print()
        print(
            "===== PRE-GRASP Feasibility Diagnostic ====="
        )

        print(
            "candidate count : {}".format(
                count
            )
        )

        print(
            "steps/candidate : {}".format(
                self.steps
            )
        )

        print(
            "gripper         : OPEN 15 mm"
        )

        print(
            "local paper     : NOT USED"
        )

        print(
            "collision       : current Planning Scene"
        )

        results = []

        if count % 2 != 0:
            print(
                "ERROR: candidate count must be even "
                "(N+/N- pairs)."
            )
            self.processing = False
            return

        print()
        print(
            "IK strategy       : "
            "N+ direct -> N- J6 flip"
        )
        print(
            "N+ IK fallback    : "
            "N- direct IK"
        )

        for plus_index in range(
            0,
            count,
            2,
        ):
            minus_index = (
                plus_index + 1
            )

            print()
            print(
                "========================================"
            )
            print(
                "PAIR: {} / {}".format(
                    self.label(plus_index),
                    self.label(minus_index),
                )
            )
            print(
                "========================================"
            )

            plus_states, plus_failure = (
                self.build_ik_states(
                    plus_index,
                    self.pre_msg.poses[
                        plus_index
                    ],
                    self.grasp_msg.poses[
                        plus_index
                    ],
                    frame_id,
                )
            )

            # N+のIK軌道そのものが作れない場合だけ
            # N-を従来どおり直接IKする。
            if plus_states is None:
                results.append(
                    plus_failure
                )

                print()
                print(
                    "N+ IK path failed."
                )
                print(
                    "Fallback: direct IK for N-."
                )

                minus_states, minus_failure = (
                    self.build_ik_states(
                        minus_index,
                        self.pre_msg.poses[
                            minus_index
                        ],
                        self.grasp_msg.poses[
                            minus_index
                        ],
                        frame_id,
                    )
                )

                if minus_states is None:
                    results.append(
                        minus_failure
                    )
                else:
                    results.append(
                        self.evaluate_states(
                            minus_index,
                            minus_states,
                            "DIRECT_IK_FALLBACK",
                        )
                    )

                continue

            # N+は通常どおり安全性評価
            results.append(
                self.evaluate_states(
                    plus_index,
                    plus_states,
                    "DIRECT_IK",
                )
            )

            # N+と同じJ1～J5を使い、
            # J6だけ180度反転してN-を作る。
            (
                minus_states,
                j6_offset,
                flip_error,
            ) = self.build_j6_flipped_states(
                plus_states
            )

            if minus_states is None:
                print()
                print(
                    "N- J6 FLIP FAILED: {}".format(
                        flip_error
                    )
                )

                results.append({
                    "label":
                        self.label(
                            minus_index
                        ),
                    "result":
                        "NO_PATH_FOUND_J6_FLIP",
                    "reason":
                        flip_error,
                    "step":
                        0,
                })

                continue

            print()
            print(
                "N- generated from N+."
            )
            print(
                "J6 offset : {:+.1f} deg".format(
                    np.degrees(
                        j6_offset
                    )
                )
            )

            results.append(
                self.evaluate_states(
                    minus_index,
                    minus_states,
                    "J6_FROM_NPLUS",
                )
            )

        print()
        print(
            "===== Summary ====="
        )

        print(
            "{:<12s} | {:<26s} | {}".format(
                "candidate",
                "result",
                "reason",
            )
        )

        print(
            "-------------+"
            "----------------------------+"
            "--------------------"
        )

        for r in results:
            print(
                "{:<12s} | {:<26s} | {}".format(
                    r["label"],
                    r["result"],
                    r["reason"],
                )
            )

        success = [
            r for r in results
            if r["result"]
            == "FEASIBLE_FOUND"
        ]

        print()
        print(
            "FEASIBLE_FOUND : {}/{}".format(
                len(success),
                count,
            )
        )

        if success:
            print(
                "At least one automatically generated "
                "PRE-GRASP approach was found."
            )
        else:
            print(
                "No path was found with the current "
                "single-seed diagnostic."
            )
            print(
                "This does NOT prove infeasibility."
            )

        # ----------------------------------------------------
        # 構造化結果をpublish
        #
        # ここでは候補の順位付け・最適化は行わない。
        # 現在の評価結果を機械可読な形で保存するだけ。
        # ----------------------------------------------------

        if p0_index == 0:
            self.result_history = {}

        candidate_records = []

        for candidate_index, r in enumerate(results):
            record = {
                "candidate_index": int(
                    candidate_index
                ),
                "edge": int(
                    candidate_index // 2
                ),
                "normal_sign": (
                    "N+"
                    if candidate_index % 2 == 0
                    else "N-"
                ),
                "result": str(
                    r["result"]
                ),
                "reason": str(
                    r["reason"]
                ),
                "step": int(
                    r["step"]
                ),
            }

            if (
                r["result"] == "FEASIBLE_FOUND"
                and "pregrasp_joints_rad" in r
                and "grasp_joints_rad" in r
            ):
                record["pregrasp_joint_names"] = list(
                    r["pregrasp_joint_names"]
                )

                record["pregrasp_joints_rad"] = [
                    float(v)
                    for v in r[
                        "pregrasp_joints_rad"
                    ]
                ]

                record["grasp_joint_names"] = list(
                    r["grasp_joint_names"]
                )

                record["grasp_joints_rad"] = [
                    float(v)
                    for v in r[
                        "grasp_joints_rad"
                    ]
                ]

                # P0到着時のcobotta_tool_link Pose。
                # N+ / N-ごとの工具姿勢を後段の折り軌道へ渡す。
                grasp_pose = (
                    self.grasp_msg.poses[
                        candidate_index
                    ]
                )

                record["grasp_tool_pose"] = {
                    "frame_id": str(
                        self.grasp_msg.header.frame_id
                    ),
                    "position": {
                        "x": float(
                            grasp_pose.position.x
                        ),
                        "y": float(
                            grasp_pose.position.y
                        ),
                        "z": float(
                            grasp_pose.position.z
                        ),
                    },
                    "orientation": {
                        "x": float(
                            grasp_pose.orientation.x
                        ),
                        "y": float(
                            grasp_pose.orientation.y
                        ),
                        "z": float(
                            grasp_pose.orientation.z
                        ),
                        "w": float(
                            grasp_pose.orientation.w
                        ),
                    },
                }

            candidate_records.append(
                record
            )

        p0_record = {
            "p0_index": int(
                p0_index
            ),
            "batch_stamp": {
                "secs": int(
                    batch_secs
                ),
                "nsecs": int(
                    batch_nsecs
                ),
            },
            "candidate_count": int(
                count
            ),
            "feasible_count": int(
                len(success)
            ),
            "steps_per_candidate": int(
                self.steps
            ),
            "candidates": candidate_records,
        }

        self.result_history[
            int(p0_index)
        ] = p0_record

        aggregate = {
            "schema_version": 1,
            "p0_results": [
                self.result_history[k]
                for k in sorted(
                    self.result_history.keys()
                )
            ],
        }

        result_msg = String()
        result_msg.data = json.dumps(
            aggregate,
            ensure_ascii=False,
            sort_keys=True,
        )

        self.result_pub.publish(
            result_msg
        )

        print()
        print(
            "Published structured feasibility results: "
            "P0[{}], stored P0 count={}".format(
                p0_index,
                len(
                    self.result_history
                ),
            )
        )

        # このP0の診断完了を記録してCandidate側へACK
        # RESULTを先にpublishし、その後DONEを送る。
        self.last_processed_key = batch_key
        self.processing = False

        done_msg = UInt64MultiArray()
        done_msg.data = [
            int(p0_index),
            int(batch_secs),
            int(batch_nsecs),
        ]

        self.done_pub.publish(
            done_msg
        )

        print()
        print(
            "Published feasibility DONE: "
            "P0[{}]".format(
                p0_index
            )
        )


def main():
    rospy.init_node(
        "cobotta_pregrasp_feasibility_diagnostic"
    )

    Diagnostic()

    rospy.spin()


if __name__ == "__main__":
    main()
