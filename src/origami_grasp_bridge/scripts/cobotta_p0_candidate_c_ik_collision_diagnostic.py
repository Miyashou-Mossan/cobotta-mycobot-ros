#!/usr/bin/env python3

import copy
import csv
import math
import os
import sys
from collections import Counter

import moveit_commander
import rospy

from geometry_msgs.msg import PoseArray, PoseStamped
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import (
    GetPositionIK,
    GetPositionIKRequest,
    GetStateValidity,
    GetStateValidityRequest,
)


DEFAULT_GROUP = "cobotta_arm"
DEFAULT_TIP = "cobotta_tool_link"
DEFAULT_FRAME = "paper_center"

DEFAULT_IK_TIMEOUT = 0.10
DEFAULT_MAX_JUMP_DEG = 10.0
DEFAULT_P0_COUNT = 7

DEFAULT_OUTPUT_CSV = os.path.expanduser(
    "~/phase1c_candidate_c_ik_collision_diagnostic.csv"
)


class CandidateCDiagnostic:

    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)

        self.group_name = rospy.get_param(
            "~group",
            DEFAULT_GROUP,
        )

        self.tip = rospy.get_param(
            "~tip",
            DEFAULT_TIP,
        )

        self.frame = rospy.get_param(
            "~frame",
            DEFAULT_FRAME,
        )

        self.ik_timeout = float(
            rospy.get_param(
                "~ik_timeout",
                DEFAULT_IK_TIMEOUT,
            )
        )

        self.max_jump_deg = float(
            rospy.get_param(
                "~max_jump_deg",
                DEFAULT_MAX_JUMP_DEG,
            )
        )

        self.max_jump_rad = math.radians(
            self.max_jump_deg
        )

        self.p0_count = int(
            rospy.get_param(
                "~p0_count",
                DEFAULT_P0_COUNT,
            )
        )

        self.output_csv = os.path.expanduser(
            rospy.get_param(
                "~output_csv",
                DEFAULT_OUTPUT_CSV,
            )
        )

        self.robot = moveit_commander.RobotCommander()

        self.group = moveit_commander.MoveGroupCommander(
            self.group_name,
            wait_for_servers=20.0,
        )

        self.active_joints = list(
            self.group.get_active_joints()
        )

        rospy.wait_for_service(
            "/compute_ik",
            timeout=30.0,
        )

        rospy.wait_for_service(
            "/check_state_validity",
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

        rospy.loginfo(
            "Phase1-C Candidate C IK/Collision diagnostic started."
        )

        rospy.loginfo(
            "group=%s tip=%s frame=%s",
            self.group_name,
            self.tip,
            self.frame,
        )

        rospy.loginfo(
            "IK timeout=%.3f s | max joint jump=%.1f deg",
            self.ik_timeout,
            self.max_jump_deg,
        )

        rospy.loginfo(
            "active joints=%s",
            str(self.active_joints),
        )

    def active_positions(self, state):
        lookup = dict(zip(
            state.joint_state.name,
            state.joint_state.position,
        ))

        missing = [
            name
            for name in self.active_joints
            if name not in lookup
        ]

        if missing:
            raise RuntimeError(
                "RobotState is missing active joints: "
                + str(missing)
            )

        return [
            float(lookup[name])
            for name in self.active_joints
        ]

    def solve_ik(
        self,
        pose_stamped,
        seed_state,
    ):
        req = GetPositionIKRequest()

        req.ik_request.group_name = self.group_name
        req.ik_request.ik_link_name = self.tip
        req.ik_request.pose_stamped = pose_stamped

        req.ik_request.robot_state = copy.deepcopy(
            seed_state
        )

        # FINISH340と同様に、
        # IKとCollision判定は分離する。
        req.ik_request.avoid_collisions = False

        req.ik_request.timeout = rospy.Duration(
            self.ik_timeout
        )

        res = self.compute_ik(req)

        if (
            res.error_code.val
            != MoveItErrorCodes.SUCCESS
        ):
            return None, res.error_code.val

        return res.solution, res.error_code.val

    def check_collision(self, state):
        req = GetStateValidityRequest()

        req.robot_state = state
        req.group_name = self.group_name

        result = self.check_validity(req)

        collision_pairs = []

        if not result.valid:
            for contact in result.contacts:
                pair = tuple(sorted([
                    contact.contact_body_1,
                    contact.contact_body_2,
                ]))

                collision_pairs.append(pair)

        return result.valid, collision_pairs

    @staticmethod
    def max_joint_delta(
        previous_joints,
        current_joints,
    ):
        if previous_joints is None:
            return 0.0

        return max(
            abs(a - b)
            for a, b in zip(
                previous_joints,
                current_joints,
            )
        )

    def diagnose_trajectory(
        self,
        p0_index,
        msg,
        initial_seed,
    ):
        seed_state = copy.deepcopy(
            initial_seed
        )

        previous_joints = None

        valid_count = 0
        prefix_count = 0
        prefix_broken = False

        ik_fail_count = 0
        collision_count = 0
        joint_jump_count = 0

        first_invalid_index = None
        first_invalid_reason = ""

        max_joint_step_rad = 0.0

        collision_counter = Counter()

        pose_count = len(msg.poses)

        print()
        print(
            "===== P0[{}] : {} poses =====".format(
                p0_index,
                pose_count,
            )
        )

        for index, pose in enumerate(msg.poses):
            target = PoseStamped()

            target.header.stamp = rospy.Time(0)
            target.header.frame_id = (
                msg.header.frame_id
                if msg.header.frame_id
                else self.frame
            )

            target.pose = pose

            solution, error_code = self.solve_ik(
                target,
                seed_state,
            )

            if solution is None:
                ik_fail_count += 1

                reasons = [
                    "IK_FAIL({})".format(
                        error_code
                    )
                ]

                is_valid = False

                if first_invalid_index is None:
                    first_invalid_index = index
                    first_invalid_reason = "+".join(
                        reasons
                    )

                prefix_broken = True

                print(
                    "index {:3d}: IK_FAIL error={}"
                    .format(
                        index,
                        error_code,
                    )
                )

                # IKが無いのでseedは前回の解を維持して、
                # 後続点の診断を継続する。
                continue

            current_joints = self.active_positions(
                solution
            )

            collision_free, collision_pairs = (
                self.check_collision(
                    solution
                )
            )

            if not collision_free:
                collision_count += 1

                for pair in collision_pairs:
                    collision_counter[pair] += 1

            # index 0は
            # 現在姿勢→STARTのjumpを評価しない。
            if previous_joints is None:
                jump_rad = 0.0
                jump_invalid = False
            else:
                jump_rad = self.max_joint_delta(
                    previous_joints,
                    current_joints,
                )

                jump_invalid = (
                    jump_rad > self.max_jump_rad
                )

                max_joint_step_rad = max(
                    max_joint_step_rad,
                    jump_rad,
                )

            if jump_invalid:
                joint_jump_count += 1

            reasons = []

            if not collision_free:
                reasons.append(
                    "COLLISION"
                )

            if jump_invalid:
                reasons.append(
                    "JOINT_JUMP({:.3f}deg)"
                    .format(
                        math.degrees(jump_rad)
                    )
                )

            point_valid = (
                collision_free
                and not jump_invalid
            )

            if point_valid:
                valid_count += 1

                if not prefix_broken:
                    prefix_count += 1
            else:
                if first_invalid_index is None:
                    first_invalid_index = index
                    first_invalid_reason = "+".join(
                        reasons
                    )

                prefix_broken = True

                pair_text = ""

                if collision_pairs:
                    pair_text = " pairs=" + ",".join(
                        [
                            "{}<->{}".format(
                                a,
                                b,
                            )
                            for a, b
                            in sorted(set(collision_pairs))
                        ]
                    )

                print(
                    "index {:3d}: {}{}"
                    .format(
                        index,
                        "+".join(reasons),
                        pair_text,
                    )
                )

            # 診断ではCollisionしていても
            # IK枝の追跡を続けるため、
            # 得られたIK解を次点seedにする。
            seed_state = copy.deepcopy(
                solution
            )

            previous_joints = current_joints

        if first_invalid_index is None:
            first_invalid_index_text = "NONE"
            first_invalid_reason = "PASS"
        else:
            first_invalid_index_text = str(
                first_invalid_index
            )

        collision_summary = []

        for pair, count in (
            collision_counter.most_common()
        ):
            collision_summary.append(
                "{}<->{}:{}".format(
                    pair[0],
                    pair[1],
                    count,
                )
            )

        print()
        print(
            "P0[{}] SUMMARY".format(
                p0_index
            )
        )

        print(
            "  valid                 = {}/{}"
            .format(
                valid_count,
                pose_count,
            )
        )

        print(
            "  continuous valid prefix = {}"
            .format(
                prefix_count
            )
        )

        print(
            "  first invalid index   = {}"
            .format(
                first_invalid_index_text
            )
        )

        print(
            "  first invalid reason  = {}"
            .format(
                first_invalid_reason
            )
        )

        print(
            "  IK_FAIL               = {}"
            .format(
                ik_fail_count
            )
        )

        print(
            "  COLLISION             = {}"
            .format(
                collision_count
            )
        )

        print(
            "  JOINT_JUMP            = {}"
            .format(
                joint_jump_count
            )
        )

        print(
            "  max joint step        = {:.3f} deg"
            .format(
                math.degrees(
                    max_joint_step_rad
                )
            )
        )

        if collision_summary:
            print(
                "  collision pairs       = "
                + " | ".join(
                    collision_summary
                )
            )
        else:
            print(
                "  collision pairs       = none"
            )

        return {
            "p0_index": p0_index,
            "pose_count": pose_count,
            "valid_count": valid_count,
            "continuous_valid_prefix": (
                prefix_count
            ),
            "first_invalid_index": (
                first_invalid_index_text
            ),
            "first_invalid_reason": (
                first_invalid_reason
            ),
            "ik_fail_count": ik_fail_count,
            "collision_count": (
                collision_count
            ),
            "joint_jump_count": (
                joint_jump_count
            ),
            "max_joint_step_deg": (
                math.degrees(
                    max_joint_step_rad
                )
            ),
            "collision_pairs": (
                " | ".join(
                    collision_summary
                )
            ),
        }

    def run(self):
        initial_seed = (
            self.robot.get_current_state()
        )

        # 初期RobotStateが使えるか確認する。
        initial_joints = self.active_positions(
            initial_seed
        )

        print()
        print(
            "===== Phase1-C Candidate C "
            "IK/Collision diagnostic ====="
        )

        print(
            "Initial seed joints [deg] ="
        )

        print(
            "  "
            + ", ".join(
                "{:.3f}".format(
                    math.degrees(q)
                )
                for q in initial_joints
            )
        )

        trajectories = []

        common_stamp = None

        for p0_index in range(
            self.p0_count
        ):
            topic = (
                "/origami/"
                "cobotta_p0_candidate_c_trajectory/"
                "lower_{}"
                .format(p0_index)
            )

            rospy.loginfo(
                "Waiting for %s",
                topic,
            )

            msg = rospy.wait_for_message(
                topic,
                PoseArray,
                timeout=15.0,
            )

            stamp = msg.header.stamp.to_nsec()

            if common_stamp is None:
                common_stamp = stamp
            elif stamp != common_stamp:
                rospy.logwarn(
                    "trajectory stamp mismatch: "
                    "P0[%d]",
                    p0_index,
                )

            trajectories.append(
                msg
            )

        results = []

        for p0_index, msg in enumerate(
            trajectories
        ):
            result = self.diagnose_trajectory(
                p0_index,
                msg,
                initial_seed,
            )

            results.append(
                result
            )

        with open(
            self.output_csv,
            "w",
            newline="",
        ) as f:
            fieldnames = [
                "p0_index",
                "pose_count",
                "valid_count",
                "continuous_valid_prefix",
                "first_invalid_index",
                "first_invalid_reason",
                "ik_fail_count",
                "collision_count",
                "joint_jump_count",
                "max_joint_step_deg",
                "collision_pairs",
            ]

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for result in results:
                writer.writerow(
                    result
                )

        print()
        print(
            "========================================"
        )
        print(
            " Phase1-C Candidate C FINAL SUMMARY"
        )
        print(
            "========================================"
        )

        for result in results:
            print(
                "P0[{p0_index}] "
                "valid={valid_count}/{pose_count} "
                "prefix={continuous_valid_prefix} "
                "first={first_invalid_index} "
                "{first_invalid_reason} "
                "IK={ik_fail_count} "
                "COL={collision_count} "
                "JUMP={joint_jump_count} "
                "maxStep={max_joint_step_deg:.3f}deg"
                .format(**result)
            )

        print()
        print(
            "CSV saved:"
        )
        print(
            self.output_csv
        )


if __name__ == "__main__":
    rospy.init_node(
        "cobotta_p0_candidate_c_ik_collision_diagnostic"
    )

    diagnostic = CandidateCDiagnostic()

    diagnostic.run()
