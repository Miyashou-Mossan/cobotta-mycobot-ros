#!/usr/bin/env python3

import os
import copy
import math
import importlib.util
from collections import Counter

import rospy
import moveit_commander

from geometry_msgs.msg import Quaternion
from moveit_msgs.srv import (
    GetPositionFK,
    GetPositionFKRequest,
    GetPositionIK,
    GetStateValidity,
)

from tf.transformations import (
    quaternion_from_euler,
    quaternion_multiply,
)
from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

# ============================================================
# Default search settings
# ============================================================

DEFAULT_Z_MM = 20.0

# Current diagonal crease
X0_MM = -75.0
Y0_MM = -50.0

X1_MM = 75.0
Y1_MM = 50.0

NUM_POINTS = 41
CENTER_INDEX = 20

# Orientation search.
# Start coarse. All combinations are automatically searched.
ANGLE_VALUES_DEG = [
    -60.0,
    -50.0,
    -40.0,
    -30.0,
    -20.0,
    -15.0,
    -10.0,
    -5.0,
    0.0,
    5.0,
    10.0,
    15.0,
    20.0,
    30.0,
    40.0,
    50.0,
    60.0,
]

JUMP_LIMIT_DEG = 30.0

# CENTER IK solutions whose maximum joint-angle difference
# is below this value are treated as the same IK branch.
CENTER_DUPLICATE_THRESHOLD_DEG = 0.01


def load_base_module():

    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "mycobot_crease_orientation_sweep.py"
    )

    spec = importlib.util.spec_from_file_location(
        "crease_base",
        path
    )

    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    return m


def make_local_orientation(
    base_orientation,
    rx_deg,
    ry_deg,
    rz_deg
):

    q_base = [
        base_orientation.x,
        base_orientation.y,
        base_orientation.z,
        base_orientation.w,
    ]

    q_delta = quaternion_from_euler(
        math.radians(rx_deg),
        math.radians(ry_deg),
        math.radians(rz_deg)
    )

    # Local rotation
    q_new = quaternion_multiply(
        q_base,
        q_delta
    )

    q = Quaternion()

    q.x = q_new[0]
    q.y = q_new[1]
    q.z = q_new[2]
    q.w = q_new[3]

    return q


def collision_text(validity):

    pairs = set()

    for contact in validity.contacts:

        pairs.add(
            (
                contact.contact_body_1,
                contact.contact_body_2
            )
        )

    return "; ".join(
        "%s<->%s" % p
        for p in sorted(pairs)
    )


def build_seed_state(
    m,
    robot,
    j1_offset_deg,
    j2_offset_deg
):

    joints = copy.deepcopy(
        m.GOOD_SEED_JOINTS
    )

    for name, offset in [
        ("mycobot_joint1", j1_offset_deg),
        ("mycobot_joint2", j2_offset_deg),
    ]:

        lower, upper = m.JOINT_LIMITS[name]

        value = (
            m.GOOD_SEED_JOINTS[name]
            + math.radians(offset)
        )

        joints[name] = m.clamp(
            value,
            lower + 0.01,
            upper - 0.01
        )

    return m.make_robot_state(
        robot,
        joints
    )


def evaluate_full_branch(
    m,
    center_state,
    orientation,
    indices,
    compute_ik,
    check_validity,
    joint_names
):

    previous_state = copy.deepcopy(
        center_state
    )

    previous_joints = m.get_joint_dict(
        center_state,
        joint_names
    )

    max_delta = 0.0

    min_margin, limiting_joint = (
        m.min_margin_from_joints(
            previous_joints,
            joint_names
        )
    )

    limiting_index = CENTER_INDEX

    states = [
        copy.deepcopy(center_state)
    ]

    for index in indices:

        ik_res = m.solve_ik(
            compute_ik,
            m.point_position(index),
            orientation,
            previous_state
        )

        if ik_res.error_code.val != 1:

            return {
                "pass": False,
                "category": "PATH_IK_FAIL",
                "detail":
                    "P%d code=%d"
                    % (
                        index,
                        ik_res.error_code.val
                    )
            }

        state = copy.deepcopy(
            ik_res.solution
        )

        validity = m.check_state(
            check_validity,
            state
        )

        if not validity.valid:

            return {
                "pass": False,
                "category": "PATH_COLLISION",
                "detail":
                    "P%d %s"
                    % (
                        index,
                        collision_text(
                            validity
                        )
                    )
            }

        joints = m.get_joint_dict(
            state,
            joint_names
        )

        point_max_delta = 0.0

        for name in joint_names:

            delta = m.wrap_to_pi(
                joints[name]
                - previous_joints[name]
            )

            delta_deg = abs(
                math.degrees(delta)
            )

            point_max_delta = max(
                point_max_delta,
                delta_deg
            )

        if point_max_delta >= JUMP_LIMIT_DEG:

            return {
                "pass": False,
                "category": "JOINT_JUMP",
                "detail":
                    "P%d %.3f deg"
                    % (
                        index,
                        point_max_delta
                    )
            }

        max_delta = max(
            max_delta,
            point_max_delta
        )

        margin, margin_joint = (
            m.min_margin_from_joints(
                joints,
                joint_names
            )
        )

        if margin < min_margin:

            min_margin = margin
            limiting_joint = margin_joint
            limiting_index = index

        previous_state = copy.deepcopy(
            state
        )

        previous_joints = copy.deepcopy(
            joints
        )

        states.append(
            copy.deepcopy(state)
        )

    return {
        "pass": True,

        "category": "PASS",

        "max_delta_deg":
            max_delta,

        "min_margin_deg":
            min_margin,

        "limiting_joint":
            limiting_joint,

        "limiting_index":
            limiting_index,

        "states":
            states,
    }


def candidate_score(result):

    # First maximize joint-limit margin.
    # Then minimize adjacent joint motion.
    return (
        result["min_margin_deg"],
        -result["max_delta_deg"],
    )


def search_branch(
    branch_name,
    m,
    robot,
    base_orientation,
    indices,
    compute_ik,
    check_validity,
    joint_names
):

    print("")
    print(
        "===== Optimizing %s ====="
        % branch_name
    )

    best = None

    stats = Counter()

    collision_stats = Counter()

    orientation_count = (
        len(ANGLE_VALUES_DEG) ** 3
    )

    seed_count = (
        len(m.J1_OFFSETS_DEG)
        * len(m.J2_OFFSETS_DEG)
    )

    total_candidates = (
        orientation_count
        * seed_count
    )

    print(
        "orientation candidates : %d"
        % orientation_count
    )

    print(
        "seed candidates        : %d"
        % seed_count
    )

    print(
        "maximum combinations   : %d"
        % total_candidates
    )

    center_position = m.point_position(
        CENTER_INDEX
    )

    tested = 0
    center_unique_count = 0
    center_valid_count = 0
    full_pass_count = 0

    for rx in ANGLE_VALUES_DEG:

        for ry in ANGLE_VALUES_DEG:

            for rz in ANGLE_VALUES_DEG:

                orientation = (
                    make_local_orientation(
                        base_orientation,
                        rx,
                        ry,
                        rz
                    )
                )

                # ====================================
                # STEP 1:
                # Solve CENTER IK for all seeds.
                # Keep only unique CENTER IK branches.
                # ====================================

                unique_center_candidates = []

                for j1_offset in (
                    m.J1_OFFSETS_DEG
                ):

                    for j2_offset in (
                        m.J2_OFFSETS_DEG
                    ):

                        tested += 1

                        seed_state = (
                            build_seed_state(
                                m,
                                robot,
                                j1_offset,
                                j2_offset
                            )
                        )

                        ik_res = m.solve_ik(
                            compute_ik,
                            center_position,
                            orientation,
                            seed_state
                        )

                        if (
                            ik_res.error_code.val
                            != 1
                        ):

                            stats[
                                "CENTER_IK_FAIL"
                            ] += 1

                            continue

                        center_state = (
                            copy.deepcopy(
                                ik_res.solution
                            )
                        )

                        center_joints = (
                            m.get_joint_dict(
                                center_state,
                                joint_names
                            )
                        )

                        duplicate = False

                        for candidate in (
                            unique_center_candidates
                        ):

                            distance_deg = (
                                m.joint_distance_deg(
                                    center_joints,
                                    candidate[
                                        "center_joints"
                                    ],
                                    joint_names
                                )
                            )

                            if (
                                distance_deg
                                <
                                CENTER_DUPLICATE_THRESHOLD_DEG
                            ):

                                duplicate = True
                                break

                        if duplicate:

                            stats[
                                "CENTER_DUPLICATE"
                            ] += 1

                            continue

                        unique_center_candidates.append(
                            {
                                "center_state":
                                    copy.deepcopy(
                                        center_state
                                    ),

                                "center_joints":
                                    copy.deepcopy(
                                        center_joints
                                    ),

                                "j1_offset_deg":
                                    j1_offset,

                                "j2_offset_deg":
                                    j2_offset,
                            }
                        )

                # ====================================
                # STEP 2:
                # Collision and PATH evaluation are
                # done only for unique CENTER IK.
                # ====================================

                center_unique_count += len(
                    unique_center_candidates
                )

                for candidate in (
                    unique_center_candidates
                ):

                    center_state = copy.deepcopy(
                        candidate["center_state"]
                    )

                    validity = m.check_state(
                        check_validity,
                        center_state
                    )

                    if not validity.valid:

                        stats[
                            "CENTER_COLLISION"
                        ] += 1

                        collision_stats[
                            collision_text(
                                validity
                            )
                        ] += 1

                        continue

                    center_valid_count += 1

                    result = evaluate_full_branch(
                        m,
                        center_state,
                        orientation,
                        indices,
                        compute_ik,
                        check_validity,
                        joint_names
                    )

                    if not result["pass"]:

                        stats[
                            result["category"]
                        ] += 1

                        if (
                            result["category"]
                            == "PATH_COLLISION"
                        ):

                            collision_stats[
                                result["detail"]
                            ] += 1

                        continue

                    full_pass_count += 1
                    stats["PASS"] += 1

                    result["rx_deg"] = rx
                    result["ry_deg"] = ry
                    result["rz_deg"] = rz

                    result[
                        "j1_offset_deg"
                    ] = candidate[
                        "j1_offset_deg"
                    ]

                    result[
                        "j2_offset_deg"
                    ] = candidate[
                        "j2_offset_deg"
                    ]

                    result[
                        "center_state"
                    ] = center_state

                    if best is None:

                        best = result

                    elif (
                        candidate_score(
                            result
                        )
                        >
                        candidate_score(
                            best
                        )
                    ):

                        best = result

                # progress after each orientation
                print(
                    "\r%s progress: %d / %d orientations"
                    % (
                        branch_name,
                        (
                            ANGLE_VALUES_DEG.index(rx)
                            * len(ANGLE_VALUES_DEG)
                            * len(ANGLE_VALUES_DEG)
                            +
                            ANGLE_VALUES_DEG.index(ry)
                            * len(ANGLE_VALUES_DEG)
                            +
                            ANGLE_VALUES_DEG.index(rz)
                            + 1
                        ),
                        orientation_count
                    ),
                    end="",
                    flush=True
                )

    print("")
    print("")
    print(
        "tested combinations : %d"
        % tested
    )

    print(
        "CENTER unique      : %d"
        % center_unique_count
    )

    print(
        "CENTER duplicate   : %d"
        % stats["CENTER_DUPLICATE"]
    )

    print(
        "CENTER valid       : %d"
        % center_valid_count
    )

    print(
        "FULL PASS          : %d"
        % full_pass_count
    )

    print("")
    print(
        "Failure summary:"
    )

    for key in [
        "CENTER_IK_FAIL",
        "CENTER_DUPLICATE",
        "CENTER_COLLISION",
        "PATH_IK_FAIL",
        "PATH_COLLISION",
        "JOINT_JUMP",
        "PASS",
    ]:

        print(
            "  %-18s : %d"
            % (
                key,
                stats[key]
            )
        )

    if best is None:

        print("")
        print(
            "BEST : NONE"
        )

        if len(collision_stats) > 0:

            print("")
            print(
                "Most common collisions:"
            )

            for text, count in (
                collision_stats.most_common(
                    5
                )
            ):

                print(
                    "  %5d x %s"
                    % (
                        count,
                        text
                    )
                )

        return None

    print("")
    print(
        "BEST solution:"
    )

    print(
        "  local_x       : %+6.1f deg"
        % best["rx_deg"]
    )

    print(
        "  local_y       : %+6.1f deg"
        % best["ry_deg"]
    )

    print(
        "  local_z       : %+6.1f deg"
        % best["rz_deg"]
    )

    print(
        "  seed J1 offset: %+6.1f deg"
        % best[
            "j1_offset_deg"
        ]
    )

    print(
        "  seed J2 offset: %+6.1f deg"
        % best[
            "j2_offset_deg"
        ]
    )

    print(
        "  min margin    : %.6f deg"
        % best[
            "min_margin_deg"
        ]
    )

    print(
        "  limiting joint: %s / P%s"
        % (
            best[
                "limiting_joint"
            ],
            str(
                best[
                    "limiting_index"
                ]
            )
        )
    )

    print(
        "  max delta     : %.6f deg"
        % best[
            "max_delta_deg"
        ]
    )

    return best

def publish_center_state(pub, state, joint_names):

    values = dict(
        zip(
            state.joint_state.name,
            state.joint_state.position
        )
    )

    traj = RobotTrajectory()

    traj.joint_trajectory.joint_names = list(
        joint_names
    )

    point = JointTrajectoryPoint()

    point.positions = [
        values[name]
        for name in joint_names
    ]

    point.velocities = [
        0.0
        for _ in joint_names
    ]

    point.accelerations = [
        0.0
        for _ in joint_names
    ]

    point.time_from_start = rospy.Duration(1.0)

    traj.joint_trajectory.points = [
        point
    ]

    msg = DisplayTrajectory()

    msg.model_id = "cobotta_mycobot_dual_robot"

    msg.trajectory_start = copy.deepcopy(
        state
    )

    msg.trajectory = [
        traj
    ]

    pub.publish(msg)

    rospy.sleep(1.0)

def main():

    moveit_commander.roscpp_initialize([])

    rospy.init_node(
        "mycobot_crease_path_optimizer"
    )

    m = load_base_module()

    # ========================================================
    # User parameters
    # ========================================================

    z_mm = rospy.get_param(
        "~z_mm",
        DEFAULT_Z_MM
    )

    x0_mm = rospy.get_param(
        "~x0_mm",
        X0_MM
    )

    y0_mm = rospy.get_param(
        "~y0_mm",
        Y0_MM
    )

    x1_mm = rospy.get_param(
        "~x1_mm",
        X1_MM
    )

    y1_mm = rospy.get_param(
        "~y1_mm",
        Y1_MM
    )

    # Update fold line used by shared module
    m.START["x"] = (
        x0_mm / 1000.0
    )

    m.START["y"] = (
        y0_mm / 1000.0
    )

    m.START["z"] = (
        z_mm / 1000.0
    )

    m.END["x"] = (
        x1_mm / 1000.0
    )

    m.END["y"] = (
        y1_mm / 1000.0
    )

    m.END["z"] = (
        z_mm / 1000.0
    )

    robot = moveit_commander.RobotCommander()

    group = moveit_commander.MoveGroupCommander(
        m.GROUP_NAME
    )

    joint_names = (
        group.get_active_joints()
    )

    rospy.wait_for_service(
        "/compute_fk"
    )

    rospy.wait_for_service(
        "/compute_ik"
    )

    rospy.wait_for_service(
        "/check_state_validity"
    )

    compute_fk = rospy.ServiceProxy(
        "/compute_fk",
        GetPositionFK
    )

    compute_ik = rospy.ServiceProxy(
        "/compute_ik",
        GetPositionIK
    )

    check_validity = rospy.ServiceProxy(
        "/check_state_validity",
        GetStateValidity
    )

    # ========================================================
    # Reference crease orientation
    # ========================================================

    reference_state = m.make_robot_state(
        robot,
        m.GOOD_SEED_JOINTS
    )

    fk_req = GetPositionFKRequest()

    fk_req.header.frame_id = (
        m.FRAME_ID
    )

    fk_req.fk_link_names = [
        m.TIP_LINK
    ]

    fk_req.robot_state = copy.deepcopy(
        reference_state
    )

    fk_res = compute_fk(
        fk_req
    )

    if fk_res.error_code.val != 1:

        print(
            "Reference FK failed."
        )

        return

    base_orientation = copy.deepcopy(
        fk_res.pose_stamped[0]
        .pose.orientation
    )

    print("")
    print(
        "=============================================="
    )
    print(
        " MyCobot automatic crease-path optimizer"
    )
    print(
        "=============================================="
    )

    print(
        "TCP    : %s"
        % m.TIP_LINK
    )

    print(
        "START  : (%.1f, %.1f, %.1f) mm"
        % (
            x0_mm,
            y0_mm,
            z_mm
        )
    )

    print(
        "CENTER : midpoint"
    )

    print(
        "END    : (%.1f, %.1f, %.1f) mm"
        % (
            x1_mm,
            y1_mm,
            z_mm
        )
    )

    print(
        "angle search : -15 to +15 deg / 5 deg"
    )

    print(
        "seed search  : %d x %d"
        % (
            len(
                m.J1_OFFSETS_DEG
            ),
            len(
                m.J2_OFFSETS_DEG
            )
        )
    )

    # ========================================================
    # LEFT and RIGHT independently optimize
    # ========================================================

    left_indices = list(
        range(
            CENTER_INDEX - 1,
            -1,
            -1
        )
    )

    right_indices = list(
        range(
            CENTER_INDEX + 1,
            NUM_POINTS
        )
    )

    left = search_branch(
        "LEFT P20 -> P0",
        m,
        robot,
        base_orientation,
        left_indices,
        compute_ik,
        check_validity,
        joint_names
    )

    right = search_branch(
        "RIGHT P20 -> P40",
        m,
        robot,
        base_orientation,
        right_indices,
        compute_ik,
        check_validity,
        joint_names
    )

    display_pub = rospy.Publisher(
   	 "/move_group/display_planned_path",
    	DisplayTrajectory,
    	queue_size=1,
    	latch=True
    )

    rospy.sleep(1.0)

    # ========================================================
    # Final result
    # ========================================================

    print("")
    print(
        "=============================================="
    )
    print(
        " FINAL OPTIMIZATION RESULT"
    )
    print(
        "=============================================="
    )

    if left is None:

        print(
            "LEFT  : NO SOLUTION"
        )

    else:

        print(
            "LEFT  : SOLUTION FOUND"
        )

        print(
            "        orientation "
            "(%+.1f, %+.1f, %+.1f) deg"
            % (
                left["rx_deg"],
                left["ry_deg"],
                left["rz_deg"]
            )
        )

        print(
            "        seed offsets "
            "J1=%+.1f J2=%+.1f deg"
            % (
                left[
                    "j1_offset_deg"
                ],
                left[
                    "j2_offset_deg"
                ]
            )
        )

        print(
            "        margin=%.3f deg "
            "max_delta=%.3f deg"
            % (
                left[
                    "min_margin_deg"
                ],
                left[
                    "max_delta_deg"
                ]
            )
        )

    if right is None:

        print(
            "RIGHT : NO SOLUTION"
        )

    else:

        print(
            "RIGHT : SOLUTION FOUND"
        )

        print(
            "        orientation "
            "(%+.1f, %+.1f, %+.1f) deg"
            % (
                right["rx_deg"],
                right["ry_deg"],
                right["rz_deg"]
            )
        )

        print(
            "        seed offsets "
            "J1=%+.1f J2=%+.1f deg"
            % (
                right[
                    "j1_offset_deg"
                ],
                right[
                    "j2_offset_deg"
                ]
            )
        )

        print(
            "        margin=%.3f deg "
            "max_delta=%.3f deg"
            % (
                right[
                    "min_margin_deg"
                ],
                right[
                    "max_delta_deg"
                ]
            )
        )

    print("")

    if (
        left is not None
        and right is not None
    ):

        print(
            "judgment : SOLUTION FOUND FOR BOTH BRANCHES"
        )

    else:

        print(
            "judgment : NO COMPLETE SOLUTION "
            "IN CURRENT SEARCH RANGE"
        )


    display_pub = rospy.Publisher(
        "/move_group/display_planned_path",
        DisplayTrajectory,
        queue_size=1,
        latch=True
    )

    rospy.sleep(1.0)


    if left is not None:

    	print("")
    	print("===== RViz LEFT best CENTER =====")

    	print("LEFT center joints:")

    	for name, value in zip(
            left["center_state"].joint_state.name,
            left["center_state"].joint_state.position
    	):
    		print(
                "  {} = {:.9f} rad".format(
                    name,
                    value
                )
    		)

    	publish_center_state(
            display_pub,
            left["center_state"],
            joint_names
    	)

    	input(
            "LEFT姿勢をRVizで確認したら Enter..."
    	)


    if right is not None:

    	print("")
    	print("===== RViz RIGHT best CENTER =====")

    	publish_center_state(
            display_pub,
            right["center_state"],
            joint_names
    	)

    	input(
            "RIGHT姿勢をRVizで確認したら Enter..."
    	)

if __name__ == "__main__":
    main()
