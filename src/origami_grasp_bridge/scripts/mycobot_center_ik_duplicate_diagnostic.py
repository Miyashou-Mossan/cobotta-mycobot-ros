#!/usr/bin/env python3

import copy

import rospy
import moveit_commander

from moveit_msgs.srv import (
    GetPositionFK,
    GetPositionFKRequest,
    GetPositionIK,
    GetStateValidity,
)

import mycobot_crease_path_optimizer as opt


THRESHOLDS_DEG = [
    0.01,
    0.1,
    1.0,
    5.0,
]


def count_unique_solutions(
    m,
    solutions,
    joint_names,
    threshold_deg
):
    clusters = []

    for solution in solutions:

        matched = False

        for cluster in clusters:

            distance = m.joint_distance_deg(
                solution["joints"],
                cluster["representative"],
                joint_names
            )

            if distance < threshold_deg:
                cluster["count"] += 1

                if solution["valid"]:
                    cluster["valid_count"] += 1
                else:
                    cluster["collision_count"] += 1
                    cluster["collision_texts"].add(
                        solution["collision"]
                    )

                cluster["members"].append(
                    (
                        solution["j1_offset_deg"],
                        solution["j2_offset_deg"]
                    )
                )
                matched = True
                break

        if not matched:
            clusters.append(
                {
                    "representative":
                        copy.deepcopy(
                            solution["joints"]
                        ),
                    "count": 1,
                    "valid_count":
                        1 if solution["valid"] else 0,
                    "collision_count":
                        0 if solution["valid"] else 1,
                    "collision_texts":
                        set(
                            [solution["collision"]]
                            if not solution["valid"]
                            else []
                        ),
                    "members": [
                        (
                            solution["j1_offset_deg"],
                            solution["j2_offset_deg"]
                        )
                    ],
                }
            )

    return clusters


def main():

    moveit_commander.roscpp_initialize([])

    rospy.init_node(
        "mycobot_center_ik_duplicate_diagnostic"
    )

    m = opt.load_base_module()

    # --------------------------------------------------------
    # Same fold-line parameters as optimizer
    # --------------------------------------------------------

    z_mm = rospy.get_param(
        "~z_mm",
        opt.DEFAULT_Z_MM
    )

    x0_mm = rospy.get_param(
        "~x0_mm",
        opt.X0_MM
    )

    y0_mm = rospy.get_param(
        "~y0_mm",
        opt.Y0_MM
    )

    x1_mm = rospy.get_param(
        "~x1_mm",
        opt.X1_MM
    )

    y1_mm = rospy.get_param(
        "~y1_mm",
        opt.Y1_MM
    )

    rx_deg = rospy.get_param(
        "~rx_deg",
        0.0
    )

    ry_deg = rospy.get_param(
        "~ry_deg",
        0.0
    )

    rz_deg = rospy.get_param(
        "~rz_deg",
        0.0
    )

    m.START["x"] = x0_mm / 1000.0
    m.START["y"] = y0_mm / 1000.0
    m.START["z"] = z_mm / 1000.0

    m.END["x"] = x1_mm / 1000.0
    m.END["y"] = y1_mm / 1000.0
    m.END["z"] = z_mm / 1000.0

    robot = moveit_commander.RobotCommander()

    group = moveit_commander.MoveGroupCommander(
        m.GROUP_NAME
    )

    joint_names = group.get_active_joints()

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

    # --------------------------------------------------------
    # Reference orientation
    # --------------------------------------------------------

    reference_state = m.make_robot_state(
        robot,
        m.GOOD_SEED_JOINTS
    )

    fk_req = GetPositionFKRequest()

    fk_req.header.frame_id = m.FRAME_ID

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

    orientation = opt.make_local_orientation(
        base_orientation,
        rx_deg,
        ry_deg,
        rz_deg
    )

    center_position = m.point_position(
        opt.CENTER_INDEX
    )

    # --------------------------------------------------------
    # 35-seed CENTER IK test
    # --------------------------------------------------------

    solutions = []

    ik_success = 0
    ik_fail = 0

    print("")
    print(
        "=============================================="
    )
    print(
        " CENTER IK duplicate diagnostic"
    )
    print(
        "=============================================="
    )

    print(
        "Z           : %.1f mm"
        % z_mm
    )

    print(
        "orientation : (%+.1f, %+.1f, %+.1f) deg"
        % (
            rx_deg,
            ry_deg,
            rz_deg
        )
    )

    print(
        "seed count  : %d x %d = %d"
        % (
            len(m.J1_OFFSETS_DEG),
            len(m.J2_OFFSETS_DEG),
            len(m.J1_OFFSETS_DEG)
            * len(m.J2_OFFSETS_DEG)
        )
    )

    print("")

    for j1_offset in m.J1_OFFSETS_DEG:

        for j2_offset in m.J2_OFFSETS_DEG:

            seed_state = opt.build_seed_state(
                m,
                robot,
                j1_offset,
                j2_offset
            )

            ik_res = m.solve_ik(
                compute_ik,
                center_position,
                orientation,
                seed_state
            )

            if ik_res.error_code.val != 1:

                ik_fail += 1
                continue

            ik_success += 1

            state = copy.deepcopy(
                ik_res.solution
            )

            joints = m.get_joint_dict(
                state,
                joint_names
            )

            validity = m.check_state(
                check_validity,
                state
            )

            solutions.append(
                {
                    "j1_offset_deg":
                        j1_offset,

                    "j2_offset_deg":
                        j2_offset,

                    "joints":
                        copy.deepcopy(
                            joints
                        ),

                    "valid":
                        validity.valid,

                    "collision":
                        opt.collision_text(
                            validity
                        ),
                }
            )

    print(
        "IK success : %d"
        % ik_success
    )

    print(
        "IK fail    : %d"
        % ik_fail
    )

    print("")

    if len(solutions) == 0:

        print(
            "No IK solutions."
        )
        return

    for threshold_deg in THRESHOLDS_DEG:

        clusters = count_unique_solutions(
            m,
            solutions,
            joint_names,
            threshold_deg
        )

        cluster_sizes = sorted(
            [
                c["count"]
                for c in clusters
            ],
            reverse=True
        )

        print(
            "threshold %5.2f deg : "
            "%2d unique / %2d IK solutions"
            % (
                threshold_deg,
                len(clusters),
                len(solutions)
            )
        )

        print(
            "                     cluster sizes = %s"
            % cluster_sizes
        )

        if threshold_deg == 0.01:

            for i, cluster in enumerate(
                clusters,
                start=1
            ):

                print(
                    "                     cluster %d: %s"
                    % (
                        i,
                        cluster["members"]
                    )
                )

                print(
                    "                                valid=%d collision=%d"
                    % (
                        cluster["valid_count"],
                        cluster["collision_count"]
                    )
                )

                if cluster["collision_texts"]:

                    print(
                        "                                collision types=%s"
                        % sorted(
                            cluster["collision_texts"]
                        )
                    )

    print("")
    print(
        "Diagnostic finished."
    )


if __name__ == "__main__":
    main()
