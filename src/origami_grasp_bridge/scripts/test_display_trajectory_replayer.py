#!/usr/bin/env python3

import copy
import rospy

from moveit_msgs.msg import DisplayTrajectory


INPUT_TOPIC = "/move_group/display_planned_path"
OUTPUT_TOPIC = "/origami/display_replay"

REPLAY_INTERVAL = 2.0


class DisplayTrajectoryReplayer:

    def __init__(self):

        self.latest_msg = None

        self.pub = rospy.Publisher(
            OUTPUT_TOPIC,
            DisplayTrajectory,
            queue_size=1,
            latch=True
        )

        self.sub = rospy.Subscriber(
            INPUT_TOPIC,
            DisplayTrajectory,
            self.callback,
            queue_size=1
        )

        rospy.loginfo(
            "Waiting for DisplayTrajectory on %s",
            INPUT_TOPIC
        )

    def callback(self, msg):

        self.latest_msg = copy.deepcopy(msg)

        rospy.loginfo(
            "Received DisplayTrajectory: %d trajectory(s)",
            len(msg.trajectory)
        )

        for i, traj in enumerate(msg.trajectory):

            point_count = len(
                traj.joint_trajectory.points
            )

            if point_count > 0:

                last_time = (
                    traj.joint_trajectory
                    .points[-1]
                    .time_from_start
                    .to_sec()
                )

            else:
                last_time = 0.0

            rospy.loginfo(
                "  trajectory[%d]: points=%d duration=%.3f s",
                i,
                point_count,
                last_time
            )

    def run(self):

        rate = rospy.Rate(10)

        next_publish = rospy.Time.now()

        while not rospy.is_shutdown():

            if (
                self.latest_msg is not None
                and rospy.Time.now() >= next_publish
            ):

                msg = copy.deepcopy(
                    self.latest_msg
                )

                self.pub.publish(msg)

                rospy.loginfo(
                    "Replay published to %s",
                    OUTPUT_TOPIC
                )

                next_publish = (
                    rospy.Time.now()
                    + rospy.Duration(
                        REPLAY_INTERVAL
                    )
                )

            rate.sleep()


def main():

    rospy.init_node(
        "display_trajectory_replayer"
    )

    node = DisplayTrajectoryReplayer()

    node.run()


if __name__ == "__main__":
    main()
