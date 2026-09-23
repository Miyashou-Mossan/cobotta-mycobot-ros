#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import threading
import time

import rospy

from std_msgs.msg import Bool


class SyncTimeoutError(RuntimeError):
    pass


def wait_for_bool_topic(
    topic,
    expected_value,
    timeout_sec,
):
    """
    この関数が呼ばれた後に届いた
    std_msgs/Bool を待つ。

    v1では安全のため、
    latched publisherは使用しない。
    """

    if not isinstance(topic, str) or not topic:
        raise ValueError(
            "topic must be a non-empty string"
        )

    if not isinstance(expected_value, bool):
        raise ValueError(
            "expected_value must be bool"
        )

    timeout_sec = float(timeout_sec)

    if (
        not math.isfinite(timeout_sec)
        or timeout_sec <= 0.0
    ):
        raise ValueError(
            "timeout_sec must be finite and > 0"
        )

    event = threading.Event()

    result = {
        "received_count": 0,
        "last_value": None,
        "matched": False,
        "error": None,
    }

    def callback(msg):
        connection_header = getattr(
            msg,
            "_connection_header",
            {},
        ) or {}

        # 過去値が即座に再送されるlatched Topicは
        # 「新しい完了通知のみ有効」というv1仕様に合わない。
        if str(
            connection_header.get(
                "latching",
                "0",
            )
        ) == "1":
            result["error"] = (
                "SYNC_EVENT does not accept "
                "latched publishers in v1"
            )
            event.set()
            return

        value = bool(msg.data)

        result["received_count"] += 1
        result["last_value"] = value

        if value == expected_value:
            result["matched"] = True
            event.set()

    subscriber = rospy.Subscriber(
        topic,
        Bool,
        callback,
        queue_size=1,
    )

    start = time.monotonic()

    rospy.loginfo(
        "SYNC_EVENT armed: topic=%s expected=%s timeout=%.3f sec",
        topic,
        expected_value,
        timeout_sec,
    )

    try:
        while not rospy.is_shutdown():

            if event.wait(timeout=0.05):
                break

            elapsed = (
                time.monotonic()
                - start
            )

            if elapsed >= timeout_sec:
                raise SyncTimeoutError(
                    "SYNC_EVENT timeout: "
                    "topic={} expected={} timeout={} sec".format(
                        topic,
                        expected_value,
                        timeout_sec,
                    )
                )

        if rospy.is_shutdown():
            raise RuntimeError(
                "ROS shutdown while waiting for SYNC_EVENT"
            )

        if result["error"] is not None:
            raise RuntimeError(
                result["error"]
            )

        if not result["matched"]:
            raise RuntimeError(
                "SYNC_EVENT ended without matching value"
            )

        elapsed = (
            time.monotonic()
            - start
        )

        return {
            "topic": topic,
            "expected_value": expected_value,
            "received_count":
                result["received_count"],
            "last_value":
                result["last_value"],
            "elapsed_sec":
                elapsed,
        }

    finally:
        subscriber.unregister()
