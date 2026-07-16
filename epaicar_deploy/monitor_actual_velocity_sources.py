#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import json
import time

import rospy
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String


class Monitor(object):
    def __init__(self):
        rospy.init_node("actual_velocity_source_monitor", anonymous=True)
        self.duration = float(rospy.get_param("~duration", 20.0))
        self.rate_hz = float(rospy.get_param("~rate_hz", 5.0))
        self.cmd = None
        self.feedback = None
        self.odom = None
        self.actual = None
        self.status = None

        rospy.Subscriber("/cmd_vel", Twist, self.cmd_cb, queue_size=20)
        rospy.Subscriber("/feedback_vel", Twist, self.feedback_cb, queue_size=20)
        rospy.Subscriber("/odom", Odometry, self.odom_cb, queue_size=20)
        rospy.Subscriber("/actual_velocity", TwistStamped, self.actual_cb, queue_size=20)
        rospy.Subscriber("/actual_velocity_status", String, self.status_cb, queue_size=20)

    def cmd_cb(self, msg):
        self.cmd = (rospy.get_time(), msg.linear.x, msg.angular.z)

    def feedback_cb(self, msg):
        self.feedback = (rospy.get_time(), msg.linear.x, msg.angular.z)

    def odom_cb(self, msg):
        self.odom = (rospy.get_time(), msg.twist.twist.linear.x, msg.twist.twist.angular.z)

    def actual_cb(self, msg):
        self.actual = (rospy.get_time(), msg.twist.linear.x, msg.twist.angular.z)

    def status_cb(self, msg):
        try:
            self.status = (rospy.get_time(), json.loads(msg.data))
        except Exception:
            self.status = (rospy.get_time(), {"raw": msg.data})

    def latest_pair(self, item):
        if item is None:
            return None, None, None
        return rospy.get_time() - item[0], item[1], item[2]

    def spin(self):
        end_time = time.time() + self.duration
        rate = rospy.Rate(self.rate_hz)
        while time.time() < end_time and not rospy.is_shutdown():
            cmd_age, cmd_v, cmd_w = self.latest_pair(self.cmd)
            fb_age, fb_v, fb_delta = self.latest_pair(self.feedback)
            odom_age, odom_v, odom_w = self.latest_pair(self.odom)
            act_age, act_v, act_w = self.latest_pair(self.actual)
            status_age = None
            source = None
            source_value = None
            if self.status is not None:
                status_age = rospy.get_time() - self.status[0]
                source = self.status[1].get("angular_source")
                source_value = self.status[1].get("source_value")
            print(
                "cmd(age={0}, v={1}, steer={2}) | feedback(age={3}, v={4}, steer={5}) | "
                "odom(age={6}, v={7}, wz={8}) | actual(age={9}, v={10}, wz={11}) | "
                "source={12} source_value={13} status_age={14}".format(
                    self.fmt(cmd_age), self.fmt(cmd_v), self.fmt(cmd_w),
                    self.fmt(fb_age), self.fmt(fb_v), self.fmt(fb_delta),
                    self.fmt(odom_age), self.fmt(odom_v), self.fmt(odom_w),
                    self.fmt(act_age), self.fmt(act_v), self.fmt(act_w),
                    source, source_value, self.fmt(status_age),
                )
            )
            rate.sleep()

    @staticmethod
    def fmt(value):
        if value is None:
            return "None"
        return "%.4f" % float(value)


if __name__ == "__main__":
    Monitor().spin()
