#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import csv
import json
import math
import os
import sys
import time
from collections import OrderedDict

import rospy
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import String


def as_bool(value):
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def stamp_to_sec(stamp):
    if stamp is None:
        return rospy.get_time()
    try:
        if stamp.to_sec() > 0.0:
            return stamp.to_sec()
    except Exception:
        pass
    return rospy.get_time()


def fmt(value, width=8, precision=3):
    if value is None:
        return " " * (width - 4) + "None"
    return ("%%%d.%df" % (width, precision)) % float(value)


class AngularSource(object):
    def __init__(self, name):
        self.name = name
        self.value = None
        self.time = None
        self.integral = 0.0
        self.samples = 0

    def update(self, value, stamp=None):
        self.value = float(value)
        self.time = stamp_to_sec(stamp)

    def fresh_value(self, now, max_age):
        if self.time is None:
            return None
        if now - self.time > max_age:
            return None
        return self.value

    def integrate(self, now, dt, max_age):
        value = self.fresh_value(now, max_age)
        if value is None:
            return
        self.integral += value * dt
        self.samples += 1


class ClockwiseTurnAngularTest(object):
    def __init__(self):
        rospy.init_node("clockwise_turn_angular_test", anonymous=True)

        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")
        self.actual_velocity_topic = rospy.get_param("~actual_velocity_topic", "/actual_velocity")
        self.imu_raw_topic = rospy.get_param("~imu_raw_topic", "/imu_raw_data")
        self.imu_data_topic = rospy.get_param("~imu_data_topic", "/imu_data")
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.feedback_vel_topic = rospy.get_param("~feedback_vel_topic", "/feedback_vel")
        self.status_topic = rospy.get_param("~status_topic", "/actual_velocity_status")

        self.angular_speed = abs(float(rospy.get_param("~angular_speed", 0.5)))
        self.turn_radians = abs(float(rospy.get_param("~turn_radians", 2.0 * math.pi)))
        self.direction = str(rospy.get_param("~direction", "cw")).strip().lower()
        self.clockwise = self.direction in ("cw", "clockwise", "right")
        self.command_w = -self.angular_speed if self.clockwise else self.angular_speed
        self.duration = self.turn_radians / max(self.angular_speed, 1e-6)

        self.rate_hz = float(rospy.get_param("~rate_hz", 30.0))
        self.print_rate_hz = float(rospy.get_param("~print_rate_hz", 5.0))
        self.max_age = float(rospy.get_param("~max_age", 0.35))
        self.countdown_sec = float(rospy.get_param("~countdown_sec", 3.0))
        self.pre_stop_sec = float(rospy.get_param("~pre_stop_sec", 1.0))
        self.post_stop_sec = float(rospy.get_param("~post_stop_sec", 2.0))
        self.wait_for_subscriber_sec = float(rospy.get_param("~wait_for_subscriber_sec", 5.0))
        self.csv_path = str(rospy.get_param("~csv_path", "/tmp/clockwise_turn_angular_test.csv")).strip()
        self.dry_run = as_bool(rospy.get_param("~dry_run", "false"))

        self.sources = OrderedDict()
        for name in ("cmd_topic", "actual", "imu_raw", "imu_data", "odom", "feedback"):
            self.sources[name] = AngularSource(name)

        self.status = {}
        self.command_integral = 0.0
        self.start_wall = None
        self.last_loop_time = None
        self.last_print_time = 0.0

        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        rospy.Subscriber(self.cmd_vel_topic, Twist, self.cmd_cb, queue_size=20)
        rospy.Subscriber(self.actual_velocity_topic, TwistStamped, self.actual_cb, queue_size=20)
        rospy.Subscriber(self.imu_raw_topic, Imu, self.imu_raw_cb, queue_size=50)
        rospy.Subscriber(self.imu_data_topic, Imu, self.imu_data_cb, queue_size=50)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=20)
        rospy.Subscriber(self.feedback_vel_topic, Twist, self.feedback_cb, queue_size=20)
        rospy.Subscriber(self.status_topic, String, self.status_cb, queue_size=20)

        self.csv_file = None
        self.csv_writer = None
        if self.csv_path:
            parent = os.path.dirname(self.csv_path)
            if parent and not os.path.isdir(parent):
                os.makedirs(parent)
            if sys.version_info[0] < 3:
                self.csv_file = open(self.csv_path, "wb")
            else:
                self.csv_file = open(self.csv_path, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                "t_sec", "phase", "w_cmd_out", "w_cmd_topic", "w_actual",
                "w_imu_raw", "w_imu_data", "w_odom", "w_feedback",
                "actual_source", "actual_raw_angular",
                "int_cmd_out", "int_cmd_topic", "int_actual", "int_imu_raw",
                "int_imu_data", "int_odom", "int_feedback",
            ])

    def cmd_cb(self, msg):
        self.sources["cmd_topic"].update(msg.angular.z)

    def actual_cb(self, msg):
        self.sources["actual"].update(msg.twist.angular.z, msg.header.stamp)

    def imu_raw_cb(self, msg):
        self.sources["imu_raw"].update(msg.angular_velocity.z, msg.header.stamp)

    def imu_data_cb(self, msg):
        self.sources["imu_data"].update(msg.angular_velocity.z, msg.header.stamp)

    def odom_cb(self, msg):
        self.sources["odom"].update(msg.twist.twist.angular.z, msg.header.stamp)

    def feedback_cb(self, msg):
        self.sources["feedback"].update(msg.angular.z)

    def status_cb(self, msg):
        try:
            self.status = json.loads(msg.data)
        except Exception:
            self.status = {"raw": msg.data}

    def make_twist(self, angular_z):
        msg = Twist()
        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = angular_z
        return msg

    def publish_cmd(self, angular_z):
        if not self.dry_run:
            self.cmd_pub.publish(self.make_twist(angular_z))

    def publish_stop(self, repeats=10):
        stop_msg = self.make_twist(0.0)
        rate = rospy.Rate(20.0)
        for _ in range(repeats):
            if rospy.is_shutdown():
                break
            if not self.dry_run:
                self.cmd_pub.publish(stop_msg)
            rate.sleep()

    def wait_for_cmd_subscriber(self):
        if self.dry_run or self.wait_for_subscriber_sec <= 0.0:
            return
        deadline = time.time() + self.wait_for_subscriber_sec
        rate = rospy.Rate(10.0)
        while time.time() < deadline and not rospy.is_shutdown():
            if self.cmd_pub.get_num_connections() > 0:
                return
            rate.sleep()
        rospy.logwarn("No subscriber detected on %s before test start", self.cmd_vel_topic)

    def integrate_sources(self, now, dt, command_w):
        self.command_integral += command_w * dt
        for source in self.sources.values():
            source.integrate(now, dt, self.max_age)

    def current_values(self, now):
        return OrderedDict((name, source.fresh_value(now, self.max_age))
                           for name, source in self.sources.items())

    def print_header(self):
        print("")
        print("Clockwise turn angular test")
        print("cmd_topic=%s actual=%s imu_raw=%s imu_data=%s odom=%s feedback=%s" % (
            self.cmd_vel_topic, self.actual_velocity_topic, self.imu_raw_topic,
            self.imu_data_topic, self.odom_topic, self.feedback_vel_topic,
        ))
        print("direction=%s command_w=%.4f rad/s target_angle=%.4f rad duration=%.2f s dry_run=%s" % (
            "cw" if self.clockwise else "ccw", self.command_w,
            -self.turn_radians if self.clockwise else self.turn_radians,
            self.duration, self.dry_run,
        ))
        if self.csv_path:
            print("csv=%s" % self.csv_path)
        print("")
        print("t phase     cmd_out cmd_top actual imu_raw imu_dat    odom feedbk src             raw_act int_cmd int_act int_imu int_odom")

    def emit_row(self, phase, command_w, force=False):
        now = rospy.get_time()
        if not force and now - self.last_print_time < (1.0 / max(self.print_rate_hz, 1e-6)):
            return
        self.last_print_time = now
        elapsed = now - self.start_wall if self.start_wall is not None else 0.0
        values = self.current_values(now)
        source = self.status.get("angular_source", "")
        raw_angular = self.status.get("raw_angular_radps", None)
        if raw_angular is None:
            source_value = self.status.get("source_value", {})
            if isinstance(source_value, dict):
                raw_angular = source_value.get("cmd_angular", None)

        print("%5.2f %-8s %s %s %s %s %s %s %s %-15s %s %s %s %s %s" % (
            elapsed,
            phase,
            fmt(command_w),
            fmt(values["cmd_topic"]),
            fmt(values["actual"]),
            fmt(values["imu_raw"]),
            fmt(values["imu_data"]),
            fmt(values["odom"]),
            fmt(values["feedback"]),
            str(source)[:15],
            fmt(raw_angular),
            fmt(self.command_integral),
            fmt(self.sources["actual"].integral),
            fmt(self.sources["imu_raw"].integral),
            fmt(self.sources["odom"].integral),
        ))

        if self.csv_writer is not None:
            self.csv_writer.writerow([
                "%.6f" % elapsed,
                phase,
                "%.9f" % command_w,
                self.csv_value(values["cmd_topic"]),
                self.csv_value(values["actual"]),
                self.csv_value(values["imu_raw"]),
                self.csv_value(values["imu_data"]),
                self.csv_value(values["odom"]),
                self.csv_value(values["feedback"]),
                str(source),
                self.csv_value(raw_angular),
                "%.9f" % self.command_integral,
                "%.9f" % self.sources["cmd_topic"].integral,
                "%.9f" % self.sources["actual"].integral,
                "%.9f" % self.sources["imu_raw"].integral,
                "%.9f" % self.sources["imu_data"].integral,
                "%.9f" % self.sources["odom"].integral,
                "%.9f" % self.sources["feedback"].integral,
            ])
            self.csv_file.flush()

    @staticmethod
    def csv_value(value):
        if value is None:
            return ""
        return "%.9f" % float(value)

    def run_phase(self, phase, duration, command_w, integrate):
        rate = rospy.Rate(self.rate_hz)
        end_time = rospy.get_time() + max(0.0, duration)
        while rospy.get_time() < end_time and not rospy.is_shutdown():
            now = rospy.get_time()
            if self.last_loop_time is None:
                dt = 0.0
            else:
                dt = max(0.0, now - self.last_loop_time)
            self.last_loop_time = now
            self.publish_cmd(command_w)
            if integrate:
                self.integrate_sources(now, dt, command_w)
            self.emit_row(phase, command_w)
            rate.sleep()

    def print_summary(self):
        target = -self.turn_radians if self.clockwise else self.turn_radians
        print("")
        print("Summary, target_angle=%.6f rad" % target)
        print("%-10s %12s %12s %12s" % ("source", "angle_rad", "error_rad", "samples"))
        rows = [("cmd_out", self.command_integral, None)]
        for name, source in self.sources.items():
            rows.append((name, source.integral, source.samples))
        for name, angle, samples in rows:
            print("%-10s %12.6f %12.6f %12s" % (
                name, angle, angle - target, "" if samples is None else str(samples)
            ))
        print("")
        print("Notes:")
        print("- cw should normally be negative angular.z in ROS. If the car turns ccw, rerun with _direction:=ccw.")
        print("- imu_raw is H30mini /imu_raw_data.angular_velocity.z.")
        print("- actual is /actual_velocity.twist.angular.z.")
        if self.csv_path:
            print("- CSV saved at %s" % self.csv_path)

    def close(self):
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None

    def spin(self):
        self.wait_for_cmd_subscriber()
        self.print_header()
        self.start_wall = rospy.get_time()
        try:
            self.run_phase("prestop", self.pre_stop_sec, 0.0, False)
            if self.countdown_sec > 0.0:
                print("Starting turn in %.1f seconds. Keep the car area clear. Press Ctrl+C to stop." % self.countdown_sec)
                self.run_phase("countdn", self.countdown_sec, 0.0, False)
            self.last_loop_time = rospy.get_time()
            self.run_phase("turn", self.duration, self.command_w, True)
            self.run_phase("stop", self.post_stop_sec, 0.0, True)
        finally:
            self.publish_stop()
            self.emit_row("done", 0.0, force=True)
            self.print_summary()
            self.close()


if __name__ == "__main__":
    try:
        ClockwiseTurnAngularTest().spin()
    except rospy.ROSInterruptException:
        pass
