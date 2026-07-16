#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import csv
import math
import os
import sys
from collections import OrderedDict

import rospy
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu


def stamp_sec(stamp):
    try:
        value = stamp.to_sec()
        if value > 0.0:
            return value
    except Exception:
        pass
    return rospy.get_time()


def mean(values):
    return sum(values) / float(len(values)) if values else None


def rmse(values):
    return math.sqrt(mean([v * v for v in values])) if values else None


def fmt(value):
    return "NA" if value is None else "%.3f" % value


class Source(object):
    def __init__(self):
        self.value = None
        self.time = None

    def update(self, value, stamp=None):
        self.value = float(value)
        self.time = stamp_sec(stamp)

    def get(self, now, max_age):
        if self.value is None or self.time is None or now - self.time > max_age:
            return None
        return self.value


class MatrixTest(object):
    def __init__(self):
        rospy.init_node("epaicar_velocity_turn_matrix_test", anonymous=True)

        self.rate_hz = float(rospy.get_param("~rate_hz", 30.0))
        self.sample_hz = float(rospy.get_param("~sample_hz", 20.0))
        self.test_duration = float(rospy.get_param("~test_duration", 4.0))
        self.pause_duration = float(rospy.get_param("~pause_duration", 2.0))
        self.countdown = float(rospy.get_param("~countdown_sec", 3.0))
        self.ignore_head = float(rospy.get_param("~ignore_head_sec", 0.5))
        self.ignore_tail = float(rospy.get_param("~ignore_tail_sec", 0.3))
        self.max_age = float(rospy.get_param("~max_age", 0.4))
        self.direction = str(rospy.get_param("~direction", "ccw")).lower()
        self.sign = -1.0 if self.direction in ("cw", "clockwise", "right") else 1.0
        self.csv_path = str(rospy.get_param("~csv_path", "/tmp/epaicar_velocity_turn_matrix.csv"))

        self.linear_values = self._numbers(rospy.get_param("~linear_values", "0.2,0.5"))
        self.angular_values = self._numbers(rospy.get_param("~angular_values", "0,1.5,3.0"))
        self.cmd = Source()
        self.sources = OrderedDict((name, Source()) for name in ("actual", "imu_data", "imu_raw"))
        self.linear_sources = OrderedDict((name, Source()) for name in ("actual", "odom", "feedback"))
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        rospy.Subscriber("/cmd_vel", Twist, lambda m: self.cmd.update(m.angular.z), queue_size=20)
        rospy.Subscriber("/actual_velocity", TwistStamped, lambda m: (self.sources["actual"].update(m.twist.angular.z, m.header.stamp), self.linear_sources["actual"].update(m.twist.linear.x, m.header.stamp)), queue_size=20)
        rospy.Subscriber("/imu_data", Imu, lambda m: self.sources["imu_data"].update(m.angular_velocity.z, m.header.stamp), queue_size=50)
        rospy.Subscriber("/imu_raw_data", Imu, lambda m: self.sources["imu_raw"].update(m.angular_velocity.z, m.header.stamp), queue_size=50)
        rospy.Subscriber("/odom", Odometry, lambda m: self.linear_sources["odom"].update(m.twist.twist.linear.x, m.header.stamp), queue_size=20)
        rospy.Subscriber("/feedback_vel", Twist, lambda m: self.linear_sources["feedback"].update(m.linear.x), queue_size=20)

        parent = os.path.dirname(self.csv_path)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent)
        self.csv = open(self.csv_path, "wb" if sys.version_info[0] < 3 else "w")
        self.writer = csv.writer(self.csv)
        self.writer.writerow(["case", "t_sec", "linear_cmd", "angular_cmd", "cmd_topic", "actual", "imu_data", "imu_raw", "actual_linear", "odom_linear", "feedback_linear"])

    def _numbers(self, value):
        if isinstance(value, (list, tuple)):
            return [float(x) for x in value]
        return [float(x.strip()) for x in str(value).split(",") if x.strip()]

    def publish(self, linear, angular):
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self.cmd_pub.publish(msg)

    def stop(self, seconds=0.3):
        end = rospy.get_time() + seconds
        while not rospy.is_shutdown() and rospy.get_time() < end:
            self.publish(0.0, 0.0)
            rospy.sleep(0.03)

    def wait_countdown(self, title):
        rospy.logwarn("即将开始 %s，车辆周围保持安全，%.1f 秒后运行", title, self.countdown)
        rospy.sleep(self.countdown)

    def run_case(self, linear, angular):
        case = "v%.1f_w%.1f" % (linear, abs(angular))
        self.wait_countdown(case)
        start = rospy.get_time()
        rows = []
        integral = {"cmd": 0.0, "actual": 0.0, "imu_data": 0.0, "imu_raw": 0.0}
        previous = start
        while not rospy.is_shutdown() and rospy.get_time() - start < self.test_duration:
            now = rospy.get_time()
            dt = max(0.0, min(now - previous, 0.2))
            previous = now
            command_w = self.sign * angular
            self.publish(linear, command_w)
            elapsed = now - start
            values = {"cmd": self.cmd.get(now, self.max_age)}
            for name, source in self.sources.items():
                values[name] = source.get(now, self.max_age)
            for name, source in self.linear_sources.items():
                values[name + "_linear"] = source.get(now, self.max_age)
            if self.ignore_head <= elapsed <= self.test_duration - self.ignore_tail:
                for name in integral:
                    if values.get(name) is not None:
                        integral[name] += values[name] * dt
                rows.append((now - start, values))
            self.writer.writerow([case, "%.4f" % elapsed, linear, command_w] + [values.get(k) for k in ("cmd", "actual", "imu_data", "imu_raw", "actual_linear", "odom_linear", "feedback_linear")])
            rospy.sleep(1.0 / self.sample_hz)
        self.stop()
        rospy.loginfo("完成 %s，停车 %.1f 秒", case, self.pause_duration)
        rospy.sleep(self.pause_duration)
        return case, linear, angular, rows, integral

    def summarize(self, case, linear, angular, rows, integral):
        target = self.sign * angular
        print("\nCASE %s: v=%.2f m/s, w_cmd=%+.2f rad/s" % (case, linear, target))
        print("angular source  mean       MAE       RMSE      gain     angle(rad)")
        for name in ("cmd", "actual", "imu_data", "imu_raw"):
            values = [r[1].get(name) for r in rows if r[1].get(name) is not None]
            if not values:
                print("%-10s  NA         NA        NA        NA       NA" % name)
                continue
            errors = [v - target for v in values]
            avg = mean(values)
            gain = avg / target if abs(target) > 1e-6 else None
            print("%-10s  %8s  %8s  %8s  %8s  %8s" % (name, fmt(avg), fmt(mean([abs(x) for x in errors])), fmt(rmse(errors)), fmt(gain), fmt(integral[name])))
        print("linear source   mean       error     gain")
        for name in ("actual_linear", "odom_linear", "feedback_linear"):
            values = [r[1].get(name) for r in rows if r[1].get(name) is not None]
            if not values:
                print("%-14s NA         NA        NA" % name)
                continue
            avg = mean(values)
            print("%-14s %8s  %8s  %8s" % (name, fmt(avg), fmt(avg - linear), fmt(avg / linear if abs(linear) > 1e-6 else None)))

    def run(self):
        print("EPAIcar 线速度/角速度矩阵测试：直行基线 + 四组转弯工况")
        print("方向=%s，单组运行 %.1f 秒；Ctrl-C 可随时停止" % (self.direction, self.test_duration))
        cases = []
        for linear in self.linear_values:
            cases.append((linear, 0.0))
            for angular in self.angular_values:
                if abs(angular) > 1e-6:
                    cases.append((linear, abs(angular)))
        results = []
        for linear, angular in cases:
            if rospy.is_shutdown():
                break
            results.append(self.run_case(linear, angular))
            self.summarize(*results[-1])
        self.stop(0.5)
        self.csv.close()
        print("\nCSV 已写入: %s" % self.csv_path)


if __name__ == "__main__":
    try:
        MatrixTest().run()
    except (rospy.ROSInterruptException, KeyboardInterrupt):
        pass
