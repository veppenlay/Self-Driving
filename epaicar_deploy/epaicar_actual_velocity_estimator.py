#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import copy
import json
import math
from collections import deque

import rospy
import tf
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import String


try:
    string_types = (basestring,)
except NameError:
    string_types = (str,)


def is_finite(value):
    return not (math.isnan(value) or math.isinf(value))


def as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, string_types):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def clamp_abs(value, limit):
    if limit is None or limit <= 0:
        return value
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


def deadband(value, threshold):
    if threshold > 0 and abs(value) < threshold:
        return 0.0
    return value


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def yaw_from_quaternion(q):
    return tf.transformations.euler_from_quaternion((q.x, q.y, q.z, q.w))[2]


class TimedValue(object):
    def __init__(self, name):
        self.name = name
        self.value = None
        self.rx_time = None
        self.header_time = None

    def update(self, value, header_time=None):
        self.value = float(value)
        self.rx_time = rospy.Time.now()
        self.header_time = header_time

    def age(self, now):
        if self.rx_time is None:
            return None
        return max(0.0, (now - self.rx_time).to_sec())

    def fresh_value(self, now, max_age, max_abs):
        age = self.age(now)
        if age is None or age > max_age:
            return None
        if self.value is None or not is_finite(self.value):
            return None
        if max_abs > 0 and abs(self.value) > max_abs:
            return None
        return self.value


class MedianEmaFilter(object):
    def __init__(self, window_size, alpha):
        self.window_size = max(1, int(window_size))
        self.alpha = max(0.0, min(1.0, float(alpha)))
        self.samples = deque(maxlen=self.window_size)
        self.state = None

    def reset(self):
        self.samples.clear()
        self.state = None

    def update(self, value):
        self.samples.append(float(value))
        ordered = sorted(self.samples)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            median = ordered[mid]
        else:
            median = 0.5 * (ordered[mid - 1] + ordered[mid])
        if self.state is None:
            self.state = median
        else:
            self.state = self.alpha * median + (1.0 - self.alpha) * self.state
        return self.state


class ActualVelocityEstimator(object):
    def __init__(self):
        rospy.init_node("epaicar_actual_velocity_estimator", anonymous=False)

        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.imu_topic = rospy.get_param("~imu_topic", "/imu_data")
        self.feedback_vel_topic = rospy.get_param("~feedback_vel_topic", "/feedback_vel")
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")
        self.actual_twist_topic = rospy.get_param("~actual_twist_topic", "/actual_velocity")
        self.actual_odom_topic = rospy.get_param("~actual_odom_topic", "/actual_odom")
        self.status_topic = rospy.get_param("~status_topic", "/actual_velocity_status")
        self.base_frame = rospy.get_param("~base_frame", "base_footprint")

        self.rate_hz = float(rospy.get_param("~rate_hz", 50.0))
        self.max_source_age = float(rospy.get_param("~max_source_age", 0.30))
        self.publish_partial = as_bool(rospy.get_param("~publish_partial", False))
        self.publish_odom = as_bool(rospy.get_param("~publish_odom", True))
        self.status_period = float(rospy.get_param("~status_period", 1.0))

        self.linear_scale = float(rospy.get_param("~linear_scale", 1.0))
        self.linear_bias = float(rospy.get_param("~linear_bias", 0.0))
        self.angular_scale = float(rospy.get_param("~angular_scale", 1.0))
        self.angular_bias = float(rospy.get_param("~angular_bias", 0.0))

        self.linear_deadband = float(rospy.get_param("~linear_deadband", 0.003))
        self.angular_deadband = float(rospy.get_param("~angular_deadband", 0.010))
        self.max_linear_mps = float(rospy.get_param("~max_linear_mps", 3.0))
        self.max_angular_radps = float(rospy.get_param("~max_angular_radps", 8.0))

        self.filter_window = int(rospy.get_param("~filter_window", 3))
        self.filter_alpha = float(rospy.get_param("~filter_alpha", 0.60))
        self.angular_fusion_tolerance = float(rospy.get_param("~angular_fusion_tolerance", 0.20))
        self.yaw_rate_fusion_tolerance = float(rospy.get_param("~yaw_rate_fusion_tolerance", 0.35))
        self.min_yaw_dt = float(rospy.get_param("~min_yaw_dt", 0.005))
        self.max_yaw_dt = float(rospy.get_param("~max_yaw_dt", 0.30))
        self.kinematic_fallback_enabled = as_bool(rospy.get_param("~kinematic_fallback_enabled", True))
        self.prefer_kinematic_when_commanded = as_bool(rospy.get_param("~prefer_kinematic_when_commanded", True))
        self.cmd_max_age = float(rospy.get_param("~cmd_max_age", 0.60))
        self.wheelbase = float(rospy.get_param("~wheelbase", 0.335))
        self.cmd_angular_mode = rospy.get_param("~cmd_angular_mode", "steering")
        self.steering_scale = float(rospy.get_param("~steering_scale", 1.0))
        self.steering_bias = float(rospy.get_param("~steering_bias", 0.0))
        self.max_steering_angle = float(rospy.get_param("~max_steering_angle", 0.70))
        self.min_linear_for_kinematic = float(rospy.get_param("~min_linear_for_kinematic", 0.02))
        self.angular_zero_epsilon = float(rospy.get_param("~angular_zero_epsilon", 0.015))
        self.cmd_angular_deadband = float(rospy.get_param("~cmd_angular_deadband", 0.015))
        self.feedback_steering_deadband = float(rospy.get_param("~feedback_steering_deadband", 0.005))
        self.use_feedback_steering = as_bool(rospy.get_param("~use_feedback_steering", False))

        self.snap_stopped_noise = as_bool(rospy.get_param("~snap_stopped_noise", True))
        self.snap_linear_threshold = float(rospy.get_param("~snap_linear_threshold", 0.015))
        self.snap_angular_threshold = float(rospy.get_param("~snap_angular_threshold", 0.030))

        self.odom_linear = TimedValue("odom_linear")
        self.odom_angular = TimedValue("odom_angular")
        self.imu_angular = TimedValue("imu_angular")
        self.feedback_linear = TimedValue("feedback_linear")
        self.feedback_steering = TimedValue("feedback_steering")
        self.yaw_rate = TimedValue("yaw_rate_from_odom_pose")
        self.cmd_linear = TimedValue("cmd_linear")
        self.cmd_angular = TimedValue("cmd_angular")

        self.linear_filter = MedianEmaFilter(self.filter_window, self.filter_alpha)
        self.angular_filter = MedianEmaFilter(self.filter_window, self.filter_alpha)

        self.last_odom_msg = None
        self.last_yaw = None
        self.last_yaw_stamp = None
        self.last_status_time = rospy.Time(0)
        self.last_quality = None

        self.twist_pub = rospy.Publisher(self.actual_twist_topic, TwistStamped, queue_size=10)
        self.odom_pub = rospy.Publisher(self.actual_odom_topic, Odometry, queue_size=10)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=3)

        rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=30)
        rospy.Subscriber(self.imu_topic, Imu, self.imu_cb, queue_size=30)
        rospy.Subscriber(self.feedback_vel_topic, Twist, self.feedback_cb, queue_size=30)
        rospy.Subscriber(self.cmd_vel_topic, Twist, self.cmd_cb, queue_size=30)

        rospy.loginfo("EPAIcar actual velocity estimator started.")
        rospy.loginfo("linear source: %s then %s; angular source: %s/%s, yaw derivative, then Ackermann fallback",
                      self.odom_topic, self.feedback_vel_topic, self.imu_topic, self.odom_topic)

    def odom_cb(self, msg):
        stamp = msg.header.stamp if msg.header.stamp != rospy.Time(0) else rospy.Time.now()
        self.last_odom_msg = msg

        lx = msg.twist.twist.linear.x
        az = msg.twist.twist.angular.z
        if is_finite(lx):
            self.odom_linear.update(lx, stamp)
        if is_finite(az):
            self.odom_angular.update(az, stamp)

        try:
            yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        except Exception:
            return
        if self.last_yaw is not None and self.last_yaw_stamp is not None:
            dt = (stamp - self.last_yaw_stamp).to_sec()
            if self.min_yaw_dt <= dt <= self.max_yaw_dt:
                rate = normalize_angle(yaw - self.last_yaw) / dt
                if is_finite(rate) and abs(rate) <= self.max_angular_radps:
                    self.yaw_rate.update(rate, stamp)
        self.last_yaw = yaw
        self.last_yaw_stamp = stamp

    def imu_cb(self, msg):
        stamp = msg.header.stamp if msg.header.stamp != rospy.Time(0) else rospy.Time.now()
        az = msg.angular_velocity.z
        if is_finite(az):
            self.imu_angular.update(az, stamp)

    def feedback_cb(self, msg):
        value = msg.linear.x
        if is_finite(value):
            self.feedback_linear.update(value, rospy.Time.now())
        if self.use_feedback_steering and is_finite(msg.angular.z):
            self.feedback_steering.update(msg.angular.z, rospy.Time.now())

    def cmd_cb(self, msg):
        if is_finite(msg.linear.x):
            self.cmd_linear.update(msg.linear.x, rospy.Time.now())
        if is_finite(msg.angular.z):
            self.cmd_angular.update(msg.angular.z, rospy.Time.now())

    def choose_linear(self, now):
        odom = self.odom_linear.fresh_value(now, self.max_source_age, self.max_linear_mps)
        if odom is not None:
            return odom, "odom.twist.linear.x"
        feedback = self.feedback_linear.fresh_value(now, self.max_source_age, self.max_linear_mps)
        if feedback is not None:
            return feedback, "feedback_vel.linear.x"
        return None, "missing"

    def choose_kinematic_angular(self, now, linear):
        if not self.kinematic_fallback_enabled:
            return None, "missing"

        steering = None
        source = None
        if self.use_feedback_steering:
            steering = self.feedback_steering.fresh_value(now, self.max_source_age, 10.0)
            if steering is not None:
                if abs(steering) > self.feedback_steering_deadband:
                    source = "feedback_steering_kinematic"
                else:
                    steering = None

        if steering is None:
            steering = self.cmd_angular.fresh_value(now, self.cmd_max_age, 10.0)
            if steering is None or abs(steering) < self.cmd_angular_deadband:
                return None, "missing"
            source = "cmd_steering_kinematic"

        mode = str(self.cmd_angular_mode).strip().lower()
        if mode in ("yaw_rate", "yaw", "omega", "angular"):
            omega = steering * self.steering_scale + self.steering_bias
            return clamp_abs(omega, self.max_angular_radps), "cmd_yaw_rate"

        if linear is None or abs(linear) < self.min_linear_for_kinematic:
            return None, "missing"
        if self.wheelbase <= 0:
            return None, "missing"

        steering_angle = steering * self.steering_scale + self.steering_bias
        steering_angle = clamp_abs(steering_angle, self.max_steering_angle)
        omega = linear * math.tan(steering_angle) / self.wheelbase
        if not is_finite(omega):
            return None, "missing"
        return clamp_abs(omega, self.max_angular_radps), source

    def choose_angular(self, now, linear):
        imu = self.imu_angular.fresh_value(now, self.max_source_age, self.max_angular_radps)
        odom = self.odom_angular.fresh_value(now, self.max_source_age, self.max_angular_radps)
        yaw = self.yaw_rate.fresh_value(now, self.max_source_age, self.max_angular_radps)
        kin, kin_source = self.choose_kinematic_angular(now, linear)

        if imu is not None and odom is not None:
            if abs(imu - odom) <= self.angular_fusion_tolerance:
                fused = 0.70 * imu + 0.30 * odom
                if (kin is not None and self.prefer_kinematic_when_commanded and
                        abs(fused) <= self.angular_zero_epsilon and abs(kin) > self.angular_zero_epsilon):
                    return kin, kin_source
                return fused, "imu+odom_twist"
            if yaw is not None:
                imu_err = abs(imu - yaw)
                odom_err = abs(odom - yaw)
                if imu_err <= self.yaw_rate_fusion_tolerance or odom_err <= self.yaw_rate_fusion_tolerance:
                    if imu_err <= odom_err:
                        return imu, "imu_angular.z"
                    return odom, "odom.twist.angular.z"
            if kin is not None and abs(imu) <= self.angular_zero_epsilon and abs(odom) <= self.angular_zero_epsilon:
                return kin, kin_source
            return imu, "imu_angular.z"
        if imu is not None:
            if (kin is not None and self.prefer_kinematic_when_commanded and
                    abs(imu) <= self.angular_zero_epsilon and abs(kin) > self.angular_zero_epsilon):
                return kin, kin_source
            return imu, "imu_angular.z"
        if odom is not None:
            if (kin is not None and self.prefer_kinematic_when_commanded and
                    abs(odom) <= self.angular_zero_epsilon and abs(kin) > self.angular_zero_epsilon):
                return kin, kin_source
            return odom, "odom.twist.angular.z"
        if yaw is not None:
            if (kin is not None and self.prefer_kinematic_when_commanded and
                    abs(yaw) <= self.angular_zero_epsilon and abs(kin) > self.angular_zero_epsilon):
                return kin, kin_source
            return yaw, "odom.pose.yaw_delta"
        if kin is not None:
            return kin, kin_source
        return None, "missing"

    def apply_calibration_and_filter(self, linear, angular):
        linear = linear * self.linear_scale + self.linear_bias
        angular = angular * self.angular_scale + self.angular_bias

        linear = clamp_abs(linear, self.max_linear_mps)
        angular = clamp_abs(angular, self.max_angular_radps)
        linear = deadband(linear, self.linear_deadband)
        angular = deadband(angular, self.angular_deadband)

        if self.snap_stopped_noise:
            if abs(linear) < self.snap_linear_threshold:
                linear = 0.0
            if abs(angular) < self.snap_angular_threshold:
                angular = 0.0

        return self.linear_filter.update(linear), self.angular_filter.update(angular)

    def estimate(self, now):
        raw_linear, linear_source = self.choose_linear(now)
        raw_angular, angular_source = self.choose_angular(now, raw_linear)

        if raw_linear is None or raw_angular is None:
            quality = "partial" if (raw_linear is not None or raw_angular is not None) else "stale"
            if not self.publish_partial:
                return None, self.make_status(now, quality, linear_source, angular_source, None, None)
            if raw_linear is None:
                raw_linear = 0.0
            if raw_angular is None:
                raw_angular = 0.0
        else:
            quality = "ok"

        linear, angular = self.apply_calibration_and_filter(raw_linear, raw_angular)
        status = self.make_status(now, quality, linear_source, angular_source, linear, angular)
        status["raw_linear_mps"] = round(raw_linear, 6)
        status["raw_angular_radps"] = round(raw_angular, 6)
        return (linear, angular), status

    def make_status(self, now, quality, linear_source, angular_source, linear, angular):
        def age_or_none(signal):
            age = signal.age(now)
            return None if age is None else round(age, 4)

        return {
            "quality": quality,
            "linear_source": linear_source,
            "angular_source": angular_source,
            "linear_mps": None if linear is None else round(linear, 6),
            "angular_radps": None if angular is None else round(angular, 6),
            "age_sec": {
                "odom_linear": age_or_none(self.odom_linear),
                "odom_angular": age_or_none(self.odom_angular),
                "imu_angular": age_or_none(self.imu_angular),
                "feedback_linear": age_or_none(self.feedback_linear),
                "feedback_steering": age_or_none(self.feedback_steering),
                "yaw_rate": age_or_none(self.yaw_rate),
                "cmd_angular": age_or_none(self.cmd_angular),
            },
            "source_value": {
                "feedback_steering": None if self.feedback_steering.value is None else round(self.feedback_steering.value, 6),
                "cmd_angular": None if self.cmd_angular.value is None else round(self.cmd_angular.value, 6),
            },
            "max_source_age_sec": self.max_source_age,
            "cmd_max_age_sec": self.cmd_max_age,
        }

    def publish(self, now, linear, angular, status):
        twist = TwistStamped()
        twist.header.stamp = now
        twist.header.frame_id = self.base_frame
        twist.twist.linear.x = linear
        twist.twist.angular.z = angular
        self.twist_pub.publish(twist)

        if self.publish_odom and self.last_odom_msg is not None:
            odom = copy.deepcopy(self.last_odom_msg)
            odom.header.stamp = now
            odom.child_frame_id = self.base_frame
            odom.twist.twist.linear.x = linear
            odom.twist.twist.angular.z = angular
            self.odom_pub.publish(odom)

        self.publish_status(now, status)

    def publish_status(self, now, status):
        if self.status_period <= 0:
            return
        if (now - self.last_status_time).to_sec() < self.status_period:
            return
        self.last_status_time = now
        self.status_pub.publish(json.dumps(status, sort_keys=True))
        if status.get("quality") != self.last_quality:
            rospy.loginfo("actual velocity quality=%s linear_source=%s angular_source=%s",
                          status.get("quality"), status.get("linear_source"), status.get("angular_source"))
            self.last_quality = status.get("quality")

    def spin(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            estimate, status = self.estimate(now)
            if estimate is None:
                self.publish_status(now, status)
            else:
                self.publish(now, estimate[0], estimate[1], status)
            rate.sleep()


if __name__ == "__main__":
    try:
        ActualVelocityEstimator().spin()
    except rospy.ROSInterruptException:
        pass
