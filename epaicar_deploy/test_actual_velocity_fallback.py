#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import time

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu


def main():
    rospy.init_node("actual_velocity_fake_source", anonymous=True)

    odom_topic = rospy.get_param("~odom_topic", "/test/odom")
    imu_topic = rospy.get_param("~imu_topic", "/test/imu")
    cmd_topic = rospy.get_param("~cmd_vel_topic", "/test/cmd_vel")
    duration = float(rospy.get_param("~duration", 4.0))
    rate_hz = float(rospy.get_param("~rate_hz", 20.0))
    linear = float(rospy.get_param("~linear", 0.20))
    steering = float(rospy.get_param("~steering", 0.50))

    odom_pub = rospy.Publisher(odom_topic, Odometry, queue_size=1)
    imu_pub = rospy.Publisher(imu_topic, Imu, queue_size=1)
    cmd_pub = rospy.Publisher(cmd_topic, Twist, queue_size=1)

    rospy.sleep(0.5)
    rate = rospy.Rate(rate_hz)
    end_time = time.time() + duration
    count = 0
    while time.time() < end_time and not rospy.is_shutdown():
        now = rospy.Time.now()

        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_footprint"
        odom.pose.pose.orientation.w = 1.0
        odom.twist.twist.linear.x = linear
        odom.twist.twist.angular.z = 0.0

        imu = Imu()
        imu.header.stamp = now
        imu.angular_velocity.z = 0.0

        cmd = Twist()
        cmd.linear.x = linear
        cmd.angular.z = steering

        odom_pub.publish(odom)
        imu_pub.publish(imu)
        cmd_pub.publish(cmd)
        count += 1
        rate.sleep()

    print("published fake samples:", count)


if __name__ == "__main__":
    main()
