#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import glob
import json
import math
import os
import struct
import sys
import time

import rospy
import serial
from sensor_msgs.msg import Imu
from std_msgs.msg import String


YIS_HEADER = [0x59, 0x53]
PROTOCOL_MIN_LEN = 7
PROTOCOL_TID_POS = 2
PROTOCOL_PAYLOAD_LEN_POS = 4
PAYLOAD_POS = 5

SENSOR_TEMP_ID = 0x01
ACC_ID = 0x10
GYRO_ID = 0x20
EULER_ID = 0x40
QUATERNION_ID = 0x41
SMP_TIMESTAMP_ID = 0x51
READY_TIMESTAMP_ID = 0x52
STATUS_ID = 0x80


def as_bool(value):
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def byte_at(data, index):
    value = data[index]
    if isinstance(value, str):
        return ord(value)
    return value


def bytes_from(data):
    if sys.version_info[0] >= 3:
        return bytes(bytearray(data))
    return "".join(chr(byte_at(data, i)) for i in range(len(data)))


def clear_front(data, count):
    del data[:count]


def find_header(data):
    limit = len(data) - 1
    for index in range(limit):
        if byte_at(data, index) == YIS_HEADER[0] and byte_at(data, index + 1) == YIS_HEADER[1]:
            return index
    return -1


def calc_crc16(data):
    check_a = 0
    check_b = 0
    for value in data:
        check_a = (check_a + byte_at([value], 0)) & 0xFF
        check_b = (check_b + check_a) & 0xFF
    return (check_b << 8) + check_a


class YesenseDecoder(object):
    def __init__(self):
        self.buffer = bytearray()

    def extend(self, data):
        if data:
            self.buffer.extend(bytearray(data))

    def pop_frame(self):
        while True:
            if len(self.buffer) < PROTOCOL_MIN_LEN:
                return None

            header_pos = find_header(self.buffer)
            if header_pos < 0:
                if len(self.buffer) > 1:
                    keep_last = byte_at(self.buffer, len(self.buffer) - 1) == YIS_HEADER[0]
                    clear_front(self.buffer, len(self.buffer) - (1 if keep_last else 0))
                return None
            if header_pos > 0:
                clear_front(self.buffer, header_pos)

            if len(self.buffer) < PROTOCOL_MIN_LEN:
                return None

            payload_len = byte_at(self.buffer, PROTOCOL_PAYLOAD_LEN_POS)
            frame_len = PROTOCOL_MIN_LEN + payload_len
            if len(self.buffer) < frame_len:
                return None

            crc_start = PROTOCOL_TID_POS
            crc_end = PAYLOAD_POS + payload_len
            crc_calc = calc_crc16(self.buffer[crc_start:crc_end])
            crc_received = struct.unpack_from("<H", bytes_from(self.buffer), crc_end)[0]
            if crc_calc != crc_received:
                clear_front(self.buffer, 2)
                continue

            frame = self.parse_frame(payload_len)
            clear_front(self.buffer, frame_len)
            return frame

    def parse_frame(self, payload_len):
        raw = bytes_from(self.buffer)
        result = {
            "tid": struct.unpack_from("<H", raw, PROTOCOL_TID_POS)[0],
            "acc_x": None,
            "acc_y": None,
            "acc_z": None,
            "gyro_x": None,
            "gyro_y": None,
            "gyro_z": None,
            "q0": None,
            "q1": None,
            "q2": None,
            "q3": None,
            "pitch": None,
            "roll": None,
            "yaw": None,
            "sensor_temp": None,
            "smp_timestamp": None,
            "ready_timestamp": None,
            "status": None,
        }

        pos = PAYLOAD_POS
        payload_end = PAYLOAD_POS + payload_len
        while pos + 2 <= payload_end:
            data_id = byte_at(self.buffer, pos)
            data_len = byte_at(self.buffer, pos + 1)
            value_pos = pos + 2
            value_end = value_pos + data_len
            if value_end > payload_end:
                break
            payload = raw[value_pos:value_end]

            if data_id == SENSOR_TEMP_ID and data_len == 2:
                result["sensor_temp"] = struct.unpack_from("<h", payload, 0)[0] * 0.01
            elif data_id == ACC_ID and data_len == 12:
                result["acc_x"] = struct.unpack_from("<i", payload, 0)[0] * 0.000001
                result["acc_y"] = struct.unpack_from("<i", payload, 4)[0] * 0.000001
                result["acc_z"] = struct.unpack_from("<i", payload, 8)[0] * 0.000001
            elif data_id == GYRO_ID and data_len == 12:
                result["gyro_x"] = struct.unpack_from("<i", payload, 0)[0] * 0.000001
                result["gyro_y"] = struct.unpack_from("<i", payload, 4)[0] * 0.000001
                result["gyro_z"] = struct.unpack_from("<i", payload, 8)[0] * 0.000001
            elif data_id == EULER_ID and data_len == 12:
                result["pitch"] = struct.unpack_from("<i", payload, 0)[0] * 0.000001
                result["roll"] = struct.unpack_from("<i", payload, 4)[0] * 0.000001
                result["yaw"] = struct.unpack_from("<i", payload, 8)[0] * 0.000001
            elif data_id == QUATERNION_ID and data_len == 16:
                result["q0"] = struct.unpack_from("<i", payload, 0)[0] * 0.000001
                result["q1"] = struct.unpack_from("<i", payload, 4)[0] * 0.000001
                result["q2"] = struct.unpack_from("<i", payload, 8)[0] * 0.000001
                result["q3"] = struct.unpack_from("<i", payload, 12)[0] * 0.000001
            elif data_id == SMP_TIMESTAMP_ID and data_len == 4:
                result["smp_timestamp"] = struct.unpack_from("<I", payload, 0)[0]
            elif data_id == READY_TIMESTAMP_ID and data_len == 4:
                result["ready_timestamp"] = struct.unpack_from("<I", payload, 0)[0]
            elif data_id == STATUS_ID and data_len == 1:
                result["status"] = byte_at(payload, 0)

            pos = value_end

        return result


class H30MiniImuNode(object):
    def __init__(self):
        rospy.init_node("h30mini_imu", anonymous=False)
        self.port = rospy.get_param("~port", "")
        self.baudrate = int(rospy.get_param("~baudrate", 460800))
        self.frame_id = rospy.get_param("~frame_id", "IMU_link")
        self.imu_topic = rospy.get_param("~imu_topic", "/imu_raw_data")
        self.status_topic = rospy.get_param("~status_topic", "/h30mini_imu_status")
        self.exclude_ports = self.parse_csv(rospy.get_param("~exclude_ports", "/dev/ttyUSB0"))
        self.scan_seconds = float(rospy.get_param("~scan_seconds", 1.5))
        self.debug = as_bool(rospy.get_param("~debug", False))

        self.decoder = YesenseDecoder()
        self.serial = None
        self.frames = 0
        self.crc_frames = 0
        self.last_frame = None
        self.last_frame_time = None
        self.last_status_time = rospy.Time(0)

        self.imu_pub = rospy.Publisher(self.imu_topic, Imu, queue_size=50)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=5)

    @staticmethod
    def parse_csv(value):
        if value is None:
            return []
        return [item.strip() for item in str(value).split(",") if item.strip()]

    def candidates(self):
        ports = []
        for pattern in ("/dev/ttyUSB*", "/dev/ttyACM*", "/dev/ttySC*"):
            ports.extend(glob.glob(pattern))
        ports = sorted(set(ports))
        return [port for port in ports if port not in self.exclude_ports]

    def open_port(self, port):
        return serial.Serial(port, self.baudrate, timeout=0.02)

    def auto_detect_port(self):
        for port in self.candidates():
            try:
                ser = self.open_port(port)
            except Exception as exc:
                rospy.logwarn("H30mini skip %s: %s", port, exc)
                continue
            decoder = YesenseDecoder()
            deadline = time.time() + self.scan_seconds
            try:
                while time.time() < deadline and not rospy.is_shutdown():
                    waiting = getattr(ser, "in_waiting", 0)
                    if not waiting:
                        try:
                            waiting = ser.inWaiting()
                        except Exception:
                            waiting = 0
                    data = ser.read(waiting or 1)
                    decoder.extend(data)
                    frame = decoder.pop_frame()
                    if frame is not None:
                        ser.close()
                        rospy.loginfo("H30mini detected on %s", port)
                        return port
                ser.close()
            except Exception as exc:
                try:
                    ser.close()
                except Exception:
                    pass
                rospy.logwarn("H30mini scan failed on %s: %s", port, exc)
        return None

    def connect(self):
        selected = self.port.strip()
        if not selected:
            selected = self.auto_detect_port()
        if not selected:
            raise RuntimeError("No Yesense H30mini serial port detected. Set _port:=/dev/ttyXXX if needed.")
        self.serial = self.open_port(selected)
        self.port = selected
        rospy.loginfo("H30mini IMU opened %s @ %d", self.port, self.baudrate)

    def publish_frame(self, frame):
        msg = Imu()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.frame_id

        if None not in (frame.get("q0"), frame.get("q1"), frame.get("q2"), frame.get("q3")):
            q0 = frame["q0"]
            q1 = frame["q1"]
            q2 = frame["q2"]
            q3 = frame["q3"]
            norm = math.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
            if norm > 0.1:
                msg.orientation.w = q0 / norm
                msg.orientation.x = q1 / norm
                msg.orientation.y = q2 / norm
                msg.orientation.z = q3 / norm
            else:
                msg.orientation_covariance[0] = -1.0
        else:
            msg.orientation_covariance[0] = -1.0

        if frame.get("gyro_x") is not None:
            msg.angular_velocity.x = math.radians(frame["gyro_x"])
        if frame.get("gyro_y") is not None:
            msg.angular_velocity.y = math.radians(frame["gyro_y"])
        if frame.get("gyro_z") is not None:
            msg.angular_velocity.z = math.radians(frame["gyro_z"])

        if frame.get("acc_x") is not None:
            msg.linear_acceleration.x = frame["acc_x"]
        if frame.get("acc_y") is not None:
            msg.linear_acceleration.y = frame["acc_y"]
        if frame.get("acc_z") is not None:
            msg.linear_acceleration.z = frame["acc_z"]

        self.imu_pub.publish(msg)
        self.last_frame = frame
        self.last_frame_time = msg.header.stamp
        self.frames += 1

    def publish_status(self):
        now = rospy.Time.now()
        if (now - self.last_status_time).to_sec() < 1.0:
            return
        self.last_status_time = now
        age = None
        if self.last_frame_time is not None:
            age = round((now - self.last_frame_time).to_sec(), 4)
        frame = self.last_frame or {}
        status = {
            "port": self.port,
            "baudrate": self.baudrate,
            "frames": self.frames,
            "last_age_sec": age,
            "last_tid": frame.get("tid"),
            "gyro_z_radps": None if frame.get("gyro_z") is None else round(math.radians(frame["gyro_z"]), 6),
            "acc_z_mps2": None if frame.get("acc_z") is None else round(frame["acc_z"], 6),
            "topic": self.imu_topic,
        }
        self.status_pub.publish(json.dumps(status, sort_keys=True))

    def spin(self):
        self.connect()
        rate = rospy.Rate(500)
        while not rospy.is_shutdown():
            try:
                waiting = getattr(self.serial, "in_waiting", 0)
                if not waiting:
                    try:
                        waiting = self.serial.inWaiting()
                    except Exception:
                        waiting = 0
                data = self.serial.read(waiting or 1)
                self.decoder.extend(data)
                while True:
                    frame = self.decoder.pop_frame()
                    if frame is None:
                        break
                    self.publish_frame(frame)
                self.publish_status()
            except serial.SerialException as exc:
                rospy.logerr("H30mini serial error: %s", exc)
                rospy.sleep(1.0)
            rate.sleep()


if __name__ == "__main__":
    try:
        H30MiniImuNode().spin()
    except Exception as exc:
        rospy.logerr("H30mini IMU node failed: %s", exc)
        raise
