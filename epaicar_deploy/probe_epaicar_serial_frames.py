#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import collections
import ctypes
import time

import serial


def crc_1byte(data):
    ret = 0
    for _ in range(8):
        if (ret ^ data) & 0x01:
            ret ^= 0x18
            ret >>= 1
            ret |= 0x80
        else:
            ret >>= 1
        data >>= 1
    return ret


def crc_byte(data):
    ret = 0
    for item in data[:-1]:
        ret = crc_1byte(ret ^ item)
    return ret


def make_frame(items):
    data = list(items)
    data[-1] = crc_byte(data)
    return bytearray(data)


def parse_frames(buffer):
    frames = []
    while True:
        start = -1
        for i, item in enumerate(buffer):
            if item == 0x5A:
                start = i
                break
        if start < 0:
            del buffer[:]
            break
        if start:
            del buffer[:start]
        if len(buffer) < 2:
            break
        length = buffer[1]
        if length <= 1:
            del buffer[0]
            continue
        if len(buffer) < length:
            break
        frame = list(buffer[:length])
        del buffer[:length]
        frames.append(frame)
    return frames


def int16_from(frame, offset):
    return ctypes.c_int16(frame[offset] * 256 + frame[offset + 1]).value


def describe(frame):
    ok = frame[-1] == crc_byte(frame)
    msg_id = frame[3] if len(frame) > 3 else None
    text = "id=0x%02x len=%d crc=%s raw=%s" % (
        msg_id if msg_id is not None else 0,
        len(frame),
        "ok" if ok else "bad",
        " ".join("%02x" % x for x in frame),
    )
    if ok and msg_id == 0x04 and len(frame) >= 10:
        text += " | vx=%.3f motor_pwm=%d steering_raw=%d steering_scaled=%.3f" % (
            int16_from(frame, 4) / 1000.0,
            int16_from(frame, 6),
            int16_from(frame, 8),
            int16_from(frame, 8) / 1000.0,
        )
    elif ok and msg_id == 0x0A and len(frame) >= 10:
        text += " | vx=%.3f yaw_deg=%.2f vyaw_radps=%.6f" % (
            int16_from(frame, 4) / 1000.0,
            -int16_from(frame, 6) / 100.0,
            (int16_from(frame, 8) / 1000.0) / 180.0 * 3.14159265358979,
        )
    elif ok and msg_id == 0x08 and len(frame) >= 8:
        text += " | voltage=%.3f current=%.3f" % (
            int16_from(frame, 4) / 1000.0,
            int16_from(frame, 6) / 1000.0,
        )
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--request-period", type=float, default=0.05)
    parser.add_argument("--mode", type=int, default=0x02, help="0x01 Express, 0x02 Boost, 0x03 Standard")
    parser.add_argument("--print-limit", type=int, default=80)
    args = parser.parse_args()

    odom_request = make_frame([0x5A, 0x07, 0x01, 0x09, args.mode, 0x00, 0x00])
    battery_request = make_frame([0x5A, 0x06, 0x01, 0x07, 0x00, 0x00])

    counts = collections.Counter()
    printed = 0
    buffer = bytearray()
    ser = serial.Serial(args.port, args.baudrate, timeout=0.01)
    try:
        try:
            ser.reset_input_buffer()
        except Exception:
            pass
        start = time.time()
        next_request = 0.0
        next_battery = 0.0
        while time.time() - start < args.duration:
            now = time.time()
            if now >= next_request:
                ser.write(odom_request)
                next_request = now + args.request_period
            if now >= next_battery:
                ser.write(battery_request)
                next_battery = now + 1.0
            data = ser.read(512)
            if data:
                buffer.extend(bytearray(data))
                for frame in parse_frames(buffer):
                    msg_id = frame[3] if len(frame) > 3 else -1
                    counts[msg_id] += 1
                    if printed < args.print_limit:
                        print(describe(frame))
                        printed += 1
            time.sleep(0.002)
    finally:
        ser.close()

    print("summary:")
    for msg_id, count in sorted(counts.items()):
        print("id=0x%02x count=%d" % (msg_id, count))


if __name__ == "__main__":
    main()
