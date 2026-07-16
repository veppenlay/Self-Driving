#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import json
import math
import os
import socket
import subprocess
import sys
import threading
import time
from collections import deque

import rospy
from geometry_msgs.msg import Twist


DEFAULT_SIGN_FILE = "/opt/nvidia/deepstream/deepstream-6.0/sources/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/yolosign.txt"


if "--check-only" in sys.argv:
    print("epaicar_drive_ros import check ok")
    sys.exit(0)


def is_finite(value):
    return not (math.isnan(value) or math.isinf(value))


class AngularReceiver(threading.Thread):
    def __init__(self, host, port):
        threading.Thread.__init__(self)
        self.daemon = True
        self.host = host
        self.port = port
        self.lock = threading.Lock()
        self.latest = None
        self.running = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.settimeout(0.2)

    def run(self):
        while self.running and not rospy.is_shutdown():
            try:
                data, _ = self.sock.recvfrom(4096)
                if not isinstance(data, str):
                    data = data.decode("ascii", "ignore")
                payload = json.loads(data)
                with self.lock:
                    self.latest = payload
            except socket.timeout:
                continue
            except Exception as exc:
                rospy.logwarn("Angular UDP receive failed: %s", exc)

    def get(self):
        with self.lock:
            if self.latest is None:
                return None
            return dict(self.latest)

    def stop(self):
        self.running = False
        try:
            self.sock.close()
        except Exception:
            pass


class SignFileMonitor(object):
    def __init__(self, path, callback):
        self.path = path
        self.callback = callback
        self.last_emit = {}
        self.last_poll = 0.0
        self.poll_interval = 0.05
        self.filter_duration = 5.0
        self.person_filter_duration = 0.5

    def poll(self):
        now = rospy.get_time()
        if now - self.last_poll < self.poll_interval:
            return
        self.last_poll = now
        try:
            with open(self.path, "r") as f:
                line = f.readline().strip()
        except Exception:
            return
        if not line:
            return
        parts = line.split()
        if len(parts) < 2:
            return
        sign = self._map_sign(parts[0], parts[1])
        if sign is None:
            return
        duration = self.person_filter_duration if sign == "person" else self.filter_duration
        last = self.last_emit.get(sign)
        if last is not None and now - last < duration:
            return
        self.last_emit[sign] = now
        self.callback(sign)

    def _map_sign(self, sign_id, area_text):
        try:
            area = float(area_text)
        except Exception:
            return None
        if sign_id == "0" and area > 2000:
            return "crossing_walk"
        if sign_id == "1" and area > 10:
            return "limit"
        if sign_id == "2":
            return "remove_limit"
        if sign_id == "3" and area > 270:
            return "red"
        if sign_id == "4":
            return "green"
        if sign_id == "5":
            return "person"
        return None


class MotionController(object):
    def __init__(self):
        rospy.init_node("epaicar_model_improving_controller", anonymous=False)

        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.engine_path = rospy.get_param("~engine_path", "/home/epaicar/results/0703_temporal3_2d_fp16.engine")
        self.hmm_model = str(
            rospy.get_param(
                "~hmm_model",
                os.path.join(self.script_dir, "steering_only_hmm_online_model.json"),
            )
        )
        # Control source: 'steering_hmm' (locked default) or 'geometry' (exploration).
        self.control_source = str(rospy.get_param("~control_source", "steering_hmm"))
        self.controller_json = str(
            rospy.get_param("~controller_json", os.path.join(self.script_dir, "controller_pid.json"))
        )
        self.camera = str(rospy.get_param("~camera", "/dev/video0"))
        self.udp_host = rospy.get_param("~udp_host", "127.0.0.1")
        self.udp_port = int(rospy.get_param("~udp_port", 15050))
        self.rate_hz = float(rospy.get_param("~rate_hz", 30.0))
        self.default_linear_speed = float(rospy.get_param("~default_linear_speed", 0.5))
        self.slow_linear_speed = float(rospy.get_param("~slow_linear_speed", 0.42))
        self.max_angular_z = float(rospy.get_param("~max_angular_z", 3.0))
        self.model_timeout = float(rospy.get_param("~model_timeout", 0.35))
        self.wait_for_start = bool(rospy.get_param("~wait_for_start", True))
        self.auto_start_without_stdin = bool(rospy.get_param("~auto_start_without_stdin", False))
        self.start_infer_server = bool(rospy.get_param("~start_infer_server", True))
        self.use_sign_file = bool(rospy.get_param("~use_sign_file", True))
        self.sign_file = rospy.get_param("~sign_file", DEFAULT_SIGN_FILE)

        self.stop1_duration = float(rospy.get_param("~stop1_duration", 2.0))
        self.stop2_duration = float(rospy.get_param("~stop2_duration", 100.0))
        self.stop_duration_person = float(rospy.get_param("~stop_duration_person", 1.5))
        self.slow_duration = float(rospy.get_param("~slow_duration", 4.0))
        self.delay_before_red = float(rospy.get_param("~delay_before_red", 2.25))

        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.receiver = AngularReceiver(self.udp_host, self.udp_port)
        self.receiver.start()
        self.infer_process = None
        if self.start_infer_server:
            self._start_infer_server()

        self.sign_queue = deque()
        self.current_action = None
        self.current_sign = None
        self.action_start_time = 0.0
        self.action_duration = 0.0
        self.limit_handled = False
        self.crosswalk_handled = False
        self.red_processed = False
        self.green_handled = False
        self.last_clear_time = 0.0
        self.last_no_model_log = 0.0

        self.interrupted_action = None
        self.interrupted_sign = None
        self.interrupted_duration = None
        self.person_delay_active = False
        self.person_delay_current_motion = None
        self.person_delay_start_time = None
        self.person_delay_motions = self._build_person_delay()

        self.initializing = bool(rospy.get_param("~run_init_motion", True))
        self.init_motions = self._build_init_motions()
        self.init_start_time = None

        self.sign_monitor = SignFileMonitor(self.sign_file, self.handle_sign) if self.use_sign_file else None
        self._try_subscribe_yolo_topic()
        self.rate = rospy.Rate(self.rate_hz)
        rospy.on_shutdown(self.shutdown)

    def _start_infer_server(self):
        script = os.path.join(self.script_dir, "epaicar_temporal3_2d_trt_udp_server.py")
        cmd = [
            "/usr/bin/python3",
            script,
            "--engine", self.engine_path,
            "--hmm-model", self.hmm_model,
            "--camera", self.camera,
            "--udp-host", self.udp_host,
            "--udp-port", str(self.udp_port),
            "--rate-hz", str(self.rate_hz),
            "--max-angular", str(self.max_angular_z),
            "--print-every", "60",
        ]
        if self.control_source == "geometry":
            cmd += [
                "--control-source", "geometry",
                "--controller-json", self.controller_json,
                "--nominal-speed", str(self.default_linear_speed),
            ]
        rospy.loginfo("Starting temporal3 2D TensorRT UDP server: %s", " ".join(cmd))
        self.infer_process = subprocess.Popen(cmd)

    def _try_subscribe_yolo_topic(self):
        if not bool(rospy.get_param("~use_yolo_topic", False)):
            return
        try:
            from e2e.msg import YoloSign
            rospy.Subscriber("/yolo_sign", YoloSign, self._yolo_callback, queue_size=10)
            rospy.loginfo("Subscribed to /yolo_sign")
        except Exception as exc:
            rospy.logwarn("Skip /yolo_sign subscriber: %s", exc)

    def _yolo_callback(self, msg):
        self.handle_sign(str(msg.sign_type))

    def _build_init_motions(self):
        return deque([
            {"duration": 0.5, "linear": 0.0, "angular": 0.0},
            {"duration": 0.5, "linear": 0.5, "angular": 0.0},
            {"duration": 0.79, "linear": 0.5, "angular": -2.59},
            {"duration": 0.12, "linear": 0.5, "angular": 0.0},
            {"duration": 2.6, "linear": 0.5, "angular": 1.4099},
            {"duration": 0.57, "linear": 0.5, "angular": 0.0},
            {"duration": 0.8, "linear": 0.5, "angular": -2.20},
            {"duration": 0.25, "linear": 0.3, "angular": 0.0},
        ])

    def _build_person_delay(self):
        return deque([
            {"duration": 1.1, "linear": 0.5, "angular": 1.6},
            {"duration": 1.0, "linear": 0.5, "angular": -1.6},
            {"duration": 2.5, "linear": 0.5, "angular": 1.6},
            {"duration": 0.65, "linear": 0.5, "angular": -1.62},
            {"duration": 1.05, "linear": 0.5, "angular": 1.76},
            {"duration": 1.25, "linear": 0.5, "angular": 0.0},
        ])

    def clear_sign_file(self):
        try:
            with open(self.sign_file, "w"):
                pass
            rospy.loginfo("Cleared sign file: %s", self.sign_file)
        except Exception as exc:
            rospy.logwarn("Failed to clear sign file %s: %s", self.sign_file, exc)

    def safe_clear_sign_file(self):
        now = rospy.get_time()
        if ("red" in self.sign_queue) or self.current_sign == "red" or self.current_action in ("delay_red", "stop2", "slow"):
            return
        if now - self.last_clear_time < 0.8:
            return
        self.clear_sign_file()
        self.last_clear_time = now

    def handle_sign(self, sign_type):
        if sign_type == "red" and not self.red_processed:
            rospy.loginfo("Sign red: enter delay_red")
            self.current_action = "delay_red"
            self.current_sign = "red"
            self.action_duration = self.delay_before_red
            self.action_start_time = rospy.get_time()
            self.red_processed = True
            return
        if self.current_sign == "red" and sign_type == "green":
            rospy.loginfo("Sign green: leave red stop")
            self.current_action = None
            self.current_sign = None
            self.green_handled = True
            self.clear_sign_file()
            return
        if sign_type == "person":
            rospy.loginfo("Sign person: stop")
            if self.current_action != "person":
                self.interrupted_action = self.current_action
                self.interrupted_sign = self.current_sign
                self.interrupted_duration = self.action_duration - (rospy.get_time() - self.action_start_time)
                self.current_action = "person"
                self.current_sign = "person"
                self.action_start_time = rospy.get_time()
                self.action_duration = self.stop_duration_person
            else:
                self.action_start_time = rospy.get_time()
                self.safe_clear_sign_file()
            return
        if sign_type == "limit" and self.limit_handled:
            return
        if sign_type == "crossing_walk" and self.crosswalk_handled:
            return
        if self.current_sign == "limit" and sign_type == "remove_limit":
            self.current_action = None
            self.current_sign = None
            self.limit_handled = False
            self.safe_clear_sign_file()
            return
        rospy.loginfo("Queued sign: %s", sign_type)
        self.sign_queue.append(sign_type)

    def clamp_angular(self, value):
        if value is None or not is_finite(value):
            return 0.0
        if self.max_angular_z and self.max_angular_z > 0:
            return max(-self.max_angular_z, min(self.max_angular_z, value))
        return value

    def latest_angular(self):
        data = self.receiver.get()
        now = time.time()
        if data is None:
            self._log_no_model("no TensorRT angular data yet")
            return None
        age = now - float(data.get("stamp", 0.0))
        if age > self.model_timeout:
            self._log_no_model("TensorRT angular data stale: %.3fs" % age)
            return None
        return self.clamp_angular(float(data.get("angular", 0.0)))

    def _log_no_model(self, text):
        now = rospy.get_time()
        if now - self.last_no_model_log > 1.0:
            rospy.logwarn(text)
            self.last_no_model_log = now

    def publish(self, linear, angular):
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = self.clamp_angular(float(angular))
        self.pub.publish(msg)

    def publish_model_drive(self, linear_speed):
        angular = self.latest_angular()
        if angular is None:
            self.publish(0.0, 0.0)
        else:
            self.publish(linear_speed, angular)

    def wait_start(self):
        if not self.wait_for_start:
            return
        rospy.loginfo("Temporal3 2D TensorRT deployment loaded. Press r then Enter to start.")
        try:
            text = raw_input("Press 'r' to start: ")
            while text.strip().lower() != "r" and not rospy.is_shutdown():
                text = raw_input("Press 'r' to start: ")
        except EOFError:
            if self.auto_start_without_stdin:
                rospy.logwarn("No interactive stdin; starting automatically by parameter.")
                return
            rospy.logerr("No interactive stdin. Keep stopped. Set ~wait_for_start:=false or ~auto_start_without_stdin:=true to run headless.")
            while not rospy.is_shutdown():
                self.publish(0.0, 0.0)
                rospy.sleep(0.2)

    def spin(self):
        self.wait_start()
        while not rospy.is_shutdown():
            now = rospy.get_time()
            if self.sign_monitor is not None:
                self.sign_monitor.poll()

            if self._run_person_delay(now):
                self.rate.sleep()
                continue
            if self._run_init_motion(now):
                self.rate.sleep()
                continue
            self._maybe_start_queued_action(now)
            if self._run_current_action(now):
                self.rate.sleep()
                continue

            self.publish_model_drive(self.default_linear_speed)
            self.rate.sleep()

    def _run_person_delay(self, now):
        if not self.person_delay_active:
            return False
        if self.person_delay_current_motion is None and self.person_delay_motions:
            self.person_delay_current_motion = self.person_delay_motions.popleft()
            self.person_delay_start_time = now
        if self.person_delay_current_motion is not None:
            elapsed = now - self.person_delay_start_time
            motion = self.person_delay_current_motion
            if elapsed < motion["duration"]:
                self.publish(motion["linear"], motion["angular"])
            else:
                self.person_delay_current_motion = None
            return True
        rospy.loginfo("Person delay trajectory completed.")
        self.person_delay_active = False
        self.green_handled = False
        self.person_delay_motions = self._build_person_delay()
        self._resume_interrupted()
        return True

    def _run_init_motion(self, now):
        if not self.initializing:
            return False
        if not self.init_motions:
            self.initializing = False
            return False
        motion = self.init_motions[0]
        if self.init_start_time is None:
            self.init_start_time = now
        elapsed = now - self.init_start_time
        if elapsed < motion["duration"]:
            self.publish(motion["linear"], motion["angular"])
        else:
            self.init_motions.popleft()
            self.init_start_time = None
        return True

    def _maybe_start_queued_action(self, now):
        if self.current_action is not None or not self.sign_queue:
            return
        sign = self.sign_queue.popleft()
        if sign == "red" and not self.red_processed:
            action = "delay_red"
            duration = self.delay_before_red
            self.red_processed = True
        elif sign == "limit":
            action = "delay_limit"
            duration = 0.01
            self.limit_handled = True
        elif sign == "crossing_walk" and not self.crosswalk_handled:
            action = "stop1"
            duration = self.stop1_duration
        else:
            return
        self.current_action = action
        self.current_sign = sign
        self.action_duration = duration
        self.action_start_time = now

    def _run_current_action(self, now):
        if self.current_action is None:
            return False
        elapsed = now - self.action_start_time

        if self.current_action == "person":
            if elapsed < self.action_duration:
                self.publish(0.0, 0.0)
            else:
                rospy.loginfo("Person stop completed.")
                self.safe_clear_sign_file()
                self.current_action = None
                self.current_sign = None
                if self.green_handled:
                    self.person_delay_active = True
                else:
                    self._resume_interrupted()
            return True

        if self.current_action == "delay_limit":
            if "red" in self.sign_queue and not self.red_processed:
                try:
                    self.sign_queue.remove("red")
                except ValueError:
                    pass
                self.current_action = "delay_red"
                self.current_sign = "red"
                self.action_duration = self.delay_before_red
                self.action_start_time = now
                self.red_processed = True
                return True
            if elapsed < self.action_duration:
                self.publish_model_drive(self.default_linear_speed)
            else:
                self.current_action = "slow"
                self.action_duration = self.slow_duration
                self.action_start_time = now
            return True

        if self.current_action == "delay_red":
            if elapsed < self.action_duration:
                self.publish_model_drive(self.default_linear_speed)
            else:
                self.current_action = "stop2"
                self.action_duration = self.stop2_duration
                self.action_start_time = now
            return True

        if self.current_action in ("stop1", "stop2"):
            if elapsed < self.action_duration:
                self.publish(0.0, 0.0)
            else:
                self.safe_clear_sign_file()
                if self.current_sign == "crossing_walk":
                    self.crosswalk_handled = True
                self.current_action = None
                self.current_sign = None
            return True

        if self.current_action == "slow":
            if "red" in self.sign_queue and not self.red_processed:
                try:
                    self.sign_queue.remove("red")
                except ValueError:
                    pass
                self.limit_handled = False
                self.current_action = "delay_red"
                self.current_sign = "red"
                self.action_duration = self.delay_before_red
                self.action_start_time = now
                self.red_processed = True
                return True
            if elapsed < self.action_duration:
                self.publish_model_drive(self.slow_linear_speed)
            else:
                self.safe_clear_sign_file()
                self.current_action = None
                self.current_sign = None
            return True

        self.current_action = None
        self.current_sign = None
        return False

    def _resume_interrupted(self):
        if self.interrupted_action:
            self.current_action = self.interrupted_action
            self.current_sign = self.interrupted_sign
            self.action_duration = max(0.0, self.interrupted_duration)
            self.action_start_time = rospy.get_time()
            self.interrupted_action = None
            self.interrupted_sign = None
            self.interrupted_duration = None

    def shutdown(self):
        rospy.loginfo("Stopping EPAIcar deployment node.")
        try:
            self.publish(0.0, 0.0)
        except Exception:
            pass
        self.receiver.stop()
        if self.infer_process is not None and self.infer_process.poll() is None:
            self.infer_process.terminate()
            try:
                self.infer_process.wait(timeout=3.0)
            except Exception:
                self.infer_process.kill()


def main():
    node = MotionController()
    node.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
