#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import ctypes
import json
import math
import os
import socket
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt

from steering_hmm import SteeringOnlyHMMOnlineFilter


NUM_FRAMES = 3
INPUT_HEIGHT = 144
INPUT_WIDTH = 192
INPUT_SHAPE = (1, 3 * NUM_FRAMES, INPUT_HEIGHT, INPUT_WIDTH)
ROI_BOTTOM_RATIO = 0.7


class CudaRuntime(object):
    HOST_TO_DEVICE = 1
    DEVICE_TO_HOST = 2

    def __init__(self):
        self.lib = self._load_cudart()
        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaMalloc.restype = ctypes.c_int
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaFree.restype = ctypes.c_int
        self.lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self.lib.cudaMemcpy.restype = ctypes.c_int
        self.lib.cudaDeviceSynchronize.argtypes = []
        self.lib.cudaDeviceSynchronize.restype = ctypes.c_int

    def _load_cudart(self):
        candidates = [
            "libcudart.so",
            "libcudart.so.10.2",
            "/usr/local/cuda/lib64/libcudart.so",
            "/usr/local/cuda-10.2/lib64/libcudart.so",
        ]
        last_error = None
        for name in candidates:
            try:
                return ctypes.CDLL(name)
            except OSError as exc:
                last_error = exc
        raise RuntimeError("failed to load libcudart: {}".format(last_error))

    def check(self, code, action):
        if code != 0:
            raise RuntimeError("{} failed with cuda error code {}".format(action, code))

    def malloc(self, nbytes):
        ptr = ctypes.c_void_p()
        self.check(self.lib.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes)), "cudaMalloc")
        return ptr

    def free(self, ptr):
        if ptr:
            self.lib.cudaFree(ptr)

    def memcpy_htod(self, dst, src_array):
        src = src_array.ctypes.data_as(ctypes.c_void_p)
        self.check(
            self.lib.cudaMemcpy(dst, src, ctypes.c_size_t(src_array.nbytes), self.HOST_TO_DEVICE),
            "cudaMemcpy H2D",
        )

    def memcpy_dtoh(self, dst_array, src):
        dst = dst_array.ctypes.data_as(ctypes.c_void_p)
        self.check(
            self.lib.cudaMemcpy(dst, src, ctypes.c_size_t(dst_array.nbytes), self.DEVICE_TO_HOST),
            "cudaMemcpy D2H",
        )

    def synchronize(self):
        self.check(self.lib.cudaDeviceSynchronize(), "cudaDeviceSynchronize")


class TensorRTInfer(object):
    def __init__(self, engine_path):
        if not os.path.exists(engine_path):
            raise RuntimeError("engine not found: {}".format(engine_path))
        self.cuda = CudaRuntime()
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("failed to deserialize engine: {}".format(engine_path))
        self.context = self.engine.create_execution_context()

        self.input_index = None
        self.output_index = None
        for idx in range(self.engine.num_bindings):
            if self.engine.binding_is_input(idx):
                self.input_index = idx
            else:
                self.output_index = idx
        if self.input_index is None or self.output_index is None:
            raise RuntimeError("engine must have one input and one output")

        self.input_shape = tuple(self.engine.get_binding_shape(self.input_index))
        self.output_shape = tuple(self.engine.get_binding_shape(self.output_index))
        if self.input_shape != INPUT_SHAPE:
            raise RuntimeError("wrong engine input shape {}, expected {}".format(self.input_shape, INPUT_SHAPE))
        # Locked steering engine outputs (1, 2) = [steering, e_y]; the P1 affordance engine may
        # output (1, 3) = [steering, e_y, theta]. Both are accepted; geometry mode reads e_y/theta.
        if self.output_shape not in ((1, 2), (1, 3)):
            raise RuntimeError("unsupported engine output shape {}, expected (1, 2) or (1, 3)".format(self.output_shape))
        self.output_dim = int(self.output_shape[1])

        self.input_dtype = trt.nptype(self.engine.get_binding_dtype(self.input_index))
        self.output_dtype = trt.nptype(self.engine.get_binding_dtype(self.output_index))
        self.host_input = np.empty(self.input_shape, dtype=self.input_dtype)
        self.host_output = np.empty(self.output_shape, dtype=self.output_dtype)
        self.device_input = self.cuda.malloc(self.host_input.nbytes)
        self.device_output = self.cuda.malloc(self.host_output.nbytes)
        self.bindings = [0] * self.engine.num_bindings
        self.bindings[self.input_index] = int(self.device_input.value)
        self.bindings[self.output_index] = int(self.device_output.value)

    def infer(self, input_array):
        if input_array.shape != self.host_input.shape:
            raise RuntimeError("input shape {} does not match {}".format(input_array.shape, self.host_input.shape))
        self.host_input[...] = np.ascontiguousarray(input_array, dtype=self.input_dtype)
        self.cuda.memcpy_htod(self.device_input, self.host_input)
        ok = self.context.execute_v2(self.bindings)
        if not ok:
            raise RuntimeError("TensorRT execute_v2 returned false")
        self.cuda.synchronize()
        self.cuda.memcpy_dtoh(self.host_output, self.device_output)
        return self.host_output.copy()

    def close(self):
        self.cuda.free(self.device_input)
        self.cuda.free(self.device_output)
        self.device_input = None
        self.device_output = None


class PostProcessor(object):
    def __init__(self, scale=1.0, offset=0.0, max_abs=3.0, deadzone=0.0, ema_alpha=1.0):
        self.scale = float(scale)
        self.offset = float(offset)
        self.max_abs = float(max_abs)
        self.deadzone = float(deadzone)
        self.ema_alpha = float(ema_alpha)
        self.prev = None

    def apply(self, raw):
        value = float(raw) * self.scale + self.offset
        if not math.isfinite(value):
            value = 0.0
        if abs(value) < self.deadzone:
            value = 0.0
        if self.max_abs > 0:
            value = max(-self.max_abs, min(self.max_abs, value))
        if self.prev is not None and 0.0 < self.ema_alpha < 1.0:
            value = self.ema_alpha * value + (1.0 - self.ema_alpha) * self.prev
        self.prev = value
        return value


def imread_bgr(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def preprocess_frame_bgr(frame_bgr):
    height, width = frame_bgr.shape[:2]
    crop_height = max(1, int(round(height * ROI_BOTTOM_RATIO)))
    roi = frame_bgr[max(0, height - crop_height):height, 0:width]
    resized = cv2.resize(roi, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    arr = hsv.astype(np.float32) * (1.0 / 255.0)
    return arr.transpose(2, 0, 1).astype(np.float32, copy=False)


def stack_chw_frames(chw_frames):
    if len(chw_frames) != NUM_FRAMES:
        raise RuntimeError("expected {} frames, got {}".format(NUM_FRAMES, len(chw_frames)))
    return np.concatenate(list(chw_frames), axis=0)[None, ...].astype(np.float32, copy=False)


def parse_frame_index(path):
    stem = Path(path).stem
    prefix = stem.split("_", 1)[0]
    return int(prefix) if prefix.isdigit() else None


def find_frame_candidate(image_path, index):
    image_path = Path(image_path)
    suffixes = [image_path.suffix, ".jpg", ".jpeg", ".png", ".bmp"]
    for suffix in suffixes:
        matches = sorted(image_path.parent.glob("{}_*{}".format(index, suffix)))
        if matches:
            return matches[0]
    return None


def load_temporal_image_stack(image_path, frame_stride=1):
    image_path = Path(image_path)
    current = parse_frame_index(image_path)
    frame_paths = []
    frames = []
    last_valid = image_path
    for offset in range(NUM_FRAMES - 1, -1, -1):
        candidate = None
        if current is not None:
            target = current - offset * int(frame_stride)
            if target >= 0:
                candidate = find_frame_candidate(image_path, target)
        if candidate is None:
            candidate = last_valid
        bgr = imread_bgr(candidate)
        if bgr is None:
            raise RuntimeError("failed to read frame: {}".format(candidate))
        frames.append(preprocess_frame_bgr(bgr))
        frame_paths.append(str(candidate))
        last_valid = candidate
    return stack_chw_frames(frames), frame_paths


def parse_camera(value):
    try:
        return int(value)
    except ValueError:
        return value


def open_camera(camera, width, height, fps):
    source = parse_camera(camera)
    if isinstance(source, str) and source.startswith("/dev/"):
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps > 0:
        cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def maybe_bind_cpu(cpu_core):
    if cpu_core < 0:
        return
    try:
        os.sched_setaffinity(0, {cpu_core})
    except Exception as exc:
        print("warning: failed to bind cpu core {}: {}".format(cpu_core, exc), file=sys.stderr)


def build_parser():
    parser = argparse.ArgumentParser(description="EPAIcar temporal3 2D TensorRT UDP inference server")
    parser.add_argument("--engine", default="/home/epaicar/results/0703_temporal3_2d_fp16.engine")
    parser.add_argument("--camera", default="/dev/video0")
    parser.add_argument("--image", default=None, help="Single image test mode. Uses previous frames by filename when available.")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--udp-host", default="127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=15050)
    parser.add_argument("--capture-width", type=int, default=640)
    parser.add_argument("--capture-height", type=int, default=480)
    parser.add_argument("--capture-fps", type=int, default=30)
    parser.add_argument("--rate-hz", type=float, default=30.0)
    parser.add_argument("--angular-scale", type=float, default=1.0)
    parser.add_argument("--angular-offset", type=float, default=0.0)
    parser.add_argument("--max-angular", type=float, default=3.0)
    parser.add_argument("--deadzone", type=float, default=0.0)
    parser.add_argument("--ema-alpha", type=float, default=1.0)
    parser.add_argument(
        "--hmm-model",
        default=str(Path(__file__).with_name("steering_only_hmm_online_model.json")),
        help="Locked steering-only HMM JSON. Pass an empty string to disable it.",
    )
    parser.add_argument(
        "--control-source",
        choices=["steering_hmm", "geometry"],
        default="steering_hmm",
        help="'steering_hmm' (locked default): angular = HMM(steering). "
             "'geometry' (exploration): angular = frozen_controller(e_y, theta).",
    )
    parser.add_argument(
        "--controller-json",
        default=None,
        help="Frozen geometry controller JSON (required when --control-source geometry).",
    )
    parser.add_argument("--nominal-speed", type=float, default=0.5, help="Speed fed to the geometry controller.")
    parser.add_argument("--cpu-core", type=int, default=-1)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--random-input", action="store_true")
    parser.add_argument("--print-every", type=int, default=30)
    return parser


def main():
    args = build_parser().parse_args()
    maybe_bind_cpu(args.cpu_core)

    infer = TensorRTInfer(args.engine)
    post = PostProcessor(
        scale=args.angular_scale,
        offset=args.angular_offset,
        max_abs=args.max_angular,
        deadzone=args.deadzone,
        ema_alpha=args.ema_alpha,
    )
    hmm = SteeringOnlyHMMOnlineFilter(args.hmm_model) if str(args.hmm_model).strip() else None

    controller = None
    if args.control_source == "geometry":
        if not str(args.controller_json).strip():
            raise RuntimeError("--control-source geometry requires --controller-json")
        from geometry_controller import GeometryController

        controller = GeometryController.from_json(args.controller_json)
        print("control_source=geometry controller={} kind={}".format(args.controller_json, controller.kind))

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = (args.udp_host, args.udp_port)

    cap = None
    history = deque(maxlen=NUM_FRAMES)
    if args.image is None and not args.random_input:
        cap = open_camera(args.camera, args.capture_width, args.capture_height, args.capture_fps)
        if not cap.isOpened():
            raise RuntimeError("failed to open camera {}".format(args.camera))

    seq = 0
    period = 1.0 / args.rate_hz if args.rate_hz > 0 else 0.0
    last_print = time.time()
    try:
        while True:
            start = time.time()
            frame_paths = []
            if args.random_input:
                inp = np.zeros(INPUT_SHAPE, dtype=np.float32)
            elif args.image is not None:
                inp, frame_paths = load_temporal_image_stack(args.image, frame_stride=args.frame_stride)
            else:
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.01)
                    continue
                chw = preprocess_frame_bgr(frame)
                history.append(chw)
                while len(history) < NUM_FRAMES:
                    history.appendleft(chw)
                inp = stack_chw_frames(history)

            output = infer.infer(inp).reshape(-1)
            raw_steering = float(output[0])
            raw_e_y = float(output[1]) if output.size > 1 else 0.0
            raw_theta = float(output[2]) if output.size > 2 else 0.0
            seq += 1
            if controller is not None:
                # Geometry mode: angular = frozen_controller(e_y, theta); steering/HMM bypassed.
                geo_cmd = controller.step(raw_e_y, raw_theta, v=args.nominal_speed, dt=period if period > 0 else 1.0 / 30.0)
                hmm_steering = raw_steering
                steering = post.apply(geo_cmd)
            else:
                # Locked default: angular = HMM(steering).
                hmm_steering = hmm.apply(raw_steering) if hmm is not None else raw_steering
                steering = post.apply(hmm_steering)
            payload = {
                "seq": seq,
                "stamp": time.time(),
                "raw": raw_steering,
                "hmm": hmm_steering,
                "steering": steering,
                "angular": steering,
                "e_y": raw_e_y,
                "theta": raw_theta,
                "control_source": args.control_source,
            }
            sock.sendto(json.dumps(payload, separators=(",", ":")).encode("ascii"), dest)

            if args.print_every > 0 and (seq == 1 or seq % args.print_every == 0):
                now = time.time()
                fps = args.print_every / max(now - last_print, 1e-6) if seq > 1 else 0.0
                print("seq={} raw={:.8f} hmm={:.8f} steering={:.8f} e_y={:.8f} input_shape={} fps={:.2f}".format(
                    seq, raw_steering, hmm_steering, steering, raw_e_y, inp.shape, fps
                ))
                if frame_paths:
                    print("frames={}".format([Path(p).name for p in frame_paths]))
                sys.stdout.flush()
                last_print = now

            if args.once:
                break
            elapsed = time.time() - start
            if period > elapsed:
                time.sleep(period - elapsed)
    finally:
        if cap is not None:
            cap.release()
        infer.close()


if __name__ == "__main__":
    main()
