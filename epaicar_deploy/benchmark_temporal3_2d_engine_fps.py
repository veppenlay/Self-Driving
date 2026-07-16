#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Benchmark raw TensorRT engine FPS without camera/preprocess/control logic."""

import argparse
import ctypes
import os
import time

import numpy as np
import tensorrt as trt


EXPECTED_INPUT_SHAPE = (1, 9, 144, 192)
EXPECTED_OUTPUT_SHAPE = (1, 2)


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
        self.lib.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        self.lib.cudaMemset.restype = ctypes.c_int
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

    def memset(self, dst, value, nbytes):
        self.check(self.lib.cudaMemset(dst, int(value), ctypes.c_size_t(nbytes)), "cudaMemset")

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


class EngineBenchmark(object):
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

        self.input_name = self.engine.get_binding_name(self.input_index)
        self.output_name = self.engine.get_binding_name(self.output_index)
        self.input_shape = tuple(self.engine.get_binding_shape(self.input_index))
        self.output_shape = tuple(self.engine.get_binding_shape(self.output_index))
        if self.input_shape != EXPECTED_INPUT_SHAPE:
            raise RuntimeError("wrong input shape {}, expected {}".format(self.input_shape, EXPECTED_INPUT_SHAPE))
        if self.output_shape != EXPECTED_OUTPUT_SHAPE:
            raise RuntimeError("wrong output shape {}, expected {}".format(self.output_shape, EXPECTED_OUTPUT_SHAPE))

        self.input_dtype = trt.nptype(self.engine.get_binding_dtype(self.input_index))
        self.output_dtype = trt.nptype(self.engine.get_binding_dtype(self.output_index))
        self.host_input = np.zeros(self.input_shape, dtype=self.input_dtype)
        self.host_output = np.empty(self.output_shape, dtype=self.output_dtype)
        self.device_input = self.cuda.malloc(self.host_input.nbytes)
        self.device_output = self.cuda.malloc(self.host_output.nbytes)
        self.bindings = [0] * self.engine.num_bindings
        self.bindings[self.input_index] = int(self.device_input.value)
        self.bindings[self.output_index] = int(self.device_output.value)
        self.cuda.memcpy_htod(self.device_input, self.host_input)
        self.cuda.memset(self.device_output, 0, self.host_output.nbytes)
        self.cuda.synchronize()

    def execute_only_once(self):
        ok = self.context.execute_v2(self.bindings)
        if not ok:
            raise RuntimeError("TensorRT execute_v2 returned false")
        self.cuda.synchronize()

    def end_to_end_once(self):
        self.cuda.memcpy_htod(self.device_input, self.host_input)
        ok = self.context.execute_v2(self.bindings)
        if not ok:
            raise RuntimeError("TensorRT execute_v2 returned false")
        self.cuda.synchronize()
        self.cuda.memcpy_dtoh(self.host_output, self.device_output)

    def close(self):
        self.cuda.free(self.device_input)
        self.cuda.free(self.device_output)
        self.device_input = None
        self.device_output = None


def percentile(values, pct):
    if not values:
        return 0.0
    arr = sorted(values)
    idx = int(round((len(arr) - 1) * pct / 100.0))
    return arr[max(0, min(len(arr) - 1, idx))]


def run_benchmark(name, fn, warmup, iterations):
    for _ in range(warmup):
        fn()

    latencies_ms = []
    start = time.time()
    for _ in range(iterations):
        t0 = time.time()
        fn()
        latencies_ms.append((time.time() - t0) * 1000.0)
    elapsed = time.time() - start
    fps = float(iterations) / elapsed if elapsed > 0 else 0.0
    mean_ms = sum(latencies_ms) / len(latencies_ms)

    print("{}: iterations={} elapsed_sec={:.6f} fps={:.3f}".format(name, iterations, elapsed, fps))
    print(
        "{} latency_ms: mean={:.6f} min={:.6f} p50={:.6f} p90={:.6f} p99={:.6f} max={:.6f}".format(
            name,
            mean_ms,
            min(latencies_ms),
            percentile(latencies_ms, 50),
            percentile(latencies_ms, 90),
            percentile(latencies_ms, 99),
            max(latencies_ms),
        )
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Benchmark pure TensorRT engine FPS")
    parser.add_argument("--engine", default="/home/epaicar/results/0703_temporal3_2d_fp16.engine")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--mode", choices=["execute", "end2end", "both"], default="both")
    return parser


def main():
    args = build_parser().parse_args()
    bench = EngineBenchmark(args.engine)
    try:
        print("engine={}".format(args.engine))
        print("input={} shape={} dtype={}".format(bench.input_name, bench.input_shape, bench.input_dtype))
        print("output={} shape={} dtype={}".format(bench.output_name, bench.output_shape, bench.output_dtype))
        print("warmup={} iterations={} mode={}".format(args.warmup, args.iterations, args.mode))
        if args.mode in ("execute", "both"):
            run_benchmark("execute_only_no_h2d_d2h", bench.execute_only_once, args.warmup, args.iterations)
        if args.mode in ("end2end", "both"):
            run_benchmark("end2end_h2d_execute_d2h", bench.end_to_end_once, args.warmup, args.iterations)
    finally:
        bench.close()


if __name__ == "__main__":
    main()
