import sys

print("python", sys.executable)
import numpy
print("numpy", numpy.__file__)
import cv2
print("cv2", cv2.__file__)
import tensorrt
print("tensorrt", tensorrt.__file__)

sys.path.append("/home/epaicar/archiconda3/envs/pycuda/lib/python3.6/site-packages")
import pycuda.driver
print("pycuda.driver", pycuda.driver.__file__)
import pycuda.autoinit
print("pycuda.autoinit", pycuda.autoinit.__file__)
