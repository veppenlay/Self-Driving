import importlib
import sys

print("python", sys.executable)
print("version", sys.version.replace("\n", " "))

for name in [
    "tensorrt",
    "pycuda.driver",
    "pycuda.autoinit",
    "cv2",
    "numpy",
    "rospy",
    "geometry_msgs.msg",
    "e2e.msg",
]:
    try:
        module = importlib.import_module(name)
        path = getattr(module, "__file__", "")
        print(name, "OK", path)
    except Exception as exc:
        print(name, "FAIL", repr(exc))
