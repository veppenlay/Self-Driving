import importlib
import sys

name = sys.argv[1]
module = importlib.import_module(name)
print(name, "OK", getattr(module, "__file__", ""))
