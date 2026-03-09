import os
import sys

_BIN_DIR = os.path.join(os.path.dirname(__file__), "kernels", "bin")

if _BIN_DIR not in sys.path:
    sys.path.append(_BIN_DIR)