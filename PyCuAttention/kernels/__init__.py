import os
import sys

# Get the path to the 'bin' subdirectory where the symlink/binary lives
_BIN_DIR = os.path.join(os.path.dirname(__file__), "bin")

# Add this internal bin to the path so the .so is discoverable
if _BIN_DIR not in sys.path:
    sys.path.append(_BIN_DIR)