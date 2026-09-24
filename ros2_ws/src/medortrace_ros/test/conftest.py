"""Make ``medortrace_ros`` and ``medortrace`` importable for plain ``pytest`` (no colcon, no ROS)."""

import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]          # ros2_ws/src/medortrace_ros
ROOT = Path(__file__).resolve().parents[4]         # repository root
for p in (PKG, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
