"""ROS 2 middleware layer for MED-OR-TRACE.

The autonomy stack (``medortrace.autonomy.stack.AutonomyStack``) is engine- and
middleware-agnostic: it consumes a :class:`medortrace.common.msgs.SensorBundle`
per control tick and returns a safety-gated
:class:`medortrace.common.msgs.VelocityCommand`.  This package only moves data
between ROS 2 topics and those dataclasses, so the *same* stack runs against
the lite simulator, Isaac Sim or the physical prototype:

    assembler   per-sensor messages -> SensorBundle at the control rate (no ROS)
    records     stack state -> publishable records (no ROS)
    mission     StackInputs <-> JSON mission description (no ROS)
    frames      static TF of the sensor rig from configs/robot/rig.yaml (no ROS)
    topics      topic registry (extends medortrace.isaac.ros2_bridge.TOPICS)
    qos         QoS profiles from config/qos.yaml (lazy rclpy import)
    convert     ROS messages <-> medortrace dataclasses (lazy ROS imports)
    *_node      rclpy nodes (lazy ROS imports; importable without ROS)

``medortrace`` itself is located, in order, on ``sys.path`` (``pip install -e``),
under ``$MEDORTRACE_ROOT``, or by walking up from this file (source checkout
or ``colcon build --symlink-install``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__version__ = "0.1.0"


def _ensure_medortrace_importable() -> None:
    try:
        import medortrace  # noqa: F401
        return
    except ImportError:
        pass
    candidates = []
    env = os.environ.get("MEDORTRACE_ROOT")
    if env:
        candidates.append(Path(env))
    candidates += list(Path(__file__).resolve().parents)
    for root in candidates:
        if (root / "medortrace" / "__init__.py").is_file():
            sys.path.insert(0, str(root))
            return


_ensure_medortrace_importable()
