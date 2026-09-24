"""MED-OR-TRACE: object-chain verifier for operating-room logistics.

The package is split into an engine-agnostic autonomy core (``common``, ``world``,
``perception``, ``belief``, ``provenance``, ``planning``, ``safety``,
``manipulation``, ``autonomy``, ``eval``, ``data``, ``usd``) and thin adapters
for NVIDIA Isaac Sim (``medortrace.isaac``) and ROS 2 (``ros2_ws``).

The core never imports ``omni``/``isaacsim``/``rclpy`` so it can be unit tested,
trained and benchmarked on a workstation or CI machine.  The ``sim`` package
provides a lightweight analytic simulator ("lite backend") that implements the
same :class:`medortrace.sim.backend.SimBackend` interface as the Isaac Sim
backend, which keeps the autonomy stack identical across the two.
"""

__version__ = "0.1.0"

# Coordinate conventions used everywhere (documented in docs/architecture.md):
#   * world frame: right-handed, x east, y north, z up, metres, radians
#   * robot base frame: x forward, y left, z up (REP-103)
#   * time: seconds of simulation time since episode start (float64)
