"""QoS profiles for the MED-OR-TRACE graph, loaded from ``config/qos.yaml``.

The YAML is parsed without ROS (``load_qos_config`` / ``resolve``) so it can be
validated in plain pytest; ``qos_profile`` lazily imports ``rclpy.qos`` and
builds the :class:`rclpy.qos.QoSProfile` for a topic key of
:data:`medortrace_ros.topics.SPECS`.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from medortrace_ros.topics import SPECS

_VALID = {
    "reliability": {"reliable", "best_effort"},
    "durability": {"volatile", "transient_local"},
    "history": {"keep_last", "keep_all"},
}


def default_qos_path() -> Path:
    try:
        from ament_index_python.packages import get_package_share_directory
        p = Path(get_package_share_directory("medortrace_ros")) / "config" / "qos.yaml"
        if p.is_file():
            return p
    except Exception:  # noqa: BLE001 - not in a ROS environment / package not installed
        pass
    return Path(__file__).resolve().parents[1] / "config" / "qos.yaml"


def load_qos_config(path: str | Path | None = None) -> dict:
    with open(path or default_qos_path()) as f:
        cfg = yaml.safe_load(f) or {}
    for name, prof in cfg.get("profiles", {}).items():
        for k, allowed in _VALID.items():
            if k in prof and prof[k] not in allowed:
                raise ValueError(f"qos profile {name}: {k}={prof[k]!r} not in {sorted(allowed)}")
    return cfg


def resolve(key: str, cfg: dict | None = None) -> dict:
    """Plain-dict QoS settings of a topic key (profile merged with per-topic overrides)."""
    cfg = cfg if cfg is not None else load_qos_config()
    prof_name = SPECS[key].qos
    prof = dict(cfg["profiles"][prof_name])
    prof.update(cfg.get("overrides", {}).get(key, {}))
    prof.setdefault("history", "keep_last")
    prof.setdefault("depth", 10)
    return prof


def qos_profile(key: str, cfg: dict | None = None):
    """``rclpy.qos.QoSProfile`` for a topic key (lazy rclpy import)."""
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

    p = resolve(key, cfg)
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE if p["reliability"] == "reliable" else ReliabilityPolicy.BEST_EFFORT,
        durability=(DurabilityPolicy.TRANSIENT_LOCAL if p["durability"] == "transient_local"
                    else DurabilityPolicy.VOLATILE),
        history=HistoryPolicy.KEEP_ALL if p["history"] == "keep_all" else HistoryPolicy.KEEP_LAST,
        depth=int(p["depth"]),
    )
