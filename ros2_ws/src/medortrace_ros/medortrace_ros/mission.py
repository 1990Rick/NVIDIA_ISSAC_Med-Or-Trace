"""Mission description: what the robot is told before a case (no ROS imports).

The autonomy stack is constructed from :class:`StackInputs` (surveyed room,
prior map, slots, sterile zones, fiducials, item list, pre-operative count
sheet, staff roster, start pose, dock, case duration) and the robot-side part
of the configuration.  On ROS 2 that information arrives as a *mission*: a
JSON document with the schema below, either published once on the latched
``/medortrace/mission`` topic (by ``sim_bridge_node`` or a facility server),
read from a file (physical prototype), or rebuilt locally from a scenario
config + seed (``medortrace.sim.episode.build_episode`` is deterministic, so
this equals what the simulator uses)::

    {"schema": "medortrace.mission/1", "scenario_id": str, "seed": int | null,
     "t0": float | null,   # ROS time [s] of mission time 0 (the /clock offset in simulation);
                           # null = the consumer's start time (missions exported for a physical robot)
     "inputs": {...},      # StackInputs, see stack_inputs_to_dict
     "cfg": {...}}         # robot-side config sections (ROBOT_CFG_KEYS)

Only survey-time knowledge is included - never the true world, the fault
schedule or the workflow truth (``stack_inputs_from_episode`` guarantees this).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from medortrace.autonomy.stack import StackInputs, stack_inputs_from_episode
from medortrace.common.config import deep_merge, load_config
from medortrace.common.geometry import OrientedBox
from medortrace.world.scene import ItemSpec, Landmark, SceneObject, Slot, StaffSpec, SterileZone

SCHEMA = "medortrace.mission/1"
# config sections read by AutonomyStack (and nothing about the true world / faults)
ROBOT_CFG_KEYS = ("autonomy", "robot", "sensors", "belief", "verifier", "mpc", "safety", "_policy_seed")


# ---------------------------------------------------------------------------------------------------------------------
def episode_config(scenario: str | dict, seed: int, policy: str | None = None, duration: float | None = None,
                   autonomy_override: dict | None = None) -> dict:
    """Scenario config with exactly the merges of ``medortrace.eval.runner.run_episode``.

    Keeping this identical makes a registry seed the same experiment whether it
    runs in-process or through the ROS 2 graph.
    """
    cfg = load_config(scenario) if isinstance(scenario, str) else dict(scenario)
    if policy:
        cfg = deep_merge(cfg, {"autonomy": {"policy": policy}})
    if autonomy_override:
        cfg = deep_merge(cfg, {"autonomy": autonomy_override})
    if duration:
        cfg = deep_merge(cfg, {"episode": {"duration_s": float(duration)}})
    return deep_merge(cfg, {"_policy_seed": int(seed) * 7919 + 17})


def robot_side_config(cfg: dict) -> dict:
    return {k: cfg[k] for k in ROBOT_CFG_KEYS if k in cfg}


# ---------------------------------------------------------------------------------------------------------------------
def _enc(o: Any) -> Any:
    if isinstance(o, OrientedBox):
        return {"center": o.center.tolist(), "half": o.half.tolist(), "yaw": float(o.yaw)}
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return {f.name: _enc(getattr(o, f.name)) for f in dataclasses.fields(o)}
    if isinstance(o, np.ndarray):
        return _enc(o.tolist())
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return _enc(o.item())
    if isinstance(o, float) and not np.isfinite(o):
        return None                                  # strict JSON (hand slots have no surveyed position)
    if isinstance(o, dict):
        return {str(k): _enc(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_enc(v) for v in o]
    return o


def _box(d: dict) -> OrientedBox:
    return OrientedBox(np.asarray(d["center"], float), np.asarray(d["half"], float), float(d.get("yaw", 0.0)))


def _fields(cls, d: dict) -> dict:
    names = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in d.items() if k in names}


def stack_inputs_to_dict(inp: StackInputs) -> dict:
    return _enc(inp)


def stack_inputs_from_dict(d: dict) -> StackInputs:
    prior = [SceneObject(**{**_fields(SceneObject, o), "box": _box(o["box"]), "tags": list(o.get("tags", []))})
             for o in d["prior_map"]]
    slots = [Slot(**{**_fields(Slot, s), "position": np.asarray(
        [np.nan if v is None else v for v in s["position"]], float)}) for s in d["slots"]]
    zones = [SterileZone(**{**_fields(SterileZone, z), "box": _box(z["box"])}) for z in d["zones"]]
    lms = [Landmark(lm["id"], np.asarray(lm["position"], float)) for lm in d["landmarks"]]
    items = [ItemSpec(**{**_fields(ItemSpec, i), "size": tuple(i.get("size", (0.1, 0.05, 0.02)))})
             for i in d["items"]]
    staff = [StaffSpec(**{**_fields(StaffSpec, s), "home": np.asarray(s["home"], float),
                          "waypoints": [np.asarray(w, float) for w in s.get("waypoints", [])]})
             for s in d["staff"]]
    return StackInputs(
        room=tuple(float(v) for v in d["room"]), prior_map=prior, slots=slots, zones=zones, landmarks=lms,
        items=items, initial_placement=dict(d["initial_placement"]), staff=staff,
        start_pose=np.asarray(d["start_pose"], float), dock=np.asarray(d["dock"], float),
        duration=float(d["duration"]), claim_grace=float(d.get("claim_grace", 25.0)),
        count_grace=float(d.get("count_grace", 90.0)))


# ---------------------------------------------------------------------------------------------------------------------
def mission_from_episode(ep, cfg: dict | None = None, t0: float | None = None) -> dict:
    return {"schema": SCHEMA, "scenario_id": ep.spec.scenario_id, "seed": int(ep.seed),
            "t0": None if t0 is None else float(t0),
            "inputs": stack_inputs_to_dict(stack_inputs_from_episode(ep)),
            "cfg": _enc(robot_side_config(cfg if cfg is not None else ep.cfg))}


def mission_for_scenario(scenario: str | dict, seed: int, policy: str | None = None, duration: float | None = None,
                         autonomy_override: dict | None = None, t0: float | None = None) -> dict:
    from medortrace.sim.episode import build_episode

    cfg = episode_config(scenario, seed, policy, duration, autonomy_override)
    return mission_from_episode(build_episode(cfg, seed), cfg, t0)


def mission_to_json(m: dict) -> str:
    return json.dumps(_enc(m), separators=(",", ":"))


def mission_from_json(s: str) -> dict:
    m = json.loads(s)
    validate_mission(m)
    return m


def load_mission_file(path: str | Path) -> dict:
    p = Path(path)
    text = p.read_text()
    m = yaml.safe_load(text) if p.suffix in (".yaml", ".yml") else json.loads(text)
    validate_mission(m)
    return m


def save_mission_file(m: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_enc(m), indent=1))
    return p


def validate_mission(m: dict) -> None:
    if m.get("schema") != SCHEMA:
        raise ValueError(f"unsupported mission schema {m.get('schema')!r} (expected {SCHEMA})")
    need = {"room", "prior_map", "slots", "zones", "landmarks", "items", "initial_placement", "staff",
            "start_pose", "dock", "duration"}
    missing = need - set(m.get("inputs", {}))
    if missing:
        raise ValueError(f"mission inputs missing {sorted(missing)}")


def stack_setup(m: dict, base_cfg: dict | None = None, overrides: dict | None = None) -> tuple[StackInputs, dict]:
    """(StackInputs, cfg) for ``AutonomyStack`` from a mission.

    Precedence: ``base_cfg`` (e.g. scenarios/default.yaml) < mission ``cfg`` < ``overrides``
    (node parameters such as ``autonomy.policy``).
    """
    validate_mission(m)
    cfg = deep_merge(deep_merge(base_cfg or {}, m.get("cfg") or {}), overrides or {})
    return stack_inputs_from_dict(m["inputs"]), cfg
