"""Fault and adversity injection with ground-truth labels.

The fault model is sampled once per episode from the ``faults`` RNG stream and
applied identically by every backend (lite or Isaac Sim), so the same seed
produces the same dropout windows, clock skews, drift and map corruption.
All injected faults are recorded (``FaultModel.labels``) so that metrics like
*calibration under sensor failure* can be stratified by fault state.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from medortrace.common.geometry import OrientedBox
from medortrace.world.scene import SceneObject, SceneSpec

SENSORS = ("lidar", "camera", "radar", "acoustic", "imu", "odom", "landmarks")


@dataclass
class FaultModel:
    dropouts: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    skew: dict[str, tuple[float, float]] = field(default_factory=dict)   # sensor -> (offset_s, drift_s_per_s)
    odom_bias: tuple[float, float] = (0.0, 0.0)                          # (v scale error, omega bias rad/s)
    odom_bias_start: float = 0.0
    specular_gain: float = 1.0
    floor_wet: bool = False
    rare_geometry: list[str] = field(default_factory=list)
    occluders: list[str] = field(default_factory=list)
    map_edits: list[dict] = field(default_factory=list)
    labels: dict = field(default_factory=dict)

    def dropped(self, sensor: str, t: float) -> bool:
        return any(a <= t < b for a, b in self.dropouts.get(sensor, []))

    def stamp(self, sensor: str, t: float) -> float:
        off, drift = self.skew.get(sensor, (0.0, 0.0))
        return t + off + drift * t

    def active(self, t: float) -> dict[str, bool]:
        out = {f"dropout_{s}": self.dropped(s, t) for s in SENSORS}
        out["skew"] = any(abs(self.stamp(s, t) - t) > 0.05 for s in self.skew)
        out["odom_drift"] = (self.odom_bias != (0.0, 0.0)) and t >= self.odom_bias_start
        return out

    def any_sensor_fault(self, t: float) -> bool:
        a = self.active(t)
        return any(v for k, v in a.items())


def sample_faults(cfg: dict, duration: float, rng: np.random.Generator, hidden_cause: dict) -> FaultModel:
    fc = cfg.get("faults", {})
    fm = FaultModel()
    # --- sensor dropout windows -------------------------------------------
    for s in SENSORS:
        rate = float(fc.get("dropout_rate_per_min", {}).get(s, 0.0))
        mean_d = float(fc.get("dropout_mean_s", 6.0))
        wins = []
        t = 0.0
        while rate > 0:
            t += float(rng.exponential(60.0 / rate))
            if t >= duration:
                break
            d = float(rng.exponential(mean_d))
            wins.append((t, min(duration, t + d)))
            t += d
        if wins:
            fm.dropouts[s] = wins
    # --- clock skew -------------------------------------------------------
    for s, spec in fc.get("timestamp_skew", {}).items():
        lo, hi = spec.get("offset_s", [0.0, 0.0])
        fm.skew[s] = (float(rng.uniform(lo, hi)), float(spec.get("drift_ppm", 0.0)) * 1e-6)
    # --- localisation drift ------------------------------------------------
    ld = fc.get("loc_drift")
    if ld:
        fm.odom_bias = (float(rng.uniform(*ld.get("v_scale_error", [0.0, 0.0]))),
                        float(rng.uniform(*ld.get("omega_bias", [0.0, 0.0]))))
        fm.odom_bias_start = float(ld.get("start_s", 0.0))
    fm.specular_gain = float(fc.get("specular_gain", 1.0))
    fm.floor_wet = bool(fc.get("floor_wet", False))
    fm.rare_geometry = list(fc.get("rare_geometry", []))
    fm.occluders = list(fc.get("adversarial_occluders", []))
    mc = fc.get("map_corruption")
    if mc:
        fm.map_edits.append({"kind": "shift_movables", "n": int(mc.get("shift_n", 1)),
                             "max_shift": float(mc.get("max_shift", 0.5))})
        if mc.get("phantom_n", 0):
            fm.map_edits.append({"kind": "phantom", "n": int(mc["phantom_n"])})
        if mc.get("drop_n", 0):
            fm.map_edits.append({"kind": "drop", "n": int(mc["drop_n"])})
    # --- CF-D hidden cause --------------------------------------------------
    if hidden_cause.get("factor") == "CF-D":
        # Both arms: the fiducial camera is blinded for a long window (e.g. a
        # boom light pointed at it), so localisation relies on odometry+lidar.
        fm.dropouts["landmarks"] = sorted(fm.dropouts.get("landmarks", []) + [(20.0, 110.0)])
        if hidden_cause.get("value") == "loc_drift":
            fm.odom_bias = (0.08, 0.03)
            fm.odom_bias_start = 20.0
        else:
            fm.map_edits.append({"kind": "cf_d_cart_moved", "object": "cart_1", "shift": [0.55, -0.35]})
    fm.labels = {
        "dropouts": fm.dropouts, "skew": fm.skew, "odom_bias": fm.odom_bias,
        "specular_gain": fm.specular_gain, "floor_wet": fm.floor_wet,
        "rare_geometry": fm.rare_geometry, "occluders": fm.occluders, "map_edits": fm.map_edits,
    }
    return fm


def apply_truth_modifications(spec: SceneSpec, fm: FaultModel, rng: np.random.Generator) -> None:
    """Modify the *true* world for rare geometry, adversarial occlusion and CF-D."""
    W, D, _ = spec.room
    tbl = spec.object("or_table").box
    for g in fm.rare_geometry:
        if g == "iv_pole_fallen":
            # thin, low obstacle across the south aisle: below most lidar rings
            y = float(tbl.center[1] - 1.9 - rng.uniform(0.2, 0.5))
            spec.objects.append(SceneObject("fallen_iv_pole", "clutter",
                                            OrientedBox((tbl.center[0] - 1.8, y, 0.03), (0.9, 0.03, 0.03), 0.15),
                                            "stainless_steel_brushed", "iv_pole", tags=["rare"]))
        elif g == "boom_lowered":
            spec.objects.append(SceneObject("lowered_boom", "monitor",
                                            OrientedBox((tbl.center[0] - 1.9, tbl.center[1] - 1.9, 1.35), (0.35, 0.2, 0.12)),
                                            "monitor_glass", "monitor", tags=["rare"]))
        elif g == "drape_trailing":
            spec.objects.append(SceneObject("trailing_drape", "drape",
                                            OrientedBox((tbl.center[0] + 0.4, tbl.center[1] - 1.35, 0.01), (0.5, 0.3, 0.01)),
                                            "surgical_drape", "drape", sterile=True, tags=["rare"]))
        elif g == "cart_tipped":
            c = spec.object("cart_2")
            c.box = OrientedBox(c.box.center * np.array([1, 1, 0]) + np.array([0, -0.4, 0.25]),
                                np.array([0.5, 0.35, 0.25]), 0.3)
            c.tags.append("rare")
    for occ in fm.occluders:
        if occ == "linen_stack_kick_bucket":
            kb = spec.object("kick_bucket_1").box.center
            spec.objects.append(SceneObject("occluder_linen", "clutter",
                                            OrientedBox((kb[0] - 0.05, kb[1] - 0.5, 0.6), (0.3, 0.15, 0.6)),
                                            "surgical_drape", "linen_cart", tags=["adversarial"]))
        elif occ == "screen_back_table":
            bt = spec.object("back_table").box.center
            spec.objects.append(SceneObject("occluder_screen", "panel",
                                            OrientedBox((bt[0] + 1.6, bt[1] - 0.2, 0.9), (0.03, 0.5, 0.9)),
                                            "stainless_steel_brushed", "cabinet", tags=["adversarial", "specular"]))
    for e in fm.map_edits:
        if e["kind"] == "cf_d_cart_moved":
            c = spec.object(e["object"])
            c.box = OrientedBox(c.box.center + np.array([*e["shift"], 0.0]), c.box.half, c.box.yaw)
            c.tags.append("moved_since_survey")
            # items and slots on the cart move with it
            for s in spec.slots:
                if s.anchor == e["object"]:
                    s.position = s.position + np.array([*e["shift"], 0.0])


def build_prior_map(true_spec: SceneSpec, surveyed_spec: SceneSpec, fm: FaultModel,
                    rng: np.random.Generator) -> list[SceneObject]:
    """The robot's prior (surveyed) static map, with corruption applied.

    ``surveyed_spec`` is the scene *before* truth modifications (the state at
    survey time); clutter, hidden-cause objects and rare geometry are absent.
    """
    prior = [copy.deepcopy(o) for o in surveyed_spec.objects if o.kind not in ("clutter",)]
    movables = [o for o in prior if o.movable]
    for e in fm.map_edits:
        if e["kind"] == "shift_movables":
            for o in rng.choice(movables, size=min(e["n"], len(movables)), replace=False) if movables else []:
                sh = rng.uniform(-e["max_shift"], e["max_shift"], 2)
                o.box = OrientedBox(o.box.center + np.array([*sh, 0.0]), o.box.half, o.box.yaw)
                o.tags.append("corrupted")
        elif e["kind"] == "phantom":
            W, D, _ = true_spec.room
            for k in range(e["n"]):
                p = rng.uniform([1.0, 1.0], [W - 1.0, D - 1.0])
                prior.append(SceneObject(f"phantom_{k}", "clutter", OrientedBox((p[0], p[1], 0.4), (0.2, 0.2, 0.4)),
                                         "plastic_hdpe", "clutter", tags=["corrupted", "phantom"]))
        elif e["kind"] == "drop":
            cands = [o for o in prior if o.kind in ("cart", "kick_bucket", "waste_bin")]
            for o in cands[: e["n"]]:
                prior.remove(o)
    return prior
