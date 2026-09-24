"""Trajectory-level dataset export.

One directory per episode::

    <out>/<scenario_id>__s<seed>__<policy>/
        meta.json            scenario, seed, config hash, hidden cause, fault labels, metrics
        trajectory.npz       per-tick arrays (observations summary, beliefs, actions, energy, truth)
        events.jsonl         workflow log, verdicts, safety events, diagnoses, operator actions
        provenance.json      PROV-JSON export of the hash-chained provenance graph
        scene_graph.json     final probabilistic scene graph
        raw_obs.npz          (optional) downsampled raw lidar ranges / radar / camera dets

The schema is documented in docs/dataset_schema.md and is identical for
lite and Isaac Sim episodes (``meta.json:backend`` says which).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MODE_CODE = {"NOMINAL": 0, "CAUTION": 1, "STOP": 2, "RETREAT": 3, "HANDOVER": 4}
SCHEMA_VERSION = "1.0"


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if hasattr(o, "value"):
        return o.value
    if hasattr(o, "__dict__"):
        return o.__dict__
    return str(o)


class TrajectoryWriter:
    def __init__(self, out_dir: str | Path, save_raw: bool = False):
        self.root = Path(out_dir)
        self.save_raw = save_raw
        self.raw: dict[str, list] = {"t": [], "lidar_ranges": [], "radar": [], "camera": []}

    def add_raw(self, t: float, bundle) -> None:
        if not self.save_raw:
            return
        self.raw["t"].append(t)
        self.raw["lidar_ranges"].append(bundle.lidar.ranges[::4].astype(np.float16) if bundle.lidar is not None else None)
        self.raw["radar"].append([(d.range, d.azimuth, d.radial_velocity, d.rcs_dbsm) for d in bundle.radar.detections]
                                 if bundle.radar is not None else [])
        self.raw["camera"].append([(d.cls, d.bearing, d.range, float(d.logits.max())) for d in bundle.camera.detections]
                                  if bundle.camera is not None else [])

    def write(self, name: str, ep, stack, truth, metrics: dict, verdicts: list, operator_log: list,
              backend_name: str, policy: str) -> Path:
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        tel = stack.telemetry
        items = list(stack.items.items)
        slot_ids = stack.items.slot_ids
        sidx = {s: i for i, s in enumerate(slot_ids)}
        n = min(len(tel), len(truth.t))
        arr = {
            "t": np.array(truth.t[:n]),
            "pose_true": np.array(truth.robot[:n]),
            "pose_est": np.array([s.pose_est for s in tel[:n]]),
            "pose_std": np.array([s.pose_std for s in tel[:n]]),
            "nis": np.array([s.nis for s in tel[:n]]),
            "action_v": np.array([s.cmd[0] for s in tel[:n]]),
            "action_omega": np.array([s.cmd[1] for s in tel[:n]]),
            "acoustic_probe": np.array([s.probe or "" for s in tel[:n]]),
            "mode": np.array([MODE_CODE[s.mode] for s in tel[:n]], dtype=np.int8),
            "mpc_min_pred_clearance": np.array([s.mpc_min_clear for s in tel[:n]]),
            "mpc_collision_prob": np.array([s.mpc_coll_prob for s in tel[:n]]),
            "path_entropy": np.array([s.path_entropy for s in tel[:n]]),
            "n_tracks": np.array([s.n_tracks for s in tel[:n]], dtype=np.int16),
            "human_clearance_est": np.array([s.human_clearance_est for s in tel[:n]]),
            "item_entropy": np.array([[s.item_entropy[i] for i in items] for s in tel[:n]]),
            "agents_true": np.array(truth.agents[:n]),
            "agents_shadow": np.array(truth.shadow[:n]),
            "battery_wh": np.array(truth.battery[:n]),
            "energy_used_wh": np.array(truth.energy[:n]),
            "collision_agent": np.array(truth.collision_agent[:n], dtype=bool),
            "collision_static": np.array(truth.collision_static[:n], dtype=bool),
            "sterile_breach": np.array(truth.in_keepout[:n], dtype=bool),
            "fault_active": np.array(truth.fault_any[:n], dtype=bool),
            "item_slot_true": np.array([[sidx.get(ts[i], -1) for i in items] for ts in truth.item_slots[:n]],
                                       dtype=np.int16),
            "belief_items": np.array(getattr(stack, "belief_history", []), dtype=np.float32),
            "belief_items_t": np.array(getattr(stack, "belief_history_t", [])),
        }
        np.savez_compressed(d / "trajectory.npz", **arr)
        meta = {
            "schema_version": SCHEMA_VERSION, "backend": backend_name, "policy": policy,
            "scenario_id": ep.spec.scenario_id, "family": ep.spec.family, "seed": ep.seed, "cfg_hash": ep.cfg_hash,
            "hidden_cause": ep.spec.hidden_cause, "hidden_notes": ep.workflow.hidden_notes,
            "faults": ep.faults.labels, "nuisance": ep.spec.nuisance, "room": ep.spec.room,
            "items": items, "slots": slot_ids, "agents": truth.agent_names, "metrics": metrics,
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=1, default=_json_default))
        with open(d / "events.jsonl", "w") as f:
            for e in ep.workflow.log:
                f.write(json.dumps({"kind": "workflow_log", "t": e.t, "type": e.type.value, "item": e.item_id,
                                    "src": e.src, "dst": e.dst, "id": e.event_id}) + "\n")
            for m in ep.workflow.truth:
                f.write(json.dumps({"kind": "truth_move", "t": m.t, "item": m.item_id, "src": m.src, "dst": m.dst,
                                    "cause": m.cause}) + "\n")
            for v in verdicts:
                f.write(json.dumps({"kind": "verdict", **v.__dict__}, default=_json_default) + "\n")
            for s in stack.sup.events:
                f.write(json.dumps({"kind": "safety", **s.__dict__}, default=_json_default) + "\n")
            for g in stack.diag.diagnoses:
                f.write(json.dumps({"kind": "diagnosis", **g.__dict__}, default=_json_default) + "\n")
            for o in operator_log:
                f.write(json.dumps({"kind": "operator", **o}, default=_json_default) + "\n")
        (d / "provenance.json").write_text(json.dumps(stack.prov.to_prov_json(), default=_json_default))
        (d / "scene_graph.json").write_text(json.dumps(stack.scene_graph(), default=_json_default))
        if self.save_raw:
            np.savez_compressed(d / "raw_obs.npz", t=np.array(self.raw["t"]),
                                lidar_ranges=np.array([r if r is not None else np.full_like(next((x for x in self.raw["lidar_ranges"] if x is not None), np.zeros(1, np.float16)), np.nan) for r in self.raw["lidar_ranges"]]),
                                radar=np.array(json.dumps(self.raw["radar"])), camera=np.array(json.dumps(self.raw["camera"])))
        return d
