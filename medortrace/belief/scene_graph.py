"""Probabilistic scene graph assembled from the belief layers.

Nodes: room, static objects (prior map, with change flags), slots, items
(with categorical beliefs), tracked people (with covariance and role), the
robot (with pose covariance) and sterile zones.  Edges carry probabilities:
``on`` / ``in`` / ``under`` (item-slot), ``held_by`` (item-person),
``part_of`` (slot-object), ``near`` (person-object), ``inside`` (person-zone).

Exported as JSON for logging and as ``medortrace_msgs/SceneGraph`` on ROS 2.
"""

from __future__ import annotations

import numpy as np

REL = {"surface": "on", "container": "in", "under_drape": "under", "floor": "on_floor", "hand": "held_by",
       "elsewhere": "unknown"}


def build_scene_graph(t: float, prior_map, slots, belief, tracks, pose, pose_cov, zones, min_p: float = 0.05) -> dict:
    nodes, edges = [], []
    nodes.append({"id": "robot", "type": "robot", "pose": np.asarray(pose).round(3).tolist(),
                  "pose_std": np.sqrt(np.diag(pose_cov)).round(3).tolist()})
    for o in prior_map:
        nodes.append({"id": o.name, "type": "object", "class": o.semantic, "sterile": o.sterile,
                      "center": o.box.center.round(3).tolist(), "tags": list(o.tags)})
    for z in zones:
        nodes.append({"id": f"zone:{z.name}", "type": "sterile_zone", "center": z.box.center[:2].round(3).tolist(),
                      "half": z.box.half[:2].round(3).tolist(), "keepout_margin": z.keepout_margin})
    for s in slots:
        nodes.append({"id": f"slot:{s.id}", "type": "slot", "kind": s.kind,
                      "position": None if not np.all(np.isfinite(s.position)) else s.position.round(3).tolist()})
        if s.kind not in ("hand", "elsewhere"):
            edges.append({"src": f"slot:{s.id}", "dst": s.anchor, "rel": "part_of", "p": 1.0})
    for tr in tracks:
        pid = f"person:{tr.identity or tr.id}"
        nodes.append({"id": pid, "type": "person", "role": tr.identity, "xy": tr.x[:2].round(3).tolist(),
                      "vel": tr.x[2:].round(3).tolist(), "pos_std": float(np.sqrt(tr.P[0, 0] + tr.P[1, 1]))})
        for z in zones:
            if z.box.contains_xy(tr.x[None, :2])[0]:
                edges.append({"src": pid, "dst": f"zone:{z.name}", "rel": "inside", "p": 1.0})
    for iid, st in belief.items.items():
        H = float(-(st.b * np.log2(st.b)).sum())
        nodes.append({"id": f"item:{iid}", "type": "item", "class": st.spec.cls, "entropy_bits": H,
                      "criticality": st.spec.criticality})
        for k in np.argsort(-st.b)[:4]:
            if st.b[k] < min_p:
                break
            s = slots[k]
            dst = f"person:{s.anchor}" if s.kind == "hand" else f"slot:{s.id}"
            edges.append({"src": f"item:{iid}", "dst": dst, "rel": REL[s.kind], "p": float(st.b[k])})
    return {"t": float(t), "nodes": nodes, "edges": edges}
