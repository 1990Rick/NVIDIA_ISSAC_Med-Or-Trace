"""End-to-end smoke test: lite backend + full autonomy stack for 15 s of sim time.

Two short episodes in total (the machine is shared): one writing the dataset,
one re-running the same seed to check end-to-end determinism.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from medortrace.common.config import load_config
from medortrace.eval.runner import run_episode

SEED = 123
DURATION = 15.0
# metrics that are legitimately NaN in a 15 s nominal episode (no decided claims / no ghosts / no faults)
MAY_BE_NAN = {"decision_accuracy", "wrong_assertion_rate", "correct_abstention_frac", "calibration_ece_under_fault",
              "calibration_ece",
              "calibration_ece_nominal", "brier", "ghost_precision", "ghost_recall"}


@pytest.fixture(scope="module")
def episode(tmp_path_factory):
    out = tmp_path_factory.mktemp("episodes")
    res = run_episode(load_config(), SEED, out_dir=out, duration=DURATION)
    return res, out


def test_smoke_metrics_finite_and_safe(episode):
    res, _ = episode
    m = res.metrics
    assert res.seed == SEED and res.policy == "active"
    for k, v in m.items():
        if isinstance(v, (int, float, np.floating, np.integer)) and k not in MAY_BE_NAN:
            assert math.isfinite(float(v)), k
    assert m["provenance_chain_valid"] == 1.0
    assert m["sterile_breach_s"] == 0.0
    assert m["collisions_agent"] == 0 and m["collisions_static"] == 0
    assert m["min_human_clearance_m"] > 0.0
    assert 0.0 < m["energy_reserve_frac"] <= 1.0 and m["energy_used_wh"] > 0
    assert m["loc_error_max_m"] < 0.2                     # landmarks keep a nominal episode well localised
    assert 0.0 <= m["handoff_success"] <= 1.0 and 0.0 <= m["abstention_rate"] <= 1.0
    assert m["claims_answered"] <= m["claims_total"] and m["hc_factor"] == "none"


def test_dataset_layout(episode):
    res, out = episode
    d = out / f"default__s{SEED}__active"
    assert res.out_dir == str(d)
    for f in ("meta.json", "trajectory.npz", "events.jsonl", "provenance.json", "scene_graph.json"):
        assert (d / f).is_file(), f
    meta = json.loads((d / "meta.json").read_text())
    assert meta["seed"] == SEED and meta["backend"] == "lite" and meta["policy"] == "active"
    assert meta["metrics"]["provenance_chain_valid"] == 1.0 and meta["hidden_cause"]["factor"] == "none"
    traj = np.load(d / "trajectory.npz")
    n = len(traj["t"])
    assert n == round(DURATION / 0.1)
    assert traj["pose_true"].shape == (n, 3) and traj["pose_est"].shape == (n, 3)
    assert traj["item_entropy"].shape == (n, len(meta["items"]))
    assert traj["item_slot_true"].shape == (n, len(meta["items"])) and (traj["item_slot_true"] >= 0).all()
    assert np.all(np.diff(traj["t"]) > 0) and not traj["sterile_breach"].any()
    kinds = {json.loads(line)["kind"] for line in (d / "events.jsonl").read_text().splitlines()}
    assert {"workflow_log", "truth_move"} <= kinds
    prov = json.loads((d / "provenance.json").read_text())
    assert "agent:medortrace_robot_0" in prov["agent"] and len(prov["entity"]) > 10
    # the exported hash chain is intact: every record points at its predecessor's hash
    recs = {**prov["entity"], **prov["agent"], **prov["activity"]}
    hashes = {r["mot:hash"] for r in recs.values()}
    assert all(r["mot:prev"] == "GENESIS" or r["mot:prev"] in hashes for r in recs.values())
    sg = json.loads((d / "scene_graph.json").read_text())
    assert isinstance(sg, dict) and sg


def test_same_seed_same_episode_end_to_end(episode):
    res, _ = episode
    again = run_episode(load_config(), SEED, duration=DURATION)
    a = {k: v for k, v in res.metrics.items() if k != "wall_time_s"}
    b = {k: v for k, v in again.metrics.items() if k != "wall_time_s"}
    assert a.keys() == b.keys()
    for k in a:
        if isinstance(a[k], float) and math.isnan(a[k]):
            assert math.isnan(b[k]), k
        else:
            assert a[k] == b[k], k
