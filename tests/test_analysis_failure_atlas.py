"""Failure atlas: taxonomy detection, ranking, reproduce commands and episode-dir enrichment (synthetic data)."""

import json
import shlex
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medortrace.eval.failure_atlas import (  # noqa: E402
    AtlasThresholds,
    EpisodeIndex,
    TruthTimeline,
    atlas_markdown,
    build_atlas,
    dedupe,
    load_episode,
    reproduce_command,
    verify_prov_chain,
    wrong_verdicts,
)
from medortrace.provenance.graph import ProvenanceGraph, Verdict  # noqa: E402

CLEAN = {"collisions_agent": 0, "collisions_static": 0, "sterile_breach_s": 0.0, "keepout_margin_violation_s": 0.0,
         "near_collision_events": 0, "near_collision_rate_per_min": 0.0, "min_human_clearance_m": 0.8,
         "wrong_assertion_rate": 0.0, "decision_accuracy": 1.0, "claims_answered": 20, "abstention_rate": 0.3,
         "handoff_success": 0.8, "safe_stop_rate_per_min": 0.5, "uncertainty_safe_stop_rate_per_min": 0.0,
         "handover_requests": 0, "intervention_cost": 0.0, "loc_error_max_m": 0.1, "loc_error_mean_m": 0.03,
         "energy_reserve_frac": 0.7, "energy_used_wh": 3.4, "distance_travelled_m": 20.0, "claims_total": 22}


def row(sid, family="nominal", seed=11, policy="active", variant="full", hc=None, pair_id=None, error=None, **m):
    return {"scenario_id": sid, "family": family, "pair_id": pair_id, "split": "test", "seed": seed, "policy": policy,
            "variant": variant, "hidden_cause": hc or {"factor": "none", "value": "none"}, "cfg_hash": "x",
            "metrics": {} if error else {**CLEAN, **m}, "error": error}


def _cats(atlas):
    return {k: v["count"] for k, v in atlas["categories"].items()}


def test_every_category_is_detected_and_clean_rows_are_not():
    rows = [
        row("clean"),
        row("contact", collisions_agent=1, min_human_clearance_m=-0.05),
        row("static", family="map_corruption", collisions_static=2),
        row("breach", sterile_breach_s=1.5, keepout_margin_violation_s=3.0),
        row("margin", keepout_margin_violation_s=2.0),
        row("near", near_collision_events=3, min_human_clearance_m=0.1),
        row("wrong", wrong_assertion_rate=0.25, decision_accuracy=0.75, abstention_rate=0.6),
        row("cf_a__p0000__under_drape", family="counterfactual", hc={"factor": "CF-A", "value": "under_drape"},
            pair_id="CF-A/p0000", cfa_retained_missed=1.0, cfa_verdicts="VERIFIED"),
        row("cf_d__p0000__loc_drift", family="counterfactual", hc={"factor": "CF-D", "value": "loc_drift"},
            pair_id="CF-D/p0000", cfd_correct=0.0, cfd_diagnosis="map_change"),
        row("cf_c__p0000__dropped_floor", family="counterfactual", hc={"factor": "CF-C", "value": "dropped_floor"},
            pair_id="CF-C/p0000", cfc_clamp_wrongly_verified=1.0, cfc_final_map_slot="mayo:top"),
        row("cf_b__p0000__real_obstacle", family="counterfactual", hc={"factor": "CF-B", "value": "real_obstacle"},
            pair_id="CF-B/p0000", cfb_correct=0.0, cfb_belief_occupied=0.1, cfb_traversed=1.0,
            cfb_min_dist_to_aisle_point=0.2, cfb_collision=0.0),
        row("stops", safe_stop_rate_per_min=5.0),
        row("handover", handover_requests=2, intervention_cost=2.5),
        row("starve", abstention_rate=0.9, handoff_success=0.1),
        row("loc", family="loc_drift", loc_error_max_m=0.9),
        row("energy", energy_reserve_frac=0.1),
        row("crash", error="Traceback...\nValueError: boom"),
    ]
    atlas = build_atlas(rows, k=3)
    c = _cats(atlas)
    assert c == {"human_contact": 1, "static_collision": 1, "sterile_breach": 2, "near_collision": 1,
                 "wrong_assertion": 1, "retained_item_signed_off": 1, "cf_misdiagnosis": 3, "safe_stop_cascade": 1,
                 "handover": 1, "mission_starvation": 1, "localization_loss": 1, "energy_depletion": 1,
                 "episode_error": 1}
    cats = atlas["categories"]
    # breach outranks margin violation; subtypes are reported
    assert [w["scenario_id"] for w in cats["sterile_breach"]["worst"]] == ["breach", "margin"]
    assert cats["sterile_breach"]["subtypes"] == {"breach": 1, "margin": 1}
    # unsafe CF misdiagnoses (clamp verified, real obstacle traversed) rank above a CF-D diagnosis error
    assert cats["cf_misdiagnosis"]["worst"][-1]["scenario_id"] == "cf_d__p0000__loc_drift"
    assert cats["cf_misdiagnosis"]["subtypes"]["CF-D loc_drift diagnosed as map_change"] == 1
    # counterfactual families are split by factor
    assert cats["retained_item_signed_off"]["by_family"] == {"counterfactual/CF-A": {"count": 1, "n": 1, "rate": 1.0}}
    assert cats["episode_error"]["worst"][0]["evidence"]["error"] == "ValueError: boom"
    assert cats["wrong_assertion"]["worst"][0]["evidence"]["wrong_verdicts_est"] == 2
    md = atlas_markdown(atlas)
    for cat in c:
        assert f"## {cat}" in md
    assert "reproduce:" in md and "hidden cause CF-A=under_drape" in md


def test_thresholds_policy_split_and_energy_outlier():
    rows = [row(f"n{i}", seed=i, energy_used_wh=3.0 + 0.01 * i) for i in range(4)]
    rows += [row("hog", seed=99, energy_used_wh=9.0)]
    rows += [row("n0", seed=0, policy="passive", keepout_margin_violation_s=0.5)]
    atlas = build_atlas(rows, thresholds=AtlasThresholds(sterile_margin_s=0.4))
    cats = atlas["categories"]
    assert cats["energy_depletion"]["count"] == 1
    w = cats["energy_depletion"]["worst"][0]
    assert w["scenario_id"] == "hog" and w["subtype"] == "energy_outlier" and w["evidence"]["ratio"] > 2.5
    assert cats["sterile_breach"]["by_policy"] == {"passive": {"count": 1, "n": 1, "rate": 1.0}}
    assert build_atlas(rows)["categories"]["sterile_breach"]["count"] == 0      # default margin threshold 1 s


def test_missing_metrics_do_not_trigger_or_crash():
    r = row("old")
    r["metrics"] = {"min_human_clearance_m": 0.5}                             # an old, sparse row
    atlas = build_atlas([r])
    assert sum(_cats(atlas).values()) == 0
    assert "## Overview" in atlas_markdown(atlas)


def test_dedupe_keeps_last_row():
    rows = [row("a", collisions_agent=1), row("a"), row("b")]
    kept, n = dedupe(rows)
    assert n == 1 and len(kept) == 2
    assert build_atlas(rows)["categories"]["human_contact"]["count"] == 0


def test_reproduce_command_is_exact_python():
    r = row("nominal__0003", seed=1234, policy="fixed_route")
    r["sim"] = {"sensors": {"camera": {"pd0": 0.69}}}
    r["duration"] = 20.0
    spec = {"autonomy": {"use_ghost_reasoning": False}, "cfg": {"verifier": {"tau_verify": 0.5}}}
    cmd = reproduce_command(r, variant_spec=spec)
    argv = shlex.split(cmd)
    assert argv[:3] == ["PYTHONPATH=.", "python", "-c"]
    code = argv[3]
    compile(code, "<repro>", "exec")
    for frag in ("scenario_id == 'nominal__0003'", "e.seed == 1234", "policy='fixed_route'", "duration=20.0",
                 "sim_overrides={'sensors': {'camera': {'pd0': 0.69}}}",
                 "autonomy_override={'use_ghost_reasoning': False}",
                 "deep_merge(e.resolve(), {'verifier': {'tau_verify': 0.5}})", "out_dir='runs/repro'"):
        assert frag in code, frag


def test_reproduce_command_uses_the_recorded_variant_spec():
    r = row("nominal__0003", seed=1234, variant="no_abstention", wrong_assertion_rate=0.2)
    r["variant_spec"] = {"cfg": {"verifier": {"require_direct_evidence": False, "tau_verify": 0.5}}}
    code = shlex.split(reproduce_command(r))[3]
    assert "deep_merge(e.resolve(), {'verifier': {'require_direct_evidence': False, 'tau_verify': 0.5}})" in code
    # the recorded spec is what ran: it wins over a (stale) --variant-specs entry
    code = shlex.split(reproduce_command(r, variant_spec={"autonomy": {"use_workflow_log": False}}))[3]
    assert "autonomy_override" not in code and "'tau_verify': 0.5" in code
    w = build_atlas([r])["categories"]["wrong_assertion"]["worst"][0]
    assert w["reproduce_exact"] and "INEXACT" not in w["reproduce"]


def test_reproduce_command_flags_unknown_variant_overrides():
    old = row("nominal__0003", seed=1234, variant="no_abstention", wrong_assertion_rate=0.2)   # no variant_spec
    cmd = reproduce_command(old)
    assert "# INEXACT" in cmd and "'no_abstention'" in cmd
    compile(shlex.split(cmd, comments=True)[3], "<repro>", "exec")               # still a runnable command
    atlas = build_atlas([old, row("nominal__0004", variant="full", wrong_assertion_rate=0.1)])
    assert atlas["meta"]["variants_overrides_unknown"] == ["no_abstention"]
    worst = {w["variant"]: w for w in atlas["categories"]["wrong_assertion"]["worst"]}
    assert not worst["no_abstention"]["reproduce_exact"] and worst["full"]["reproduce_exact"]
    md = atlas_markdown(atlas)
    assert "**Warning:** the overrides of variant(s) no_abstention" in md
    assert "reproduce (INEXACT: variant overrides unknown)" in md
    # a --variant-specs entry makes it exact again
    atlas = build_atlas([old], variant_specs={"no_abstention": {"cfg": {"verifier": {"tau_verify": 0.5}}}})
    w = atlas["categories"]["wrong_assertion"]["worst"][0]
    assert w["reproduce_exact"] and "'tau_verify': 0.5" in w["reproduce"]
    assert atlas["meta"]["variants_overrides_unknown"] == []


# ---------------------------------------------------------------------------------------------------------------------
def _write_episode(root: Path, variant: str = "full") -> Path:
    """A minimal exported episode (writer.py layout) with one wrong VERIFIED verdict."""
    d = root / variant / "nominal__0001__s7__active"
    d.mkdir(parents=True)
    items = ["sponge_1", "clamp_1"]
    slots = ["back_table:tray", "field:top", "kick_bucket_1:inside", "field:under_drape", "mayo:top"]
    meta = {"schema_version": "1.0", "backend": "lite", "policy": "active", "scenario_id": "nominal__0001",
            "family": "nominal", "seed": 7, "hidden_cause": {"factor": "none", "value": "none"}, "hidden_notes": {},
            "faults": {"dropouts": {"camera": [[10.0, 14.0]]}, "skew": {}, "odom_bias": [0.0, 0.0],
                       "specular_gain": 1.0, "floor_wet": False, "rare_geometry": [], "occluders": [], "map_edits": []},
            "nuisance": {"haze": 0.05, "glare_gain": 1.0}, "items": items, "slots": slots,
            "agents": ["surgeon", "circulator"], "metrics": {}}
    (d / "meta.json").write_text(json.dumps(meta))
    ev = [
        {"kind": "workflow_log", "t": 20.0, "type": "discard", "item": "sponge_1", "src": "field:top",
         "dst": "kick_bucket_1:inside", "id": "wf_001"},
        {"kind": "truth_move", "t": 5.0, "item": "sponge_1", "src": "back_table:tray", "dst": "field:top",
         "cause": "workflow"},
        {"kind": "truth_move", "t": 20.0, "item": "sponge_1", "src": "field:top", "dst": "field:under_drape",
         "cause": "hidden_cause"},
        # writer.py merges VerdictRecord.__dict__ after "kind": "verdict", so the claim kind overwrites it
        {"kind": "handoff", "claim_id": "c_wf_001", "t": 40.0, "verdict": "VERIFIED", "posterior": 0.93,
         "reason": "posterior above verify threshold with direct evidence", "item_id": "sponge_1",
         "slot_id": "kick_bucket_1:inside", "t_ref": 26.0, "map_slot": "kick_bucket_1:inside", "direct": True},
        {"kind": "count", "claim_id": "count1_clamp_1", "t": 50.0, "verdict": "VERIFIED", "posterior": 0.97,
         "reason": "ok", "item_id": "clamp_1", "slot_id": "mayo:top", "t_ref": 45.0, "map_slot": "mayo:top",
         "direct": True},
        {"kind": "count", "claim_id": "count1_sponge_1", "t": 50.0, "verdict": "ABSTAIN", "posterior": 0.5,
         "reason": "no direct sensor evidence of the claimed slot", "item_id": "sponge_1",
         "slot_id": "kick_bucket_1:inside", "t_ref": 45.0, "map_slot": "field:top", "direct": False},
        {"kind": "safety", "t": 30.0, "mode_from": "NOMINAL", "mode_to": "STOP",
         "reasons": ["predicted collision p=0.40 c=0.10"], "category": "collision_risk", "values": {}},
    ]
    (d / "events.jsonl").write_text("\n".join(json.dumps(e) for e in ev) + "\n")
    g = ProvenanceGraph()
    g.add_evidence("camera:1", 30.0, "camera", {"n_det": 2, "tags": []}, np.array([1.0, 2.0, 0.1]))
    g.add_evidence("acoustic:2", 32.0, "acoustic", {"energy": 0.2, "region": "kick_bucket_1", "nlos": False})
    g.add_evidence("radar:3", 33.0, "radar", {"metal_hits": [0.0] * 8})
    g.add_verdict("verdict:c_wf_001", 40.0, {"claim_id": "c_wf_001", "item_id": "sponge_1",
                                            "slot_id": "kick_bucket_1:inside", "t_ref": 26.0, "kind": "handoff"},
                  Verdict.VERIFIED, 0.93, "posterior above verify threshold with direct evidence",
                  [("camera:1", 1.2), ("acoustic:2", 0.4), ("radar:3", -0.3)])
    assert g.verify_chain()
    (d / "provenance.json").write_text(json.dumps(g.to_prov_json()))
    T = np.arange(1, 61) * 1.0
    agents = np.stack([np.stack([np.full(60, 3.0), np.full(60, 3.0)], 1),
                       np.stack([5.0 - T / 30, np.full(60, 2.0)], 1)], 1)
    pose = np.stack([np.full(60, 4.0), np.full(60, 2.0), np.zeros(60)], 1)
    slot_true = np.array([[0 if t < 5 else (1 if t < 20 else 3), 4] for t in T], dtype=np.int16)
    np.savez_compressed(d / "trajectory.npz", t=T, pose_true=pose, pose_est=pose + [0.02, 0.0, 0.0],
                        mode=np.zeros(60, np.int8), action_v=np.full(60, 0.3), human_clearance_est=np.full(60, 1.0),
                        agents_true=agents, collision_agent=T >= 45, collision_static=np.zeros(60, bool),
                        sterile_breach=np.zeros(60, bool), fault_active=(T >= 10) & (T < 14), item_slot_true=slot_true)
    return d


def test_episode_enrichment_wrong_verdict_and_provenance(tmp_path):
    d = _write_episode(tmp_path)
    ep = load_episode(d)
    tt = TruthTimeline(ep)
    assert tt.slot("sponge_1", 4.0) == "back_table:tray"
    assert tt.slot("sponge_1", 26.0) == "field:under_drape"
    assert tt.slot("clamp_1", 45.0) == "mayo:top"                  # never moved: taken from trajectory.npz
    wv = wrong_verdicts(ep)
    assert [w["claim_id"] for w in wv] == ["c_wf_001"]            # the correct clamp VERIFIED is not listed
    w = wv[0]
    assert w["truth_slot_at_t_ref"] == "field:under_drape" and w["claim_kind"] == "handoff"
    ex = w["explanation"]
    assert [e["evidence"] for e in ex["supporting"]] == ["camera:1", "acoustic:2"]
    assert [e["evidence"] for e in ex["contradicting"]] == ["radar:3"]
    assert ex["supporting"][0]["sensor"] == "camera" and ex["verdict"]["verdict"] == "VERIFIED"
    assert any("unlogged move" in h for h in w["hints"])
    assert any("log/truth mismatch" in h for h in w["hints"])
    assert verify_prov_chain(ep.prov) is True

    r = row("nominal__0001", seed=7, wrong_assertion_rate=0.5, collisions_agent=1, safe_stop_rate_per_min=4.0)
    atlas = build_atlas([r], EpisodeIndex([tmp_path]))
    wa = atlas["categories"]["wrong_assertion"]["worst"][0]
    assert wa["episode_dir"] == str(d)
    assert "dropout:camera (1 windows, 4.0s)" in wa["fault_labels"]
    assert wa["wrong_verdicts"][0]["claim_id"] == "c_wf_001"
    assert wa["evidence"]["provenance_chain_recomputed"] is True
    hc = atlas["categories"]["human_contact"]["worst"][0]
    assert any(h.startswith("first event at t=45.0s: mode=NOMINAL") for h in hc["hints"])
    ss = atlas["categories"]["safe_stop_cascade"]["worst"][0]
    assert any("predicted collision x1" in h for h in ss["hints"])
    md = atlas_markdown(atlas)
    assert "supporting: camera:1 (camera, t=30.0s, w=+1.20" in md
    assert "contradicting: radar:3" in md


def test_prov_chain_tamper_is_detected(tmp_path):
    ep = load_episode(_write_episode(tmp_path))
    ep.prov["entity"]["camera:1"]["mot:n_det"] = 5
    assert verify_prov_chain(ep.prov) is False


def test_episode_index_prefers_matching_variant(tmp_path):
    d_full = _write_episode(tmp_path, "full")
    d_abl = _write_episode(tmp_path, "no_radar")
    idx = EpisodeIndex([tmp_path])
    assert idx.find(row("nominal__0001", seed=7, variant="no_radar")) == d_abl
    assert idx.find(row("nominal__0001", seed=7, variant="full")) == d_full
    assert idx.find(row("nominal__0001", seed=8)) is None
