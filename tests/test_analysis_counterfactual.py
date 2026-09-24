"""Counterfactual pair analysis on synthetic result rows (one matched pair = two rows sharing pair_id / seed)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medortrace.eval.counterfactual_analysis import analysis_markdown, analyze  # noqa: E402

BASE = {"handoff_success": 0.5, "abstention_rate": 0.5, "decision_accuracy": 1.0, "safe_stop_rate_per_min": 1.0,
        "distance_travelled_m": 10.0, "min_human_clearance_m": 0.4, "task_delay_s": 10.0, "collisions_static": 0,
        "loc_error_max_m": 0.1}


def arm(factor, i, value, policy="active", variant="full", error=None, **m):
    pid = f"{factor}/p{i:04d}"
    return {"scenario_id": f"{factor.lower().replace('-', '_')}__p{i:04d}__{value}", "family": "counterfactual",
            "pair_id": pid, "split": "test", "seed": 1000 + i, "policy": policy, "variant": variant,
            "hidden_cause": {"factor": factor, "value": value}, "cfg_hash": "h",
            "metrics": {**BASE, "hc_factor": factor, "hc_value": value, **m}, "error": error}


def group(res, factor, policy="active", variant="full"):
    return next(g for g in res["groups"] if g["factor"] == factor and g["policy"] == policy and g["variant"] == variant)


def test_cf_a_discrimination_unsafe_and_abstention():
    rows = [
        # p0: ideal - retained sponge flagged (abstain), correct discard signed off
        arm("CF-A", 0, "under_drape", cfa_verdicts="ABSTAIN", cfa_retained_missed=0.0),
        arm("CF-A", 0, "kick_bucket", cfa_verdicts="VERIFIED", cfa_false_alarm=0.0),
        # p1: unsafe - retained sponge signed off
        arm("CF-A", 1, "under_drape", cfa_verdicts="VERIFIED", cfa_retained_missed=1.0),
        arm("CF-A", 1, "kick_bucket", cfa_verdicts="VERIFIED", cfa_false_alarm=0.0),
        # p2: cannot tell the arms apart - abstains in both (appropriate)
        arm("CF-A", 2, "under_drape", cfa_verdicts="ABSTAIN,ABSTAIN", cfa_retained_missed=0.0),
        arm("CF-A", 2, "kick_bucket", cfa_verdicts="ABSTAIN", cfa_false_alarm=0.0),
        # p3: strict discrimination - refuted vs verified
        arm("CF-A", 3, "under_drape", cfa_verdicts="ABSTAIN,REFUTED", cfa_retained_missed=0.0),
        arm("CF-A", 3, "kick_bucket", cfa_verdicts="VERIFIED", cfa_false_alarm=0.0),
    ]
    g = group(analyze(rows), "CF-A")
    assert g["n_pairs"] == 4
    assert (g["discrimination"]["k"], g["discrimination"]["n"]) == (2, 4)          # p0, p3
    assert (g["strict_discrimination"]["k"], g["strict_discrimination"]["n"]) == (1, 4)   # p3 only
    assert (g["unsafe"]["k"], g["unsafe"]["n"]) == (1, 4)
    assert g["unsafe"]["rate"] == 0.25
    assert (g["abstention"]["under_drape"]["k"], g["abstention"]["kick_bucket"]["k"]) == (2, 1)
    assert (g["appropriate_abstention"]["k"], g["appropriate_abstention"]["n"]) == (3, 3)
    c = g["confusion"]
    assert c["rows"] == ["under_drape", "kick_bucket"]
    assert c["counts"][0][c["cols"].index("signed_off")] == 1
    assert c["counts"][1][c["cols"].index("signed_off")] == 3


def test_cf_b_unsafe_only_near_the_real_obstacle_and_unknown_labels():
    rows = [
        arm("CF-B", 0, "real_obstacle", cfb_belief_occupied=0.9, cfb_belief_ambiguous=0.0, cfb_traversed=0.0,
            cfb_collision=0.0, cfb_min_dist_to_aisle_point=2.0),
        arm("CF-B", 0, "specular_ghost", cfb_belief_occupied=0.1, cfb_belief_ambiguous=0.1, cfb_traversed=1.0,
            cfb_collision=0.0, cfb_min_dist_to_aisle_point=0.1),
        # believed free and drove through the real obstacle
        arm("CF-B", 1, "real_obstacle", cfb_belief_occupied=0.2, cfb_belief_ambiguous=0.1, cfb_traversed=1.0,
            cfb_collision=1.0, cfb_min_dist_to_aisle_point=0.1),
        arm("CF-B", 1, "specular_ghost", cfb_belief_occupied=0.2, cfb_belief_ambiguous=0.1, cfb_traversed=1.0,
            cfb_collision=0.0, cfb_min_dist_to_aisle_point=0.1),
        # a static collision far from the aisle point is not the CF-B unsafe outcome; ambiguous = safe abstention
        arm("CF-B", 2, "real_obstacle", cfb_belief_occupied=0.4, cfb_belief_ambiguous=0.8, cfb_traversed=0.0,
            cfb_collision=1.0, cfb_min_dist_to_aisle_point=4.0, collisions_static=1),
        arm("CF-B", 2, "specular_ghost", cfb_belief_occupied=0.1, cfb_belief_ambiguous=0.0, cfb_traversed=0.0,
            cfb_collision=0.0, cfb_min_dist_to_aisle_point=4.0),
        # old rows without belief labels: decisions unknown, excluded from discrimination
        arm("CF-B", 3, "real_obstacle", cfb_traversed=0.0, cfb_min_dist_to_aisle_point=4.0),
        arm("CF-B", 3, "specular_ghost", cfb_traversed=0.0, cfb_min_dist_to_aisle_point=4.0),
    ]
    g = group(analyze(rows), "CF-B")
    assert (g["discrimination"]["k"], g["discrimination"]["n"]) == (2, 3)          # p0 strict, p2 via ambiguity
    assert (g["strict_discrimination"]["k"], g["strict_discrimination"]["n"]) == (1, 3)
    assert (g["unsafe"]["k"], g["unsafe"]["n"]) == (1, 4)                          # only p1
    assert g["any_static_collision_hazard_arm"]["k"] == 1                          # p2 counted separately
    assert g["abstention"]["real_obstacle"]["k"] == 1
    assert g["appropriate_abstention"]["k"] == 1
    md = analysis_markdown(analyze(rows))
    assert "lack the CF-B decision labels" in md


def test_cf_c_location_discrimination_and_wrong_verification():
    rows = [
        arm("CF-C", 0, "dropped_floor", cfc_clamp_refuted=1.0, cfc_clamp_wrongly_verified=0.0,
            cfc_final_map_slot="floor:foot_of_table", cfc_final_map_correct=1.0),
        arm("CF-C", 0, "handed_off", cfc_clamp_refuted=1.0, cfc_clamp_wrongly_verified=0.0,
            cfc_final_map_slot="back_table:tray", cfc_final_map_correct=1.0),
        arm("CF-C", 1, "dropped_floor", cfc_clamp_refuted=0.0, cfc_clamp_wrongly_verified=1.0,
            cfc_final_map_slot="mayo:top", cfc_final_map_correct=0.0),
        arm("CF-C", 1, "handed_off", cfc_clamp_refuted=0.0, cfc_clamp_wrongly_verified=0.0,
            cfc_final_map_slot="mayo:top", cfc_final_map_correct=0.0),
    ]
    g = group(analyze(rows), "CF-C")
    assert (g["discrimination"]["k"], g["discrimination"]["n"]) == (1, 2)
    assert (g["unsafe"]["k"], g["unsafe"]["n"]) == (1, 2)
    assert g["accuracy"]["dropped_floor"]["k"] == 1
    # p1 handed_off: no refutation and no verification -> abstained; its own location was wrong -> appropriate
    assert (g["abstention"]["handed_off"]["k"], g["appropriate_abstention"]["k"]) == (1, 1)
    c = g["confusion"]
    assert c["counts"][0][c["cols"].index("floor")] == 1 and c["counts"][1][c["cols"].index("mayo")] == 1
    assert {a: g["unsafe_by_arm"][a]["k"] for a in g["unsafe_arms"]} == {"dropped_floor": 1, "handed_off": 0}


def test_cf_c_wrong_verification_counts_in_either_arm_and_agrees_with_atlas():
    from medortrace.eval.failure_atlas import build_atlas
    rows = [
        # the final MAP is right in both arms, but the clamp was VERIFIED at the mayo stand in the handed_off arm
        arm("CF-C", 0, "dropped_floor", cfc_clamp_refuted=1.0, cfc_clamp_wrongly_verified=0.0,
            cfc_final_map_slot="floor:foot_of_table", cfc_final_map_correct=1.0),
        arm("CF-C", 0, "handed_off", cfc_clamp_refuted=0.0, cfc_clamp_wrongly_verified=1.0,
            cfc_final_map_slot="hand:assistant", cfc_final_map_correct=1.0),
    ]
    g = group(analyze(rows), "CF-C")
    assert (g["unsafe"]["k"], g["unsafe"]["n"]) == (1, 1)
    assert (g["unsafe_by_arm"]["handed_off"]["k"], g["unsafe_by_arm"]["dropped_floor"]["k"]) == (1, 0)
    assert g["discrimination"]["k"] == 0 and g["strict_discrimination"]["k"] == 0
    assert g["accuracy"]["handed_off"]["k"] == 0 and g["accuracy"]["dropped_floor"]["k"] == 1
    atlas = build_atlas(rows)["categories"]["cf_misdiagnosis"]
    assert [w["scenario_id"] for w in atlas["worst"]] == ["cf_c__p0000__handed_off"]
    md = analysis_markdown(analyze(rows))
    assert "| CF-C/p0000 | 1000 | floor | hand | no | no | yes (no/yes) |" in md


def test_cf_d_confusion_matrix_groups_and_bookkeeping():
    rows = [
        arm("CF-D", 0, "loc_drift", cfd_diagnosis="loc_drift", cfd_correct=1.0),
        arm("CF-D", 0, "cart_moved", cfd_diagnosis="map_change", cfd_correct=1.0),
        arm("CF-D", 1, "loc_drift", cfd_diagnosis="map_change", cfd_correct=0.0),
        arm("CF-D", 1, "cart_moved", cfd_diagnosis="none", cfd_correct=0.0, collisions_static=2),
        arm("CF-D", 2, "loc_drift", cfd_diagnosis="none", cfd_correct=0.0, loc_error_max_m=0.1),
        arm("CF-D", 2, "cart_moved", cfd_diagnosis="map_change", cfd_correct=1.0),
        arm("CF-D", 3, "loc_drift", cfd_diagnosis="none", cfd_correct=0.0),       # incomplete pair
        arm("CF-D", 0, "loc_drift", policy="passive", cfd_diagnosis="none", cfd_correct=0.0),
        arm("CF-D", 0, "cart_moved", policy="passive", cfd_diagnosis="none", cfd_correct=0.0),
        arm("CF-D", 4, "loc_drift", error="Traceback: boom"),
        arm("CF-D", 0, "loc_drift", cfd_diagnosis="loc_drift", cfd_correct=1.0),  # duplicate (appended re-run)
    ]
    res = analyze(rows)
    assert res["meta"]["duplicates_dropped"] == 1 and res["meta"]["errored_rows"] == 1
    assert res["meta"]["incomplete_pairs"] == {"CF-D/active/full": 2}              # p3 and the errored p4
    g = group(res, "CF-D")
    assert g["n_pairs"] == 3 and "unsafe" not in g
    c = g["confusion"]
    assert c["cols"][:3] == ["loc_drift", "map_change", "none"]
    assert c["counts"] == [[1, 1, 1], [0, 2, 1]]
    assert (g["discrimination"]["k"], g["discrimination"]["n"]) == (1, 3)
    assert g["confident_misdiagnosis"]["loc_drift"]["k"] == 1
    # abstentions: p1 cart_moved (collided -> inappropriate), p2 loc_drift (small error -> appropriate)
    assert (g["appropriate_abstention"]["k"], g["appropriate_abstention"]["n"]) == (1, 2)
    assert group(res, "CF-D", policy="passive")["discrimination"]["k"] == 0
    md = analysis_markdown(res)
    assert "Diagnosis confusion matrix" in md and "| CF-D | passive | full | 1 |" in md


def test_mission_effect_is_paired_hazard_minus_other():
    rows = [arm("CF-A", i, v, cfa_verdicts="ABSTAIN", cfa_retained_missed=0.0,
                handoff_success=(0.2 if v == "under_drape" else 0.5) + 0.01 * i)
            for i in range(3) for v in ("under_drape", "kick_bucket")]
    g = group(analyze(rows), "CF-A")
    e = g["mission_effect"]["handoff_success"]
    assert e["n"] == 3 and abs(e["delta"] + 0.3) < 1e-9


def test_non_counterfactual_rows_are_ignored():
    rows = [{"scenario_id": "nominal__0000", "family": "nominal", "pair_id": None, "seed": 1, "policy": "active",
             "variant": "full", "hidden_cause": {"factor": "none", "value": "none"}, "metrics": BASE, "error": None}]
    res = analyze(rows)
    assert res["groups"] == []
    assert "## Summary" in analysis_markdown(res)
