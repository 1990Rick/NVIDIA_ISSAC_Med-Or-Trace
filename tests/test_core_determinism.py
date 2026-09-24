"""Reproducibility: RNG streams, episode construction, counterfactual pairs and
the seed registry.

The whole experimental design rests on three invariants tested here:

1. a scenario seed fully determines an episode (scene, workflow log, faults);
2. both arms of a counterfactual pair share every nuisance factor and differ
   only through the hidden cause;
3. the registry keeps pair members together (same seed, same split).
"""

from __future__ import annotations

import collections
from pathlib import Path

import numpy as np
import pytest

from medortrace.common.config import CONFIG_DIR, config_hash, deep_merge, get, load_config, load_yaml
from medortrace.common.rng import STREAMS, RngStreams
from medortrace.eval import registry as reg_mod
from medortrace.eval.registry import COUNTERFACTUALS, FAMILIES, build_registry, load_registry, select
from medortrace.sim.episode import build_episode
from medortrace.world.agents import StaffPopulation


def _log_key(wf):
    return [(e.t, e.type.value, e.item_id, e.src, e.dst, e.reporter, e.confidence, e.event_id, sorted(e.payload))
            for e in wf.log]


def _truth_key(wf):
    return [(m.t, m.item_id, m.src, m.dst, m.cause) for m in wf.truth]


def _tasks_key(wf):
    return {k: [(s.t_start, tuple(np.round(s.goal, 12)), s.dwell, s.purpose) for s in v]
            for k, v in wf.staff_tasks.items()}


def _moved_offsets(a_objs, b_objs) -> list[np.ndarray]:
    """xy displacement of every object present in both arms that the hidden cause moved."""
    ob = {o.name: o for o in b_objs}
    return [np.asarray(ob[o.name].box.center[:2], float) - np.asarray(o.box.center[:2], float) for o in a_objs
            if o.name in ob and not np.allclose(ob[o.name].box.center, o.box.center)]


def _objects_key(objs, skip_tags=()):
    return [(o.name, o.kind, o.material, tuple(o.box.center), tuple(o.box.half), o.box.yaw, tuple(o.tags))
            for o in objs if not set(skip_tags) & set(o.tags)]


# ---------------------------------------------------------------------------
# RNG streams
# ---------------------------------------------------------------------------
def test_streams_are_reproducible_and_independent():
    a, b = RngStreams(7), RngStreams(7)
    for s in STREAMS:
        assert np.array_equal(a[s].random(5), b[s].random(5))
    # consuming one stream never perturbs another
    c, d = RngStreams(99), RngStreams(99)
    c["layout"].random(1000)
    assert np.array_equal(c["agents"].random(8), d["agents"].random(8))
    # distinct streams (and seeds) give distinct draws
    e = RngStreams(99)
    assert not np.array_equal(e["layout"].random(8), e["agents"].random(8))
    assert not np.array_equal(RngStreams(1)["layout"].random(8), RngStreams(2)["layout"].random(8))


def test_stream_override_and_fork():
    base, ov = RngStreams(5), RngStreams(5, overrides={"hidden": 123})
    assert not np.array_equal(base["hidden"].random(4), ov["hidden"].random(4))
    assert np.array_equal(base["workflow"].random(4), ov["workflow"].random(4))
    # fork is deterministic, tag-specific and independent of how much the parent was consumed
    s1, s2 = RngStreams(5), RngStreams(5)
    s1["agents"].random(100)
    assert np.array_equal(s1.fork("agents", "x").random(4), s2.fork("agents", "x").random(4))
    assert not np.array_equal(s1.fork("agents", "x").random(4), s1.fork("agents", "y").random(4))


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
def test_config_extends_merge_and_hash():
    cf = load_config("scenarios/cf_b.yaml")
    assert cf["family"] == "counterfactual" and cf["hidden_cause"]["factor"] == "CF-B"
    assert cf["episode"]["dt"] == load_config()["episode"]["dt"]          # inherited from default.yaml
    assert "extends" not in cf
    merged = deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}, "d": 4})
    assert merged == {"a": {"b": 1, "c": 3}, "d": 4}
    assert get(cf, "sensors.lidar.rings") == 12 and get(cf, "nope.x", 5) == 5
    assert config_hash(cf) == config_hash(load_config("scenarios/cf_b.yaml"))
    assert config_hash(cf) != config_hash(load_config("scenarios/cf_b.yaml", {"items": {"sponges": 5}}))


def test_load_yaml_repo_relative_path_independent_of_cwd(tmp_path, monkeypatch):
    """Scenario files reference ``configs/policies/...``; resolution must not depend on the cwd."""
    monkeypatch.chdir(tmp_path)
    w = load_yaml("configs/policies/nbv_default.yaml")["weights"]
    assert w["w_eig"] == pytest.approx(4.0)
    assert load_yaml("scenarios/default.yaml")["family"] == "nominal"
    assert load_yaml(CONFIG_DIR / "robot" / "rig.yaml")["footprint_radius"] == pytest.approx(0.28)
    with pytest.raises(FileNotFoundError):
        load_yaml("configs/does_not_exist.yaml")


# ---------------------------------------------------------------------------
# episodes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scenario", ["scenarios/default.yaml", "scenarios/sensor_dropout.yaml",
                                      "scenarios/map_corruption.yaml"])
def test_same_seed_identical_episode(scenario):
    cfg = load_config(scenario)
    a, b = build_episode(cfg, 2024), build_episode(load_config(scenario), 2024)
    assert a.spec.to_json() == b.spec.to_json()
    assert a.survey_spec.to_json() == b.survey_spec.to_json()
    assert _objects_key(a.prior_map) == _objects_key(b.prior_map)
    assert _log_key(a.workflow) == _log_key(b.workflow)
    assert _truth_key(a.workflow) == _truth_key(b.workflow)
    assert _tasks_key(a.workflow) == _tasks_key(b.workflow)
    assert [(c.id, c.t_ref, c.t_due, c.slot_id) for c in a.workflow.claims] == \
        [(c.id, c.t_ref, c.t_due, c.slot_id) for c in b.workflow.claims]
    assert a.faults.labels == b.faults.labels
    assert a.materials == b.materials
    assert a.cfg_hash == b.cfg_hash
    # a different seed changes the world
    c = build_episode(load_config(scenario), 2025)
    assert c.spec.to_json() != a.spec.to_json()
    assert _log_key(c.workflow) != _log_key(a.workflow)


def test_episode_basic_invariants(default_cfg):
    ep = build_episode(default_cfg, 11)
    W, D, _ = ep.spec.room
    lo, hi = default_cfg["layout"]["room_width"], default_cfg["layout"]["room_depth"]
    assert lo[0] <= W <= lo[1] and hi[0] <= D <= hi[1]
    slot_ids = set(ep.spec.slot_ids())
    assert all(m.dst in slot_ids and m.src in slot_ids for m in ep.workflow.truth)
    assert [m.t for m in ep.workflow.truth] == sorted(m.t for m in ep.workflow.truth)
    assert [e.t for e in ep.workflow.log] == sorted(e.t for e in ep.workflow.log)
    assert all(c.t_ref <= c.t_due <= ep.workflow.duration for c in ep.workflow.claims)
    # the robot starts outside every keep-out zone and the dock is inside the room
    assert not ep.spec.in_keepout(ep.spec.robot_start[None, :2])[0]
    assert 0 < ep.spec.dock[0] < W and 0 < ep.spec.dock[1] < D
    # no clutter / hidden-cause objects in the robot's prior map
    assert all(o.kind != "clutter" for o in ep.prior_map)


# ---------------------------------------------------------------------------
# counterfactual pairs
# ---------------------------------------------------------------------------
PAIRS = [(fac, vals) for fac, (_, vals) in COUNTERFACTUALS.items()]


@pytest.mark.parametrize("factor,values", PAIRS)
def test_counterfactual_arms_share_nuisance(cf_episode, factor, values):
    a, b = (cf_episode(factor, v, 31337) for v in values)
    # --- shared nuisance ---------------------------------------------------
    assert a.spec.room == b.spec.room
    assert a.spec.nuisance == b.spec.nuisance
    assert a.materials == b.materials
    assert [(li.name, li.intensity) for li in a.spec.lights] == [(li.name, li.intensity) for li in b.spec.lights]
    assert [(s.name, tuple(s.home), s.speed) for s in a.spec.staff] == [(s.name, tuple(s.home), s.speed)
                                                                       for s in b.spec.staff]
    assert _objects_key(a.survey_spec.objects) == _objects_key(b.survey_spec.objects)
    assert _objects_key(a.prior_map) == _objects_key(b.prior_map)
    # staff schedules are shared; a task goal may differ only by the exact displacement of an
    # object the hidden cause moved (CF-D cart_moved: staff walk to where the cart actually is)
    ta, tb = _tasks_key(a.workflow), _tasks_key(b.workflow)
    assert ta.keys() == tb.keys()
    shifts = _moved_offsets(a.spec.objects, b.spec.objects)
    for who in ta:
        assert [(x[0], x[2], x[3]) for x in ta[who]] == [(y[0], y[2], y[3]) for y in tb[who]]
        for x, y in zip(ta[who], tb[who]):
            d = np.subtract(y[1], x[1])
            assert np.allclose(d, 0.0, atol=1e-9) or any(np.allclose(d, s, atol=1e-9) for s in shifts), (who, x, y)
    assert _log_key(a.workflow) == _log_key(b.workflow)             # identical reported log
    assert [c.id for c in a.workflow.claims] == [c.id for c in b.workflow.claims]
    # --- differing hidden cause ---------------------------------------------
    assert a.spec.hidden_cause["value"] == values[0] and b.spec.hidden_cause["value"] == values[1]
    assert a.spec.hidden_cause["factor"] == b.spec.hidden_cause["factor"] == factor
    geometry_differs = _objects_key(a.spec.objects) != _objects_key(b.spec.objects)
    truth_differs = _truth_key(a.workflow) != _truth_key(b.workflow)
    faults_differ = a.faults.labels != b.faults.labels
    assert geometry_differs or truth_differs or faults_differ


def test_cf_a_retained_sponge_truth_differs_only_for_that_sponge(cf_episode):
    a, b = cf_episode("CF-A", "under_drape", 77), cf_episode("CF-A", "kick_bucket", 77)
    sp = a.workflow.hidden_notes["retained_sponge"]
    assert sp == b.workflow.hidden_notes["retained_sponge"]
    diff = set(_truth_key(a.workflow)) ^ set(_truth_key(b.workflow))
    assert diff and {d[1] for d in diff} == {sp}
    T = a.workflow.duration
    assert a.workflow.truth_slot(sp, T) == "field:under_drape"
    assert b.workflow.truth_slot(sp, T) == a.workflow.hidden_notes["claimed_bucket"]
    # identical geometry: CF-A acts only through the custody ground truth
    assert _objects_key(a.spec.objects) == _objects_key(b.spec.objects)


def test_cf_b_only_the_hidden_obstacle_differs(cf_episode):
    g, r = cf_episode("CF-B", "specular_ghost", 5), cf_episode("CF-B", "real_obstacle", 5)
    assert _objects_key(g.spec.objects) == _objects_key(r.spec.objects, skip_tags=("hidden_cause",))
    extra = [o for o in r.spec.objects if "hidden_cause" in o.tags]
    assert [o.name for o in extra] == ["aisle_obstacle"]
    # the obstacle sits at the mirror image of the hamper across the steel screen
    ap = np.array(r.spec.hidden_cause["aisle_point"])
    assert np.allclose(extra[0].box.center[:2], ap)
    assert g.spec.hidden_cause["aisle_point"] == r.spec.hidden_cause["aisle_point"]
    sx = r.spec.object("steel_screen").box.center[0]
    hamper = r.spec.object("linen_hamper").box.center
    assert ap[0] == pytest.approx(2 * sx - hamper[0]) and ap[1] == pytest.approx(hamper[1])
    # the screen is equally (strongly) specular in both arms: matched evidence
    assert "strong_specular" in g.spec.object("steel_screen").tags
    assert "strong_specular" in r.spec.object("steel_screen").tags
    assert _truth_key(g.workflow) == _truth_key(r.workflow)
    assert all("hidden_cause" not in o.tags for o in r.prior_map)


@pytest.mark.parametrize("factor,values", [("CF-A", ["under_drape", "kick_bucket"]),
                                           ("CF-C", ["dropped_floor", "handed_off"])])
def test_identical_staff_trajectories_when_geometry_is_shared(cf_episode, factor, values):
    """Robot-free staff rollouts are identical across arms whose physical world is identical."""
    pops = []
    for v in values:
        ep = cf_episode(factor, v, 404, {"episode": {"duration_s": 60.0}})
        pops.append(StaffPopulation(ep.spec, ep.workflow.staff_tasks, ep.streams, robot_aware=False,
                                    params=ep.cfg.get("agents")))
    for k in range(120):
        for p in pops:
            p.step(k * 0.1, 0.1, None, None)
    assert np.array_equal(pops[0].positions(), pops[1].positions())
    assert np.linalg.norm(pops[0].positions() - np.array([s.home for s in pops[0].spec.staff])) > 0.1


# ---------------------------------------------------------------------------
# seed registry
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def registry():
    return load_registry(Path(CONFIG_DIR) / "seed_registry.yaml")


def test_registry_shape_and_uniqueness(registry):
    assert len(registry) == 1200
    ids = [e.scenario_id for e in registry]
    assert len(set(ids)) == len(ids)
    fam = collections.Counter(e.family for e in registry)
    assert fam["counterfactual"] == 400 and all(fam[f] == 100 for f in FAMILIES)
    splits = collections.Counter(e.split for e in registry)
    assert set(splits) == {"train", "val", "test"}
    assert 0.6 < splits["train"] / len(registry) < 0.8


def test_registry_pairs_share_seed_split_and_nuisance(registry):
    pairs = collections.defaultdict(list)
    for e in registry:
        if e.pair_id:
            pairs[e.pair_id].append(e)
    assert len(pairs) == 200
    for pid, members in pairs.items():
        fac = pid.split("/")[0]
        assert len(members) == 2, pid
        a, b = members
        assert a.seed == b.seed and a.split == b.split and a.config == b.config
        arms = {a.overrides["hidden_cause"]["value"], b.overrides["hidden_cause"]["value"]}
        assert arms == set(COUNTERFACTUALS[fac][1])
        if pid.endswith("p0000"):                       # resolving is slow-ish: one pair per factor
            ra, rb = a.resolve(), b.resolve()
            for r in (ra, rb):
                r.pop("scenario_id")
                r["hidden_cause"].pop("value")
            assert ra == rb                              # configs differ only in hidden_cause.value
    assert len(select(registry, factor="CF-B")) == 100
    assert all(e.split == "test" for e in select(registry, split="test", limit=5))


def test_registry_regenerates_from_master_seed(registry, monkeypatch):
    """Seeds, splits, pair ids and overrides are a pure function of the master seed;
    stored config hashes match the current config files (sampled)."""
    monkeypatch.setattr(reg_mod.RegistryEntry, "resolve", lambda self: {})   # skip 1200 YAML loads
    monkeypatch.setattr(reg_mod, "config_hash", lambda cfg: "")
    regen = build_registry(per_family=100, pairs_per_cf=50, master_seed=20260924)
    assert [(e.scenario_id, e.seed, e.split, e.pair_id, e.overrides) for e in regen] == \
        [(e.scenario_id, e.seed, e.split, e.pair_id, e.overrides) for e in registry]
    monkeypatch.undo()
    for e in registry[::97]:
        assert config_hash(e.resolve()) == e.cfg_hash, e.scenario_id


def test_registry_entry_builds_reproducible_episode(registry):
    e = next(x for x in registry if x.pair_id == "CF-D/p0003")
    cfg = e.resolve()
    assert cfg["scenario_id"] == e.scenario_id and cfg["hidden_cause"]["factor"] == "CF-D"
    a, b = build_episode(cfg, e.seed), build_episode(e.resolve(), e.seed)
    assert a.spec.to_json() == b.spec.to_json() and _log_key(a.workflow) == _log_key(b.workflow)
    assert a.spec.scenario_id == e.scenario_id
