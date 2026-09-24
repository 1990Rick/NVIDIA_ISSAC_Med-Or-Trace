"""sim_overrides plumbing (runner / batch) and the sim-to-real sensitivity analysis, without running a simulation."""

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import medortrace.eval.runner as runner  # noqa: E402
from medortrace.eval.batch import run_batch  # noqa: E402
from medortrace.eval.registry import RegistryEntry  # noqa: E402

_spec = importlib.util.spec_from_file_location("sim_to_real_sensitivity",
                                               ROOT / "scripts" / "sim_to_real_sensitivity.py")
s2r = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = s2r                  # dataclasses resolve annotations through sys.modules
_spec.loader.exec_module(s2r)


# ---------------------------------------------------------------------------------------------------------------------
def _fake_loop(monkeypatch):
    """Replace the heavy parts of run_episode and record the config each component receives."""
    seen = {}

    class Stack:
        def __init__(self, inputs, cfg):
            seen["stack"] = cfg
            self.telemetry = [SimpleNamespace(verdicts=[])]
            self.prov = SimpleNamespace(verify_chain=lambda: True)
            self.operator_request_open = False

        def step(self, bundle, dt):
            self.telemetry.append(SimpleNamespace(verdicts=[]))

        def finalize(self, t):
            return []

    class Backend:
        dt, t = 0.1, 1.0

        def reset(self, ep):
            seen["reset_ep_cfg"] = ep.cfg

        def close(self):
            pass

    def build_episode(cfg, seed):
        seen["episode"] = cfg
        return SimpleNamespace(cfg=cfg, workflow=SimpleNamespace(duration=0.0), spec=SimpleNamespace(scenario_id="x"),
                               streams=SimpleNamespace(fork=lambda *a: np.random.default_rng(0)))

    def make_backend(name, cfg):
        seen["backend"] = cfg
        return Backend()

    monkeypatch.setattr(runner, "build_episode", build_episode)
    monkeypatch.setattr(runner, "make_backend", make_backend)
    monkeypatch.setattr(runner, "AutonomyStack", Stack)
    monkeypatch.setattr(runner, "stack_inputs_from_episode", lambda ep: None)
    monkeypatch.setattr(runner, "compute_metrics", lambda *a, **k: {"m": 1.0})
    return seen


def test_sim_overrides_reach_the_world_only(monkeypatch):
    seen = _fake_loop(monkeypatch)
    cfg = {"sensors": {"camera": {"pd0": 0.92, "max_range": 5.0}}, "agents": {"A_robot": 4.0}}
    ov = {"sensors": {"camera": {"pd0": 0.7}}, "agents": {"A_robot": 3.0}}
    r = runner.run_episode(cfg, 5, policy="passive", duration=12.0, sim_overrides=ov)
    for k in ("episode", "backend", "reset_ep_cfg"):
        assert seen[k]["sensors"]["camera"] == {"pd0": 0.7, "max_range": 5.0}
        assert seen[k]["agents"]["A_robot"] == 3.0
        assert seen[k]["autonomy"]["policy"] == "passive" and seen[k]["episode"]["duration_s"] == 12.0
    assert seen["stack"]["sensors"]["camera"]["pd0"] == 0.92 and seen["stack"]["agents"]["A_robot"] == 4.0
    assert seen["stack"]["_policy_seed"] == seen["episode"]["_policy_seed"] == 5 * 7919 + 17
    assert cfg["sensors"]["camera"]["pd0"] == 0.92            # caller's dict untouched
    assert r.policy == "passive"


def test_without_sim_overrides_everyone_sees_the_same_cfg(monkeypatch):
    seen = _fake_loop(monkeypatch)
    runner.run_episode({"agents": {"A_robot": 4.0}}, 1)
    assert seen["stack"] is seen["episode"] is seen["backend"]


def test_batch_threads_sim_variant_and_records_it(monkeypatch):
    calls = []

    def fake_run_episode(cfg, seed, **kw):
        calls.append(kw)
        return SimpleNamespace(metrics={"x": 1.0, "y": float("nan")})

    monkeypatch.setattr(runner, "run_episode", fake_run_episode)
    e = RegistryEntry("nominal__t", "nominal", "scenarios/nominal.yaml", 3)
    rows = run_batch([e], ("active",), {"nominal": {}, "pd": {"sim": {"sensors": {"camera": {"pd0": 0.5}}}}},
                     workers=1, duration=15.0, progress=False)
    by = {r["variant"]: r for r in rows}
    assert by["pd"]["sim"] == {"sensors": {"camera": {"pd0": 0.5}}} and "sim" not in by["nominal"]
    assert by["pd"]["duration"] == 15.0 and by["pd"]["metrics"]["y"] is None
    assert [c["sim_overrides"] for c in calls] == [None, {"sensors": {"camera": {"pd0": 0.5}}}]
    assert by["nominal"]["variant_spec"] == {}
    assert by["pd"]["variant_spec"] == {"sim": {"sensors": {"camera": {"pd0": 0.5}}}}


def test_batch_records_the_full_variant_spec(monkeypatch):
    monkeypatch.setattr(runner, "run_episode", lambda cfg, seed, **kw: SimpleNamespace(metrics={}))
    e = RegistryEntry("nominal__t", "nominal", "scenarios/nominal.yaml", 3)
    spec = {"autonomy": {"modalities": ["lidar", "camera"]}, "cfg": {"verifier": {"tau_verify": 0.5}}}
    rows = run_batch([e], ("active",), {"abl": spec}, workers=1, progress=False)
    assert rows[0]["variant_spec"] == spec and "sim" not in rows[0]


# ---------------------------------------------------------------------------------------------------------------------
def test_perturbation_overrides_clip_and_scale_ranges():
    ov, p0, p1 = s2r.perturb(s2r.PARAMS["camera_pd0"], {}, 0.25)
    assert ov == {"sensors": {"camera": {"pd0": 1.0}}} and p0 == 0.92 and p1 == 1.0     # clipped probability
    ov, p0, p1 = s2r.perturb(s2r.PARAMS["haze"], {"nuisance": {"haze": [0.1, 0.3]}}, -0.5)
    assert ov == {"nuisance": {"haze": [0.05, 0.15]}} and math.isclose(p0, 0.2) and math.isclose(p1, 0.1)
    ov, p0, p1 = s2r.perturb(s2r.PARAMS["specular_gain"], {"faults": {"specular_gain": 1.6}}, 0.25)
    assert ov == {"faults": {"specular_gain": 2.0}} and p0 == 1.6
    assert s2r.variant_name("haze", -0.25) == "haze-0.25"
    for p in s2r.PARAMS.values():                  # every parameter path exists in the simulator config schema
        ov, p0, p1 = s2r.perturb(p, {}, 0.1)
        assert p0 > 0 and p1 != p0


def test_elasticity_and_zero_baseline():
    e = s2r.elasticity(np.array([10.0, 10.0, 10.0]), np.array([12.0, 11.0, 13.0]), np.full(3, 0.2), B=200)
    assert math.isclose(e["S"], 1.0) and e["ci95"][0] <= 1.0 <= e["ci95"][1] and e["n"] == 3
    z = s2r.elasticity(np.zeros(3), np.array([0.0, 1.0, 0.0]), np.full(3, 0.2), B=200)
    assert z["S"] is None and math.isclose(z["dM"], 1 / 3)
    c = s2r.central_elasticity(np.full(3, 10.0), np.full(3, 9.0), np.full(3, 11.0), np.full(3, -0.25),
                               np.full(3, 0.25), B=200)
    assert math.isclose(c["S"], 0.4)


def _rows(metric_fn):
    rows = []
    for s in ("a", "b", "c"):
        for pol in ("active", "fixed_route"):
            for v in ("nominal", "camera_pd0-0.25", "camera_pd0+0.25"):
                rows.append({"scenario_id": s, "policy": pol, "variant": v, "metrics": metric_fn(s, pol, v),
                             "error": None})
    return rows


def test_analyze_detects_conclusion_flips_and_central_estimate():
    design = {"levels_by_param": {"camera_pd0": [-0.25, 0.25]},
              "entries": {s: {"camera_pd0-0.25": {"rel": -0.25, "p0": 0.92, "p1": 0.69},
                              "camera_pd0+0.25": {"rel": 0.087, "p0": 0.92, "p1": 1.0}} for s in "abc"}}
    jitter = {"a": 0.0, "b": 0.01, "c": -0.01}

    def metric(s, pol, v):
        # active has higher handoff_success than the baseline at nominal, lower when pd0 drops (a flip)
        hs = {"nominal": 0.6, "camera_pd0-0.25": 0.3, "camera_pd0+0.25": 0.65}[v] if pol == "active" else 0.5
        return {"handoff_success": hs + jitter[s], "task_delay_s": 10.0, "collisions_agent": 0.0}

    res = s2r.analyze(_rows(metric), design, metrics=["handoff_success", "task_delay_s", "collisions_agent"], B=300)
    flips = res["flips"]
    assert [(f["param"], f["level"], f["metric"]) for f in flips] == [("camera_pd0", -0.25, "handoff_success")]
    assert flips[0]["significant"] and flips[0]["active_better_nominal"]
    pr = res["params"]["camera_pd0"]
    lm = pr["levels"]["-0.25"]["metrics"]["handoff_success"]
    assert math.isclose(lm["S"], (0.3 - 0.6) / 0.6 / -0.25, rel_tol=1e-6)
    assert math.isclose(pr["central"]["handoff_success"]["S"], (0.65 - 0.3) / 0.6 / (0.087 + 0.25), rel_tol=1e-6)
    assert pr["levels"]["+0.25"]["metrics"]["task_delay_s"]["S"] == 0.0
    assert pr["levels"]["+0.25"]["metrics"]["collisions_agent"]["S"] is None      # zero baseline: no elasticity
    md = s2r.markdown(res, {"duration": 20.0})
    assert "| camera_pd0 | -0.25 | handoff_success |" in md and "Most sim-sensitive parameters" in md
