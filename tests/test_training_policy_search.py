"""CEM policy search: objective, entry sampling, common random numbers, update rule, CLI output (no simulation)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from medortrace.common.config import load_yaml  # noqa: E402
from medortrace.eval.policy_search import (  # noqa: E402
    DEFAULT_OBJECTIVE,
    CemConfig,
    CemSearch,
    episode_objective,
    load_base_weights,
    sample_entries,
)
from medortrace.eval.registry import RegistryEntry  # noqa: E402

BASE = load_base_weights()


def pool(n_per=6):
    out = []
    for fam in ("nominal", "loc_drift", "reflective"):
        for i in range(n_per):
            out.append(RegistryEntry(f"{fam}__{i:04d}", fam, f"scenarios/{fam}.yaml", 1000 + i, {}, None, "train",
                                     f"h{fam}{i}"))
    for i in range(n_per):
        for v in ("under_drape", "kick_bucket"):
            out.append(RegistryEntry(f"cf_a__p{i:04d}__{v}", "counterfactual", "scenarios/cf_a.yaml", 5000 + i,
                                     {}, f"CF-A/p{i:04d}", "train", f"hcf{i}{v}"))
    return out


def test_objective_matches_documented_formula():
    m = {"handoff_success": 0.75, "abstention_rate": 0.2, "wrong_assertion_rate": 0.1,
         "near_collision_rate_per_min": 0.4, "task_delay_s": 12.0, "collisions_agent": 1, "sterile_breach_s": 0.3}
    want = 0.75 + 0.5 * (1 - 0.2) - 2 * 0.1 - 0.5 * 0.4 - 0.01 * 12.0 - 5 * 1 - 5 * 0.3
    assert episode_objective(m) == pytest.approx(want)
    assert episode_objective({**m, "wrong_assertion_rate": None, "task_delay_s": float("nan")}) == \
        pytest.approx(want + 2 * 0.1 + 0.01 * 12.0)
    assert episode_objective(m, {"handoff_success": 2.0, "1-abstention_rate": 1.0}) == pytest.approx(1.5 + 0.8)
    assert set(DEFAULT_OBJECTIVE) == {"handoff_success", "1-abstention_rate", "wrong_assertion_rate",
                                      "near_collision_rate_per_min", "task_delay_s", "collisions_agent",
                                      "sterile_breach_s"}


def test_sample_entries_stratified_distinct_and_seeded():
    P = pool()
    a = sample_entries(P, 8, np.random.default_rng(1))
    b = sample_entries(P, 8, np.random.default_rng(1))
    assert [e.scenario_id for e in a] == [e.scenario_id for e in b]
    assert len({e.scenario_id for e in a}) == 8
    strata = {e.family if e.family != "counterfactual" else "CF-A" for e in a}
    assert strata == {"nominal", "loc_drift", "reflective", "CF-A"}       # round-robin covers every stratum
    assert len(sample_entries(P, 10_000, np.random.default_rng(0))) == len(P)


class FakeEval:
    """Objective peaks at w_eig = 8, w_risk = 1 (others irrelevant); records the entries of every call."""

    def __init__(self, fail: str | None = None):
        self.calls = []
        self.fail = fail

    def __call__(self, entries, variants):
        self.calls.append(([e.scenario_id for e in entries], variants))
        rows = []
        for e in entries:
            for name, v in variants.items():
                w = v["autonomy"]["nbv_weight_overrides"]
                q = -(math.log(w["w_eig"] / 8.0)) ** 2 - (math.log(w["w_risk"] / 1.0)) ** 2
                err = "boom" if (self.fail == name and e is entries[0]) else None
                rows.append({"scenario_id": e.scenario_id, "variant": name, "error": err,
                             "metrics": {} if err else {"handoff_success": q, "abstention_rate": 1.0}})
        return rows


def test_cem_common_random_numbers_and_convergence(tmp_path: Path):
    fe = FakeEval()
    cfg = CemConfig(iters=12, pop=16, elite=4, episodes=3, sigma0=0.8, keys=["w_eig", "w_risk"], seed=5)
    res = CemSearch(pool(), BASE, cfg, out_dir=tmp_path, evaluate=fe, log=None).run()
    assert len(fe.calls) == 12 and len(res.history) == 12
    for entries, variants in fe.calls:
        assert len(entries) == 3 and len(variants) == 16 and set(variants) == {f"cand_{i}" for i in range(16)}
    assert fe.calls[0][0] != fe.calls[1][0]                    # fresh seeds per iteration ...
    rec = res.history[0]                                       # ... shared by every candidate of an iteration
    assert rec.candidates[0] == pytest.approx(BASE)            # cand_0 is the incumbent mean
    assert res.weights["w_eig"] == pytest.approx(8.0, rel=0.25)
    assert res.weights["w_risk"] == pytest.approx(1.0, rel=0.25)
    for k in BASE:
        if k not in ("w_eig", "w_risk"):
            assert res.weights[k] == BASE[k]                   # unsearched keys stay at the base value
    hist = [r.summary()["incumbent"] for r in res.history]
    assert hist[-1] > hist[0]
    assert (tmp_path / "history.jsonl").read_text().count("\n") == 12
    assert (tmp_path / "results.jsonl").read_text().count("\n") == 12 * 16 * 3
    prov = res.provenance()
    yaml.safe_dump(prov)                                       # plain python types only
    assert prov["iterations"] == 12 and len(prov["seeds"]) == 12 and prov["cfg_hash"] == res.cfg_hash
    assert [h["iteration"] for h in prov["objective_history"]] == list(range(12))


def test_cem_error_rows_are_penalised():
    fe = FakeEval(fail="cand_1")
    cfg = CemConfig(iters=1, pop=3, elite=1, episodes=2, keys=["w_eig"], error_penalty=-7.0)
    res = CemSearch(pool(), BASE, cfg, evaluate=fe, log=None).run()
    rec = res.history[0]
    assert rec.n_errors == 1 and rec.per_episode[1][0] == -7.0 and 1 not in rec.elite
    with pytest.raises(ValueError):
        CemSearch(pool(), BASE, CemConfig(pop=2, elite=3), evaluate=fe, log=None)
    with pytest.raises(KeyError):
        CemSearch(pool(), BASE, CemConfig(keys=["w_nope"]), evaluate=fe, log=None)
    with pytest.raises(ValueError):
        CemSearch([], BASE, CemConfig(), evaluate=fe, log=None)


def test_train_nbv_policy_cli_writes_loadable_policy(tmp_path: Path, monkeypatch):
    import train_nbv_policy
    monkeypatch.setattr(CemSearch, "_run_batch", lambda self, entries, variants: FakeEval()(entries, variants))
    out = tmp_path / "nbv_trained.yaml"
    rc = train_nbv_policy.main(["--iters", "2", "--pop", "4", "--elite", "2", "--episodes", "2", "--duration", "20",
                                "--out", str(tmp_path / "run"), "--output", str(out),
                                "--objective-weights", "task_delay_s=-0.02", "sterile_breach_s=0"])
    assert rc == 0
    doc = yaml.safe_load(out.read_text())
    assert set(doc["weights"]) == set(BASE) and all(v > 0 for v in doc["weights"].values())
    assert load_yaml(out)["weights"] == doc["weights"]         # what AutonomyStack reads (autonomy.nbv_weights)
    prov = doc["provenance"]
    assert prov["objective"]["task_delay_s"] == -0.02 and "sterile_breach_s" not in prov["objective"]
    assert prov["smoke_scale"] and prov["iterations"] == 2 and len(prov["seeds"]) == 2
    assert all(e["split"] == "train" for s in prov["seeds"] for e in s["entries"])
    assert "WARNING: smoke-scale" in out.read_text().splitlines()[3]
    assert (tmp_path / "run" / "history.jsonl").exists()
