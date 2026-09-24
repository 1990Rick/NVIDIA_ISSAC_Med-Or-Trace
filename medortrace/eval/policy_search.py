"""Cross-Entropy Method (CEM) search over the next-best-view objective weights.

The NBV planner (``medortrace.planning.nbv``) scores viewpoints with a weighted
sum of information-gain, cost and risk terms whose weights live in
``configs/policies/nbv_default.yaml``.  This module tunes those weights
end-to-end against *episode-level* outcomes measured in the lite simulator.

Search space
    ``theta = log(w)`` for the searched keys (all weights are non-negative
    scales, so a Gaussian over log-weights keeps them positive and makes the
    step size relative).  Keys not searched stay at their base value.

Iteration ``k``
    1. sample ``pop`` candidates ``theta_i ~ N(mu_k, diag sigma_k^2)``; with
       ``include_mean`` candidate 0 *is* ``mu_k`` (so the incumbent is always
       re-measured on the new seeds);
    2. draw ``episodes`` registry entries from the TRAIN split (stratified over
       families / counterfactual factors) with an iteration-keyed RNG;
    3. evaluate every candidate on the *same* entries with
       :func:`medortrace.eval.batch.run_batch` using variants
       ``{"cand_i": {"autonomy": {"nbv_weight_overrides": w_i}}}``.  This is
       common random numbers: the scenario seed fixes layout, staff, faults,
       sensor noise and the stack's policy seed, so candidate differences are
       paired and the ranking reflects the weights, not the scenario lottery;
    4. ``J_i`` = mean episode objective; the ``elite`` best candidates update
       ``mu <- (1-a) mu + a mean(theta_elite)`` and
       ``sigma <- (1-a) sigma + a std(theta_elite) + noise_k`` (``noise_k``
       decays linearly to 0; sigma is clipped to ``[sigma_min, sigma_max]``).

Objective (per episode, metrics from ``medortrace.eval.metrics``; defaults in
:data:`DEFAULT_OBJECTIVE`, every coefficient is configurable)::

    J = 1.0  * handoff_success                  (claims answered correctly and on time)
      + 0.5  * (1 - abstention_rate)            (answering is worth something ...)
      - 2.0  * wrong_assertion_rate             (... but a wrong assertion costs 4x more)
      - 0.5  * near_collision_rate_per_min
      - 0.01 * task_delay_s                     (staff delay vs the robot-free shadow twin)
      - 5.0  * collisions_agent
      - 5.0  * sterile_breach_s

A term named ``"1-<metric>"`` uses ``1 - metric``.  Missing / NaN metrics count
as 0; a crashed episode scores ``error_penalty``.

The search is deliberately budget-agnostic: ``--iters 1 --pop 2 --elite 1
--episodes 1 --duration 20`` is a smoke test, a real run uses tens of
episodes per iteration.  ``scripts/train_nbv_policy.py`` is the CLI.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from medortrace.common.config import config_hash, load_yaml
from medortrace.common.rng import stable_hash
from medortrace.eval.registry import RegistryEntry

DEFAULT_OBJECTIVE: dict[str, float] = {
    "handoff_success": 1.0,
    "1-abstention_rate": 0.5,
    "wrong_assertion_rate": -2.0,
    "near_collision_rate_per_min": -0.5,
    "task_delay_s": -0.01,
    "collisions_agent": -5.0,
    "sterile_breach_s": -5.0,
}


def load_base_weights(path: str | Path = "policies/nbv_default.yaml") -> dict[str, float]:
    """``weights:`` mapping of an NBV policy file (paths relative to ``configs/`` work from any cwd)."""
    return {k: float(v) for k, v in load_yaml(path).get("weights", {}).items()}


# ---------------------------------------------------------------------------
def _metric(m: dict, name: str) -> float:
    v = m.get(name)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    return v if math.isfinite(v) else 0.0


def episode_objective(metrics: dict, terms: dict[str, float] | None = None) -> float:
    """Weighted sum of episode metrics (see module docstring)."""
    J = 0.0
    for name, c in (DEFAULT_OBJECTIVE if terms is None else terms).items():
        if name.startswith("1-"):
            J += float(c) * (1.0 - _metric(metrics, name[2:]))
        else:
            J += float(c) * _metric(metrics, name)
    return float(J)


def objective_breakdown(metrics: dict, terms: dict[str, float] | None = None) -> dict[str, float]:
    terms = DEFAULT_OBJECTIVE if terms is None else terms
    return {name: (1.0 - _metric(metrics, name[2:])) if name.startswith("1-") else _metric(metrics, name)
            for name in terms}


# ---------------------------------------------------------------------------
def _stratum(e: RegistryEntry) -> str:
    if e.family == "counterfactual" and e.pair_id:
        return e.pair_id.split("/")[0]
    return e.family


def sample_entries(pool: list[RegistryEntry], n: int, rng: np.random.Generator,
                   stratify: bool = True) -> list[RegistryEntry]:
    """``n`` distinct entries; round-robin over randomly ordered strata (families / CF factors)."""
    if n >= len(pool):
        return list(pool)
    if not stratify:
        return [pool[i] for i in sorted(rng.choice(len(pool), size=n, replace=False))]
    groups: dict[str, list[RegistryEntry]] = defaultdict(list)
    for e in pool:
        groups[_stratum(e)].append(e)
    keys = sorted(groups)
    order = [keys[i] for i in rng.permutation(len(keys))]
    queues = {k: [groups[k][i] for i in rng.permutation(len(groups[k]))] for k in keys}
    out: list[RegistryEntry] = []
    while len(out) < n:
        for k in order:
            if queues[k] and len(out) < n:
                out.append(queues[k].pop())
    return out


# ---------------------------------------------------------------------------
@dataclass
class CemConfig:
    iters: int = 8
    pop: int = 10
    elite: int = 3
    episodes: int = 8                       # registry entries per iteration (shared by all candidates)
    duration: float | None = 60.0           # episode sim time override (None: scenario default)
    workers: int | None = None
    keys: list[str] | None = None           # searched weights (default: all base keys)
    sigma0: float = 0.5                     # initial std of log-weights (0.5 ~ x1.65)
    sigma_min: float = 0.05
    sigma_max: float = 1.5
    alpha: float = 0.7                      # smoothing of the mean / std update
    extra_noise: float = 0.1                # added to sigma, decays linearly to 0 at the last iteration
    log_bounds: tuple[float, float] = (-4.0, 4.0)   # clip theta - log(base) (x0.018 .. x55)
    include_mean: bool = True
    resample_entries: bool = True           # new seeds every iteration (False: one fixed subset)
    stratify: bool = True
    seed: int = 0
    split: str = "train"
    family: str | None = None
    factor: str | None = None
    policy: str = "active"
    backend: str = "lite"
    error_penalty: float = -10.0
    final: str = "mean"                     # mean: exp(mu) after the last update | best: best evaluated candidate
    objective: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_OBJECTIVE))

    def validate(self) -> None:
        if self.pop < 1 or self.elite < 1 or self.elite > self.pop:
            raise ValueError(f"need 1 <= elite <= pop (got elite={self.elite}, pop={self.pop})")
        if self.iters < 1 or self.episodes < 1:
            raise ValueError("iters and episodes must be >= 1")
        if self.final not in ("mean", "best"):
            raise ValueError("final must be 'mean' or 'best'")
        if self.split != "train":
            print(f"[policy_search] WARNING: searching on split={self.split!r}; tuned weights must be "
                  "reported on a split they were not tuned on")


@dataclass
class IterationRecord:
    iteration: int
    entries: list[dict]                     # scenario_id, seed, cfg_hash, split, family
    candidates: list[dict[str, float]]      # full weight dicts
    objectives: list[float]                 # mean J per candidate
    objective_std: list[float]
    per_episode: list[list[float]]          # [candidate][entry]
    elite: list[int]
    mu: dict[str, float]                    # log-weights after the update
    sigma: dict[str, float]
    n_errors: int
    wall_s: float

    def summary(self) -> dict:
        j = np.asarray(self.objectives)
        return {"iteration": self.iteration, "best": float(j.max()), "mean": float(j.mean()),
                "elite_mean": float(j[self.elite].mean()), "incumbent": float(j[0]),
                "n_errors": self.n_errors, "wall_s": round(self.wall_s, 2)}


@dataclass
class CemResult:
    weights: dict[str, float]
    best: dict
    history: list[IterationRecord]
    mu: dict[str, float]
    sigma: dict[str, float]
    base_weights: dict[str, float]
    config: CemConfig
    cfg_hash: str

    def provenance(self, extra: dict | None = None) -> dict:
        cfg = asdict(self.config)
        cfg["log_bounds"] = list(cfg["log_bounds"])
        seeds = [{"iteration": r.iteration, "entries": r.entries} for r in self.history]
        prov = {
            "method": "cross_entropy_method",
            "objective": dict(self.config.objective),
            "search_config": cfg,
            "cfg_hash": self.cfg_hash,
            "iterations": len(self.history),
            "episodes_evaluated": int(sum(len(r.entries) * len(r.candidates) for r in self.history)),
            "episode_errors": int(sum(r.n_errors for r in self.history)),
            "base_weights": dict(self.base_weights),
            "final": self.config.final,
            "final_sigma_log": {k: float(v) for k, v in self.sigma.items()},
            "best_candidate": self.best,
            "objective_history": [r.summary() for r in self.history],
            "seeds": seeds,
        }
        prov.update(extra or {})
        return prov


class CemSearch:
    """CEM over log NBV weights; see the module docstring for the algorithm."""

    def __init__(self, pool: list[RegistryEntry], base_weights: dict[str, float], cfg: CemConfig,
                 out_dir: str | Path | None = None, evaluate: Callable | None = None, log=print):
        cfg.validate()
        if not pool:
            raise ValueError("empty registry pool (check --split/--family/--factor)")
        self.pool = list(pool)
        self.base = {k: float(v) for k, v in base_weights.items()}
        self.keys = list(cfg.keys or self.base)
        unknown = [k for k in self.keys if k not in self.base]
        if unknown:
            raise KeyError(f"weights {unknown} not in base policy {sorted(self.base)}")
        bad = [k for k in self.keys if self.base[k] <= 0]
        if bad:
            raise ValueError(f"log-space search needs positive base weights: {bad}")
        self.cfg = cfg
        self.out = Path(out_dir) if out_dir else None
        self.log = log or (lambda *a, **k: None)
        self._evaluate = evaluate or self._run_batch
        self.theta0 = np.log(np.array([self.base[k] for k in self.keys]))
        self.cfg_hash = config_hash({"search": asdict(cfg), "base": self.base, "keys": self.keys,
                                     "pool": [e.cfg_hash for e in self.pool]})

    # ------------------------------------------------------------------
    def weights_of(self, theta: np.ndarray) -> dict[str, float]:
        w = dict(self.base)
        w.update({k: float(np.exp(t)) for k, t in zip(self.keys, theta)})
        return w

    def _clip(self, theta: np.ndarray) -> np.ndarray:
        lo, hi = self.cfg.log_bounds
        return np.clip(theta, self.theta0 + lo, self.theta0 + hi)

    def _entries(self, k: int) -> list[RegistryEntry]:
        key = k if self.cfg.resample_entries else 0
        rng = np.random.default_rng([int(self.cfg.seed) & (2**63 - 1), stable_hash(f"cem:entries:{key}")])
        return sample_entries(self.pool, self.cfg.episodes, rng, self.cfg.stratify)

    def _run_batch(self, entries: list[RegistryEntry], variants: dict) -> list[dict]:
        from medortrace.eval.batch import run_batch
        return run_batch(entries, (self.cfg.policy,), variants, workers=self.cfg.workers, duration=self.cfg.duration,
                         backend=self.cfg.backend, progress=False)

    # ------------------------------------------------------------------
    def run(self) -> CemResult:
        cfg = self.cfg
        mu = self.theta0.copy()
        sigma = np.full(len(self.keys), float(cfg.sigma0))
        history: list[IterationRecord] = []
        best = {"objective": -np.inf}
        if self.out:
            self.out.mkdir(parents=True, exist_ok=True)
        for k in range(cfg.iters):
            t0 = time.time()
            rng = np.random.default_rng([int(cfg.seed) & (2**63 - 1), stable_hash(f"cem:candidates:{k}")])
            thetas = mu + sigma * rng.standard_normal((cfg.pop, len(self.keys)))
            if cfg.include_mean:
                thetas[0] = mu
            thetas = self._clip(thetas)
            cands = [self.weights_of(th) for th in thetas]
            entries = self._entries(k)
            variants = {f"cand_{i}": {"autonomy": {"nbv_weight_overrides": w}} for i, w in enumerate(cands)}
            self.log(f"[cem] iter {k + 1}/{cfg.iters}: {len(cands)} candidates x {len(entries)} episodes "
                     f"({', '.join(e.scenario_id for e in entries)})")
            rows = self._evaluate(entries, variants)
            J, n_err = self._score(rows, entries, len(cands))
            Jm = J.mean(axis=1)
            elite = [int(i) for i in np.argsort(-Jm, kind="stable")[: cfg.elite]]
            # --- CEM update in log space ------------------------------------
            a = float(cfg.alpha)
            noise = cfg.extra_noise * max(0.0, 1.0 - k / max(cfg.iters - 1, 1))
            mu = self._clip(((1 - a) * mu + a * thetas[elite].mean(axis=0))[None])[0]
            sigma = np.clip((1 - a) * sigma + a * thetas[elite].std(axis=0) + noise, cfg.sigma_min, cfg.sigma_max)
            rec = IterationRecord(
                iteration=k, entries=[{"scenario_id": e.scenario_id, "seed": int(e.seed), "cfg_hash": e.cfg_hash,
                                       "split": e.split, "family": e.family} for e in entries],
                candidates=cands, objectives=[float(x) for x in Jm], objective_std=[float(x) for x in J.std(axis=1)],
                per_episode=J.tolist(), elite=elite, mu=dict(zip(self.keys, map(float, mu))),
                sigma=dict(zip(self.keys, map(float, sigma))), n_errors=n_err, wall_s=time.time() - t0)
            history.append(rec)
            i_best = elite[0]
            if Jm[i_best] > best["objective"]:
                best = {"objective": float(Jm[i_best]), "iteration": k, "candidate": f"cand_{i_best}",
                        "weights": cands[i_best]}
            self._log_iteration(rec, rows)
        weights = self.weights_of(mu) if cfg.final == "mean" else dict(best["weights"])
        return CemResult(weights, best, history, dict(zip(self.keys, map(float, mu))),
                         dict(zip(self.keys, map(float, sigma))), self.base, cfg, self.cfg_hash)

    def _score(self, rows: list[dict], entries: list[RegistryEntry], n_cand: int) -> tuple[np.ndarray, int]:
        col = {e.scenario_id: j for j, e in enumerate(entries)}
        J = np.full((n_cand, len(entries)), float(self.cfg.error_penalty))
        seen = np.zeros_like(J, dtype=bool)
        n_err = 0
        for r in rows:
            i = int(str(r["variant"]).split("_")[-1])
            j = col.get(r["scenario_id"])
            if j is None:
                continue
            seen[i, j] = True
            if r.get("error") or not r.get("metrics"):
                n_err += 1
                self.log(f"[cem] episode error ({r['scenario_id']}, {r['variant']}):\n{r.get('error')}")
                continue
            J[i, j] = episode_objective(r["metrics"], self.cfg.objective)
        n_err += int((~seen).sum())          # missing rows count as failures
        return J, n_err

    def _log_iteration(self, rec: IterationRecord, rows: list[dict]) -> None:
        s = rec.summary()
        self.log(f"[cem] iter {rec.iteration + 1}: best={s['best']:.3f} mean={s['mean']:.3f} "
                 f"incumbent={s['incumbent']:.3f} elite={['cand_%d' % i for i in rec.elite]} "
                 f"errors={rec.n_errors} ({rec.wall_s:.1f}s)")
        if not self.out:
            return
        with open(self.out / "history.jsonl", "a") as f:
            f.write(json.dumps({**asdict(rec), "summary": s, "cfg_hash": self.cfg_hash}) + "\n")
        with open(self.out / "results.jsonl", "a") as f:
            for r in rows:
                obj = episode_objective(r["metrics"], self.cfg.objective) if r.get("metrics") else None
                f.write(json.dumps({**r, "iteration": rec.iteration, "objective": obj,
                                    "objective_terms": objective_breakdown(r.get("metrics") or {},
                                                                           self.cfg.objective)}) + "\n")
