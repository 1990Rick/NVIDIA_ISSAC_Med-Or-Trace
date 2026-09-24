"""Parallel batch execution of registry entries (policies x variants)."""

from __future__ import annotations

import json
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from medortrace.common.config import deep_merge
from medortrace.eval.registry import RegistryEntry


def _json_safe(o):
    if isinstance(o, float) and (o != o):
        return None
    return o


def _run_one(entry: dict, policy: str, variant: str, autonomy_override: dict, cfg_override: dict,
             export_dir: str | None, duration: float | None, backend: str, save_raw: bool) -> dict:
    from medortrace.eval.runner import run_episode
    e = RegistryEntry(**entry)
    cfg = deep_merge(e.resolve(), cfg_override or {})
    out = None
    if export_dir:
        out = str(Path(export_dir) / variant)
    try:
        r = run_episode(cfg, e.seed, backend=backend, out_dir=out, policy=policy, duration=duration,
                        autonomy_override=autonomy_override, save_raw=save_raw)
        metrics = {k: _json_safe(v) for k, v in r.metrics.items()}
        err = None
    except Exception:  # keep the sweep alive; failures are reported
        metrics, err = {}, traceback.format_exc()
    return {"scenario_id": e.scenario_id, "family": e.family, "pair_id": e.pair_id, "split": e.split,
            "seed": e.seed, "policy": policy, "variant": variant, "hidden_cause": cfg.get("hidden_cause"),
            "cfg_hash": e.cfg_hash, "metrics": metrics, "error": err}


def run_batch(entries: list[RegistryEntry], policies=("active",), variants: dict | None = None,
              workers: int | None = None, out_jsonl: str | Path | None = None, export_dir: str | None = None,
              duration: float | None = None, backend: str = "lite", save_raw: bool = False,
              progress: bool = True) -> list[dict]:
    """``variants``: name -> {"autonomy": {...}, "cfg": {...}} (ablations / sensitivity)."""
    variants = variants or {"full": {}}
    jobs = []
    for e in entries:
        for p in policies:
            for vname, v in variants.items():
                jobs.append((e.__dict__, p, vname, v.get("autonomy", {}), v.get("cfg", {}), export_dir, duration,
                             backend, save_raw))
    rows = []
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    fh = open(out_jsonl, "a") if out_jsonl else None
    try:
        if workers == 1:
            it = (_run_one(*j) for j in jobs)
            for k, row in enumerate(it):
                rows.append(row)
                if fh:
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                if progress:
                    print(f"[{k + 1}/{len(jobs)}] {row['scenario_id']} {row['policy']}/{row['variant']}"
                          f"{' ERROR' if row['error'] else ''}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_run_one, *j) for j in jobs]
                for k, f in enumerate(as_completed(futs)):
                    row = f.result()
                    rows.append(row)
                    if fh:
                        fh.write(json.dumps(row) + "\n")
                        fh.flush()
                    if progress:
                        print(f"[{k + 1}/{len(jobs)}] {row['scenario_id']} {row['policy']}/{row['variant']}"
                              f"{' ERROR' if row['error'] else ''}", flush=True)
    finally:
        if fh:
            fh.close()
    return rows
