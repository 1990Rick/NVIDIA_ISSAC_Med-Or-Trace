#!/usr/bin/env bash
# The full MED-OR-TRACE experiment program on the lite backend (the numbers in
# docs/results.md).  About 6 h on 4 cores; every step is resumable by hand and
# every episode is fixed by configs/seed_registry.yaml.
#
#   WORKERS=4 OUT=runs/final bash scripts/run_experiments.sh [step ...]
#
# Steps: cem benchmark ablation cf sensitivity atlas (default: all, in order).
set -euo pipefail
cd "$(dirname "$0")/.."
W=${WORKERS:-4}
OUT=${OUT:-runs/final}
STEPS=${*:-cem benchmark ablation cf sensitivity atlas}
mkdir -p "$OUT"

for step in $STEPS; do
  echo "=== $step ($(date -u +%H:%M:%S))"
  case $step in
    cem)          # NBV objective weights, cross-entropy method on the TRAIN split (common random numbers)
      python scripts/train_nbv_policy.py --iters 6 --pop 8 --elite 3 --episodes 6 --duration 120 \
        --workers "$W" --out "$OUT/cem" --output configs/policies/nbv_trained.yaml ;;
    benchmark)    # held-out TEST split: active (hand-set and trained NBV) vs fixed-route vs passive
      python scripts/run_benchmark.py --split test --limit-per-family 4 --workers "$W" --out "$OUT/benchmark" \
        --nbv-variant trained=configs/policies/nbv_trained.yaml ;;
    ablation)     # VAL split, one capability removed per variant, paired against the full stack
      python scripts/run_ablation.py --split val --limit 24 --duration 150 --workers "$W" --out "$OUT/ablation" ;;
    cf)           # matched counterfactual pairs: discrimination, unsafe outcomes, appropriate abstention
      python scripts/analyze_counterfactuals.py "$OUT/benchmark/results.jsonl" "$OUT/ablation/results.jsonl" \
        --out "$OUT/cf" ;;
    sensitivity)  # sim-to-real: +/-25 % on simulator parameters; does the active-vs-baseline ranking flip?
      python scripts/sim_to_real_sensitivity.py --families nominal reflective sensor_dropout --n-seeds 2 \
        --params camera_pd0 specular_gain glare_gain agents_A_robot workflow_p_log_missing haze \
        --duration 120 --workers "$W" --out "$OUT/sensitivity" ;;
    atlas)        # failure-case atlas with exact reproduce commands
      python scripts/build_failure_atlas.py "$OUT/benchmark/results.jsonl" "$OUT/ablation/results.jsonl" \
        --out "$OUT/atlas" ;;
    *) echo "unknown step $step" >&2; exit 2 ;;
  esac
done
echo "=== done ($(date -u +%H:%M:%S))"
