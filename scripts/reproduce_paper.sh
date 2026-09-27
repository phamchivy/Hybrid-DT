#!/usr/bin/env bash
# Runs, in parallel, every five-seed robustness sweep this repo defines:
#   - controlled (synthetic) and TelecomTS, each with the paper's fixed
#     fusion weights and with --learnable-gate (4 jobs)
#   - 2 structural ablations (single-plane graph, no queue features), each
#     on both datasets, fixed-gate only -- the ablations isolate MP-Graph's
#     architecture, so they are not crossed with the gate ablation (4 jobs)
# That is 8 independent, CPU-only (numpy/pandas, no GPU) jobs, launched in
# parallel; each writes its own log under logs/.
#
# There is deliberately no separate single-seed run here: seed 7 (controlled)
# and seed 17 (TelecomTS) are the paper's original single-split seeds, and
# both are already included as one of the five seeds in the sweeps below
# (outputs/*_multiseed*/seed_7/ and .../seed_17/ respectively), so a
# standalone single-seed run would just duplicate that one seed's result.
set -uo pipefail

# Each run below is independent; cap every process to a single BLAS thread so
# 8 parallel jobs times numpy's own internal multi-threading doesn't
# oversubscribe the machine's cores.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p outputs logs

SEEDS="7 11 17 23 29"
TELECOMTS_SAMPLES=3200
TELECOMTS_CACHE=data/telecomts_3200_even.jsonl.gz

pids=()
names=()

launch() {
  local name="$1"; shift
  (
    echo "[$(date +%T)] START $name: $*"
    if "$@" > "logs/${name}.log" 2>&1; then
      echo "[$(date +%T)] OK    $name"
    else
      status=$?
      echo "[$(date +%T)] FAIL  $name (exit $status) -- see logs/${name}.log" >&2
      exit "$status"
    fi
  ) &
  pids+=("$!")
  names+=("$name")
}

# --- Controlled (synthetic), 5-seed sweep ---
launch controlled_multiseed_fixed \
  python -m benchmark.run_multiseed --dataset controlled \
    --seeds $SEEDS --timesteps 1800 --window 12 --horizon 3 \
    --outdir outputs/controlled_multiseed

launch controlled_multiseed_learnable \
  python -m benchmark.run_multiseed --dataset controlled \
    --seeds $SEEDS --timesteps 1800 --window 12 --horizon 3 --learnable-gate \
    --outdir outputs/controlled_multiseed_learnable

# --- TelecomTS, 5-seed sweep, 3200 samples ---
# The first job to touch $TELECOMTS_CACHE downloads and caches those rows;
# --skip-checksum is required because the pinned paper checksum only covers
# the tracked 800-row snapshot (data/telecomts_800_even.jsonl.gz), not this
# larger 3200-row pull.
launch telecomts_multiseed_fixed \
  python -m benchmark.run_multiseed --dataset telecomts \
    --seeds $SEEDS --samples "$TELECOMTS_SAMPLES" --input-len 96 \
    --cache "$TELECOMTS_CACHE" --download --skip-checksum \
    --outdir outputs/telecomts_multiseed

launch telecomts_multiseed_learnable \
  python -m benchmark.run_multiseed --dataset telecomts \
    --seeds $SEEDS --samples "$TELECOMTS_SAMPLES" --input-len 96 --learnable-gate \
    --cache "$TELECOMTS_CACHE" --download --skip-checksum \
    --outdir outputs/telecomts_multiseed_learnable

# --- Ablations, 5-seed sweep, fixed gate ---
# Single-plane graph: collapse control/user/slice down to MP-Graph's "all"
# plane (pure data substitution, no model code change).
launch controlled_ablation_single_plane \
  python -m benchmark.run_multiseed --dataset controlled \
    --seeds $SEEDS --timesteps 1800 --window 12 --horizon 3 --single-plane-graph \
    --outdir outputs/controlled_ablation_single_plane

launch telecomts_ablation_single_plane \
  python -m benchmark.run_multiseed --dataset telecomts \
    --seeds $SEEDS --samples "$TELECOMTS_SAMPLES" --input-len 96 --single-plane-graph \
    --cache "$TELECOMTS_CACHE" --download --skip-checksum \
    --outdir outputs/telecomts_ablation_single_plane

# No queue features: drop MP-Graph's queueing-theory utilization/delay block.
launch controlled_ablation_no_queue \
  python -m benchmark.run_multiseed --dataset controlled \
    --seeds $SEEDS --timesteps 1800 --window 12 --horizon 3 --no-queue-features \
    --outdir outputs/controlled_ablation_no_queue

launch telecomts_ablation_no_queue \
  python -m benchmark.run_multiseed --dataset telecomts \
    --seeds $SEEDS --samples "$TELECOMTS_SAMPLES" --input-len 96 --no-queue-features \
    --cache "$TELECOMTS_CACHE" --download --skip-checksum \
    --outdir outputs/telecomts_ablation_no_queue

fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    fail=1
  fi
done

if [ "$fail" -ne 0 ]; then
  echo ""
  echo "One or more runs failed. Logs are in logs/*.log. Aborting before summary."
  exit 1
fi

echo ""
echo "All 8 runs finished."
echo ""
echo "Note: benchmark.verify_results checks against artifacts/expected/*.csv,"
echo "which are pinned to the paper's original 800-row TelecomTS snapshot and"
echo "single controlled/telecomts runs. This script now runs TelecomTS at"
echo "--samples $TELECOMTS_SAMPLES and only produces multiseed sweeps, so that"
echo "check does not apply here and is intentionally skipped. If you still"
echo "want to confirm the original 800-sample numbers reproduce exactly, run"
echo "run_benchmark.py / run_telecomts_benchmark.py / verify_results.py"
echo "separately (see README.md)."

echo ""
echo "=== Fixed-gate vs. learnable-gate: Hybrid-DT and MP-Graph, at seed 7/17 ==="
python - <<'PY'
import pandas as pd

pairs = [
    ("Controlled, seed 7", "outputs/controlled_multiseed/seed_7/metrics.csv", "outputs/controlled_multiseed_learnable/seed_7/metrics.csv"),
    ("TelecomTS, seed 17", "outputs/telecomts_multiseed/seed_17/metrics.csv", "outputs/telecomts_multiseed_learnable/seed_17/metrics.csv"),
]
rows = []
for label, fixed_path, learnable_path in pairs:
    fixed = pd.read_csv(fixed_path).set_index("model")
    learnable = pd.read_csv(learnable_path).set_index("model")
    for model in ("mp_graph", "hybrid_dt"):
        rows.append(
            {
                "dataset": label,
                "model": model,
                "fixed_mae": fixed.loc[model, "latency_mae"],
                "learnable_mae": learnable.loc[model, "latency_mae"],
                "fixed_f1": fixed.loc[model, "violation_f1"],
                "learnable_f1": learnable.loc[model, "violation_f1"],
            }
        )
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== Multi-seed (5 seeds): Hybrid-DT, fixed vs. learnable ===")
for label, fixed_path, learnable_path in [
    ("Controlled", "outputs/controlled_multiseed/summary.csv", "outputs/controlled_multiseed_learnable/summary.csv"),
    ("TelecomTS", "outputs/telecomts_multiseed/summary.csv", "outputs/telecomts_multiseed_learnable/summary.csv"),
]:
    fixed_ms = pd.read_csv(fixed_path).set_index("model")
    learnable_ms = pd.read_csv(learnable_path).set_index("model")
    print(f"\n-- {label} --")
    print(
        pd.DataFrame(
            {
                "fixed_mae_mean": [fixed_ms.loc["hybrid_dt", "latency_mae_mean"]],
                "fixed_mae_std": [fixed_ms.loc["hybrid_dt", "latency_mae_std"]],
                "learnable_mae_mean": [learnable_ms.loc["hybrid_dt", "latency_mae_mean"]],
                "learnable_mae_std": [learnable_ms.loc["hybrid_dt", "latency_mae_std"]],
                "fixed_f1_mean": [fixed_ms.loc["hybrid_dt", "violation_f1_mean"]],
                "learnable_f1_mean": [learnable_ms.loc["hybrid_dt", "violation_f1_mean"]],
            },
            index=["hybrid_dt"],
        ).to_string()
    )

print("\n=== Ablations (fixed gate, 5-seed mean): MP-Graph vs. full model ===")
ablation_pairs = [
    ("Controlled", "outputs/controlled_multiseed/summary.csv",
     "single-plane graph", "outputs/controlled_ablation_single_plane/summary.csv",
     "no queue features", "outputs/controlled_ablation_no_queue/summary.csv"),
    ("TelecomTS", "outputs/telecomts_multiseed/summary.csv",
     "single-plane graph", "outputs/telecomts_ablation_single_plane/summary.csv",
     "no queue features", "outputs/telecomts_ablation_no_queue/summary.csv"),
]
for label, full_path, name_a, path_a, name_b, path_b in ablation_pairs:
    full = pd.read_csv(full_path).set_index("model")
    rows = [{"variant": "full (paper)", "latency_mae_mean": full.loc["mp_graph", "latency_mae_mean"], "violation_f1_mean": full.loc["mp_graph", "violation_f1_mean"]}]
    for name, path in [(name_a, path_a), (name_b, path_b)]:
        ablated = pd.read_csv(path).set_index("model")
        rows.append({"variant": name, "latency_mae_mean": ablated.loc["mp_graph", "latency_mae_mean"], "violation_f1_mean": ablated.loc["mp_graph", "violation_f1_mean"]})
    print(f"\n-- {label}, MP-Graph --")
    print(pd.DataFrame(rows).to_string(index=False))
PY