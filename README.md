# Hybrid-DT Reproducibility Artifact

This repository reproduces the experiments for:

> **Hybrid-DT: A Queueing-Aware Multi-Plane Digital Twin for SLA Risk
> Prediction in 5G Core Networks**

It contains the complete NumPy implementation of Hybrid-DT and all reported
baselines, the controlled 5G Core generator, the TelecomTS-to-5GC projection,
the exact open-data snapshot, seeded split logic, metric code, expected
outputs, tests, and an automated result checker.

The artifact is CPU-only and does not require PyTorch or a GPU.

## Double-Blind Review

The metadata intentionally identifies the authors as `Anonymous Authors`.
During double-blind review, do not link the manuscript to a repository hosted
under a personal or institutional account. Upload this directory as
supplementary material or publish it through an anonymity-preserving artifact
service. Replace `Anonymous Authors` in `LICENSE`, `CITATION.cff`, and
`pyproject.toml` only after the review policy permits de-anonymization.

## Quick Start

Python 3.12 is the reference environment. Python 3.11-3.13 is supported.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m unittest discover -s tests -v
```

Run lightweight smoke experiments:

```bash
make smoke
```

## Reproduce the Paper

The tracked 2.5 MB TelecomTS snapshot makes the default run offline.

```bash
./scripts/reproduce_paper.sh
```

Equivalent commands:

```bash
# Controlled 5GC benchmark, Table 4
python -m benchmark.run_benchmark \
  --timesteps 1800 --window 12 --horizon 3 --seed 7 \
  --outdir outputs/controlled_seed7

# TelecomTS main comparison, Table 5
python -m benchmark.run_telecomts_benchmark \
  --samples 800 --input-len 96 --seed 17 \
  --cache data/telecomts_800_even.jsonl.gz \
  --outdir outputs/telecomts_seed17

# Five-split robustness study, Table 6
python -m benchmark.run_multiseed \
  --seeds 7 11 17 23 29 \
  --samples 800 --input-len 96 \
  --cache data/telecomts_800_even.jsonl.gz \
  --outdir outputs/telecomts_multiseed

# Check every reported metric
python -m benchmark.verify_results
```

The checker uses absolute tolerance `5e-4`, which requires all four decimal
places printed in the paper to agree. Raw predictions, metrics, selected
hyperparameters, split sizes, positive rates, and environment metadata are
saved below `outputs/`.

## Expected Headline Results

| Experiment | Model | Latency MAE | Violation F1 |
|---|---|---:|---:|
| Controlled, seed 7 | MP-Graph | 1.1948 | 0.9899 |
| Controlled, seed 7 | Hybrid-DT | 1.2572 | 0.9265 |
| TelecomTS, seed 17 | Hybrid-DT | 0.6963 | 0.9686 |
| TelecomTS, five seeds | Hybrid-DT | 0.7698 +/- 0.0420 | 0.9504 +/- 0.0198 |

These are best-among-implemented-baseline results under the documented
protocol, not a claim of universal state of the art.

## Repository Layout

```text
benchmark/
  dataset.py                  controlled 5GC generator and graph
  telecomts.py                open-data snapshot, projection, checksum
  models.py                   baselines, MP-Graph, and Hybrid-DT
  metrics.py                  regression and violation metrics
  experiments.py              shared train/evaluate/artifact pipeline
  run_benchmark.py            controlled experiment CLI
  run_telecomts_benchmark.py  main TelecomTS CLI
  run_multiseed.py            five-seed robustness CLI
  verify_results.py           paper-output checker
artifacts/expected/            immutable CSV values reported in the paper
configs/paper.json             complete paper protocol and hyperparameters
data/                          exact compressed TelecomTS subset and manifest
tests/                         deterministic unit and data-integrity tests
```

The code names this model **MP-Graph** directly (`mp_graph`), matching how
it is described in the paper: a transparent, fixed multi-plane diffusion +
ridge model, not an end-to-end trained GNN. `hybrid_dt` combines MP-Graph
with the Ridge-flat latency head and Temporal-MLP violation head. Both the
controlled (synthetic) and TelecomTS benchmarks now evaluate Hybrid-DT
alongside MP-Graph.

## Data Provenance

[TelecomTS](https://huggingface.co/datasets/AliMaatouk/TelecomTS) is an
MIT-licensed observability dataset derived from a 5G testbed. The source has
32,000 records of 128 KPI steps. This repository tracks the exact 800 records
used in the paper after removing unused text, descriptions, statistics, and
Q&A fields:

```text
SHA-256 f410765a0cdd4e4802967de4304e5d138455524389e83e132e682e71f9c509b6
```

See `data/manifest.json` and `PROTOCOL.md`. To rebuild the cache from the
upstream datasets server when it is absent, add `--download`; the runner will
reject content that does not match the paper checksum.

## Docker

```bash
docker build -t hybrid-dt-artifact .
docker run --rm hybrid-dt-artifact
docker run --rm -v "$PWD/outputs:/artifact/outputs" \
  hybrid-dt-artifact ./scripts/reproduce_paper.sh
```

## License and Citation

The implementation is released under the MIT license. The included TelecomTS
subset retains the source dataset's MIT terms. Citation metadata remains
anonymous for review and should be updated in the camera-ready repository.