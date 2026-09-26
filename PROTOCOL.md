# Experimental Protocol

## 1. Controlled 5G Core Experiment

The seeded generator creates 1,800 observations over 14 nodes:

- network functions: AMF, SMF, UPF1, UPF2, PCF, NSSF, NRF, and UDM;
- logical slices: eMBB, URLLC, and mMTC;
- UE groups: one per slice.

Each node has 11 features. The graph contains separate normalized control,
user, and slice-dependency planes. Windows contain 12 observations and predict
slice latency and SLA violations three steps ahead. Samples are split
chronologically 65/15/20 with generator seed 7.

The controlled experiment is a mechanism test. Because its generator follows
the graph and queueing assumptions used by MP-Graph, it must not be interpreted
as independent external validation.

## 2. TelecomTS Projection

The source is the `train` split of `AliMaatouk/TelecomTS`. The artifact uses
800 rows obtained as eight evenly spaced pages of 100 rows. Each record has
128 KPI steps:

- steps 0-95 form the model input;
- steps 96-127 form three deterministic QoE-risk latency proxies;
- source anomaly and congestion fields form three binary violation targets.

Radio and traffic KPIs are mapped onto the fixed 14-node 5GC graph by
`benchmark/telecomts.py`. This mapping is fully deterministic and has no
learned parameters. TelecomTS is a 5G observability proxy; it does not contain
native AMF, SMF, or UPF telemetry.

Rows are randomly partitioned 65/15/20. The main split seed is 17. Robustness
uses seeds 7, 11, 17, 23, and 29. The study tests sensitivity to row
partitions, not temporal or site-level generalization.

## 3. Models

- **Persistence** copies the latest slice latency and converts SLA excess to a
  violation score.
- **Ridge-flat** uses the flattened window plus last, mean, and standard
  deviation features.
- **Temporal-MLP** is a one-hidden-layer NumPy model over recent, mean, and
  trend features.
- **MP-Graph** (`mp_graph` in code) applies fixed multi-plane graph
  diffusion, temporal weighting, queueing features, and residual ridge heads.
- **Hybrid-DT** fuses 25% MP-Graph with 75% Ridge-flat for latency, and 15%
  MP-Graph with 85% Temporal-MLP for violation risk.

The graph-model hyperparameter grid is selected on the validation split using:

```text
validation latency MAE - 6 * validation violation F1
```

All values are listed in `configs/paper.json`.

## 4. Metrics

Latency metrics are MAE, RMSE, and MAPE over all records and slices. Violation
scores are thresholded at 0.5 and micro-aggregated over all records and slices
to obtain accuracy, precision, recall, and F1. Multi-seed tables report the
arithmetic mean and sample standard deviation (`ddof=1`).

## 5. Determinism

- dataset snapshot: SHA-256 pinned;
- a single `--seed` drives every stochastic component of a run: the
  controlled generator, the TelecomTS row split, MP-Graph's random
  features, and every model's weight initialization (standalone
  Temporal-MLP and Hybrid-DT's internal MLP alike). Earlier releases let
  these drift independently (fixed at 7/11/17), which silently kept the
  standalone Temporal-MLP baseline's initialization pinned at seed 17 even
  during the multi-seed sweep; this is now fixed;
- multi-seed robustness sweeps vary this one seed across runs
  (`run_multiseed.py`, default seeds 7/11/17/23/29);
- dependencies: exact NumPy and Pandas versions in `requirements.txt`.

Linear algebra libraries may differ in low-order floating-point bits across
platforms. The verification threshold of `5e-4` is stricter than the precision
displayed in the paper.