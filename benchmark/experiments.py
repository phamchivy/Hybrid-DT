from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from benchmark.dataset import generate_synthetic_5gc, train_val_test_split
from benchmark.metrics import combined_metrics
from benchmark.models import (
    BaseModel,
    HybridDigitalTwin,
    PersistenceBaseline,
    RidgeBaseline,
    TemporalMLP,
    tune_mp_graph,
)
from benchmark.telecomts import PAPER_CACHE_SHA256, load_telecomts_bundle


METRIC_COLUMNS = [
    "latency_mae",
    "latency_rmse",
    "latency_mape",
    "violation_accuracy",
    "violation_precision",
    "violation_recall",
    "violation_f1",
]

PAPER_MODEL_NAMES = {
    "hybrid_dt": "Hybrid-DT",
    "mp_graph": "MP-Graph",
    "persistence": "Persistence",
    "ridge_flat": "Ridge-flat",
    "temporal_mlp": "Temporal-MLP",
}


def random_split(
    n: int,
    seed: int = 17,
    train: float = 0.65,
    val: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_train = int(n * train)
    n_val = int(n * val)
    return (
        order[:n_train],
        order[n_train : n_train + n_val],
        order[n_train + n_val :],
    )


def evaluate(
    model: BaseModel,
    x: np.ndarray,
    y_latency: np.ndarray,
    y_violation: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    pred = model.predict(x)
    metrics = combined_metrics(
        y_latency,
        pred.latency,
        y_violation,
        pred.violation,
    )
    row = {
        "model": model.name,
        "paper_name": PAPER_MODEL_NAMES.get(model.name, model.name),
        **metrics,
    }
    arrays = {
        f"{model.name}__latency": pred.latency,
        f"{model.name}__violation": pred.violation,
    }
    return row, arrays


def environment_metadata() -> dict:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "byteorder": sys.byteorder,
    }


def save_experiment(
    rows: Iterable[dict],
    predictions: dict[str, np.ndarray],
    metadata: dict,
    outdir: Path,
) -> pd.DataFrame:
    outdir.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(
        ["latency_mae", "violation_f1"],
        ascending=[True, False],
    ).reset_index(drop=True)
    frame.to_csv(outdir / "metrics.csv", index=False)
    frame.to_csv(outdir / "benchmark_results.csv", index=False)
    (outdir / "metrics.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (outdir / "metadata.json").write_text(
        json.dumps(
            {**metadata, "environment": environment_metadata()},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    np.savez_compressed(outdir / "predictions.npz", **predictions)
    return frame


def run_controlled_experiment(
    outdir: Path,
    timesteps: int = 1800,
    window: int = 12,
    horizon: int = 3,
    seed: int = 7,
    quick: bool = False,
    learnable_gate: bool = False,
) -> pd.DataFrame:
    if quick:
        timesteps = min(timesteps, 650)
    bundle = generate_synthetic_5gc(
        timesteps=timesteps,
        window=window,
        horizon=horizon,
        seed=seed,
    )
    train_sl, val_sl, test_sl = train_val_test_split(len(bundle.x))
    x_train, x_val, x_test = (
        bundle.x[train_sl],
        bundle.x[val_sl],
        bundle.x[test_sl],
    )
    yl_train, yl_val, yl_test = (
        bundle.y_latency[train_sl],
        bundle.y_latency[val_sl],
        bundle.y_latency[test_sl],
    )
    yv_train, yv_val, yv_test = (
        bundle.y_violation[train_sl],
        bundle.y_violation[val_sl],
        bundle.y_violation[test_sl],
    )

    models: list[BaseModel] = [
        PersistenceBaseline().fit(x_train, yl_train, yv_train),
        RidgeBaseline(alpha=25.0).fit(x_train, yl_train, yv_train),
        TemporalMLP(
            hidden=128,
            lr=0.016,
            epochs=180,
            seed=seed,
        ).fit(x_train, yl_train, yv_train),
    ]
    graph_model, validation = tune_mp_graph(
        graph=bundle.graph,
        x_train=x_train,
        y_lat_train=yl_train,
        y_vio_train=yv_train,
        x_val=x_val,
        y_lat_val=yl_val,
        y_vio_val=yv_val,
        metric_fn=combined_metrics,
        seed=seed,
        learnable_gate=learnable_gate,
    )
    models.append(graph_model)
    # Reuse the exact tuned graph_model as Hybrid-DT's graph_head instead of
    # building a second MPGraph with untuned hyperparameters: otherwise
    # Hybrid-DT's graph component is never validated against the same grid
    # the standalone MP-Graph row gets, an unfair comparison.
    hybrid_model = HybridDigitalTwin(
        graph_head=graph_model,
        ridge_alpha=25.0,
        seed=seed,
        learnable_gate=learnable_gate,
    ).fit(x_train, yl_train, yv_train, x_val=x_val, y_latency_val=yl_val, y_violation_val=yv_val)
    models.append(hybrid_model)

    rows: list[dict] = []
    predictions: dict[str, np.ndarray] = {
        "target__latency": yl_test,
        "target__violation": yv_test,
    }
    for model in models:
        row, model_predictions = evaluate(
            model,
            x_test,
            yl_test,
            yv_test,
        )
        rows.append(row)
        predictions.update(model_predictions)

    metadata = {
        **bundle.metadata,
        "seed": seed,
        "experiment": "controlled_5gc",
        "n_samples": int(len(bundle.x)),
        "split": {
            "type": "chronological",
            "train": int(len(x_train)),
            "validation": int(len(x_val)),
            "test": int(len(x_test)),
        },
        "graph_model_best_validation": validation,
        "quick": quick,
        "learnable_gate": learnable_gate,
        "fitted_gates": {
            # mp_graph_gate_vio is shared: Hybrid-DT reuses the exact same
            # tuned graph_model as its graph_head (see comment above), so
            # there is no separate "internal" gate value to report.
            "mp_graph_gate_vio": graph_model.gate_vio,
            "hybrid_dt_gate_latency": hybrid_model.gate_latency,
            "hybrid_dt_gate_violation": hybrid_model.gate_violation,
        },
    }
    return save_experiment(rows, predictions, metadata, outdir)


def run_telecomts_experiment(
    outdir: Path,
    cache_path: Path,
    samples: int = 800,
    input_len: int = 96,
    sampling: str = "even",
    seed: int = 17,
    offline: bool = True,
    verify_snapshot: bool = True,
    learnable_gate: bool = False,
) -> pd.DataFrame:
    expected_hash = (
        PAPER_CACHE_SHA256
        if verify_snapshot and sampling == "even" and samples <= 800
        else None
    )
    bundle = load_telecomts_bundle(
        cache_path=cache_path,
        samples=samples,
        input_len=input_len,
        sampling=sampling,
        allow_download=not offline,
        expected_sha256=expected_hash,
    )
    train_ix, val_ix, test_ix = random_split(len(bundle.x), seed=seed)
    x_train, x_val, x_test = (
        bundle.x[train_ix],
        bundle.x[val_ix],
        bundle.x[test_ix],
    )
    yl_train, yl_val, yl_test = (
        bundle.y_latency[train_ix],
        bundle.y_latency[val_ix],
        bundle.y_latency[test_ix],
    )
    yv_train, yv_val, yv_test = (
        bundle.y_violation[train_ix],
        bundle.y_violation[val_ix],
        bundle.y_violation[test_ix],
    )

    models: list[BaseModel] = [
        PersistenceBaseline().fit(x_train, yl_train, yv_train),
        RidgeBaseline(alpha=35.0).fit(x_train, yl_train, yv_train),
        # Unified seed policy: every stochastic component of every model in
        # a run shares the same single seed (the run's --seed), rather than
        # each sub-model picking its own fixed value. Multi-seed robustness
        # sweeps still vary this one seed across runs (see run_multiseed.py).
        TemporalMLP(
            hidden=96,
            lr=0.012,
            epochs=160,
            seed=seed,
        ).fit(x_train, yl_train, yv_train),
    ]
    graph_model, validation = tune_mp_graph(
        graph=bundle.graph,
        x_train=x_train,
        y_lat_train=yl_train,
        y_vio_train=yv_train,
        x_val=x_val,
        y_lat_val=yl_val,
        y_vio_val=yv_val,
        metric_fn=combined_metrics,
        seed=seed,
        learnable_gate=learnable_gate,
    )
    # Reuse the exact tuned graph_model as Hybrid-DT's graph_head instead of
    # building a second MPGraph with untuned hyperparameters (see comment in
    # run_controlled_experiment for why).
    hybrid_model = HybridDigitalTwin(
        graph_head=graph_model,
        ridge_alpha=35.0,
        seed=seed,
        learnable_gate=learnable_gate,
    ).fit(x_train, yl_train, yv_train, x_val=x_val, y_latency_val=yl_val, y_violation_val=yv_val)
    models.extend([graph_model, hybrid_model])

    rows: list[dict] = []
    predictions: dict[str, np.ndarray] = {
        "target__latency": yl_test,
        "target__violation": yv_test,
        "test_indices": test_ix,
    }
    for model in models:
        row, model_predictions = evaluate(
            model,
            x_test,
            yl_test,
            yv_test,
        )
        rows.append(row)
        predictions.update(model_predictions)

    metadata = {
        **bundle.metadata,
        "experiment": "telecomts_projection",
        "n_samples": int(len(bundle.x)),
        "split": {
            "type": "seeded_random_rows",
            "seed": seed,
            "train": int(len(x_train)),
            "validation": int(len(x_val)),
            "test": int(len(x_test)),
        },
        "positive_rate": {
            "train": float(yv_train.mean()),
            "validation": float(yv_val.mean()),
            "test": float(yv_test.mean()),
        },
        "graph_model_best_validation": validation,
        "seed": seed,
        "seed_policy": "single unified seed drives data split, MP-Graph "
        "random features, and every model's weight initialization",
        "learnable_gate": learnable_gate,
        "fitted_gates": {
            # mp_graph_gate_vio is shared: Hybrid-DT reuses the exact same
            # tuned graph_model as its graph_head (see comment above), so
            # there is no separate "internal" gate value to report.
            "mp_graph_gate_vio": graph_model.gate_vio,
            "hybrid_dt_gate_latency": hybrid_model.gate_latency,
            "hybrid_dt_gate_violation": hybrid_model.gate_violation,
        },
    }
    return save_experiment(rows, predictions, metadata, outdir)