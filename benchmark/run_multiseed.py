from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from benchmark.experiments import (
    METRIC_COLUMNS,
    PAPER_MODEL_NAMES,
    run_telecomts_experiment,
)


PAPER_SEEDS = (7, 11, 17, 23, 29)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and aggregate the five TelecomTS splits in the paper",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(PAPER_SEEDS),
    )
    parser.add_argument("--samples", type=int, default=800)
    parser.add_argument("--input-len", type=int, default=96)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("data/telecomts_800_even.jsonl.gz"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs/telecomts_multiseed"),
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--skip-checksum", action="store_true")
    parser.add_argument(
        "--learnable-gate",
        action="store_true",
        help="Fit the fusion gates (MP-Graph internal, Hybrid-DT latency/violation) "
        "from training data instead of using the paper's fixed weights.",
    )
    return parser.parse_args()


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    summary = (
        raw.groupby("model", sort=False)[METRIC_COLUMNS]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "model"
        if column[0] == "model"
        else f"{column[0]}_{column[1]}"
        for column in summary.columns
    ]
    summary.insert(
        1,
        "paper_name",
        summary["model"].map(PAPER_MODEL_NAMES),
    )
    return summary


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    frames = []
    for seed in args.seeds:
        seed_dir = args.outdir / f"seed_{seed}"
        frame = run_telecomts_experiment(
            outdir=seed_dir,
            cache_path=args.cache,
            samples=args.samples,
            input_len=args.input_len,
            seed=seed,
            offline=not args.download,
            verify_snapshot=not args.skip_checksum,
            learnable_gate=args.learnable_gate,
        )
        frame["seed"] = seed
        frames.append(frame)
        print(
            f"seed={seed}: "
            f"Hybrid-DT MAE="
            f"{frame.loc[frame.model == 'hybrid_dt', 'latency_mae'].iloc[0]:.4f}, "
            f"F1="
            f"{frame.loc[frame.model == 'hybrid_dt', 'violation_f1'].iloc[0]:.4f}"
        )

    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(args.outdir / "all_seeds.csv", index=False)
    summary = summarize(raw)
    summary.to_csv(args.outdir / "summary.csv", index=False)
    (args.outdir / "run.json").write_text(
        json.dumps(
            {
                "seeds": args.seeds,
                "samples": args.samples,
                "input_len": args.input_len,
                "std": "sample standard deviation (pandas ddof=1)",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n" + summary.to_string(index=False))
    print(f"\nArtifacts: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()