from __future__ import annotations

import argparse
from pathlib import Path

from benchmark.experiments import run_telecomts_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid-DT benchmark on the TelecomTS projection",
    )
    parser.add_argument("--samples", type=int, default=800)
    parser.add_argument("--input-len", type=int, default=96)
    parser.add_argument(
        "--sampling",
        choices=["even", "sequential"],
        default="even",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("data/telecomts_800_even.jsonl.gz"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs/telecomts_seed17"),
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Fetch from Hugging Face when the tracked snapshot is absent.",
    )
    parser.add_argument(
        "--skip-checksum",
        action="store_true",
        help="Allow a cache other than the paper's exact 800-row snapshot.",
    )

    parser.add_argument("--transport", choices=["rows_api", "datasets"], default="rows_api")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = run_telecomts_experiment(
        outdir=args.outdir,
        cache_path=args.cache,
        samples=args.samples,
        input_len=args.input_len,
        sampling=args.sampling,
        seed=args.seed,
        offline=not args.download,
        verify_snapshot=not args.skip_checksum,
        transport=args.transport,
    )
    print(frame.to_string(index=False))
    print(f"\nArtifacts: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
