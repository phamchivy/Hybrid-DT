from __future__ import annotations

import argparse
from pathlib import Path

from benchmark.experiments import run_controlled_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Controlled 5G Core Digital Twin benchmark",
    )
    parser.add_argument("--timesteps", type=int, default=1800)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs/controlled_seed7"),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Cap the generated trace at 650 steps for a smoke test.",
    )
    parser.add_argument(
        "--learnable-gate",
        action="store_true",
        help="Fit the fusion gates (MP-Graph internal, Hybrid-DT latency/violation) "
        "from training data instead of using the paper's fixed weights.",
    )
    parser.add_argument(
        "--no-queue-features",
        dest="use_queue_features",
        action="store_false",
        help="Ablation: drop MP-Graph's queueing-theory utilization/delay "
        "features entirely (default: included, matching the paper).",
    )
    parser.add_argument(
        "--single-plane-graph",
        action="store_true",
        help="Ablation: collapse the 3 control/user/slice graph planes into "
        "MP-Graph's single 'all' plane (default: multi-plane, matching the paper).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = run_controlled_experiment(
        outdir=args.outdir,
        timesteps=args.timesteps,
        window=args.window,
        horizon=args.horizon,
        seed=args.seed,
        quick=args.quick,
        learnable_gate=args.learnable_gate,
        use_queue_features=args.use_queue_features,
        single_plane_graph=args.single_plane_graph,
    )
    print(frame.to_string(index=False))
    print(f"\nArtifacts: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()