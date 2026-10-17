#!/usr/bin/env python3
"""Compare raw, deconfounded, and counterfactual graphs for one recommender."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from esb.experiment import resolve_device, result_stem, run_one, summarize_runs


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(
        description=(
            "Test whether exposure-biased edges make a LightGCN learn the exposure "
            "neighborhood instead of the preference neighborhood."
        )
    )
    cli.add_argument(
        "--dataset",
        required=True,
        choices=("synthetic", "movielens", "kuairec", "kuairec_natural", "kuairand"),
    )
    cli.add_argument("--data-dir", default="data")
    cli.add_argument("--results-dir", default="results")
    cli.add_argument("--download", action="store_true")
    cli.add_argument("--dry-run", action="store_true")
    cli.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    cli.add_argument("--seeds", default="0", help="Comma-separated seeds, for example 0,1,2.")
    cli.add_argument("--epochs", type=int, default=80)
    cli.add_argument("--patience", type=int, default=10)
    cli.add_argument("--eval-every", type=int, default=1)
    cli.add_argument("--dim", type=int, default=64)
    cli.add_argument("--layers", type=int, default=3)
    cli.add_argument("--lr", type=float, default=1e-3)
    cli.add_argument("--l2", type=float, default=1e-4)
    cli.add_argument("--batch-size", type=int, default=2048)
    cli.add_argument("--ks", default="10,20", help="Ranking cutoffs, comma-separated.")
    cli.add_argument("--neigh-k", type=int, default=10)
    cli.add_argument("--min-degree", type=int, default=5)
    cli.add_argument("--policy", choices=("popularity", "uniform"), default="popularity")
    cli.add_argument("--alpha", type=float, default=1.0)
    cli.add_argument(
        "--exposure-align",
        type=float,
        default=0.0,
        help="Synthetic only. Fraction of the exposure head taken from preference community 0.",
    )
    cli.add_argument(
        "--shift-keep",
        type=float,
        default=0.5,
        help="Fraction of held-out preferences kept for the uniform shift test.",
    )
    cli.add_argument("--min-prop", type=float, default=0.1)
    cli.add_argument("--keep-prob", type=float, default=0.5)
    cli.add_argument("--propensity", choices=("auto", "known", "estimated"), default="auto")
    cli.add_argument("--ips-clip", type=float, default=10.0)
    cli.add_argument("--resample-keep", type=float, default=0.5)
    cli.add_argument("--no-resampled", action="store_false", dest="include_resampled")
    cli.add_argument("--include-mf", action="store_true", help="Also train a 0-layer model on the raw edges.")
    cli.add_argument("--watch-threshold", type=float, default=2.0)
    cli.add_argument("--rating-threshold", type=float, default=4.0)
    cli.add_argument("--feedback", choices=("is_click", "long_view", "is_like"), default="is_click")
    cli.add_argument("--val-ratio", type=float, default=0.1)
    cli.add_argument("--test-ratio", type=float, default=0.2)
    cli.add_argument("--min-train", type=int, default=2)
    cli.add_argument("--min-user", type=int, default=5)
    cli.add_argument("--min-item", type=int, default=5)
    cli.set_defaults(include_resampled=True)
    return cli


def main() -> None:
    cfg = parser().parse_args()
    cfg.ks = tuple(int(part) for part in str(cfg.ks).split(",") if part)
    if not cfg.ks:
        raise SystemExit("--ks needs at least one cutoff.")
    seeds = [int(part) for part in str(cfg.seeds).split(",") if part]
    device = resolve_device(cfg.device)
    runs = [run_one(cfg, seed, device) for seed in seeds]
    if len(runs) > 1:
        summary = summarize_runs(runs)
        path = Path(cfg.results_dir) / f"{result_stem(cfg, seeds[0]).rsplit('_seed', 1)[0]}_summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Wrote {path}", flush=True)


if __name__ == "__main__":
    main()
