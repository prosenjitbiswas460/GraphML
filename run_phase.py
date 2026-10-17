#!/usr/bin/env python3
"""Sweep the exposure policy and write the phase diagram."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from esb.experiment import _jsonify, resolve_device
from esb.phase import run_phase
from run_experiment import parser


def main() -> None:
    cli = parser()
    cli.add_argument("--alphas", default="0,0.25,0.5,1", help="Exposure skews to sweep.")
    cli.add_argument(
        "--aligns",
        default="0,0.5,1",
        help="Synthetic only. Fractions of the exposure head taken from one preference community.",
    )
    cli.add_argument("--layer-grid", default="1,3", help="LightGCN depths to train.")
    # The single-run default floor clips every alpha at or above 0.5 on the
    # synthetic head/tail scores, so the sweep would repeat one mask.
    cli.set_defaults(min_prop=0.01)
    cfg = cli.parse_args()
    cfg.ks = tuple(int(part) for part in str(cfg.ks).split(",") if part)
    if not cfg.ks:
        raise SystemExit("--ks needs at least one cutoff.")
    if not 0 <= cfg.exposure_align <= 1:
        raise SystemExit("--exposure-align must lie in [0, 1]. The sweep uses --aligns.")
    alphas = _floats(cfg.alphas)
    aligns = _floats(cfg.aligns)
    layers = _ints(cfg.layer_grid)
    seeds = _ints(cfg.seeds)
    if cfg.dataset == "synthetic":
        _warn_if_alphas_share_a_mask(alphas, cfg.min_prop)
    if cfg.dataset != "synthetic":
        aligns = [0.0]
        print(
            "Exposure alignment is part of the synthetic generator. "
            "This sweep varies alpha and depth.",
            flush=True,
        )
    if cfg.dataset == "kuairand":
        print(
            "On KuaiRand, alpha changes the propensity estimate, not the logged edges. "
            "The random-exposure test is already the policy shift.",
            flush=True,
        )
    device = resolve_device(cfg.device)
    summary = run_phase(cfg, device, alphas, aligns, layers, seeds)
    path = Path(cfg.results_dir) / "phase_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonify(summary), indent=2) + "\n")
    print(f"\nWrote {path}", flush=True)
    _print_summary(summary)


def _warn_if_alphas_share_a_mask(alphas, min_prop: float) -> None:
    """Synthetic tail score is 1 and the head score is 100."""
    tail = [float(np.clip(0.01 ** alpha, min_prop, 1.0)) for alpha in alphas]
    distinct = {round(value, 6) for value in tail}
    shown = ", ".join(f"alpha {alpha:g}→p {prob:.4f}" for alpha, prob in zip(alphas, tail))
    print(f"Synthetic tail propensity at --min-prop {min_prop:g}: {shown}", flush=True)
    if len(distinct) < len(alphas):
        print(
            "Some alphas hit the propensity floor and build the same mask. "
            "Lower --min-prop, or keep alpha at or below 1.",
            flush=True,
        )


def _floats(text: str) -> list[float]:
    values = [float(part) for part in str(text).split(",") if part]
    if not values:
        raise SystemExit("The sweep needs at least one value.")
    return values


def _ints(text: str) -> list[int]:
    values = [int(part) for part in str(text).split(",") if part]
    if not values:
        raise SystemExit("The sweep needs at least one value.")
    return values


def _print_summary(summary: dict) -> None:
    print("\nFrontier: smallest alpha where raw leakage exceeds the density-matched graph", flush=True)
    for row in summary["frontier"]:
        print(
            f"  align={row['exposure_align']:g}  layers={row['layers']}  "
            f"alpha={row['alpha_when_neighborhood_flips']}",
            flush=True,
        )
    print("Untrained propagation, support gap versus weight gap from 1 to 3 layers", flush=True)
    seen = set()
    for row in summary["propagation"]:
        key = (row["alpha"], row["exposure_align"])
        if key in seen:
            continue
        seen.add(key)
        print(
            f"  alpha={row['alpha']:g} align={row['exposure_align']:g}  "
            f"support {row['support_gap_L1']} → {row['support_gap_L3']}  "
            f"ips {row['ips_gap_L1']} → {row['ips_gap_L3']}  "
            f"support_grew={row['support_gap_grew']}  "
            f"ips_grew_less={row['ips_gap_grew_less_than_support']}",
            flush=True,
        )
    selector = summary["selector"]
    print(
        "Shift selector: "
        f"mean Spearman val→shift {selector['mean_spearman_val_vs_shift']}  "
        f"leakage→shift {selector['mean_spearman_leakage_vs_shift']}  "
        f"leakage wins {selector['n_cells_leakage_ranks_shift']}"
        f"/{selector['n_cells_compared']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
