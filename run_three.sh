#!/usr/bin/env bash
# Train the three study datasets. Override DEVICE, SEEDS, or DATA_DIR if needed.
set -euo pipefail

DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-0}"
DATA_DIR="${DATA_DIR:-data}"

python run_experiment.py --dataset synthetic --epochs 15 --dim 32 --layers 2 --device cpu
python run_experiment.py --dataset kuairec --download --data-dir "$DATA_DIR" --device "$DEVICE" --seeds "$SEEDS"
python run_experiment.py --dataset kuairand --download --data-dir "$DATA_DIR" --device "$DEVICE" --seeds "$SEEDS"
python run_experiment.py --dataset movielens --download --data-dir "$DATA_DIR" --device "$DEVICE" --seeds "$SEEDS"
