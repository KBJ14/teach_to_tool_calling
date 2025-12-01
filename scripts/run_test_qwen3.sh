#!/usr/bin/env bash
# Simple runner for the Qwen test script
set -e
PYTHON=python3
SCRIPT_DIR=$(dirname "$0")
$PYTHON "$SCRIPT_DIR/test_qwen3_sample.py" --cache-dir /models/huggingface_cache --dataset-dir "$(pwd)/dataset_experiments/hybrid_ours" --max-tokens 256 "$@"
