#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/bjk/tool_learning/teach_to_tool_calling"
EPISODE_ROOT="$REPO_ROOT/episode_data_wo_state"
OUTPUT_ROOT="$REPO_ROOT/processed_dataset"

cd "$REPO_ROOT"

# 필요하면 여기서 conda/venv 활성화
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate teach_env

python build_dataset.py \
  --episode-root "$EPISODE_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  "$@"

# ./run_build_dataset.sh --task-ids 0


