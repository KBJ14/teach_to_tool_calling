#!/usr/bin/env bash
# Convenience wrapper - uses default example paths from the dataset README
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EP="${REPO_ROOT}/episode_data_wo_state/task_0/episode_0.json"
IMAGES_DIR="/teach_dataset/images"
IDX=0

# You can override defaults by providing parameters: ./run_test_fetch_one_default.sh /path/to/episode.json /path/to/images_dir idx
if [[ $# -ge 1 ]]; then
  EP="$1"
fi
if [[ $# -ge 2 ]]; then
  IMAGES_DIR="$2"
fi
if [[ $# -ge 3 ]]; then
  IDX="$3"
fi

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

$REPO_ROOT/scripts/test_fetch_one.sh --episode "$EP" --images_dir "$IMAGES_DIR" --idx "$IDX"
