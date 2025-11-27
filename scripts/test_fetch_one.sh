#!/usr/bin/env bash
set -euo pipefail

# Wrapper script for running teach_to_tool_calling/test_fetch_one.py
# usage: test_fetch_one.sh --episode EPJSON --images_dir IMGD --idx N [--edh_idx M]

print_usage() {
  echo "Usage: $0 --episode <episode.json> --images_dir <images_root> [--idx <turn_idx>] [--edh_idx <edh_idx>]"
  echo "Example: $0 --episode ../episode_data_wo_state/task_0/episode_0.json --images_dir /teach_dataset/images --idx 0"
}

EPISODE=""
IMAGES_DIR=""
IDX=0
EDH_IDX=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --episode)
      EPISODE="$2"; shift 2;;
    --images_dir)
      IMAGES_DIR="$2"; shift 2;;
    --idx)
      IDX="$2"; shift 2;;
    --edh_idx)
      EDH_IDX="$2"; shift 2;;
    -h|--help)
      print_usage; exit 0;;
    *)
      echo "Unknown argument: $1"; print_usage; exit 1;;
  esac
done

if [[ -z "$EPISODE" || -z "$IMAGES_DIR" ]]; then
  echo "Missing required arguments."; print_usage; exit 1
fi

# Determine repository root (parent of the directory this script sits in)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Run the Python CLI
python3 "$REPO_ROOT/test_fetch_one.py" \
  --episode "$EPISODE" \
  --images_dir "$IMAGES_DIR" \
  --idx "$IDX" \
  --edh_idx "$EDH_IDX"
