#!/usr/bin/env bash
# Usage: ./run_passk.sh /path/to/accuracy/dir experiment_name [/path/to/out.jsonl]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_INPUT_DIR="$SCRIPT_DIR/accuracy/gpt-4o_semantic_edh"
DEFAULT_NAME="gpt-4o_semantic_edh"

if [ "$#" -eq 0 ]; then
  INPUT_DIR="$DEFAULT_INPUT_DIR"
  NAME="$DEFAULT_NAME"
  OUT=""
else
  INPUT_DIR="$1"
  NAME="${2:-}"
  OUT="${3:-}"
fi
PYTHON=/bin/python3
if [ -z "$NAME" ]; then
  # let calculate_passk infer name from dir
  if [ -z "$OUT" ]; then
    $PYTHON "$SCRIPT_DIR/calculate_passk.py" --exp_dir "$INPUT_DIR"
  else
    $PYTHON "$SCRIPT_DIR/calculate_passk.py" --exp_dir "$INPUT_DIR" --out "$OUT"
  fi
else
  if [ -z "$OUT" ]; then
    $PYTHON "$SCRIPT_DIR/calculate_passk.py" --exp_dir "$INPUT_DIR" --name "$NAME"
  else
    $PYTHON "$SCRIPT_DIR/calculate_passk.py" --exp_dir "$INPUT_DIR" --name "$NAME" --out "$OUT"
  fi
fi

exit 0
