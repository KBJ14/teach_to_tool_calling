#!/bin/bash
set -euo pipefail

# 1. Python 파일 이름 (이름을 변경했다면 수정하세요)
PYTHON_SCRIPT="compute_first_action_accuracy.py"

# 2. 기본 경로 및 모델 설정 (사용자 환경에 맞게 수정)
EXPERIMENT_RESULTS_DIR="/home/bjk/tool_learning/teach_to_tool_calling/experiment_results_edh"
OUTPUT_BASE_DIR="/home/bjk/tool_learning/teach_to_tool_calling/scripts/accuracy"
EXPERIMENT_NAME="gpt-4o_semantic_edh"
IDS_FILE="/home/bjk/tool_learning/teach_to_tool_calling/scripts/selected_500_instance_ids.json"
SUMMARY_FILE="/home/bjk/tool_learning/teach_to_tool_calling/scripts/all_experiments_accuracy.jsonl"
API_MODEL="openai/gpt-4o"

echo "=== Starting evaluation for Model: $API_MODEL ==="

# Iterate trial numbers 0..9
for TRIAL_NUMBER in {0..9}; do
    BASE_INPUT="$EXPERIMENT_RESULTS_DIR/trial_${TRIAL_NUMBER}/openai/${EXPERIMENT_NAME}/valid_unseen"
    OUTPUT_DIR_TRIAL="$OUTPUT_BASE_DIR/${EXPERIMENT_NAME}/trial${TRIAL_NUMBER}"
    mkdir -p "$OUTPUT_DIR_TRIAL"
    BASE_OUTPUT="$OUTPUT_DIR_TRIAL/selected_500_gpt-4o_semantic_edh_accuracy.jsonl"

    echo "--- Running trial $TRIAL_NUMBER ---"

    if [ ! -d "$BASE_INPUT" ]; then
        echo "Warning: base input directory does not exist: $BASE_INPUT -- skipping trial $TRIAL_NUMBER"
        continue
    fi

    python3 "$PYTHON_SCRIPT" \
        --base-dir "$BASE_INPUT" \
        --ids-file "$IDS_FILE" \
        --out-file "$BASE_OUTPUT" \
        --summary-file "$SUMMARY_FILE" \
        --experiment-name "$EXPERIMENT_NAME" \
        --trial-number "$TRIAL_NUMBER" \
        --debug-missing
done

echo "=== All trials completed ==="

