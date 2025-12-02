#!/bin/bash

# 1. Python 파일 이름 (이름을 변경했다면 수정하세요)
PYTHON_SCRIPT="compute_first_action_accuracy.py"

# 2. 기본 경로 및 모델 설정 (사용자 환경에 맞게 수정)
BASE_INPUT="/home/bjk/tool_learning/teach_to_tool_calling/experiment_results_fc_summ_plus/openai/gpt-4o_semantic_fc_summ_plus/valid_unseen"
BASE_OUTPUT="/home/bjk/tool_learning/teach_to_tool_calling/scripts/selected_500_gpt4o_semantic_fc_summ_plus_accuracy.jsonl"
ids_FILE="/home/bjk/tool_learning/teach_to_tool_calling/scripts/selected_500_instance_ids.json"
summary_FILE="/home/bjk/tool_learning/teach_to_tool_calling/scripts/all_experiments_accuracy.jsonl"
API_MODEL="gpt-4o"

echo "=== Starting evaluation for Model: $API_MODEL ==="

python3 $PYTHON_SCRIPT \
    --base-dir $BASE_INPUT \
    --ids-file $ids_FILE \
    --out-file $BASE_OUTPUT \
    --summary-file $summary_FILE \
    --experiment-name gpt4o_semantic_fc_summ_plus \
    --debug-missing

