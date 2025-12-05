#!/bin/bash

# 1. Python 파일 이름 (이름을 변경했다면 수정하세요)
PYTHON_SCRIPT="batch_inference.py"

# 2. 기본 경로 및 모델 설정 (사용자 환경에 맞게 수정)
BASE_INPUT="/home/bjk/tool_learning/teach_to_tool_calling/dataset_experiments_edh"
BASE_OUTPUT="/home/bjk/tool_learning/teach_to_tool_calling/experiment_results_edh"
API_MODEL="openai/gpt-4o"

echo "=== Starting API Inference for Model: $API_MODEL ==="

# Helper: create next trial dir under a parent path
get_trial_dir() {
    # usage: get_trial_dir <parent> <exp_subdir>
    parent="$1"
    exp_subdir="$2"
    mkdir -p "${parent}"
    trial=0
    while true; do
        trial_dir="${parent}/trial_${trial}"
        if [ ! -d "${trial_dir}" ]; then
            # trial dir doesn't exist yet -> reuse (new slot)
            echo "${trial_dir}"
            return
        fi
        # If experiment subdir doesn't exist in this trial directory, reuse it
        if [ -n "${exp_subdir}" ] && [ ! -d "${trial_dir}/${exp_subdir}" ]; then
            echo "${trial_dir}"
            return
        fi
        trial=$((trial+1))
    done
}


# # 3. Hybrid EDH 
# echo "Running Hybrid EDH..."
# # Choose trial folder under top-level BASE_OUTPUT (e.g., experiment_results_edh/trial_0)
# TRIAL_DIR=$(get_trial_dir "${BASE_OUTPUT}" "${API_MODEL}_hybrid_edh")
# mkdir -p "${TRIAL_DIR}"
# BASE_OUT_DIR="${TRIAL_DIR}/${API_MODEL}_hybrid_edh/valid_unseen"
# mkdir -p "${BASE_OUT_DIR}"
# echo "Writing results to: ${BASE_OUT_DIR}"
# python3 $PYTHON_SCRIPT \
#     --model-name $API_MODEL \
#     --input-dir "${BASE_INPUT}/hybrid_edh/valid_unseen" \
#     --output-dir "${BASE_OUT_DIR}" \
#     --ids-file selected_500_instance_ids.json


# # 4. Semantic EDH
# echo "Running Semantic EDH..."
# TRIAL_DIR=$(get_trial_dir "${BASE_OUTPUT}" "${API_MODEL}_semantic_edh")
# mkdir -p "${TRIAL_DIR}"
# BASE_OUT_DIR="${TRIAL_DIR}/${API_MODEL}_semantic_edh/valid_unseen"
# mkdir -p "${BASE_OUT_DIR}"
# echo "Writing results to: ${BASE_OUT_DIR}"
# python3 $PYTHON_SCRIPT \
#     --model-name $API_MODEL \
#     --input-dir "${BASE_INPUT}/semantic_edh/valid_unseen" \
#     --output-dir "${BASE_OUT_DIR}" \
#     --ids-file selected_500_instance_ids.json

# 5. Spatial EDH
echo "Running Spatial EDH..."
TRIAL_DIR=$(get_trial_dir "${BASE_OUTPUT}" "${API_MODEL}_spatial_edh")
mkdir -p "${TRIAL_DIR}"
BASE_OUT_DIR="${TRIAL_DIR}/${API_MODEL}_spatial_edh/valid_unseen"
mkdir -p "${BASE_OUT_DIR}"
echo "Writing results to: ${BASE_OUT_DIR}"
python3 $PYTHON_SCRIPT \
    --model-name $API_MODEL \
    --input-dir "${BASE_INPUT}/spatial_edh/valid_unseen" \
    --output-dir "${BASE_OUT_DIR}" \
    --ids-file selected_500_instance_ids.json
echo "All experiments queued. Check output directories for results."

