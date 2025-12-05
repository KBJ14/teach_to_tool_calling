#!/bin/bash

# Script Name: run_inference_fc.sh
# Description: Runs batch inference using OpenAI Function Calling.
#              Uses 'dataset_experiments_fc' (prompts without tool text)
#              and passes 'tools.json' via --tools-file.

# 1. 설정 (Configuration)
PYTHON_SCRIPT="batch_inference_fc.py"
API_MODEL="openai/gpt-4o"

# [중요] FC 전용 데이터셋 경로 (build_dataset_fc.py로 생성한 곳)
BASE_INPUT="/home/bjk/tool_learning/teach_to_tool_calling/dataset_experiments_fc_summ_plus"

# 결과 저장 경로 (기존 결과와 섞이지 않게 _fc 접미사 권장)
BASE_OUTPUT="/home/bjk/tool_learning/teach_to_tool_calling/experiment_results_fc_summ_plus"

# 리소스 파일 경로
TOOLS_FILE="/home/bjk/tool_learning/teach_to_tool_calling/dataset/prompts/tools.json"
IDS_FILE="/home/bjk/tool_learning/teach_to_tool_calling/scripts/selected_500_instance_ids.json"


echo "=========================================================="
echo " [Start] Function Calling Inference: $API_MODEL"
echo "=========================================================="

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
            echo "${trial_dir}"
            return
        fi
        if [ -n "${exp_subdir}" ] && [ ! -d "${trial_dir}/${exp_subdir}" ]; then
            echo "${trial_dir}"
            return
        fi
        trial=$((trial+1))
    done
}

# 3. semantic (Function Calling) 실행
echo "[Running] Semantic (FC Mode)..."
TRIAL_DIR=$(get_trial_dir "${BASE_OUTPUT}" "${API_MODEL}_semantic_fc_summ_plus")
mkdir -p "${TRIAL_DIR}"
BASE_OUT_DIR="${TRIAL_DIR}/${API_MODEL}_semantic_fc_summ_plus/valid_unseen"
mkdir -p "${BASE_OUT_DIR}"
echo "Writing results to: ${BASE_OUT_DIR}"
python3 $PYTHON_SCRIPT \
    --model-name "$API_MODEL" \
    --input-dir "${BASE_INPUT}/semantic_fc_summ_plus/valid_unseen" \
    --output-dir "${BASE_OUT_DIR}" \
    --tools-file "$TOOLS_FILE" \
    --ids-file "$IDS_FILE"

# 4. Hybrid (Function Calling) 실행
# echo "[Running] Hybrid (FC Mode)..."
# TRIAL_DIR=$(get_trial_dir "${BASE_OUTPUT}" "${API_MODEL}_hybrid_fc_edh")
# mkdir -p "${TRIAL_DIR}"
# BASE_OUT_DIR="${TRIAL_DIR}/${API_MODEL}_hybrid_fc_edh/valid_unseen"
# mkdir -p "${BASE_OUT_DIR}"
# echo "Writing results to: ${BASE_OUT_DIR}"
# python3 $PYTHON_SCRIPT \
#     --model-name "$API_MODEL" \
#     --input-dir "${BASE_INPUT}/hybrid_fc_edh/valid_unseen" \
#     --output-dir "${BASE_OUT_DIR}" \
#     --tools-file "$TOOLS_FILE" \
#     --ids-file "$IDS_FILE"

# 5. spatial (Function Calling) 실행
# echo "[Running] Spatial (FC Mode)..."
# TRIAL_DIR=$(get_trial_dir "${BASE_OUTPUT}" "${API_MODEL}_spatial_fc_edh")
# mkdir -p "${TRIAL_DIR}"
# BASE_OUT_DIR="${TRIAL_DIR}/${API_MODEL}_spatial_fc_edh/valid_unseen"
# mkdir -p "${BASE_OUT_DIR}"
# echo "Writing results to: ${BASE_OUT_DIR}"
# python3 $PYTHON_SCRIPT \
#     --model-name "$API_MODEL" \
#     --input-dir "${BASE_INPUT}/spatial_fc_edh/valid_unseen" \
#     --output-dir "${BASE_OUT_DIR}" \
#     --tools-file "$TOOLS_FILE" \
#     --ids-file "$IDS_FILE"

echo "=========================================================="
echo " All tasks finished."
echo " Check results in: $BASE_OUTPUT"
echo "=========================================================="
