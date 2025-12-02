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

# 2. 파일 존재 확인 (Safety Checks)
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: $PYTHON_SCRIPT not found."
    exit 1
fi
if [ ! -f "$TOOLS_FILE" ]; then
    echo "Error: Tools file not found at $TOOLS_FILE"
    exit 1
fi
if [ ! -d "$BASE_INPUT/semantic_fc_summ_plus/valid_unseen" ]; then
    echo "Error: Input dataset directory not found at $BASE_INPUT/semantic_fc_summ_plus/valid_unseen"
    echo "Did you run 'generate_datasets_fc.sh'?"
    exit 1
fi

echo "=========================================================="
echo " [Start] Function Calling Inference: $API_MODEL"
echo "=========================================================="

# 3. Hybrid Ours (Function Calling) 실행
echo "[Running] Semantic (FC Mode)..."
python3 $PYTHON_SCRIPT \
    --model-name "$API_MODEL" \
    --input-dir "${BASE_INPUT}/semantic_fc_summ_plus/valid_unseen" \
    --output-dir "${BASE_OUTPUT}/${API_MODEL}_semantic_fc_summ_plus/valid_unseen" \
    --tools-file "$TOOLS_FILE" \
    --ids-file "$IDS_FILE"

echo "=========================================================="
echo " All tasks finished."
echo " Check results in: $BASE_OUTPUT"
echo "=========================================================="
