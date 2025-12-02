#!/bin/bash

# 1. Python 파일 이름 (이름을 변경했다면 수정하세요)
PYTHON_SCRIPT="batch_inference.py"

# 2. 기본 경로 및 모델 설정 (사용자 환경에 맞게 수정)
BASE_INPUT="/home/bjk/tool_learning/teach_to_tool_calling/dataset_experiments_edh"
BASE_OUTPUT="/home/bjk/tool_learning/teach_to_tool_calling/experiment_results_edh"
API_MODEL="gpt-4o"

echo "=== Starting API Inference for Model: $API_MODEL ==="


3. Hybrid Ours 
echo "Running Hybrid Ours..."
python3 $PYTHON_SCRIPT \
    --model-name $API_MODEL \
    --input-dir "${BASE_INPUT}/hybrid_ours/valid_unseen" \
    --output-dir "${BASE_OUTPUT}/${API_MODEL}_hybrid_ours/valid_unseen" \
    --ids-file selected_500_instance_ids.json


# # 4. Semantic Ablation
# echo "Running Semantic Ablation..."
# python3 $PYTHON_SCRIPT \
#     --model-name $API_MODEL \
#     --input-dir "${BASE_INPUT}/semantic_ablation/valid_unseen" \
#     --output-dir "${BASE_OUTPUT}/${API_MODEL}_semantic_ablation/valid_unseen" \
#     --ids-file selected_500_instance_ids.json

# # 5. Spatial Baseline
# echo "Running Spatial Baseline..."
# python3 $PYTHON_SCRIPT \
#     --model-name $API_MODEL \
#     --input-dir "${BASE_INPUT}/spatial_baseline/valid_unseen" \
#     --output-dir "${BASE_OUTPUT}/${API_MODEL}_spatial_baseline/valid_unseen" \
#     --ids-file selected_500_instance_ids.json
# echo "All experiments queued. Check output directories for results."

