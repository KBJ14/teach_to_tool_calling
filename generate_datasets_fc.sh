#!/bin/bash

# Script Name: generate_datasets_fc.sh
# Description: Generates datasets specifically for Function Calling experiments.
#              Calls 'build_dataset_fc.py' which excludes tool definitions from the prompt.

# [Config] 사용자 경로 설정
# 실제 데이터가 있는 경로로 수정해주세요.
EPISODE_ROOT="/teach_dataset/edh_instances"
GAME_ROOT="/teach_dataset/games"
OUTPUT_BASE="/home/bjk/tool_learning/teach_to_tool_calling/dataset_experiments_fc_edh"

# 실행할 Python 스크립트 이름 (Function Calling용)
BUILD_SCRIPT="build_dataset_fc_edh.py"

# 경로 체크
if [ ! -d "$EPISODE_ROOT" ]; then
    echo "Error: Directory $EPISODE_ROOT not found."
    exit 1
fi
if [ ! -d "$GAME_ROOT" ]; then
    echo "Error: Directory $GAME_ROOT not found."
    exit 1
fi

# 파이썬 스크립트 존재 확인
if [ ! -f "$BUILD_SCRIPT" ]; then
    echo "Error: Python script '$BUILD_SCRIPT' not found in current directory."
    exit 1
fi

echo "=========================================================="
echo " [Start] TEACh Dataset Generation (Function Calling Ver.)"
echo "=========================================================="

# 1. Hybrid (Ours)
echo "[1/1] Generating 'semantic' Dataset for Function Calling..."
python3 $BUILD_SCRIPT \
    --episode-root "$EPISODE_ROOT" \
    --game-root "$GAME_ROOT" \
    --output-root "$OUTPUT_BASE/semantic_fc_edh" \
    --filter-mode semantic \
    --embedding-model "all-MiniLM-L6-v2"

# (필요시 Semantic, Spatial 등 다른 모드도 아래에 추가 가능)
# python3 $BUILD_SCRIPT ... --filter-mode semantic
# python3 $BUILD_SCRIPT ... --filter-mode spatial

echo "=========================================================="
echo " Generation Complete."
echo " Output Directory: $OUTPUT_BASE/semantic_fc_edh"
echo "=========================================================="