#!/bin/bash

# Script Name: generate_datasets.sh

# [Config] 사용자 경로 설정
EPISODE_ROOT="/teach_dataset/edh_instances"
GAME_ROOT="/teach_dataset/games"  # <-- 게임 파일이 들어있는 실제 경로
OUTPUT_BASE="/home/bjk/tool_learning/teach_to_tool_calling/dataset_experiments"

# 경로 체크
if [ ! -d "$EPISODE_ROOT" ]; then
    echo "Error: Directory $EPISODE_ROOT not found."
    exit 1
fi
if [ ! -d "$GAME_ROOT" ]; then
    echo "Error: Directory $GAME_ROOT not found."
    exit 1
fi

echo "=========================================================="
echo " [Start] TEACh Dataset Generation (EDH Based)"
echo "=========================================================="

# 1. Hybrid (Ours)
echo "[1/3] Generating 'Hybrid' Dataset..."
python build_dataset.py \
    --episode-root "$EPISODE_ROOT" \
    --game-root "$GAME_ROOT" \
    --output-root "$OUTPUT_BASE/hybrid_ours" \
    --filter-mode hybrid

# 2. Spatial Baseline
echo "[2/3] Generating 'Spatial' Dataset..."
python build_dataset.py \
    --episode-root "$EPISODE_ROOT" \
    --game-root "$GAME_ROOT" \
    --output-root "$OUTPUT_BASE/spatial_baseline" \
    --filter-mode spatial

# 3. Semantic Ablation
echo "[3/3] Generating 'Semantic' Dataset..."
python build_dataset.py \
    --episode-root "$EPISODE_ROOT" \
    --game-root "$GAME_ROOT" \
    --output-root "$OUTPUT_BASE/semantic_ablation" \
    --filter-mode semantic

echo "=========================================================="
echo " All Done."
echo "=========================================================="