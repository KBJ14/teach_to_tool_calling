#!/bin/bash

# Script Name: generate_datasets.sh
# Description: Generates training/validation datasets for Embodied AI experiments.

# [Config] 경로를 사용자 환경에 맞게 수정하세요.
EPISODE_ROOT="/home/bjk/tool_learning/teach_to_tool_calling/episode_data_wo_state"
OUTPUT_BASE="/home/bjk/tool_learning/teach_to_tool_calling/dataset_experiments"

echo "=========================================================="
echo " [Start] TEACh Dataset Generation Pipeline"
echo " Target Output Directory: $OUTPUT_BASE"
echo "=========================================================="

# ---------------------------------------------------------
# Configuration 1: Spatial Baseline Dataset
# - Description: Naive distance-based filtering (Standard Robot FOV).
# - Use Case: Main Baseline for comparison.
# ---------------------------------------------------------
echo "[1/3] Generating 'Spatial Baseline' Dataset..."
python build_dataset.py \
    --episode-root "$EPISODE_ROOT" \
    --output-root "$OUTPUT_BASE/spatial_baseline" \
    --filter-mode spatial

echo " -> Done. Saved to $OUTPUT_BASE/spatial_baseline"
echo "----------------------------------------------------------"

# ---------------------------------------------------------
# Configuration 2: Semantic Ablation Dataset
# - Description: Pure semantic filtering without proximity safety.
# - Use Case: Ablation study to prove necessity of spatial awareness.
# ---------------------------------------------------------
echo "[2/3] Generating 'Semantic Ablation' Dataset..."
python build_dataset.py \
    --episode-root "$EPISODE_ROOT" \
    --output-root "$OUTPUT_BASE/semantic_ablation" \
    --filter-mode semantic

echo " -> Done. Saved to $OUTPUT_BASE/semantic_ablation"
echo "----------------------------------------------------------"

# ---------------------------------------------------------
# Configuration 3: Hybrid (Ours) Dataset
# - Description: Proposed method (Semantic + Local Safety + Landmarks).
# - Use Case: Main method to evaluate efficiency and safety.
# ---------------------------------------------------------
echo "[3/3] Generating 'Hybrid (Ours)' Dataset..."
python build_dataset.py \
    --episode-root "$EPISODE_ROOT" \
    --output-root "$OUTPUT_BASE/hybrid_ours" \
    --filter-mode hybrid

echo " -> Done. Saved to $OUTPUT_BASE/hybrid_ours"
echo "=========================================================="
echo " [Complete] All datasets are ready for training!"
echo "=========================================================="