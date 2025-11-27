# Embodied Agent Data Generation Pipeline

This repository contains the data generation pipeline for the research:
**"Robot Action as an External Tool for LLMs: Multi-turn Embodied Reasoning"**.

This pipeline transforms raw TEACh robot states (JSON) into optimized textual prompts suitable for small LLMs (e.g., Llama-3-8B), enabling efficient tool learning and reasoning.

---

## 🚀 Methodology: Two-Stage Perception Pipeline

We propose a modular approach to handle long-context robot observations.

### Stage 1: Perception Module (Deterministic)
Before feeding data to the LLM, a Python-based module processes the raw JSON state to reduce token usage and computational load.
1.  **State Merging:** Merges `initial_state` (Map) and `state_diff` (Updates) to maintain a complete and up-to-date world model.
2.  **Relative Coordinate Calculation:** Computes distance ($r$) and relative angle ($\theta$) to provide precise parameters for `motion_delta` (e.g., "1.5m, 45° Right").
3.  **Task-Aware Filtering:** Prunes irrelevant objects based on `spatial` or `semantic` logic using `all-MiniLM-L6-v2`.
4.  **State Abstraction:** Converts JSON to natural language, retaining only 11 dynamic states (e.g., `Open`, `Dirty`, `Visible`) and discarding static capabilities.

### Stage 2: Reasoning Module (LLM)
The LLM receives the refined text from Stage 1 and generates robot actions (Function Calls).

---

## 🧪 Dataset Configurations

We generate three distinct datasets to validate the efficiency of our Hybrid strategy.

| Dataset Name | Filter Mode | Logic (Parallel Union) | Thresholds | Role |
| :--- | :--- | :--- | :--- | :--- |
| **Spatial** | `spatial` | `Landmark` $\cup$ `Visible` $\cup$ `Dist < 5.0m` | $r=5.0m$ | **Strong Baseline** (Standard Robot FOV) |
| **Semantic** | `semantic` | `Landmark` $\cup$ `Visible` $\cup$ `Sim > 0.3` | $Sim=0.3$ | **Ablation Study** (No proximity safety) |
| **Hybrid** | `hybrid` | `Landmark` $\cup$ `Visible` $\cup$ `Sim > 0.35` $\cup$ `Dist < 2.5m` | $r=2.5m, Sim=0.35$ | **Proposed Method** (Efficient & Safe) |

*Note: The Naive Baseline (No Filter) is excluded as it causes Context Overflow (OOM) on 8B models.*

---

## 💻 Usage

### 1. Install Dependencies
This project requires `sentence-transformers` for semantic similarity calculation.

pip install sentence-transformers

### 2. Generate All Datasets (Recommended)
Use the provided shell script to generate all three experimental datasets (Spatial, Semantic, Hybrid) in one go. This is the standard way to prepare data for the experiments.

**Steps:**
1. Open `generate_datasets.sh` and modify the `EPISODE_ROOT` and `OUTPUT_BASE` variables to match your local file paths.
2. Grant execution permission and run the script:

# Give execution permission (only needed once)
chmod +x generate_datasets.sh

# Run the generation pipeline
./generate_datasets.sh

### 3. (Optional) Run individual Mode
python build_dataset.py \
    --episode-root /path/to/episode_data_wo_state \
    --output-root /path/to/output_dir \
    --filter-mode semantic \
    --embedding-model all-MiniLM-L6-v2

--episode-root: Path to the input directory containing TEACh episode JSON files.

--output-root: Target directory where the .jsonl files will be saved.

--filter-mode: Filtering strategy. Choose between spatial, semantic, or hybrid.

--embedding-model: (Optional) HuggingFace model name for embeddings. Default is all-MiniLM-L6-v2.

## Implementation details
### State Abstraction (JSON to Text)

We remove 95% of token-heavy raw data (e.g., 8-point bounding boxes, mass) and retain only actionable states to reduce cognitive load on the 8B model.Kept States (11 Types): Visible, Held, Open, On, Dirty, Filled, Sliced, Cooked, Broken, Empty, Hot/Cold.Navigation Info: Converted from absolute $(x, y, z)$ to Relative Polar Coordinates (e.g., "1.5m, 45° Right") to facilitate motion_delta prediction.

### Parallel filter logic
Unlike sequential filtering, we use a Union (OR) logic to prevent information loss.Spatial Mode (Baseline): Retains objects within $5.0m$ radius or visible ones.Hybrid Mode (Ours): Retains objects if they match any of the following:Semantically Relevant (Similarity > 0.35)Very Close ($< 2.5m$, for collision avoidance)Visible (in camera view)Landmarks (Fixed furniture for map grounding)