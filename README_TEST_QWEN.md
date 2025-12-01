# Test Qwen-3-8b on hybrid_ours dataset

This mini harness finds a sample from `dataset_experiments/hybrid_ours` where:
- `answer_all_motion_or_objectinteraction` is true
- `turn_last_success` is true

It loads `qwen/qwen-3-8b` from the local HF cache `--cache-dir` (default `/models/huggingface_cache`) or downloads it there and runs a generation for the `prompt` in the chosen sample, extracts everything after `</think>`, and compares the generated output (parsed as JSON actions) to the ground-truth `answer['actions']`.

Usage example:

```bash
# Create virtual env and install
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run test
bash scripts/run_test_qwen3.sh --max-tokens 200

# Run a specific jsonl file (sample-file) or index (sample-index)
python3 scripts/test_qwen3_sample.py \
	--cache-dir /models/huggingface_cache \
	--sample-file /home/bjk/tool_learning/teach_to_tool_calling/dataset_experiments/hybrid_ours/valid_unseen/task_1451/episode_0.jsonl \
	--max-tokens 256

python3 scripts/test_qwen3_sample.py \
	--cache-dir /models/huggingface_cache \
	--sample-file /home/bjk/tool_learning/teach_to_tool_calling/dataset_experiments/hybrid_ours/valid_unseen/task_1451/episode_0.jsonl \
	--sample-index 0 \
	--max-tokens 256
```

Notes:
- The script uses `transformers` and will attempt to use a GPU (CUDA) if available.
- If model download is needed, it will go to `--cache-dir`.
- The script takes the first matching sample it finds; you can tune it to pick a specific one if needed.
- If the generated output cannot be parsed as JSON, it prints both the raw generation and the ground truth for inspection.
The script will always attempt to load the model (downloading to the given cache if missing) and perform an actual generation; there is no `--dry-run` option anymore.

