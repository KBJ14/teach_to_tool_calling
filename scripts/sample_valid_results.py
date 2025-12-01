#!/usr/bin/env python3
"""
Count how many valid outputs exist across experiment_results and randomly sample N valid ones into a list.

Usage:
    python scripts/sample_valid_results.py --base-dir experiment_results --out-file selected_500.jsonl --n 500 --seed 42

The script uses reservoir sampling to avoid storing all records in memory.

A "valid" result is defined as:
- JSON object contains a non-empty `prediction` with `actions` (list) and at least one action, OR
- `model_output_raw` is non-empty (fallback)

Output format: JSONL file with full JSON record (one per line) for each sampled instance.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional


def is_valid_instance(obj: Dict[str, Any], min_actions: int = 1) -> bool:
    """Return True if object contains a valid prediction (non-empty actions) or non-empty raw output."""
    # Check structured prediction
    pred = obj.get("prediction")
    if pred:
        # if prediction has .actions and it's a list with >= 1 elements
        a = pred.get("actions") if isinstance(pred, dict) else None
        if isinstance(a, list) and len(a) >= min_actions:
            return True
        # otherwise, if prediction itself is non-empty dict, accept
        if isinstance(pred, dict) and len(pred) > 0:
            return True
        if isinstance(pred, list) and len(pred) > 0:
            return True

    # fallback: check model_output_raw for non-empty plausible string
    raw = obj.get("model_output_raw")
    if raw and isinstance(raw, str) and raw.strip() != "":
        return True

    return False


def iter_jsonl_files(base_dir: str, ext: str = ".jsonl"):
    for root, _dirs, files in os.walk(base_dir):
        for fn in files:
            if fn.endswith(ext):
                yield os.path.join(root, fn)


def count_and_sample(base_dir: str, sample_k: int, seed: Optional[int] = None, model_name: Optional[str] = None, min_actions: int = 1, ids_only: bool = False):
    if seed is not None:
        random.seed(seed)

    total_lines = 0
    valid_count = 0
    invalid_count = 0
    reservoir: List[Any] = []

    for file_path in iter_jsonl_files(base_dir):
        with open(file_path, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                raw_line = raw_line.strip()
                if raw_line == "":
                    continue
                total_lines += 1
                try:
                    obj = json.loads(raw_line)
                except Exception:
                    # not json -> treat as invalid
                    invalid_count += 1
                    continue

                if model_name is not None:
                    if obj.get("model_name") != model_name:
                        invalid_count += 1
                        continue

                if is_valid_instance(obj, min_actions=min_actions):
                    # valid
                    valid_count += 1
                    # Reservoir sampling (store instance_id or full object)
                    item = obj.get("instance_id") if ids_only else obj
                    if len(reservoir) < sample_k:
                        reservoir.append(item)
                    else:
                        # randomly replace each item with slowly decreasing probability
                        j = random.randint(0, valid_count - 1)
                        if j < sample_k:
                            reservoir[j] = item
                else:
                    invalid_count += 1

    return {
        "total_lines": total_lines,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "sample": reservoir,
    }


def save_jsonl(records: List[Dict[str, Any]], out_file: str):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as fw:
        for rec in records:
            fw.write(json.dumps(rec, ensure_ascii=False) + "\n")


def save_json_list(ids: List[str], out_file: str):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as fw:
        json.dump(ids, fw, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Count valid outputs and sample random valid ones")
    parser.add_argument("--base-dir", default="experiment_results", help="Base experiment_results dir")
    parser.add_argument("--n", type=int, default=500, help="Number of random valid examples to sample")
    parser.add_argument("--model-name", type=str, default=None, help="Optional: filter results only for a specific model_name")
    parser.add_argument("--min-actions", type=int, default=1, help="Optional: minimum number of actions in prediction for it to count as valid")
    parser.add_argument("--ids-only", action="store_true", help="If set, only save the sampled `instance_id`s as a JSON array (default: save full records as JSONL)")
    parser.add_argument("--out-file", default="selected_500.jsonl", help="Output JSONL file path for the sample")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (optional)")
    args = parser.parse_args()

    print(f"Scanning base directory: {args.base_dir}")
    stats = count_and_sample(args.base_dir, args.n, seed=args.seed, model_name=args.model_name, min_actions=args.min_actions, ids_only=args.ids_only)

    print("--- scan completed ---")
    print(f"Total instances lines: {stats['total_lines']}")
    print(f"Valid outputs: {stats['valid_count']}")
    print(f"Invalid outputs: {stats['invalid_count']}")

    # Write the sample to out-file
    print(f"Saving sample (size = {len(stats['sample'])}) to {args.out_file}")
    if args.ids_only:
        save_json_list(stats["sample"], args.out_file)
    else:
        save_jsonl(stats["sample"], args.out_file)
    print("Done.")


if __name__ == "__main__":
    main()
