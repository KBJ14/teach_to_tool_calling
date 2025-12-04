#!/usr/bin/env python3
"""
Compute accuracy of first predicted action vs first ground truth action for a list of instance IDs.

Usage:
  python scripts/compute_first_action_accuracy.py --base-dir experiment_results/gpt-4o_hybrid_ours --ids-file scripts/selected_500_instance_ids.json --out-file scripts/selected_500_gpt4o_accuracy.jsonl

Output: JSONL lines of {"instance_id": ..., "predicted_action": {...}, "ground_truth_action": {...}, "correct": true/false}
Also prints a summary: total, correct, accuracy.

Notes:
- Only compares first action: `prediction.actions[0]` vs `ground_truth.actions[0]`.
- If a prediction or ground truth is missing or has no actions, it's considered incorrect.
- Only instance IDs present in both the selected list and the `base-dir` will be evaluated.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional, List


def iter_jsonl_files(base_dir: str, ext: str = ".jsonl"):
    for root, _dirs, files in os.walk(base_dir):
        for fn in files:
            if fn.endswith(ext):
                yield os.path.join(root, fn)


def load_records_by_id(base_dir: str) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    for file_path in iter_jsonl_files(base_dir):
        with open(file_path, 'r', encoding='utf-8') as fh:
            for raw_line in fh:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    obj = json.loads(raw_line)
                except Exception:
                    continue
                inst_id = obj.get('instance_id')
                if inst_id:
                    mapping[inst_id] = obj
    return mapping


def action_from(obj: Dict[str, Any], field: str = 'prediction') -> Optional[Dict[str, Any]]:
    if not obj:
        return None
    f = obj.get(field)
    if isinstance(f, dict):
        actions = f.get('actions')
        if isinstance(actions, list) and actions:
            return actions[0]
    return None


def compare_actions(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> bool:
    # Both None -> correct? No: treat missing as incorrect
    if a is None or b is None:
        return False
    # compare tool_name
    if a.get('tool_name') != b.get('tool_name'):
        return False
    # compare parameters via JSON structural equality
    return a.get('parameters') == b.get('parameters')


def main():
    parser = argparse.ArgumentParser(description='Compute first-action accuracy for selected instance ids.')
    parser.add_argument('--base-dir', required=True, help='Dataset directory for model results (e.g., experiment_results/gpt-4o_hybrid_ours)')
    parser.add_argument('--ids-file', required=True, help='JSON file containing list of instance_id strings')
    parser.add_argument('--out-file', required=True, help='Output JSONL per-instance truth/pred/score file')
    parser.add_argument('--debug-missing', action='store_true', help='Print instance IDs not found in base-dir')
    parser.add_argument('--summary-file', type=str, default=None, help='Append a JSON summary line to this file (JSONL)')
    parser.add_argument('--experiment-name', type=str, default=None, help='Optional experiment name to record in the summary')
    parser.add_argument('--trial-number', type=int, default=None, help='Optional trial number to record in the summary')
    
    args = parser.parse_args()

    # Load selected ids
    with open(args.ids_file, 'r', encoding='utf-8') as f:
        selected_ids = json.load(f)

    print(f'Loaded {len(selected_ids)} instance IDs from {args.ids_file}')

    print(f'Indexing records under {args.base_dir} ...')
    records_map = load_records_by_id(args.base_dir)
    print(f'Indexed {len(records_map)} records from {args.base_dir}')

    found = 0
    total = 0
    correct = 0
    results: List[Dict[str, Any]] = []
    missing_ids = []

    for inst_id in selected_ids:
        total += 1
        rec = records_map.get(inst_id)
        if rec is None:
            # not found under base dir -> skip but record as missing if requested
            missing_ids.append(inst_id)
            continue
        found += 1
        pred_action = action_from(rec, 'prediction')
        gt_action = action_from(rec, 'ground_truth')
        is_cor = compare_actions(pred_action, gt_action)
        if is_cor:
            correct += 1
        results.append({
            'instance_id': inst_id,
            'predicted_action': pred_action,
            'ground_truth_action': gt_action,
            'correct': is_cor,
        })

    out_dir = os.path.dirname(args.out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    # Write per-instance JSONL
    with open(args.out_file, 'w', encoding='utf-8') as fw:
        for r in results:
            fw.write(json.dumps(r, ensure_ascii=False) + '\n')

    print('--- Done ---')
    print(f'Total selected IDs: {total}')
    print(f'Found in base-dir: {found}')
    print(f'Correct first-action: {correct}')
    accuracy = correct / found if found > 0 else 0.0
    print(f'Accuracy (on found instances): {accuracy:.4%}')

    # Optionally write a summary record to `--summary-file` containing: experiment_name/base_dir, selected_total, found, correct, accuracy, and optionally seed/filters
    if args.summary_file:
        summary = {
            'experiment_name': args.experiment_name or os.path.basename(args.base_dir.rstrip('/')),
            'trial' : args.trial_number,
            'accuracy': accuracy
        }
        # add optional debug info
        if args.debug_missing:
            summary['missing_ids'] = missing_ids
        # append summary as a single JSON line
        os.makedirs(os.path.dirname(args.summary_file), exist_ok=True)
        with open(args.summary_file, 'a', encoding='utf-8') as sf:
            sf.write(json.dumps(summary, ensure_ascii=False) + '\n')

    if args.debug_missing and missing_ids:
        print('Missing instance IDs (skipped):')
        for mid in missing_ids:
            print(' -', mid)

if __name__ == '__main__':
    main()
