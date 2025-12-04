#!/usr/bin/env python3
"""
Compute average number of objects in the 'state' for selected instance IDs across experiment result dataset folders.

For each experiment folder under `--experiments-root` whose name matches 'hybrid', 'spatial', or 'semantic' (or provided via --categories),
this script will index JSONL dataset files, find entries whose `instance_id` is in the provided ids-file, and compute the number of objects
in the state.

State object counting logic (robust):
- If JSON record has a `state` field with 'objects' list: count len(state['objects']).
- Else, fallback to parse the 'prompt' text: find the "Environmental Context" block (header starting with '###', e.g. "### Environmental Context"),
  and count the bullet lines starting with '-' in the block.

Output: JSON file that maps experiment_name -> { 'selected_total', 'found', 'avg_num_objects', 'counts': [{instance_id, count}] }

Usage:
    python3 scripts/compute_avg_objects_by_experiment.py \
        --experiments-root teach_to_tool_calling/dataset_experiments \
        --ids-file teach_to_tool_calling/scripts/selected_500_instance_ids.json \
        --out-file teach_to_tool_calling/scripts/selected_500_avg_objects.json \
        --categories hybrid,spatial,semantic

"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List, Optional


def iter_jsonl_files(root_dir: str):
    for root, _dirs, files in os.walk(root_dir):
        for fn in files:
            if fn.endswith('.jsonl'):
                yield os.path.join(root, fn)


def count_objects_in_state(obj: Dict) -> Optional[int]:
    # 1) Try explicit state->objects list
    state = obj.get('state')
    if isinstance(state, dict):
        objs = state.get('objects')
        if isinstance(objs, list):
            return len(objs)
    # 2) If 'prompt' exists, parse Environmental Context block
    prompt = obj.get('prompt')
    if isinstance(prompt, str):
        # Search for 'Environmental Context' header
        # Using regex to find the block starting at a header like '### Environmental Context' and ending before the next header '###' or 'Interaction History'
        m = re.search(r"###\s*Environmental Context[\s\S]*?(?=###|$)", prompt, re.IGNORECASE)
        if m:
            block = m.group(0)
        else:
            # fallback: capture from 'Environmental Context' without '###'
            m2 = re.search(r"Environmental Context[\s\S]*?(?=Interaction History|$)", prompt, re.IGNORECASE)
            block = m2.group(0) if m2 else ''
        if block:
            # Count bullet lines starting with '-' (trim whitespace)
            count = 0
            for line in block.splitlines():
                line = line.strip()
                if line.startswith('-'):
                    count += 1
            # If we got zero but block contains words separated by newlines without bullets, count lines that look like object entries
            if count == 0:
                for line in block.splitlines():
                    if '(' in line and ')' in line:
                        count += 1
            return count
    # 3) Nothing found
    return None


def index_files_by_instance(root_dir: str, ids_set: set):
    # Return dict instance_id -> record for only matching ids
    mapping = {}
    for p in iter_jsonl_files(root_dir):
        try:
            with open(p, 'r', encoding='utf-8') as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    inst = obj.get('instance_id')
                    if inst and inst in ids_set:
                        mapping[inst] = obj
        except Exception:
            continue
    return mapping


def compute_avg_for_experiment(exp_dir: str, ids: List[str]) -> Dict:
    ids_set = set(ids)
    mapping = index_files_by_instance(exp_dir, ids_set)
    results = []
    found = 0
    total_count = 0
    for inst in ids:
        rec = mapping.get(inst)
        if not rec:
            continue
        found += 1
        obj_count = count_objects_in_state(rec)
        if obj_count is None:
            # Could not determine -> treat as 0 or skip? we'll skip counting in average
            # but record as None
            results.append({'instance_id': inst, 'objects': None})
            continue
        total_count += obj_count
        results.append({'instance_id': inst, 'objects': obj_count})
    avg = (total_count / found) if found > 0 else 0.0
    return {'selected_total': len(ids), 'found': found, 'avg_num_objects': avg, 'counts': results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments-root', default='teach_to_tool_calling/dataset_experiments_edh', help='Root where experiment datasets live')
    parser.add_argument('--ids-file', required=True, help='JSON array selected instance IDs')
    parser.add_argument('--out-file', default='teach_to_tool_calling/scripts/selected_500_avg_objects.json', help='Output JSON file for results')
    parser.add_argument('--categories', default='hybrid,spatial,semantic', help='Comma-separated substrings to match experiment folders')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    with open(args.ids_file, 'r', encoding='utf-8') as f:
        ids = json.load(f)

    categories = [c.strip().lower() for c in args.categories.split(',') if c.strip()]
    experiments = []
    for name in sorted(os.listdir(args.experiments_root)):
        p = os.path.join(args.experiments_root, name)
        if not os.path.isdir(p):
            continue
        lname = name.lower()
        if any(c in lname for c in categories):
            experiments.append((name, p))

    results_all = {}
    for name, p in experiments:
        if args.verbose:
            print(f'Computing for {name} at {p}...')
        res = compute_avg_for_experiment(p, ids)
        results_all[name] = res
        if args.verbose:
            print(json.dumps(res, ensure_ascii=False))

    # Save out file
    out_dir = os.path.dirname(args.out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_file, 'w', encoding='utf-8') as fw:
        json.dump(results_all, fw, ensure_ascii=False, indent=2)

    print('Done. Results written to', args.out_file)

if __name__ == '__main__':
    main()
