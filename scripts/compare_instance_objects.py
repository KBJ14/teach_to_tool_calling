#!/usr/bin/env python3
"""
Given an instance_id, find the corresponding sample in dataset_experiments_edh/*
for the requested categories (e.g. spatial_baseline, semantic_ablation) and extract
objects that appear in the Environmental Context (or state->objects). Save JSON with
both sets and differences.

Usage:
  python3 scripts/compare_instance_objects.py \
      --instance-id 57862376ad3394a1_c9dc.edh4 \
      --experiments-root dataset_experiments_edh \
      --categories spatial_baseline,semantic_ablation \
      --out scripts/instance_objects_compare.json

Output structure:
{
  "instance_id": "...",
  "found": {
    "spatial_baseline": ["Sink","Table",...],
    "semantic_ablation": ["Sink","Knife",...]
  },
  "diff": {
    "only_spatial": [...],
    "only_semantic": [...],
    "both": [...]
  },
  "records": {category: record}
}

If multiple records for a category (same instance_id), we dedupe the object lists.
"""

from __future__ import annotations
import argparse
import json
import os
import re
from typing import List, Dict, Optional


def iter_jsonl_files(root_dir: str):
    for root, _dirs, files in os.walk(root_dir):
        for fn in files:
            if fn.endswith('.jsonl'):
                yield os.path.join(root, fn)


def read_jsonl_index(root_dir: str, instance_id: str):
    # returns list of records found for given instance_id
    recs = []
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
                    if obj.get('instance_id') == instance_id:
                        recs.append(obj)
        except Exception:
            continue
    return recs


def extract_objects_from_record(record: Dict) -> List[str]:
    # Try to parse 'state' field first
    objs = set()
    state = record.get('state')
    if isinstance(state, dict):
        oblist = state.get('objects')
        if isinstance(oblist, list):
            for o in oblist:
                name = o.get('objectType') or o.get('name')
                if isinstance(name, str):
                    # some names might be 'Mug|0'; split
                    name = name.split('|')[0]
                    objs.add(name.strip())
            if objs:
                return sorted(objs)

    # Fallback: parse `prompt` for Environmental Context bullet list
    prompt = record.get('prompt')
    if isinstance(prompt, str):
        m = re.search(r"###\s*Environmental Context[\s\S]*?(?=###|$)", prompt, re.IGNORECASE)
        if m:
            block = m.group(0)
        else:
            m2 = re.search(r"Environmental Context[\s\S]*?(?=Interaction History|$)", prompt, re.IGNORECASE)
            block = m2.group(0) if m2 else ''
        if block:
            for line in block.splitlines():
                line = line.strip()
                if line.startswith('-'):
                    # e.g. '- Sink (1.2m, 164° Back, Visible)'
                    content = line.lstrip('-').strip()
                    # split by ' (' if present
                    parts = content.split(' (', 1)
                    obj_name = parts[0].strip()
                    # sometimes object names include ' (' or ','; keep up to first comma?
                    obj_name = obj_name.split(',')[0].strip()
                    if obj_name:
                        objs.add(obj_name)
            if objs:
                return sorted(objs)

    # As a last fallback, try a `sample` wrapper
    sample = record.get('sample')
    if isinstance(sample, dict):
        return extract_objects_from_record(sample)

    return []


def find_experiment_folders(root_dir: str, categories: List[str]):
    experiments = []
    for name in sorted(os.listdir(root_dir)):
        p = os.path.join(root_dir, name)
        if not os.path.isdir(p):
            continue
        lname = name.lower()
        if any(c in lname for c in categories):
            experiments.append((name, p))
    return experiments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance-id', required=True)
    parser.add_argument('--experiments-root', default='dataset_experiments_edh')
    parser.add_argument('--categories', default='spatial_baseline,semantic_ablation',
                        help='Comma separated substrings to match experiment folder names')
    parser.add_argument('--out', default='scripts/instance_objects_compare.json')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    categories = [c.strip().lower() for c in args.categories.split(',') if c.strip()]
    experiments = find_experiment_folders(args.experiments_root, categories)
    if not experiments:
        print('No experiments matched in', args.experiments_root)
        return

    result = {
        'instance_id': args.instance_id,
        'found': {},
        'records': {}
    }

    for name, path in experiments:
        if args.verbose:
            print('Searching in', name, path)
        recs = read_jsonl_index(path, args.instance_id)
        if not recs:
            if args.verbose:
                print('  not found')
            continue
        # In many experiment folders there should be a single matching record
        # But handle multiple (choose first or aggregate objects from all)
        all_objs = set()
        for rec in recs:
            objs = extract_objects_from_record(rec)
            for o in objs:
                all_objs.add(o)
        result['found'][name] = sorted(all_objs)
        # store first record for inspection
        result['records'][name] = recs[0]

    # Compute diffs between categories (we'll compare the first two
    # categories provided if both exist, otherwise compute pairwise comparisons
    names = list(result['found'].keys())
    diffs = {}
    if names:
        # For pairwise convenience, choose spatial vs semantic if present
        # Find keys that contain 'spatial' and 'semantic' prefer them
        spatial_key = None
        semantic_key = None
        for k in names:
            if 'spatial' in k.lower():
                spatial_key = k
            if 'semantic' in k.lower():
                semantic_key = k
        # fallback to first two names
        if not spatial_key and len(names) >= 1:
            spatial_key = names[0]
        if not semantic_key and len(names) >= 2:
            semantic_key = names[1]

        if spatial_key and semantic_key and spatial_key in result['found'] and semantic_key in result['found']:
            sset = set(result['found'][spatial_key])
            tset = set(result['found'][semantic_key])
            diffs['only_spatial'] = sorted(sset - tset)
            diffs['only_semantic'] = sorted(tset - sset)
            diffs['both'] = sorted(sset & tset)
            diffs['spatial_key'] = spatial_key
            diffs['semantic_key'] = semantic_key

    result['diff'] = diffs

    # Save
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as fw:
        json.dump(result, fw, ensure_ascii=False, indent=2)

    print('Done. Wrote result for', args.instance_id, 'to', args.out)


if __name__ == '__main__':
    main()
