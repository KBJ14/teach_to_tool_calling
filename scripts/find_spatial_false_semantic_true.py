#!/usr/bin/env python3
"""
Find cases where spatial correct == false but semantic correct == true
and save the pair of records for inspection.

Usage:
    python3 scripts/find_spatial_false_semantic_true.py \
      --spatial scripts/selected_500_gpt4o_spatial_baseline_edh_accuracy.jsonl \
      --semantic scripts/selected_500_gpt4o_semantic_ablation_edh_accuracy.jsonl \
      --out scripts/spatial_false_semantic_true.json
"""

import argparse
import json
from typing import Dict, Any


def iter_jsonl(path: str):
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line=line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spatial', required=True, help='Spatial JSONL file')
    parser.add_argument('--semantic', required=True, help='Semantic JSONL file')
    parser.add_argument('--out', default='scripts/spatial_false_semantic_true.json', help='Output JSON file')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    spatial_map: Dict[str, Dict[str, Any]] = {}
    for rec in iter_jsonl(args.spatial):
        iid = rec.get('instance_id')
        if iid:
            spatial_map[iid] = rec

    matches = []
    for rec in iter_jsonl(args.semantic):
        iid = rec.get('instance_id')
        if not iid:
            continue
        srec = spatial_map.get(iid)
        if not srec:
            continue
        # spatial correct false and semantic correct true
        if (srec.get('correct') is False) and (rec.get('correct') is True):
            matches.append({
                'instance_id': iid,
                'spatial': srec,
                'semantic': rec
            })

    if args.verbose:
        print(f'Found {len(matches)} matches')

    # Save output
    with open(args.out, 'w', encoding='utf-8') as fw:
        json.dump(matches, fw, ensure_ascii=False, indent=2)

    print('Done, wrote', len(matches), 'matches to', args.out)


if __name__ == '__main__':
    main()
