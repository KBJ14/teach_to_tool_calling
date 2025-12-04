#!/usr/bin/env python3
"""
Find cases where semantic_fc_summ correct == false but semantic_fc_summ_plus correct == true
and save matched records to JSON for inspection.

Usage:
    python3 scripts/find_semantic_fc_summ_to_plus.py \
      --summ scripts/selected_500_gpt4o_semantic_fc_summ_accuracy.jsonl \
      --summ-plus scripts/selected_500_gpt4o_semantic_fc_summ_plus_accuracy.jsonl \
      --out scripts/semantic_fc_summ_to_plus_matches.json
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
    parser.add_argument('--summ', required=True, help='semantic_fc_summ JSONL file')
    parser.add_argument('--summ-plus', required=True, help='semantic_fc_summ_plus JSONL file')
    parser.add_argument('--out', default='scripts/semantic_fc_summ_to_plus_matches.json', help='Output JSON file')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    summ_map: Dict[str, Dict[str, Any]] = {}
    for rec in iter_jsonl(args.summ):
        iid = rec.get('instance_id')
        if iid:
            summ_map[iid] = rec

    matches = []
    for rec in iter_jsonl(args.summ_plus):
        iid = rec.get('instance_id')
        if not iid:
            continue
        srec = summ_map.get(iid)
        if not srec:
            continue
        # summ was false, summ_plus is true
        if (srec.get('correct') is False) and (rec.get('correct') is True):
            matches.append({
                'instance_id': iid,
                'summ': srec,
                'summ_plus': rec
            })

    if args.verbose:
        print(f'Found {len(matches)} matches')

    with open(args.out, 'w', encoding='utf-8') as fw:
        json.dump(matches, fw, ensure_ascii=False, indent=2)

    print('Done, wrote', len(matches), 'matches to', args.out)


if __name__ == '__main__':
    main()
