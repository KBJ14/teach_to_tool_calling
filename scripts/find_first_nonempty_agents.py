#!/usr/bin/env python3
"""
Scan TEACh images directory for the first JSON file where the top-level key 'agents' is non-empty.
By default this checks files matching 'statediff*.json' recursively under the images dir but
can be told to check any pattern or all '.json' files.

Usage: find_first_nonempty_agents.py --images_dir /teach_dataset/images
       find_first_nonempty_agents.py --images_dir /teach_dataset/images --pattern "**/*.json" --all

By default the script prints the first json found and exits with code 0; if not found it returns 1.
"""
import argparse
import glob
import json
import os
import sys
from typing import Iterable


def iter_files(images_dir: str, pattern: str) -> Iterable[str]:
    # Use glob with recursive support; if the pattern is relative (no /) make it starred under images_dir
    if os.path.isabs(pattern):
        base = pattern
    else:
        base = os.path.join(images_dir, pattern)
    for fn in glob.glob(base, recursive=True):
        if os.path.isfile(fn):
            yield fn


def check_file(fn: str) -> bool:
    try:
        with open(fn, 'r', encoding='utf-8') as f:
            j = json.load(f)
    except Exception:
        return False
    if not isinstance(j, dict):
        return False
    agents = j.get('agents')
    if agents is None:
        return False
    if isinstance(agents, dict) or isinstance(agents, list):
        return len(agents) > 0
    # if agents is truthy and not list/dict
    return bool(agents)


def pretty_print_agents(fn: str):
    with open(fn, 'r', encoding='utf-8') as f:
        j = json.load(f)
    print("Found file:", fn)
    print("agents:")
    print(json.dumps(j.get('agents'), indent=2, ensure_ascii=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description='Find the first JSON with non-empty agents in images dir')
    parser.add_argument('--images_dir', default='/teach_dataset/images', help='Root of images dir')
    parser.add_argument('--pattern', default='**/statediff*.json', help='glob pattern to search (recursive).')
    parser.add_argument('--all', action='store_true', help='Print all files whose agents is non-empty')
    parser.add_argument('--verbose', action='store_true', help='Print progress')
    args = parser.parse_args(argv)

    files = list(iter_files(args.images_dir, args.pattern))
    total = len(files)
    if total == 0:
        print(f"No files found for pattern {args.pattern} under {args.images_dir}")
        return 1

    found = []
    # progress interval to avoid flooding stderr
    progress_interval = max(1, total // 100)

    for idx, fn in enumerate(files, start=1):
        # print progress occasionally or always if verbose
        if args.verbose and (idx % progress_interval == 0 or idx <= 5):
            print(f"Progress: checked {idx}/{total} files ({idx*100//total}%)", file=sys.stderr)

        if check_file(fn):
            if args.all:
                found.append(fn)
                continue
            pretty_print_agents(fn)
            # stop immediately when first match found
            return 0

    if args.all:
        if not found:
            print('No files found with non-empty agents')
            return 1
        for fn in found:
            pretty_print_agents(fn)
        return 0

    print('No file found with non-empty agents')
    return 1


if __name__ == '__main__':
    sys.exit(main())
