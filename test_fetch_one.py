#!/usr/bin/env python3
import argparse
import json
import os
import sys
import os

# Make local package import resilient: when running this script from the repo root (or elsewhere)
# ensure the directory containing this script is on sys.path so we can import local modules.
this_dir = os.path.dirname(os.path.abspath(__file__))
if this_dir not in sys.path:
    sys.path.insert(0, this_dir)

try:
    # preferred import when the package is installed or repo root is in PYTHONPATH
    from teach_to_tool_calling.state_helpers import load_json, find_state_for_interaction
except Exception:
    # fallback to local import if the package name isn't available
    from state_helpers import load_json, find_state_for_interaction


def pretty_short_print(json_obj, max_items=10):
    # Print a short preview with top-level keys and first N items
    if isinstance(json_obj, dict):
        keys = list(json_obj.keys())[:max_items]
        out = {k: json_obj[k] for k in keys}
        print(json.dumps(out, indent=2) + ("\n..." if len(json_obj) > max_items else ""))
    else:
        print(json.dumps(json_obj, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", required=True, help="Episode JSON (per-task episode output) or path to single episode file")
    parser.add_argument("--images_dir", required=True, help="Root path containing TEACh images directories")
    parser.add_argument("--idx", type=int, default=0, help="Turn index in the target EDH entry to inspect")
    parser.add_argument("--edh_idx", type=int, default=0, help="Which edh entry inside the episode json (if it's a list) to use")
    args = parser.parse_args()

    ep = load_json(args.episode)
    print(f"Loaded episode file: {args.episode}")
    print(f"EDH entries in file: {len(ep) if isinstance(ep, list) else 1}")

    try:
        state_path = find_state_for_interaction(ep, turn_idx=args.idx, images_dir=args.images_dir, edh_idx=args.edh_idx)
    except Exception as e:
        print(f"Error: {e}")
        return

    if state_path:
        print(f"Found state file: {state_path}")
        try:
            with open(state_path, "r") as f:
                state = json.load(f)
            print("State preview:")
            pretty_short_print(state)
        except Exception as e:
            print(f"State found but could not load JSON: {e}")
    else:
        print("No state file found for that turn. Consider using --edh_idx to try different edh entries or check images_dir match.")


if __name__ == "__main__":
    main()
