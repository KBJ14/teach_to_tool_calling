#!/usr/bin/env python3
"""
Preview data for a single TEACh turn (EDH + corresponding game + state JSON) and save as JSON.

Usage examples:

python preview_raw_turn.py --data_root /teach_dataset --edh_index 0 --turn_idx 3 --out /tmp/preview.json
python preview_raw_turn.py --data_root /teach_dataset --edh_fn /teach_dataset/edh_instances/valid_seen/abcd1234_0.edh0.json --turn_idx 2

Notes:
- If --edh_fn is not provided, the EDH file will be picked by --edh_index from the edh_instances folder.
- The script tries to find matching game file and a state JSON file heuristically.
"""

import argparse
import json
import os
import re
from glob import glob


def load_json(fn):
    with open(fn, "r", encoding="utf-8") as f:
        return json.load(f)


def find_edh_files(data_root):
    edh_dir = os.path.join(data_root, "edh_instances")
    if os.path.isdir(edh_dir):
        hits = glob(os.path.join(edh_dir, "**", "*.edh*.json"), recursive=True)
    else:
        hits = glob(os.path.join(data_root, "**", "*.edh*.json"), recursive=True)
    return sorted(hits)


def find_first_edh(data_root):
    hits = find_edh_files(data_root)
    if not hits:
        raise FileNotFoundError("No EDH instances found under data_root")
    return hits[0]


def get_game_id_from_edh(edh_fn, edh_json):
    # Check common metadata fields
    for key in ("game_id", "structured_log_fn", "orig_game_id", "original_game_fn"):
        if key in edh_json:
            v = edh_json[key]
            if isinstance(v, str) and v.endswith(".game.json"):
                return os.path.splitext(os.path.basename(v))[0]
            if isinstance(v, str):
                return os.path.splitext(os.path.basename(v))[0].split("_")[0]
    # fallback: filename prefix before underscore
    base = os.path.basename(edh_fn)
    m = re.match(r"([0-9a-fA-F]+)", base)
    if m:
        return m.group(1)
    return base.split("_")[0].split(".")[0]


def find_game_file_for_gameid(data_root, game_id):
    candidate_dirs = ["experiment_games", "all_games", "games", ""]
    for d in candidate_dirs:
        p = os.path.join(data_root, d, f"{game_id}.game.json")
        if os.path.isfile(p):
            return p
    # fallback: look recursively
    hits = glob(os.path.join(data_root, "**", f"{game_id}.game.json"), recursive=True)
    if hits:
        return hits[0]
    # else search by containing game id
    hits = glob(os.path.join(data_root, "**", f"*{game_id}*.game.json"), recursive=True)
    return hits[0] if hits else None


def find_state_file_for_turn(data_root, game_id, turn_idx):
    patterns = [
        os.path.join(data_root, "images", "**", f"*{game_id}*.json"),
        os.path.join(data_root, "images_and_states", "**", f"*{game_id}*.json"),
        os.path.join(data_root, "**", f"*{game_id}*state*.json"),
        os.path.join(data_root, "**", f"*{game_id}*_{turn_idx}.json"),
        os.path.join(data_root, "images", "**", "*.json"),
    ]
    candidates = []
    for pat in patterns:
        candidates += glob(pat, recursive=True)
    
    # heuristic: prefer 'state' in filename and containing turn index
    for fn in sorted(set(candidates)):
        name = os.path.basename(fn).lower()
        if "state" in name and f"_{turn_idx}" in name:
            return fn
    for fn in sorted(set(candidates)):
        if "state" in os.path.basename(fn).lower():
            return fn
    for fn in sorted(set(candidates)):
        if game_id in os.path.basename(fn):
            return fn
    return None


def find_episode_in_game(game_json, edh_json):
    tidx = edh_json.get("task_idx")
    eidx = edh_json.get("episode_idx")
    if game_json is None:
        return None, None
    if "tasks" in game_json and isinstance(game_json["tasks"], list):
        if tidx is not None and 0 <= tidx < len(game_json["tasks"]):
            task = game_json["tasks"][tidx]
        else:
            task = game_json["tasks"][0]
        if eidx is not None and "episodes" in task and 0 <= eidx < len(task["episodes"]):
            return task, task["episodes"][eidx]
        if "episodes" in task and task["episodes"]:
            return task, task["episodes"][0]
    if "episodes" in game_json:
        return None, game_json["episodes"][0]
    return None, None


def get_interaction_for_turn(edh_json, turn_idx):
    arr = edh_json.get("interactions", [])
    if not arr:
        return None
    if turn_idx < 0 or turn_idx >= len(arr):
        try:
            return arr[turn_idx]
        except Exception:
            return None
    return arr[turn_idx]


def main():
    p = argparse.ArgumentParser(description="Preview TEACh EDH + game + state for a single turn")
    p.add_argument("--data_root", default="/teach_dataset", help="Root of TEACh dataset (e.g. /teach_dataset)")
    p.add_argument("--edh_fn", default=None, help="Path to EDH instance file (optional)")
    p.add_argument("--edh_index", type=int, default=None, help="Index into edh_instances (optional)")
    p.add_argument("--turn_idx", type=int, default=0, help="Turn index within EDH interactions")
    p.add_argument("--out", default="./preview_raw_turn.json", help="Output preview JSON")
    args = p.parse_args()

    data_root = args.data_root

    if args.edh_fn is None:
        edh_files = find_edh_files(data_root)
        if not edh_files:
            raise FileNotFoundError("No EDH files found under data_root")
        if args.edh_index is not None:
            if args.edh_index < 0 or args.edh_index >= len(edh_files):
                raise IndexError("--edh_index out of range")
            edh_fn = edh_files[args.edh_index]
        else:
            edh_fn = edh_files[0]
    else:
        edh_fn = args.edh_fn

    edh = load_json(edh_fn)
    game_id = get_game_id_from_edh(edh_fn, edh)
    game_fn = find_game_file_for_gameid(data_root, game_id)
    game_json = load_json(game_fn) if game_fn else None

    task, episode = find_episode_in_game(game_json, edh)
    interaction = get_interaction_for_turn(edh, args.turn_idx)

    game_interaction = None
    if episode and "interactions" in episode:
        if 0 <= args.turn_idx < len(episode["interactions"]):
            game_interaction = episode["interactions"][args.turn_idx]

    state_fn = find_state_file_for_turn(data_root, game_id, args.turn_idx)
    state_json = load_json(state_fn) if state_fn and os.path.isfile(state_fn) else None

    preview = {
        "edh_fn": edh_fn,
        "game_id": game_id,
        "game_fn": game_fn,
        "edh": edh,
        "episode_meta": {
            "task_idx": edh.get("task_idx"),
            "episode_idx": edh.get("episode_idx"),
        },
        "selected_interaction_idx": args.turn_idx,
        "edh_interaction": interaction,
        "game_interaction_at_same_index": game_interaction,
        "state_fn": state_fn,
        "state": state_json,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(preview, f, indent=2, ensure_ascii=False)

    print("Preview written to", args.out)
    if not state_fn:
        print("Warning: state JSON not found for game_id/turn by heuristics. Try specifying different turn_idx or check /teach_dataset/images directories.")


if __name__ == "__main__":
    main()
