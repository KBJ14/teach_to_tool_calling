#!/usr/bin/env python3
"""
Extract TEACh dataset into Task->Episode->Turn mapping and compress consecutive driver actions.

- A 'turn' is defined here as a block of consecutive interactions by the agent (driver) that is
  separated by Commander messages/utterances.
-- State snapshot is not included by default; use the `game_id` and the first compressed action timestamp
-- to fetch a pre-turn state separately if required.
- Consecutive actions of the same type are compressed into a single action entry with a repeat `count`.

Usage examples:

python extract_turns.py --data_root /teach_dataset --out-dir /tmp/turns_by_task --compress

"""

import argparse
import json
import os
from glob import glob
try:
    # optional progress bar for long runs
    from tqdm import tqdm
except Exception:
    # fallback: no progress bar
    def tqdm(iterable, **kwargs):
        return iterable
import re


def load_json(fn):
    with open(fn, "r", encoding="utf-8") as f:
        return json.load(f)


# --- Utility functions (copied / adapted from preview_raw_turn.py) ---

def find_edh_files(data_root):
    edh_dir = os.path.join(data_root, "edh_instances")
    if os.path.isdir(edh_dir):
        hits = glob(os.path.join(edh_dir, "**", "*.edh*.json"), recursive=True)
    else:
        hits = glob(os.path.join(data_root, "**", "*.edh*.json"), recursive=True)
    return sorted(hits)


def get_game_id_from_edh(edh_fn, edh_json):
    # Try typical fields
    for key in ("game_id", "structured_log_fn", "orig_game_id", "original_game_fn"):
        if key in edh_json:
            v = edh_json[key]
            if isinstance(v, str) and v.endswith(".game.json"):
                return os.path.splitext(os.path.basename(v))[0]
            if isinstance(v, str):
                return os.path.splitext(os.path.basename(v))[0].split("_")[0]
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
    hits = glob(os.path.join(data_root, "**", f"*{game_id}*.game.json"), recursive=True)
    return hits[0] if hits else None


def find_state_file_for_turn(data_root, game_id, turn_idx):
    patterns = [
        os.path.join(data_root, "images", "**", f"*{game_id}*.json"),
        os.path.join(data_root, "**", f"*{game_id}*state*.json"),
        os.path.join(data_root, "**", f"*{game_id}*_{turn_idx}.json"),
        os.path.join(data_root, "images", "**", "*.json"),
    ]
    candidates = []
    for pat in patterns:
        candidates += glob(pat, recursive=True)
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


_state_index_cache = {}


def _build_state_index_for_game(data_root, game_id, game_key=None):
    """Return list of candidate state files for a game; cached."""
    key = (data_root, game_id)
    if key in _state_index_cache:
        return _state_index_cache[key]

    patterns = [
        os.path.join(data_root, "images_and_states", "**", f"*{game_id}*.json"),
        os.path.join(data_root, "images", "**", f"*{game_id}*.json"),
        os.path.join(data_root, "**", f"*{game_id}*state*.json"),
    ]

    # If a more specific game_key (usually game file base like 0008f3c95e006303_2053) is
    # supplied, add explicit patterns for directories that may be named by game key.
    if game_key:
        patterns += [
            os.path.join(data_root, "images", "**", f"{game_key}", "**", "*.json"),
            os.path.join(data_root, "images", "**", f"*{game_key}*", "*.json"),
            os.path.join(data_root, "**", f"*{game_key}*.json"),
        ]
    candidates = []
    for pat in patterns:
        candidates += glob(pat, recursive=True)
    # remove duplicates and sort
    candidates = sorted(set(candidates))
    _state_index_cache[key] = candidates
    return candidates


def find_state_file_before_time(data_root, game_id, lookup_time):
    """Find a state file whose timestamp is the nearest time <= lookup_time.

    This function loads candidate state JSONs and attempts to extract a timestamp
    from common keys. If no numeric timestamp is found, fallback to file name heuristics.
    Returns a path or None.
    """
    if lookup_time is None:
        return None
    candidates = _build_state_index_for_game(data_root, game_id)
    best_fn = None
    best_ts = None
    for fn in candidates:
        try:
            j = load_json(fn)
        except Exception:
            continue
        # heuristics: try common keys
        ts = None
        for key in ("time_start", "timestamp", "time", "t"):
            v = j.get(key) if isinstance(j, dict) else None
            if isinstance(v, (int, float)):
                ts = float(v)
                break
            # sometimes timestamp is nested under metadata
            if isinstance(j, dict) and "metadata" in j and isinstance(j["metadata"], dict):
                v2 = j["metadata"].get(key)
                if isinstance(v2, (int, float)):
                    ts = float(v2)
                    break
        # If no numeric timestamp, try to parse from filename for an integer suffix
        if ts is None:
            m = re.search(r"_(\d+(?:\.\d+)?)\.json$", fn)
            if m:
                try:
                    ts = float(m.group(1))
                except Exception:
                    ts = None
        if ts is None:
            continue
        # pick the nearest <= lookup_time
        if ts <= lookup_time and (best_ts is None or ts > best_ts):
            best_ts = ts
            best_fn = fn

    return best_fn


# --- Compression and turn extraction logic ---

def compress_actions(actions):
    """
    Compress a list of action dicts by merging consecutive actions with same action_id.
    Adds a 'count' field to indicate repetition.
    - input: list of action dicts as in EDH/preview (each typically contains action_id, action_name, time_start,...)
    - output: list of compressed dicts with optional 'count'
    """

    if not actions:
        return []

    compressed = []
    prev = None
    for a in actions:
        key = (a.get("action_id"), a.get("action_name"))
        if prev is None:
            entry = {**a}
            entry["count"] = 1
            # keep list of times optionally
            entry["time_starts"] = [a.get("time_start")]
            compressed.append(entry)
            prev = (key, entry)
            continue
        curkey, entry = prev
        if key == curkey:
            entry["count"] += 1
            if a.get("time_start") is not None:
                entry["time_starts"].append(a.get("time_start"))
            # record underlying raw success if present
            raw = a.get("raw") or {}
            s = raw.get("success") if isinstance(raw, dict) else None
            entry.setdefault("successes", []).append(s)
        else:
            entry = {**a}
            entry["count"] = 1
            entry["time_starts"] = [a.get("time_start")]
            raw = a.get("raw") or {}
            s = raw.get("success") if isinstance(raw, dict) else None
            entry.setdefault("successes", []).append(s)
            compressed.append(entry)
            prev = (key, entry)

    # You might prefer not to carry 'time_starts' if not needed — keep it for debugging
    # Derive aggregated success fields for convenience
    for e in compressed:
        succs = [x for x in e.get("successes", []) if x is not None]
        if succs:
            e["any_success"] = any(int(x) == 1 for x in succs)
            e["all_success"] = all(int(x) == 1 for x in succs)
            e["last_success"] = int(succs[-1]) == 1
        else:
            e["any_success"] = None
            e["all_success"] = None
            e["last_success"] = None
    return compressed


def extract_raw_turns_from_edh(edh_fn):
    """Extract turns from EDH without compressing or adding state.

    Returns a list of turns where each turn has: start_interaction_idx, end_interaction_idx,
    agent_id, actions (list of raw interaction dicts), commander_context.
    """
    edh = load_json(edh_fn)
    interactions = edh.get("interactions", [])

    turns = []
    i = 0
    last_commander_dialogs = []

    while i < len(interactions):
        cur = interactions[i]
        if cur.get("agent_id", 0) == 0:
            last_commander_dialogs.append(cur)
            i += 1
            continue

        start_i = i
        block = []
        while i < len(interactions) and interactions[i].get("agent_id", 0) != 0:
            block.append(interactions[i])
            i += 1

        turn = {
            "start_interaction_idx": start_i,
            "end_interaction_idx": i - 1,
            "agent_id": block[0].get("agent_id"),
            "actions": block,
            "commander_context": last_commander_dialogs.copy(),
            # Note: state_fn/state removed — state is obtained separately by game_id + timestamps
        }

        # Add pre-turn history: dialogues and actions up to the start of this turn (chronological order).
        # Merge commander_dialogs and previous actions into a single `history` list to provide full
        # dialogue+action context before the turn. This makes it easier for models to consume a single
        # history stream.
        pre = interactions[:start_i]
        turn["history"] = [x for x in pre]  # preserve chronology (both commander and driver events)
        # Keep backwards-compatible fields as well
        turn["pre_turn_dialogs"] = [x for x in pre if x.get("agent_id") == 0]
        turn["pre_turn_actions"] = [x for x in pre if x.get("agent_id") != 0]

        turns.append(turn)
        last_commander_dialogs = []

    return turns


def compress_turns(turns):
    """Compress actions for each turn in place using `compress_actions`.

    Input: turns list as produced by `extract_raw_turns_from_edh`.
    After call, 'actions' in each turn is replaced by compressed actions.
    """
    for t in turns:
        raw_actions = []
        # convert interaction dicts to compact action dicts expected by compress_actions
        for a in t.get("actions", []):
            # EDH interaction may already contain action_name/action_id fields or 'action_name'
            action = {
                "action_id": a.get("action_id") or a.get("action_idx") or a.get("action", {}).get("action_id"),
                "action_name": a.get("action_name") or a.get("action", {}).get("action_name") or a.get("action", {}).get("action"),
                "time_start": a.get("time_start") or a.get("time") or a.get("timestamp"),
                # keep a reference to raw interaction in case caller needs it
                "raw": a,
            }
            raw_actions.append(action)
        # replace 'actions' with compressed form
        t["actions"] = compress_actions(raw_actions)


def add_state_metadata_to_turns(turns, data_root, game_id, include_state=False, game_key=None):
    """Add `state_fn` and `state` to each turn using current start interaction index.

    This function assumes turns have `start_interaction_idx`.
    It prefers to look up a state file by using the **first compressed action's timestamp** as
    the reference; this ensures the state used for the turn reflects the environment immediately
    before that action. If no time-based state file is found, it falls back to the start index
    lookup and then to the previous turn's state (or episode initial state for the first turn).
    """
    # initial state lookup for the episode (fallback for first turn)
    # Try a more specific wildcard for builds that use the game filename base as a folder
    initial_state_fn = None
    if game_key:
        initial_state_fn = _build_state_index_for_game(data_root, game_id, game_key)
        initial_state_fn = initial_state_fn[0] if initial_state_fn else None
    if not initial_state_fn:
        initial_state_fn = find_state_file_for_turn(data_root, game_id, 0)

    for i, t in enumerate(turns):
        # prefer the compressed first action's time for lookup (as requested)
        state_fn = None
        first_time = None
        if t.get("actions"):
            # first compressed action
            first_act = t["actions"][0]
            # try the earliest timestamp from compressed action
            if first_act.get("time_starts"):
                first_time = first_act.get("time_starts")[0]
            else:
                first_time = first_act.get("time_start")

        if first_time is not None:
            state_fn = find_state_file_before_time(data_root, game_id, first_time)
            # If we didn't find a good candidate and a game_key was supplied, try building a more
            # directed index for this game_key and rerun the nearest lookup on that subset.
            if not state_fn and game_key:
                candidates = _build_state_index_for_game(data_root, game_id, game_key)
                # try a nearest <= lookup_time pass but only for these candidates
                best_fn = None
                best_ts = None
                for fn in candidates:
                    try:
                        j = load_json(fn)
                    except Exception:
                        continue
                    ts = None
                    for key in ("time_start", "timestamp", "time", "t"):
                        v = j.get(key) if isinstance(j, dict) else None
                        if isinstance(v, (int, float)):
                            ts = float(v)
                            break
                        if isinstance(j, dict) and "metadata" in j and isinstance(j["metadata"], dict):
                            v2 = j["metadata"].get(key)
                            if isinstance(v2, (int, float)):
                                ts = float(v2)
                                break
                    if ts is None:
                        m = re.search(r"_(\d+(?:\.\d+)?)\.json$", fn)
                        if m:
                            try:
                                ts = float(m.group(1))
                            except Exception:
                                ts = None
                    if ts is None:
                        continue
                    if ts <= first_time and (best_ts is None or ts > best_ts):
                        best_ts = ts
                        best_fn = fn
                state_fn = best_fn

        # fallback to start index based if time-based lookup fails
        if not state_fn:
            state_fn = find_state_file_for_turn(data_root, game_id, t.get("start_interaction_idx"))
        # If there's no state file for this specific start index, fall back to the previous
        # turn's state (which is the environment after the previous turn), or the episode
        # initial state if no previous turn.
        if not state_fn:
            if i > 0:
                state_fn = turns[i-1].get("state_fn")
            else:
                state_fn = initial_state_fn
        t["state_fn"] = state_fn
        if include_state and state_fn and os.path.isfile(state_fn):
            try:
                t["state"] = load_json(state_fn)
            except Exception:
                t["state"] = None


def extract_turns_from_edh(edh_fn, data_root, compress=True, include_state=False, compress_first=True, game_file=None):
    """Backward-compatible wrapper: extract turns, optionally compress then add state metadata."""
    # get game id and raw turns
    edh = load_json(edh_fn)
    game_id = get_game_id_from_edh(edh_fn, edh)

    raw_turns = extract_raw_turns_from_edh(edh_fn)

    # compress prior to metadata lookup if requested
    if compress and compress_first:
        compress_turns(raw_turns)

    # We no longer automatically add state metadata — callers can use game_id and
    # the first compressed action time to look up the proper state externally.

    # compress after metadata lookup if requested but compress_first==False
    if compress and not compress_first:
        compress_turns(raw_turns)

    return raw_turns


def iter_edh_and_extract_all(data_root, compress=True, include_state=False, limit=None, compress_first=True):
    results = []
    edh_files = find_edh_files(data_root)
    if limit is not None:
        edh_files = edh_files[:limit]

    for edh_fn in tqdm(edh_files, desc="Processing EDH", unit="file"):
        edh = load_json(edh_fn)
        game_id = get_game_id_from_edh(edh_fn, edh)
        game_file = find_game_file_for_gameid(data_root, game_id)
        game = load_json(game_file) if game_file else None

        # get task/episode indexes from edh
        tidx = edh.get("task_idx")
        eidx = edh.get("episode_idx")

        turns = extract_turns_from_edh(edh_fn, data_root, compress=compress, include_state=include_state, compress_first=compress_first, game_file=game_file)

        results.append(
            {
                "edh_fn": edh_fn,
                "game_id": game_id,
                "game_fn": game_file,
                "task_idx": tidx,
                "episode_idx": eidx,
                "turns": turns,
            }
        )
    return results


def compute_stats(results):
    """Compute dataset-level statistics from iteration results."""
    stats = {
        "num_edh": 0,
        "num_tasks": 0,
        "num_episodes": 0,
        "num_turns": 0,
        "turns_per_edh": [],
        "episodes_per_task": {},
        "action_counts": {},
    }

    stats["num_edh"] = len(results)
    task_episode_pairs = set()
    for r in results:
        stats["num_turns"] += len(r.get("turns", []))
        stats["turns_per_edh"].append(len(r.get("turns", [])))
        tidx = r.get("task_idx")
        eidx = r.get("episode_idx")
        if tidx is None:
            tidx = "unknown"
        if eidx is None:
            eidx = "unknown"
        task_episode_pairs.add((tidx, eidx))
        stats["episodes_per_task"].setdefault(str(tidx), set()).add(str(eidx))

        # count compressed actions
        for t in r.get("turns", []):
            for act in t.get("actions", []):
                name = act.get("action_name") or str(act.get("action_id"))
                stats["action_counts"][name] = stats["action_counts"].get(name, 0) + (act.get("count", 1) or 1)

    stats["num_tasks"] = len(stats["episodes_per_task"])
    stats["num_episodes"] = len(task_episode_pairs)
    # convert sets to counts
    stats["episodes_per_task"] = {k: len(v) for k, v in stats["episodes_per_task"].items()}
    stats["mean_turns_per_edh"] = float(sum(stats["turns_per_edh"]) / max(1, len(stats["turns_per_edh"])))
    # most frequent actions
    stats["top_actions"] = sorted(stats["action_counts"].items(), key=lambda x: x[1], reverse=True)[:20]
    return stats


def write_grouped_by_task_episode(results, out_dir):
    """Write mapping task_idx -> episode_idx -> list(edh) into files under out_dir.

    Directory format:
      out_dir/task_<task_idx>/episode_<episode_idx>.json
    If multiple EDH for same episode exist, they are appended to the JSON array in that file.
    """
    os.makedirs(out_dir, exist_ok=True)
    mapping = group_by_task_episode(results)
    for tidx, episodes in mapping.items():
        dir_task = os.path.join(out_dir, f"task_{tidx}")
        os.makedirs(dir_task, exist_ok=True)
        for eidx, entries in episodes.items():
            out_fn = os.path.join(dir_task, f"episode_{eidx}.json")
            with open(out_fn, "w", encoding="utf-8") as f:
                json.dump(entries, f, indent=2, ensure_ascii=False)



def group_by_task_episode(results):
    """Return mapping: task_idx -> episode_idx -> list of edh entries with turns."""
    mapping = {}
    # Assign small sequential numeric names for unknown task/episode entries
    unknown_task_map = {}
    unknown_task_counter = 0
    unknown_ep_counters = {}

    for r in results:
        raw_tidx = r.get("task_idx")
        raw_eidx = r.get("episode_idx")

        if raw_tidx is None:
            # use (game_id, episode_idx) as a stable key when task_idx is missing
            key = (r.get("game_id"), raw_eidx)
            if key not in unknown_task_map:
                unknown_task_map[key] = unknown_task_counter
                unknown_task_counter += 1
            tidx = str(unknown_task_map[key])
        else:
            tidx = str(raw_tidx)

        # handle unknown episode indices: use per-task sequential numbering
        if raw_eidx is None:
            unknown_ep_counters.setdefault(tidx, 0)
            eidx = str(unknown_ep_counters[tidx])
            unknown_ep_counters[tidx] += 1
        else:
            eidx = str(raw_eidx)

        mapping.setdefault(tidx, {})
        mapping[tidx].setdefault(eidx, []).append(r)
    return mapping


# --- CLI ---


def main():
    p = argparse.ArgumentParser(description="Extract TEACh episodes into turns and compress consecutive driver actions")
    p.add_argument("--data_root", default="/teach_dataset", help="Path to TEACh root data folder")
    # Single-file output removed. Require --out-dir for per-task/episode split output.
    p.add_argument("--out-dir", required=True, help="Directory to save per-task/episode JSONs (required)")
    p.add_argument("--grouped-out", default=None, help="Optional JSON filename for grouped by task->episode")
    p.add_argument("--compress", action="store_true", help="Compress consecutive actions")
    # --include-state is deprecated; state is now resolved separately by ID+timestamp
    p.add_argument("--limit", type=int, default=None, help="Limit number of EDH files to process")
    p.add_argument("--stats-out", default=None, help="Optional filename to write dataset stats JSON")
    p.add_argument("--no-compress-first", dest="compress_first", action="store_false",
                   help="Do not compress actions before metadata lookup (default: compress first)")
    p.set_defaults(compress_first=True)
    args = p.parse_args()

    print("Scanning EDH files under", args.data_root)
    results = iter_edh_and_extract_all(args.data_root, compress=args.compress, include_state=False, limit=args.limit, compress_first=args.compress_first)

    print("Done. Found turns for %d EDH files." % len(results))
    if args.grouped_out:
        grouped = group_by_task_episode(results)
        with open(args.grouped_out, "w", encoding="utf-8") as f:
            json.dump(grouped, f, indent=2, ensure_ascii=False)
        print("Grouped data written to", args.grouped_out)
    print("Writing grouped per-task/episode JSONs to", args.out_dir)
    write_grouped_by_task_episode(results, args.out_dir)
    print("Wrote per-task/episode data to", args.out_dir)

    stats = compute_stats(results)
    # print a small summary
    print("Dataset stats: edh=%d tasks=%d episodes=%d turns=%d mean_turns/edh=%.2f" % (
        stats["num_edh"], stats["num_tasks"], stats["num_episodes"], stats["num_turns"], stats["mean_turns_per_edh"]))
    if args.stats_out:
        with open(args.stats_out, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print("Stats written to", args.stats_out)


if __name__ == "__main__":
    main()
