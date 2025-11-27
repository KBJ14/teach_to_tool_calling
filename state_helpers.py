import os
import glob
import json
import re
from typing import Optional, Tuple, List


def load_json(fn: str):
    """Load JSON from file with a helpful error message."""
    with open(fn, "r") as f:
        return json.load(f)


def _parse_float_tokens(s: str) -> List[float]:
    """Find all float-looking tokens in a filename like 'statediff.302.36096763.json' and return as floats.

    Returns list of floats in the order they appear.
    """
    # find tokens with decimals first
    floats = [float(m.group(0)) for m in re.finditer(r"\d+\.\d+", s)]
    # also accept integers (fall back, e.g., 'statediff.302')
    ints = [int(m.group(0)) for m in re.finditer(r"(?<!\.)\b\d+\b", s)]
    # ints will include floats' integer parts; remove those that are part of floats
    if floats:
        # remove int tokens that are substrings of floats
        float_strs = set(m.group(0).split(".")[0] for m in re.finditer(r"\d+\.\d+", s))
        ints = [x for x in ints if str(x) not in float_strs]
    # combine. prefer floats then ints
    return floats + ints


def parse_statediff_timestamp(fname: str) -> Optional[float]:
    """Return a timestamp float associated with a statediff file name.

    - 'statediff.end.json' -> returns float('inf')
    - tries to find last float token in filename
    - otherwise tries integer tokens
    - returns None if no numeric tokens present
    """
    base = os.path.basename(fname)
    if base.startswith("statediff.end") or base.endswith("statediff.end.json"):
        return float("inf")
    # remove extension
    key = os.path.splitext(base)[0]
    # get tokens after the staged 'statediff.' prefix
    if key.startswith("statediff."):
        key_body = key[len("statediff.") :]
    else:
        key_body = key
    numbers = _parse_float_tokens(key_body)
    if not numbers:
        # nothing numeric in the name -> None
        return None
    # prefer the last numeric token (commonly timestamp)
    return float(numbers[-1])


def _find_game_folder(images_dir: str, game_fn_or_id: str) -> Optional[str]:
    """Try to find the subdirectory in `images_dir` that corresponds to the game.

    `game_fn_or_id` can be a full `game_fn` or just the `game_id` or a basename like
    '0008f3c95e006303_2053'. We'll search recursively under images_dir for a directory with
    that basename.
    """
    # If it's a path to a game json, extract basename without extension
    candidate = os.path.splitext(os.path.basename(game_fn_or_id))[0]
    # Candidate might already equal the folder name; check direct
    direct = os.path.join(images_dir, candidate)
    if os.path.isdir(direct):
        return direct
    # Otherwise search recursively for a folder name that equals candidate
    pattern = os.path.join(images_dir, "**", candidate)
    matches = glob.glob(pattern, recursive=True)
    if matches:
        return matches[0]
    # If candidate is just a game id and the folder names include suffixes like _134b,
    # try to find a path that startswith the game_id
    base_id = candidate.split("_")[0]
    pattern2 = os.path.join(images_dir, "**", base_id + "*")
    matches2 = glob.glob(pattern2, recursive=True)
    if matches2:
        return matches2[0]
    return None


def find_state_for_time(images_dir: str, game_folder: str, target_time: float, pattern: str = "statediff.*.json") -> Optional[str]:
    """Find the statediff file inside `game_folder` whose timestamp is closest **<=** target_time.

    If `target_time` is None, returns the earliest statediff (or statediff.end if present). Returns path or None.
    """
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"images_dir not found: {images_dir}")
    # allow passing either absolute game_folder or basename
    abs_game_dir = game_folder if os.path.isabs(game_folder) else os.path.join(images_dir, game_folder)
    if not os.path.isdir(abs_game_dir):
        # fallback: try searching for a directory matching that basename
        found = _find_game_folder(images_dir, game_folder)
        if not found:
            return None
        abs_game_dir = found

    candidates = glob.glob(os.path.join(abs_game_dir, pattern))
    if not candidates:
        return None

    # try exact match for the target_time if present
    if target_time is not None:
        exact = os.path.join(abs_game_dir, f"statediff.{target_time}.json")
        if os.path.exists(exact):
            return exact

    # Build list of (time, path) and select the closest <= target
    entries: List[Tuple[float, str]] = []
    for c in candidates:
        t = parse_statediff_timestamp(c)
        if t is None:
            continue
        entries.append((t, c))

    # parsed candidate timestamps available in `entries`

    if not entries:
        return None

    # sort by timestamp
    entries.sort(key=lambda x: x[0])
    if target_time is None:
        # return earliest statediff, except prefer statediff.end
        for t, p in entries:
            if os.path.basename(p).startswith("statediff.end"):
                return p
        return entries[0][1]

    # else pick the largest timestamp <= target_time
    le = [e for e in entries if e[0] <= target_time]
    if le:
        return le[-1][1]

    # if none <= target_time, fallback to the earliest (or end)
    for t, p in entries:
        if os.path.basename(p).startswith("statediff.end"):
            return p
    return entries[0][1]


def find_state_for_interaction(ep: dict, turn_idx: int = 0, images_dir: str = "/teach_dataset/images", edh_idx: int = 0, state_suffixes: List[str] = None) -> Optional[str]:
    """Find and return a statediff JSON path that corresponds to the given turn index in an episode file.

    Parameters:
    - ep: Loaded episode JSON. This may be either a list of EDH entries or a single EDH dict.
    - turn_idx: index of the turn (in `turns`) for which to find state.
    - images_dir: root images directory where `game_folder` lives
    - edh_idx: if `ep` is a list of EDH items, which one to select.

    Returns: path to statediff file, or None
    """
    # Normalize ep to a single EDH entry
    entry = None
    if isinstance(ep, list):
        if edh_idx < 0 or edh_idx >= len(ep):
            raise IndexError("edh_idx out of range for episode file")
        entry = ep[edh_idx]
    elif isinstance(ep, dict):
        entry = ep
    else:
        raise ValueError("ep must be a dict or list of dicts")

    turns = entry.get("turns", [])
    if turn_idx < 0 or turn_idx >= len(turns):
        raise IndexError("turn_idx out of range for episode turns")

    turn = turns[turn_idx]
    # prefer first compressed action's first timestamp
    actions = turn.get("actions", [])
    target_time = None
    if actions:
        # some outputs use `time_start` or `time_starts` plural
        a0 = actions[0]
        # Use time_starts array if available
        if a0.get("time_starts"):
            target_time = a0["time_starts"][0]
        elif a0.get("time_start"):
            target_time = a0["time_start"]

    # game folder: use game_fn or game_id to find images directory
    game_fn = entry.get("game_fn") or entry.get("game_id") or entry.get("edh_fn")
    if not game_fn:
        raise ValueError("Could not determine game folder from ep entry")
    game_folder = os.path.splitext(os.path.basename(game_fn))[0]

    # find state file
    sf = find_state_for_time(images_dir, game_folder, target_time)
    # attach to the turn for convenience
    if sf:
        try:
            turn["state_fn"] = sf
            # optional: populate state JSON (may be large) if desired
            with open(sf, "r") as f:
                turn["state"] = json.load(f)
        except Exception:
            # if JSON loading failed, still return path
            pass
    return sf
