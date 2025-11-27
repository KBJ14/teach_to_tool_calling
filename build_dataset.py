import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Try to import TEACh utilities for accurate task checks. If unavailable, we'll fall back to
# the simpler action-flag heuristics.
try:
    from teach.utils import create_task_thor_from_state_diff, update_objs_with_custom_metadata
    _HAS_TEACH_UTILS = True
except Exception:
    _HAS_TEACH_UTILS = False
import argparse


# ---------- CONFIG ----------

REPO_ROOT = Path("/home/bjk/tool_learning/teach_to_tool_calling")
EPISODE_ROOT = REPO_ROOT / "episode_data_wo_state"   # task_*/episode_*.json
TOOLS_JSON = REPO_ROOT / "dataset/prompts/tools.json"
DATASET_OUTPUT = REPO_ROOT / "dataset/teach_robot_tools_train.jsonl"

TEACH_ROOT = Path("/teach_dataset")  # game / edh / images 경로의 공통 prefix

# ---------- BASE PROMPT TEMPLATE (고정) ----------

BASE_PROMPT = """
You are a high-level robot control assistant in a simulated home environment.

The following information describes the current context:

Initial scene state:
{initial_state}

Dialogue history so far:
{dialogue_history}

Previous tool calls executed:
{previous_actions}

State changes after previous actions (state_diff):
{state_diff}

Available robot control tools:
{tool_list}

Your task:
Based on the latest Commander utterance in the dialogue history and the context above,
decide the next robot actions.

Output format:
Return ONLY a JSON object with:
- "actions": a list of tool calls to execute in sequence.
- Each action includes:
  - "tool_name": string
  - "parameters": object (empty if the tool has no parameters)

If the appropriate response is pure natural language and no tool use is required,
return "actions": [].

Example output format:
 {{
     "actions": [
         {{
             "tool_name": "motion_delta",
             "parameters": {{
                 "x": dx,
                 "y": dy,
                 "z": dz,
                 "rot_x": drot_x,
                 "rot_y": drot_y,
                 "rot_z": drot_z
             }}
         }}
     ]
 }}
"""


# ---------- PARSE ARGS UTILS ----------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--episode-root",
        type=str,
        required=True,
        help="episode_data_wo_state 루트 경로"
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="생성된 데이터셋을 저장할 루트 경로"
    )

    parser.add_argument(
        "--task-ids",
        type=str,
        default=None,
        help="쉼표로 구분된 task id들 (예: '0' 또는 '0,1,3')"
    )

    return parser.parse_args()


# ---------- IO UTILS ----------

def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _round_floats_in_obj(obj: Any, ndigits: int = 2) -> Any:
    """
    Recursively walk a JSON-like structure (dict/list/primitive) and round
    floats to `ndigits` decimal places. Returns a new object with the same
    structure but with floats rounded. Non-float primitives are left as-is.

    Note: This function returns lists/dicts by value (mutable copies) so it's
    safe to use on parsed JSON structures before further processing.
    """
    # primitives
    if isinstance(obj, float):
        # round float to given precision
        # keep as float (e.g. 1.0 -> 1.0)
        return round(obj, ndigits)

    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[k] = _round_floats_in_obj(v, ndigits)
        return out

    if isinstance(obj, list):
        return [_round_floats_in_obj(e, ndigits) for e in obj]

    # ints, strings, bools, None remain unchanged
    return obj


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------- TOOLS + MAPPINGS ----------

def load_tools_and_mappings(tools_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    data = read_json(tools_path)
    tools = data["tools"]
    mappings = data["mappings"]["action_idx_to_tool"]
    return tools, mappings


def get_tool_schema_by_action_id(tools: List[Dict[str, Any]], tool_action_id: int) -> Optional[Dict[str, Any]]:
    for t in tools:
        if t.get("action_id") == tool_action_id:
            return t
    return None


def get_tool_schema_by_name(tools: List[Dict[str, Any]], tool_name: str) -> Optional[Dict[str, Any]]:
    """Return the tool schema dict for a given tool name, or None if not found."""
    for t in tools:
        if t.get("name") == tool_name:
            return t
    return None


# ---------- STATEDIFF 처리 ----------

def parse_statediff_time(path: Path) -> Optional[float]:
    """
    statediff.200.98502206802368.json -> 200.98502206802368
    """
    name = path.name  # statediff.200.9850.json
    if not name.startswith("statediff."):
        return None
    core = name[len("statediff."):]
    if core.endswith(".json"):
        core = core[:-len(".json")]
    try:
        return float(core)
    except ValueError:
        return None


def load_statediff_index(images_dir: Path) -> List[Tuple[float, Path]]:
    if not images_dir.exists():
        return []
    entries: List[Tuple[float, Path]] = []
    for p in images_dir.glob("statediff*.json"):
        t = parse_statediff_time(p)
        if t is not None:
            entries.append((t, p))
    entries.sort(key=lambda x: x[0])
    return entries


# ----- Utilities for selecting initial / final state files and filtering -----
def _extract_object_keys_from_diff(diff: Dict[str, Any]) -> List[str]:
    """
    Collect object keys (objectId or name) that appear in a final_state_diff's objects
    so we can filter the initial_state objects to only those that matter.
    """
    obj_keys: List[str] = []

    objects = diff.get("objects")
    if objects is None:
        return obj_keys

    if isinstance(objects, dict):
        obj_keys.extend(list(objects.keys()))
    elif isinstance(objects, list):
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            oid = obj.get("objectId") or obj.get("name")
            if oid:
                obj_keys.append(oid)

    return sorted(set(obj_keys))


def _filter_initial_objects(initial_objects: Any, keys_of_interest: List[str]) -> Any:
    """
    From initial_state["objects"], keep only objects that appear in keys_of_interest.
    Works with both dict and list structures.
    """
    if not keys_of_interest:
        return initial_objects

    key_set = set(keys_of_interest)

    if isinstance(initial_objects, dict):
        return {k: v for k, v in initial_objects.items() if k in key_set}

    if isinstance(initial_objects, list):
        filtered = []
        for obj in initial_objects:
            if not isinstance(obj, dict):
                continue
            oid = obj.get("objectId") or obj.get("name")
            if oid in key_set:
                filtered.append(obj)
        return filtered

    return initial_objects


def derive_keys_of_interest_from_turns(turns_list: List[Dict[str, Any]]) -> List[str]:
    """
    Walk through episode turns/actions and collect object ids (oids/objectId/etc.)
    that represent objects directly acted upon. Return a sorted list of unique keys.
    """
    keys = set()
    for t in turns_list:
        for agg in (t.get("actions", []) or []):
            raw = agg.get("raw", {}) or {}
            # Common fields that refer to object ids
            for kname in ("oid", "objectId", "object_id"):
                val = raw.get(kname)
                if not val:
                    continue
                if isinstance(val, list):
                    for v in val:
                        if v:
                            keys.add(str(v))
                else:
                    keys.add(str(val))

            # arrays or nested dicts (safely search a few likely shapes)
            for key, v in raw.items():
                if isinstance(v, list):
                    for e in v:
                        if isinstance(e, str) and "|" in e:
                            keys.add(e)
                        if isinstance(e, dict):
                            if e.get("objectId"):
                                keys.add(str(e.get("objectId")))
                            if e.get("oid"):
                                keys.add(str(e.get("oid")))
                elif isinstance(v, dict):
                    if v.get("objectId"):
                        keys.add(str(v.get("objectId")))
                    if v.get("oid"):
                        keys.add(str(v.get("oid")))

    return sorted(keys)


def build_filtered_initial_state(
    initial_state_path: Optional[Path], final_state_diff_path: Optional[Path]
) -> Optional[Dict[str, Any]]:
    """
    Read initial_state.json and final_state_diff.json and return a filtered
    initial_state containing only the objects that appear in the final diff.
    Returns None if either path is missing.
    """
    if initial_state_path is None or final_state_diff_path is None:
        return None

    if not initial_state_path.exists():
        print(f"[WARN] initial_state missing: {initial_state_path}")
        return None
    if not final_state_diff_path.exists():
        print(f"[WARN] final_state_diff missing: {final_state_diff_path}")
        return None

    init = read_json(initial_state_path)
    final = read_json(final_state_diff_path)

    keys = _extract_object_keys_from_diff(final)
    filtered_objects = _filter_initial_objects(init.get("objects"), keys)

    out = {
        "time_start": init.get("time_start"),
        "agents": init.get("agents"),
        "objects": filtered_objects,
    }

    # Round any float values in the returned initial_state for compactness
    return _round_floats_in_obj(out, ndigits=2)


def derive_state_paths_from_edh_fn(edh_fn: Optional[str], split: Optional[str] = None) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Given an edh filename (e.g. '/teach_dataset/edh/train/0008f3c95e006303_2053.edh0.json'),
    derive the initial_state.json and the final statediff.*.json for the scene.

    If `split` is provided ("train"/"val"/"test"), prefer TEACH_ROOT/images/{split}/{scene_id}.
    Otherwise, search under TEACH_ROOT/images/*/{scene_id}.
    We pick the final statediff file as the one with the highest frame index.
    """
    if not edh_fn:
        return None, None

    edh_path = Path(edh_fn)
    stem = edh_path.stem
    if ".edh" in stem:
        scene_id = stem.split(".edh")[0]
    else:
        scene_id = stem

    # Try the given split first
    roots: List[Path] = []
    if split:
        roots.append(TEACH_ROOT / "images" / split)
    # fallback: search all splits under images
    roots.append(TEACH_ROOT / "images")

    initial_state_path: Optional[Path] = None
    final_state_diff_path: Optional[Path] = None
    max_frame = -1

    for r in roots:
        state_dir = r / scene_id
        if not state_dir.exists():
            continue
        cand_init = state_dir / "initial_state.json"
        if cand_init.exists():
            initial_state_path = cand_init

        for cand in state_dir.glob("statediff.*.json"):
            parts = cand.name.split(".")
            if len(parts) < 3:
                continue
            try:
                frame = int(parts[1])
            except ValueError:
                continue
            if frame > max_frame:
                max_frame = frame
                final_state_diff_path = cand

        # if we found anything, stop scanning other candidate roots
        if initial_state_path or final_state_diff_path:
            break

    return initial_state_path, final_state_diff_path


def find_nearest_statediff(t: float, index: List[Tuple[float, Path]]) -> Optional[Path]:
    """
    t 시각과 가장 가까운 statediff.*.json 을 선택.
    필요하면 여기서 't 이하만 허용' 같은 정책으로 바꿔도 됨.
    """
    if not index:
        return None
    best_path = None
    best_dist = math.inf
    for tt, p in index:
        d = abs(tt - t)
        if d < best_dist:
            best_dist = d
            best_path = p
    return best_path


# ---------- STATE (pose + objects) BEFORE TURN k ----------

def build_state_before_turn(
    turn_idx: int,
    turns: List[Dict[str, Any]],
    statediff_index: List[Tuple[float, Path]],
    game_json: Dict[str, Any],
    keys_of_interest: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    turn_idx 번째 턴을 예측할 때 LLM에게 보여줄 '현재 상태'를 계산.

    - turn_idx == 0:
        game_json["tasks"][0]["episodes"][0]["initial_state"] 사용
    - turn_idx > 0:
        바로 이전 턴(turn_idx-1)의 마지막 action 이후 상태
        (pose + 그 시각과 가장 가까운 statediff.*.json 의 objects)
    """
    # Always include the episode's initial_state for the turn
    initial_state = game_json["tasks"][0]["episodes"][0].get("initial_state", {})
    # If requested, keep only the objects that are in keys_of_interest
    if keys_of_interest:
        try:
            initial_state_objects = initial_state.get("objects")
            filtered = _filter_initial_objects(initial_state_objects, keys_of_interest)
            initial_state = {**initial_state, "objects": filtered}
        except Exception:
            pass

    # Default values
    pose_after_prev = None
    last_time_prev = None

    # If first turn, there are no prior actions, so no state_diff
    if turn_idx == 0:
        # Round floats in the initial state before returning for the dataset
        return {
            "initial_state": _round_floats_in_obj(initial_state, ndigits=2),
            "pose": None,
            "state_diff": [],
        }

    # 2) 그 외: 이전 턴 기반

        
    prev_turn = turns[turn_idx - 1]
    prev_actions = prev_turn.get("actions", [])

    pose_after_prev = None
    last_time_prev = None

    for agg in prev_actions:
        raw = agg.get("raw", {})

        # pose: 가장 최신 raw.pose
        if "pose" in raw:
            pose_after_prev = raw["pose"]

        # 시간 후보들 모으기 (압축 action 고려)
        t_candidates: List[float] = []
        time_starts = agg.get("time_starts")
        if time_starts:
            t_candidates.extend(float(t) for t in time_starts)

        if agg.get("time_start") is not None:
            t_candidates.append(float(agg["time_start"]))
        if raw.get("time_start") is not None:
            t_candidates.append(float(raw["time_start"]))

        if t_candidates:
            t_max = max(t_candidates)
            if (last_time_prev is None) or (t_max > last_time_prev):
                last_time_prev = t_max


    # state_diff: pick the single statediff entry nearest to last_time_prev
    state_diff: Optional[Dict[str, Any]] = None
    if last_time_prev is None:
        # no action timestamps found in previous actions
        pass

    if last_time_prev is not None and statediff_index:
        # choose the latest statediff with timestamp <= last_time_prev
        sd_path = None
        # choose the latest statediff with timestamp <= last_time_prev
        for tt, p in statediff_index:
            if tt <= last_time_prev:
                sd_path = p
            else:
                break
        if sd_path is not None:
            # selected statediff path available in sd_path
            try:
                state_diff = read_json(sd_path)
                # Filter statediff objects if we have a keys_of_interest list
                if keys_of_interest and isinstance(state_diff, dict):
                    try:
                        objs = state_diff.get("objects")
                        if isinstance(objs, dict):
                            allowed = set(keys_of_interest)
                            state_diff["objects"] = {k: v for k, v in objs.items() if k in allowed}
                    except Exception:
                        pass
            except Exception:
                state_diff = None

    if last_time_prev is not None and not statediff_index:
        # last_time_prev available but no statediff candidates were found
        pass

    # If agents are missing in the selected statediff (or statediff absent), try to
    # synthesize an agent entry from the previous action pose relative to the
    # episode initial_state agent.
    if pose_after_prev is not None:
        agents_present = False
        if isinstance(state_diff, dict) and state_diff.get("agents"):
            agents_present = True

        if not agents_present:
            try:
                init_agents = game_json["tasks"][0]["episodes"][0].get("initial_state", {}).get("agents", [])
            except Exception:
                init_agents = []

            init_agent = None
            if isinstance(init_agents, list):
                for cand in init_agents:
                    if isinstance(cand, dict) and (cand.get("name") == "agent" or cand.get("cameraHorizon") is not None or cand.get("isStanding") is not None):
                        init_agent = cand
                        break

            if init_agent is not None:
                ipos = init_agent.get("position", {})
                irot = init_agent.get("rotation", {})
                cam_horizon = init_agent.get("cameraHorizon", None)

                init_pose = [
                    float(ipos.get("x", 0.0)),
                    float(ipos.get("y", 0.0)),
                    float(ipos.get("z", 0.0)),
                    float(irot.get("x", 0.0)),
                    float(cam_horizon if cam_horizon is not None else irot.get("y", 0.0)),
                    float(irot.get("y", irot.get("z", 0.0))),
                ]

                pap = pose_after_prev
                if isinstance(pap, list) and len(pap) >= 6:
                    pap6 = [float(x) for x in pap[:6]]
                    pose_delta = [pap6[i] - init_pose[i] for i in range(6)]

                    agent_diff = {
                        "initial_pose": init_pose,
                        "pose_after_prev": pap6,
                        "pose_delta": pose_delta,
                        "position": {"x": pap6[0], "y": pap6[1], "z": pap6[2]},
                        "rotation": {"x": pap6[3], "cameraHorizon": pap6[4], "yaw": pap6[5]},
                    }

                    if state_diff is None:
                        state_diff = {"agents": {"agent": agent_diff}, "objects": {}}
                    else:
                        if not isinstance(state_diff.get("agents"), dict):
                            state_diff["agents"] = {}
                        state_diff["agents"].setdefault("agent", agent_diff)

    # Round floats in the returned structures before dataset creation
    rounded_initial = _round_floats_in_obj(initial_state, ndigits=2)
    rounded_pose = None
    if isinstance(pose_after_prev, list):
        rounded_pose = _round_floats_in_obj(pose_after_prev, ndigits=2)
    rounded_state_diff = _round_floats_in_obj(state_diff or {}, ndigits=2)

    return {
        "initial_state": rounded_initial,
        "pose": rounded_pose,
        "state_diff": rounded_state_diff,
    }


# ---------- ACTION → TOOL 변환 ----------

def convert_aggregated_action_to_tool_calls(
    agg: Dict[str, Any],
    tools: List[Dict[str, Any]],
    mappings: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    episode_data_wo_state 의 aggregated action 하나를 tool list로 변환.
    - motion류: motion_delta 하나로 합치되, pose_delta * count 로 net delta 계산
    - 나머지: count와 상관 없이 1개 tool로 표현 (pickup, place 등)
    """
    raw = agg.get("raw", {})
    # action_idx 우선 사용, fallback 으로 action_id 사용
    idx = raw.get("action_idx", agg.get("action_idx", agg.get("action_id")))
    if idx is None:
        # Missing action index: nothing to map to tools
        return []

    idx_str = str(idx)
    if idx_str not in mappings:
        # 매핑 없는 action은 스킵 (필요시 로그 추가 가능)
        return []

    map_info = mappings[idx_str]
    tool_name = map_info["tool_name"]
    tool_action_id = map_info["tool_action_id"]

    tool_schema = get_tool_schema_by_action_id(tools, tool_action_id)
    has_params = bool(tool_schema and tool_schema.get("parameters"))

    count = agg.get("count", 1) or 1

    tool_calls: List[Dict[str, Any]] = []

    if tool_name == "motion_delta":
        # pose_delta: [dx, dy, dz, drot_x, drot_y, drot_z]
        pose_delta = raw.get("pose_delta") or [0, 0, 0, 0, 0, 0]
        if len(pose_delta) != 6:
            # 안전하게 길이 보정
            pose_delta = (pose_delta + [0, 0, 0, 0, 0, 0])[:6]

        # 연속된 motion action이면 하나로 합침: pose_delta * count
        total_delta = [pose_delta[i] * count for i in range(6)]
        tool_calls.append({
            "tool_name": tool_name,
            "parameters": {
                "x": total_delta[0],
                "y": total_delta[1],
                "z": total_delta[2],
                "rot_x": total_delta[3],
                "rot_y": total_delta[4],
                "rot_z": total_delta[5],
            }
        })
    else:
        # Parameter 없는 tool 이라면 빈 object
        if not has_params:
            tool_calls.append({
                "tool_name": tool_name,
                "parameters": {}
            })
        else:
            # 나중에 parameter 있는 ProgressCheck 등 다룰 때 확장
            # 일단은 raw 정보로부터 적당히 복사하거나, 필요 없으면 빈 object
            tool_calls.append({
                "tool_name": tool_name,
                "parameters": {}
            })

    return tool_calls


def convert_turn_actions_to_tools(
    turn_actions: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    mappings: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    하나의 turn 안에 있는 aggregated actions 전체를 tool list로 변환.
    """
    tool_list: List[Dict[str, Any]] = []
    for agg in turn_actions:
        tool_list.extend(convert_aggregated_action_to_tool_calls(agg, tools, mappings))
    return tool_list


# ---------- HISTORY 구축 ----------

def build_history_until_turn(
    edh_json: Dict[str, Any],
    turns: List[Dict[str, Any]],
    turn_idx: int,
    tools: List[Dict[str, Any]],
    mappings: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    간단 버전:
    - edh["dialog_history_cleaned"] 전체를 dialogue history로 사용
    - 이전 turn(0 ~ turn_idx-1)의 action들을 tool list로 변환해 history에 추가
    """
    history: List[Dict[str, Any]] = []

    # Build dialogue history only up to (but not including) the current turn.
    # Prefer turn-local fields (pre_turn_dialogs, commander_context) and action utterances.
    for t in turns[:turn_idx]:
        # commander utterances that happened right before this turn
        for cand in (t.get("pre_turn_dialogs") or t.get("commander_context") or t.get("history") or []):
            if not isinstance(cand, dict):
                continue
            # candidate object may have different keys depending on source
            role = "Commander" if int(cand.get("agent_id", 0)) == 0 else "Driver"
            utt = cand.get("corrected_utterance") or cand.get("utterance") or cand.get("query")
            if utt:
                history.append({
                    "type": "dialogue",
                    "role": role,
                    "utterance": utt,
                })

        # driver utterances from actions in this turn
        for agg in t.get("actions", []):
            raw = agg.get("raw", {})
            utt = raw.get("corrected_utterance") or raw.get("utterance")
            if utt:
                history.append({
                    "type": "dialogue",
                    "role": "Driver",
                    "utterance": utt,
                })

    # 2) 이전 turn들의 action history (unchanged behaviour) - only insert Driver turn actions
    for j in range(turn_idx):
        t = turns[j]
        # Driver turn만 action history에 넣는다 (agent_id == 1)
        if t.get("agent_id") != 1:
            continue
        tool_actions = convert_turn_actions_to_tools(
            t.get("actions", []),
            tools,
            mappings
        )
        if not tool_actions:
            continue
        history.append({
            "type": "action",
            "role": "Driver",
            "actions": tool_actions
        })

    return history


def find_latest_commander_utterance(history: List[Dict[str, Any]]) -> str:
    """
    history에서 가장 최근 Commander 발화를 찾는다.
    """
    for h in reversed(history):
        if h.get("type") == "dialogue" and str(h.get("role", "")).lower() == "commander":
            return h.get("utterance", "")
    return ""


# ---------- LLM INPUT 생성 ----------

def build_llm_input(
    state_before: Dict[str, Any],
    tools: List[Dict[str, Any]],
    history: List[Dict[str, Any]],
) -> str:
    """
    - BASE_PROMPT 에 initial_state / tool_list 채우기
    - 그 뒤에 history + state_diff를 텍스트로 이어붙여 최종 prompt 생성
    """
    # initial_state in BASE_PROMPT should only show the scene's initial_state
    # Use compact JSON serialization (no indent/newlines) so prompt strings stay short
    initial_state_text = json.dumps(state_before.get("initial_state", {}), ensure_ascii=False, separators=(",", ":"))
    tool_list_text = json.dumps(tools, ensure_ascii=False, separators=(",", ":"))

    # Split history into dialogue history (for the dialogue_history placeholder)
    # and previous action list (for the previous_actions placeholder)
    dialogue_lines: List[str] = []
    prev_actions_for_prompt: List[Dict[str, Any]] = []
    for h in history:
        if h["type"] == "dialogue":
            dialogue_lines.append(f'{h["role"]}: {h["utterance"]}')
        elif h["type"] == "action":
            prev_actions_for_prompt.append({"role": h.get("role"), "actions": h.get("actions")})

    dialogue_history_text = "\n".join(dialogue_lines)
    previous_actions_text = json.dumps(prev_actions_for_prompt, ensure_ascii=False, separators=(",", ":"))
    state_diff_text = json.dumps(state_before.get("state_diff", {}), ensure_ascii=False, separators=(",", ":"))

    # Fill all placeholders in the BASE_PROMPT so '{}' slots are correctly populated
    head = BASE_PROMPT.format(
        initial_state=initial_state_text,
        dialogue_history=dialogue_history_text,
        previous_actions=previous_actions_text,
        state_diff=state_diff_text,
        tool_list=tool_list_text,
    )

    # Build a compact representation of history (dialogue + actions) for the section after the template
    hist_lines: List[str] = []
    for h in history:
        if h["type"] == "dialogue":
            hist_lines.append(f'{h["role"]}: {h["utterance"]}')
        elif h["type"] == "action":
            # keep action list compact in-line
            hist_lines.append(f'{h["role"]} (actions): {json.dumps(h["actions"], ensure_ascii=False, separators=(",", ":"))}')

    history_text = "\n".join(hist_lines)

    latest_cmd = find_latest_commander_utterance(history)

    # 최종 프롬프트
    full_input = (
        head
        + "\n\n--- Dialogue + Action History ---\n"
        + history_text
        + "\n\nCommander: " + latest_cmd + "\n"
        + "Now decide the next robot actions and respond with a JSON object."
    )
    # Normalize whitespace so the prompt doesn't contain excessive indentation
    # or repeated blank lines that end up in the saved dataset files.
    # - strip leading/trailing whitespace
    # - collapse runs of blank-only lines to a single blank line
    # - strip leading/trailing whitespace on each non-empty line
    lines = full_input.splitlines()
    out_lines: List[str] = []
    seen_blank = False
    for ln in lines:
        stripped = ln.strip()
        if stripped == "":
            if not seen_blank:
                out_lines.append("")
            seen_blank = True
        else:
            out_lines.append(stripped)
            seen_blank = False

    full_input = "\n".join(out_lines).strip()
    return full_input


# ---------- MAIN: EPISODE → DATASET ----------

def build_dataset_from_episode(
    episode_path: Path,
    tools: List[Dict[str, Any]],
    mappings: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    하나의 episode_x.json (episode_data_wo_state)을 여러 sample(turn 단위)로 변환.
    주의: episode_x.json의 최상위가 dict일 수도 있고 list일 수도 있음.
          - dict  이면: 그 dict 하나를 하나의 episode로 처리
          - list 이면: 리스트 안의 각 element(=episode dict)를 순회하며 처리
    """
    ep_raw = read_json(episode_path)

    # 최상위가 list면 여러 episode, dict면 단일 episode로 처리
    if isinstance(ep_raw, list):
        episodes = ep_raw
    else:
        episodes = [ep_raw]

    # Filter requirement: include only episodes that meet task-level success.
    # Prefer explicit episode-level flags when available (last_success/all_success/any_success).
    # Otherwise, fall back to a conservative check:
    #   - the last action in the last non-empty turn must have last_success==True
    #   - and there must be NO action anywhere in the episode with last_success==False (explicit failures)
    def _episode_task_success(ep: Dict[str, Any]) -> bool:
        # If the episode includes explicit top-level success flags, prefer them
        if ep.get("last_success") is not None:
            return bool(ep.get("last_success"))
        if ep.get("all_success") is not None:
            return bool(ep.get("all_success"))
        if ep.get("any_success") is not None:
            return bool(ep.get("any_success"))

        # Fallback: check final turn's last aggregated action and ensure no explicit failures exist
        turns = ep.get("turns", [])
        if not turns:
            return False

        # find the last turn that has any actions
        last_turn = None
        for t in reversed(turns):
            if t.get("actions"):
                last_turn = t
                break
        if last_turn is None:
            return False

        # find last action in the last_turn with a last_success flag
        last_action = None
        for a in reversed(last_turn.get("actions", [])):
            if "last_success" in a:
                last_action = a
                break

        if last_action is None or not last_action.get("last_success", False):
            return False

        # exclude episodes with any explicit last_success == False anywhere
        for t in turns:
            for a in t.get("actions", []):
                if a.get("last_success") is False:
                    return False

        return True

    # We'll build a new list of kept_episodes using the strongest available check.
    kept: List[Dict[str, Any]] = []
    orig_count = len(episodes)

    def _episode_contains_move_to(ep: Dict[str, Any]) -> bool:
        """Return True if any aggregated action in the episode has action_idx/action_id == 1 (Move to)."""
        for t in ep.get("turns", []):
            for agg in t.get("actions", []):
                raw = agg.get("raw", {}) or {}
                idx = raw.get("action_idx", agg.get("action_idx", agg.get("action_id")))
                # robust check for numeric or string index
                try:
                    if int(idx) == 1:
                        return True
                except Exception:
                    if str(idx) == "1":
                        return True
        return False

    for e in episodes:
        # Exclude any episode that contains a ground-truth 'Move to' action (action_idx == 1)
        if _episode_contains_move_to(e):
            print(f"[INFO] skipping episode {e.get('edh_fn') or e.get('game_fn') or 'unknown'}: contains Move to ground-truth action")
            continue
        # By default, prefer using TEACh state-diff based check (most accurate) when utilities and files exist.
        edh_path = Path(e.get("edh_fn")) if e.get("edh_fn") else None
        game_path = Path(e.get("game_fn")) if e.get("game_fn") else None

        used_state_check = False
        if _HAS_TEACH_UTILS and edh_path and edh_path.exists() and game_path and game_path.exists():
            try:
                edh_json = read_json(edh_path)
                game_json = read_json(game_path)
                state_changes = edh_json.get("state_changes")
                # final_state might be stored in game_json under tasks[0].episodes[0].final_state
                fin_ep = None
                try:
                    fin_ep = game_json["tasks"][0]["episodes"][0].get("final_state")
                except Exception:
                    fin_ep = None

                if state_changes and fin_ep and fin_ep.get("objects"):
                    task_obj = create_task_thor_from_state_diff(state_changes)
                    final_objs = update_objs_with_custom_metadata(fin_ep.get("objects", []), fin_ep.get("custom_object_metadata", {}))
                    progress = task_obj.check_episode_progress(final_objs)
                    if progress and progress.get("success"):
                        kept.append(e)
                    # if progress indicates failure, skip episode
                    used_state_check = True
            except Exception:
                # If any error occurs during TEACh-based check, fall back to heuristics below
                used_state_check = False

        if used_state_check:
            continue

        # Fallback to previous episode-level heuristics
        if _episode_task_success(e):
            kept.append(e)

    episodes = kept
    if len(episodes) != orig_count:
        print(f"[INFO] filtered out {orig_count - len(episodes)} episode(s) that didn't meet task-level success criteria")

    all_samples: List[Dict[str, Any]] = []

    for ep in episodes:
        edh_path = Path(ep["edh_fn"])
        game_path = Path(ep["game_fn"])

        edh_json = read_json(edh_path)
        game_json = read_json(game_path)

        # images 폴더: /teach_dataset/images/{split}/{game_id}/
        # game_fn: /teach_dataset/games/train/0008f3c95e006303_2053.game.json
        split = game_path.parent.name  # "train"
        game_id = game_path.stem       # may be e.g. "001296256b543d60_11e0.game" or "0008f3c95e006303_2053"
        # If game files are named with a .game.json suffix, remove the extra '.game' stem
        if game_id.endswith(".game"):
            game_id = game_id[: -len(".game")]
        images_dir = TEACH_ROOT / "images" / split / game_id
        statediff_index = load_statediff_index(images_dir)
        # debug prints removed (candidates suppressed to keep output clean)

        # We'll always include the episode/game initial_state (no filtering by final_state_diff)
        filtered_initial_state = game_json.get("initial_state")
        # We no longer load or attach final_state_diff (we instead build state_diff up to the turn in build_state_before_turn)
        final_state_diff_json = None

        turns = ep.get("turns", [])
        game_id_full = ep.get("game_id", game_id)

        # derive keys_of_interest for the episode: union of object ids referenced by any action
        keys_of_interest_episode = derive_keys_of_interest_from_turns(turns)

        for k, turn in enumerate(turns):
            # Driver가 아닌 턴은 스킵
            if turn.get("agent_id") != 1:
                continue

            # 1) 이 턴 직전 상태
            # For the per-turn state, we filter to only objects relevant to the episode
            state_before = build_state_before_turn(
                turn_idx=k,
                turns=turns,
                statediff_index=statediff_index,
                game_json=game_json,
                keys_of_interest=keys_of_interest_episode,
            )

            # 2) history (dialogue + 이전 actions)
            history = build_history_until_turn(
                edh_json=edh_json,
                turns=turns,
                turn_idx=k,
                tools=tools,
                mappings=mappings,
            )

            # 3) 정답 action (현재 턴의 aggregated actions → tool list)
            gt_actions = convert_turn_actions_to_tools(
                turn.get("actions", []),
                tools,
                mappings,
            )

            # 4) LLM input prompt
            llm_input = build_llm_input(
                state_before=state_before,
                tools=tools,
                history=history,
            )

            # Minimal sample fields required by the user:
            # only 'game_id', 'instance_id', 'turn_index', 'prompt', 'answer'
            # add: 'turn_last_success' boolean reflecting this turn's last aggregated action result
            turn_actions = turn.get("actions", [])
            turn_last_success = False
            if turn_actions:
                for a in reversed(turn_actions):
                    if "last_success" in a:
                        turn_last_success = bool(a.get("last_success"))
                        break

            # compute answer action-type flag: True if answer is empty OR all actions
            # have action_type in {"Motion","ObjectInteraction"}
            def _answer_actions_are_motion_or_objectinteraction(actions: List[Dict[str, Any]]) -> bool:
                if not actions:
                    return True
                allowed = {"Motion", "ObjectInteraction"}
                for a in actions:
                    tname = a.get("tool_name")
                    schema = get_tool_schema_by_name(tools, tname)
                    atype = schema.get("action_type") if schema else None
                    if atype not in allowed:
                        return False
                return True

            answer_flag = _answer_actions_are_motion_or_objectinteraction(gt_actions)

            sample = {
                "game_id": game_id_full,
                "instance_id": edh_json.get("instance_id"),
                "turn_index": k,
                "prompt": llm_input,
                "answer": {
                    "actions": gt_actions
                },
                "answer_all_motion_or_objectinteraction": answer_flag,
                "turn_last_success": turn_last_success,
                # include the split so callers can write outputs grouped per-split
                "_split": split,
            }
            all_samples.append(sample)

    return all_samples


def main():
    args = parse_args()

    EPISODE_ROOT = Path(args.episode_root)
    OUTPUT_ROOT = Path(args.output_root)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    tools, mappings = load_tools_and_mappings(TOOLS_JSON)

    total_samples = 0
    # accumulator for per-split / per-task summary statistics
    SUMMARY: Dict[str, Any] = {}

    # 사용할 task 디렉토리 결정
    if args.task_ids is None:
        task_dirs = sorted(EPISODE_ROOT.glob("task_*"))
    else:
        ids = [x.strip() for x in args.task_ids.split(",") if x.strip() != ""]
        task_dirs = []
        for tid in ids:
            task_dir = EPISODE_ROOT / f"task_{tid}"
            if task_dir.exists():
                task_dirs.append(task_dir)
            else:
                print(f"[WARN] task_{tid} not found: {task_dir}")

    # task / episode 별로 각각 jsonl 파일 생성
    for task_dir in task_dirs:
        task_name = task_dir.name  # 예: "task_0"
        for ep_path in sorted(task_dir.glob("episode_*.json")):
            print(f"[INFO] Processing episode: {ep_path}")

            ep_samples = build_dataset_from_episode(ep_path, tools, mappings)
            total_samples += len(ep_samples)
            if not ep_samples:
                print(f"[INFO]   -> Skipping {ep_path.name}: no samples after filtering (no output file created)")
                continue

            # We'll write outputs only into split-specific directories: OUTPUT_ROOT/{split}/{task_name}/
            # Group samples by split so we can write to OUTPUT_ROOT/{split}/{task_name}/...
            samples_by_split = {}
            for s in ep_samples:
                sp = s.get("_split", "unknown")
                samples_by_split.setdefault(sp, []).append(s)

            for sp, samples in samples_by_split.items():
                split_out_dir = OUTPUT_ROOT / sp / task_name
                split_out_dir.mkdir(parents=True, exist_ok=True)

                # If the episode file only contains a single split, keep the original filename.
                # If multiple splits are present in the same episode file, suffix the filename with the split
                # to avoid clobbering different-split outputs.
                filename = f"{ep_path.stem}.jsonl" if len(samples_by_split) == 1 else f"{ep_path.stem}.{sp}.jsonl"
                split_out_path = split_out_dir / filename
                write_jsonl(split_out_path, samples)
                print(f"[INFO]   -> Saved {len(samples)} samples to {split_out_path}")

                # -----------------
                # Per-episode statistics
                # Count how many samples in this episode/split have both flags True
                both_true_count = 0
                for s in samples:
                    if bool(s.get("answer_all_motion_or_objectinteraction")) and bool(s.get("turn_last_success")):
                        both_true_count += 1

                # We count per-episode 'both flags true' and accumulate into SUMMARY,
                # but we no longer write per-episode stats files to avoid clutter.

                # accumulate global summary across this run (per-split only)
                summary_for_split = SUMMARY.setdefault(sp, {"total_samples": 0, "both_flags_true": 0})
                summary_for_split["total_samples"] += len(samples)
                summary_for_split["both_flags_true"] += both_true_count

    # After processing all episodes, write out a summary JSON per split and an overall summary
    # Write a single consolidated summary file containing overall totals and
    # per-split breakdowns (including per-task values within each split).
    if SUMMARY:
        # Compute overall aggregates across splits
        overall = {"total_samples": 0, "both_flags_true": 0}
        for sp, vals in SUMMARY.items():
            overall["total_samples"] += int(vals.get("total_samples", 0))
            overall["both_flags_true"] += int(vals.get("both_flags_true", 0))

        consolidated = {
            "overall": overall,
            "per_split": SUMMARY,
        }

        summary_path = OUTPUT_ROOT / "summary.json"
        with summary_path.open("w", encoding="utf-8") as sf:
            json.dump(consolidated, sf, ensure_ascii=False, indent=2)
        print(f"[INFO] Wrote consolidated summary to {summary_path}")

    print(f"[INFO] Total samples across all episodes: {total_samples}")



if __name__ == "__main__":
    main()
