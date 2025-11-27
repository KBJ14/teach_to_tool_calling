import json
import math
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

# Try to import TEACh utilities for accurate task checks.
try:
    from teach.utils import create_task_thor_from_state_diff, update_objs_with_custom_metadata
    _HAS_TEACH_UTILS = True
except Exception:
    _HAS_TEACH_UTILS = False

# Try to import SentenceTransformer
try:
    from sentence_transformers import SentenceTransformer, util
    _HAS_SBERT = True
except ImportError:
    _HAS_SBERT = False


# ---------- CONFIG ----------

REPO_ROOT = Path("/home/bjk/tool_learning/teach_to_tool_calling")
TOOLS_JSON = REPO_ROOT / "dataset/prompts/tools.json"
TEACH_ROOT = Path("/teach_dataset")

# ---------- PROMPT TEMPLATE ----------

BASE_PROMPT = """
You are a high-level robot control assistant in a simulated home environment.

The following information describes the current context:

### Environmental Context (Perception & Map):
{initial_state}

### Dialogue History:
{dialogue_history}

### Previous Actions:
{previous_actions}

### Available Tools:
{tool_list}

Your task:
Based on the latest Commander utterance and the environmental context,
decide the next robot actions.

Output format:
Return ONLY a JSON object with "actions" list.
"""

# ---------- PARSE ARGS ----------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-root", type=str, required=True, help="episode_data_wo_state root path")
    parser.add_argument("--output-root", type=str, required=True, help="Output root path")
    parser.add_argument("--task-ids", type=str, default=None, help="Comma separated task ids")
    
    parser.add_argument(
        "--filter-mode", 
        type=str, 
        default="spatial", 
        choices=["spatial", "semantic", "hybrid"],
        help="spatial (Baseline), semantic, hybrid (Ours)"
    )
    parser.add_argument(
        "--embedding-model", 
        type=str, 
        default="all-MiniLM-L6-v2",
        help="Embedding model for semantic filtering"
    )
    return parser.parse_args()


# ---------- CONTEXT PROCESSOR (Perception Module) ----------

class ContextProcessor:
    def __init__(self, mode: str, model_name: str):
        self.mode = mode
        self.model = None
        
        # Thresholds
        self.SPATIAL_THRESHOLD = 5.0
        self.NEARBY_THRESHOLD = 2.5
        self.SEMANTIC_THRESHOLD = 0.3 
        
        # Landmarks
        self.LANDMARKS = {
            "Bed", "Desk", "Dresser", "ArmChair", "SideTable", "CounterTop", 
            "Floor", "Wall", "Door", "Window", "Drawer", "GarbageCan", 
            "LaundryHamper", "Fridge", "Stove", "Sink", "Bathtub", "Toilet",
            "Sofa", "Table", "Cabinet", "Shelf", "Microwave", "CoffeeMachine",
            "StoveBurner", "DiningTable", "TVStand", "Safe"
        }

        if "semantic" in mode or "hybrid" in mode:
            if not _HAS_SBERT:
                raise ImportError("sentence-transformers is required.")
            self.model = SentenceTransformer(model_name)
    
    def _get_relative_coords(self, agent_pos: Dict, agent_rot: Dict, obj_pos: Dict) -> Tuple[float, float, str]:
        if not agent_pos or not obj_pos: return 999.9, 0.0, ""
        
        dx = obj_pos['x'] - agent_pos['x']
        dz = obj_pos['z'] - agent_pos['z']
        dist = math.sqrt(dx*dx + dz*dz)
        
        target_angle_global = math.degrees(math.atan2(dx, dz))
        agent_yaw = agent_rot.get('y', 0.0)
        
        rel_angle = (target_angle_global - agent_yaw) % 360
        if rel_angle > 180: rel_angle -= 360
        
        if -45 <= rel_angle <= 45: direction = "Front"
        elif 45 < rel_angle <= 135: direction = "Right"
        elif -135 <= rel_angle < -45: direction = "Left"
        else: direction = "Back"
            
        return dist, rel_angle, direction

    def _format_object_text(self, obj: Dict, dist: float, rel_angle: float, direction: str) -> str:
        obj_type = obj.get("objectType")
        if not obj_type:
            obj_type = obj.get("name", "").split('_')[0].split('(')[0]
        
        states = []
        if obj.get("visible"): states.append("Visible")
        else: states.append("Map Knowledge")
        
        if obj.get("isPickedUp"): states.append("Held")
        if obj.get("isOpen"): states.append("Open")
        if obj.get("isToggled"): states.append("On")
        
        if obj.get("isDirty"): states.append("Dirty")
        if obj.get("isFilledWithLiquid"): states.append("Filled")
        if obj.get("isSliced"): states.append("Sliced")
        if obj.get("isCooked"): states.append("Cooked")
        if obj.get("isBroken"): states.append("Broken")
        if obj.get("isUsedUp"): states.append("Empty")
        
        temp = obj.get("ObjectTemperature")
        if temp == "Hot": states.append("Hot")
        elif temp == "Cold": states.append("Cold")

        state_str = ", ".join(states)
        
        loc_str = ""
        parents = obj.get("parentReceptacles")
        if parents and len(parents) > 0:
            p_name = parents[0].split('|')[0]
            loc_str = f"in/on {p_name}"

        if "Held" in states:
            nav_info = "Held by Agent"
        else:
            nav_info = f"{dist:.1f}m, {rel_angle:.0f}° {direction}"
        
        details = [nav_info]
        if loc_str: details.append(loc_str)
        if state_str: details.append(state_str)
        
        return f"- {obj_type} ({', '.join(details)})"

    def process_state(self, state: Dict[str, Any], instruction: str) -> str:
        objects = state.get("objects", [])
        agent_data = state.get("agents", {}) or {}
        
        agent_pos = {"x": 0, "y": 0, "z": 0}
        agent_rot = {"x": 0, "y": 0, "z": 0}
        
        if isinstance(agent_data, list) and len(agent_data) > 0:
            for ag in agent_data:
                if ag.get("name") == "agent":
                    agent_pos = ag.get("position", agent_pos)
                    agent_rot = ag.get("rotation", agent_rot)
                    break
        elif isinstance(agent_data, dict):
            ag = agent_data.get("agent", {})
            agent_pos = ag.get("position", agent_pos)
            agent_rot = ag.get("rotation", agent_rot)

        if not objects: return "(No objects information available)"

        query_embedding = None
        if self.model and instruction and instruction.strip():
            query_embedding = self.model.encode(instruction, convert_to_tensor=True)

        landmarks_text = []
        relevant_text = []
        nearby_text = []

        for obj in objects:
            if not isinstance(obj, dict): continue
            
            obj_type = obj.get("objectType")
            if not obj_type: obj_type = obj.get("name", "").split('_')[0].split('(')[0]
            
            dist, rel_angle, direction = self._get_relative_coords(agent_pos, agent_rot, obj.get("position"))
            
            is_visible = bool(obj.get("visible", False))
            is_landmark = obj_type in self.LANDMARKS
            is_nearby = dist < self.NEARBY_THRESHOLD
            
            should_keep = False
            is_target_candidate = False

            if self.mode == "spatial":
                if is_landmark or dist < self.SPATIAL_THRESHOLD or is_visible:
                    should_keep = True
                    if (dist < self.SPATIAL_THRESHOLD or is_visible) and not is_landmark:
                        is_target_candidate = True
            
            elif self.mode == "semantic":
                if is_landmark or is_visible:
                    should_keep = True
                    if is_visible and not is_landmark: is_target_candidate = True
                
                if self.model and query_embedding is not None:
                    target_name = obj_type
                    obj_emb = self.model.encode(target_name, convert_to_tensor=True)
                    score = util.cos_sim(query_embedding, obj_emb).item()
                    if score > self.SEMANTIC_THRESHOLD: 
                        should_keep = True
                        is_target_candidate = True
            
            elif self.mode == "hybrid":
                if is_landmark:
                    should_keep = True
                else:
                    semantic_hit = False
                    if self.model and query_embedding is not None:
                        target_name = obj_type
                        obj_emb = self.model.encode(target_name, convert_to_tensor=True)
                        score = util.cos_sim(query_embedding, obj_emb).item()
                        if score > 0.35: semantic_hit = True
                    
                    if semantic_hit or is_nearby or is_visible:
                        should_keep = True
                        if semantic_hit: is_target_candidate = True
                        elif is_visible and not is_nearby: is_target_candidate = False 

            if should_keep:
                desc = self._format_object_text(obj, dist, rel_angle, direction)
                if is_landmark: landmarks_text.append(desc)
                elif is_target_candidate: relevant_text.append(desc)
                else: nearby_text.append(desc)

        sections = []
        if landmarks_text: sections.append("[Fixed Furniture / Map]\n" + "\n".join(sorted(landmarks_text)))
        if relevant_text: sections.append("[Target Candidates]\n" + "\n".join(sorted(relevant_text)))
        if nearby_text: sections.append("[Nearby & Visible Items]\n" + "\n".join(sorted(nearby_text)))
        
        if not sections: return "(No relevant objects visible)"
        return "\n\n".join(sections)


# ---------- UTILS ----------

def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f: return json.load(f)

def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False) + "\n")

def load_tools_and_mappings(tools_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    data = read_json(tools_path)
    return data["tools"], data["mappings"]["action_idx_to_tool"]

def get_tool_schema_by_name(tools: List[Dict[str, Any]], tool_name: str) -> Optional[Dict[str, Any]]:
    for t in tools:
        if t.get("name") == tool_name: return t
    return None

def parse_statediff_time(path: Path) -> Optional[float]:
    if not path.name.startswith("statediff."): return None
    core = path.name[len("statediff."):].replace(".json", "")
    try: return float(core)
    except ValueError: return None

def load_statediff_index(images_dir: Path) -> List[Tuple[float, Path]]:
    if not images_dir.exists(): return []
    entries = [(parse_statediff_time(p), p) for p in images_dir.glob("statediff*.json") if parse_statediff_time(p) is not None]
    entries.sort(key=lambda x: x[0])
    return entries

def convert_aggregated_action_to_tool_calls(agg, tools, mappings):
    raw = agg.get("raw", {})
    idx = raw.get("action_idx", agg.get("action_idx", agg.get("action_id")))
    if idx is None or str(idx) not in mappings: return []
    map_info = mappings[str(idx)]
    tool_name = map_info["tool_name"]
    count = agg.get("count", 1) or 1
    if tool_name == "motion_delta":
        pose_delta = raw.get("pose_delta") or [0]*6
        if len(pose_delta) != 6: pose_delta = (pose_delta + [0]*6)[:6]
        total_delta = [pose_delta[i]*count for i in range(6)]
        return [{"tool_name": tool_name, "parameters": dict(zip(["x","y","z","rot_x","rot_y","rot_z"], total_delta))}]
    else:
        return [{"tool_name": tool_name, "parameters": {}}]

def convert_turn_actions_to_tools(turn_actions, tools, mappings):
    tool_list = []
    for agg in turn_actions: tool_list.extend(convert_aggregated_action_to_tool_calls(agg, tools, mappings))
    return tool_list

def build_history_until_turn(turns, turn_idx, tools, mappings):
    history = []
    for t in turns[:turn_idx]:
        for cand in (t.get("pre_turn_dialogs") or t.get("commander_context") or t.get("history") or []):
            if not isinstance(cand, dict): continue
            role = "Commander" if int(cand.get("agent_id", 0)) == 0 else "Driver"
            utt = cand.get("corrected_utterance") or cand.get("utterance") or cand.get("query")
            if utt: history.append({"type": "dialogue", "role": role, "utterance": utt})
        for agg in t.get("actions", []):
            raw = agg.get("raw", {})
            utt = raw.get("corrected_utterance") or raw.get("utterance")
            if utt: history.append({"type": "dialogue", "role": "Driver", "utterance": utt})
    for j in range(turn_idx):
        t = turns[j]
        if t.get("agent_id") != 1: continue
        acts = convert_turn_actions_to_tools(t.get("actions", []), tools, mappings)
        if acts: history.append({"type": "action", "role": "Driver", "actions": acts})
    return history

def get_accumulated_instructions(history: List[Dict[str, Any]]) -> str:
    instructions = []
    for h in history:
        if h.get("type") == "dialogue" and str(h.get("role", "")).lower() == "commander":
            utt = h.get("utterance", "")
            if utt:
                instructions.append(utt)
    return " ".join(instructions)

def build_state_before_turn(turn_idx, turns, statediff_index, game_json):
    initial_state = game_json["tasks"][0]["episodes"][0].get("initial_state", {})
    if turn_idx == 0: return {"initial_state": initial_state, "state_diff": {}}

    prev_turn = turns[turn_idx - 1]
    last_time_prev = None
    for agg in prev_turn.get("actions", []):
        raw = agg.get("raw", {})
        t_vals = [float(x) for x in (agg.get("time_starts") or [])]
        if agg.get("time_start"): t_vals.append(float(agg["time_start"]))
        if raw.get("time_start"): t_vals.append(float(raw["time_start"]))
        if t_vals:
            t_max = max(t_vals)
            if last_time_prev is None or t_max > last_time_prev: last_time_prev = t_max

    state_diff = None
    if last_time_prev is not None and statediff_index:
        sd_path = None
        for tt, p in statediff_index:
            if tt <= last_time_prev: sd_path = p
            else: break
        if sd_path:
            try: state_diff = read_json(sd_path)
            except: pass

    return {"initial_state": initial_state, "state_diff": state_diff or {}}

# ---------- PROMPT BUILDER ----------

def build_llm_input(state_text: str, tools, history, latest_cmd: str):
    tool_list_text = json.dumps(tools, ensure_ascii=False)
    dialogue_lines = []
    prev_actions = []
    for h in history:
        if h["type"] == "dialogue": dialogue_lines.append(f'{h["role"]}: {h["utterance"]}')
        elif h["type"] == "action": prev_actions.append({"role": h["role"], "actions": h["actions"]})
    
    head = BASE_PROMPT.format(
        initial_state=state_text,
        dialogue_history="\n".join(dialogue_lines) if dialogue_lines else "(No dialogue history)",
        previous_actions=json.dumps(prev_actions, ensure_ascii=False),
        tool_list=tool_list_text,
    )
    
    hist_lines = []
    for h in history:
        if h["type"] == "dialogue": hist_lines.append(f'{h["role"]}: {h["utterance"]}')
        elif h["type"] == "action": hist_lines.append(f'{h["role"]} (actions): {json.dumps(h["actions"], ensure_ascii=False)}')
        
    return f"{head}\n\n--- Current Context ---\n" + "\n".join(hist_lines) + \
           f"\n\nCommander: {latest_cmd}\nNow decide the next robot actions and respond with a JSON object."


# ---------- MAIN LOOP ----------

def build_dataset_from_episode(episode_path, tools, mappings, processor):
    ep_raw = read_json(episode_path)
    episodes = ep_raw if isinstance(ep_raw, list) else [ep_raw]
    
    # --- [RESTORED] Episode Filtering Logic ---
    def _episode_task_success(ep: Dict[str, Any]) -> bool:
        if ep.get("last_success") is not None: return bool(ep.get("last_success"))
        if ep.get("all_success") is not None: return bool(ep.get("all_success"))
        if ep.get("any_success") is not None: return bool(ep.get("any_success"))
        
        turns = ep.get("turns", [])
        if not turns: return False
        
        last_turn = None
        for t in reversed(turns):
            if t.get("actions"): last_turn = t; break
        if last_turn is None: return False
        
        last_action = None
        for a in reversed(last_turn.get("actions", [])):
            if "last_success" in a: last_action = a; break
        
        if last_action is None or not last_action.get("last_success", False): return False
        
        for t in turns:
            for a in t.get("actions", []):
                if a.get("last_success") is False: return False
        return True

    def _episode_contains_move_to(ep: Dict[str, Any]) -> bool:
        for t in ep.get("turns", []):
            for agg in t.get("actions", []):
                raw = agg.get("raw", {})
                idx = raw.get("action_idx", agg.get("action_idx", agg.get("action_id")))
                try:
                    if int(idx) == 1: return True # Move to
                except:
                    if str(idx) == "1": return True
        return False

    kept = []
    for e in episodes:
        if _episode_contains_move_to(e): continue
        
        edh_path = Path(e.get("edh_fn")) if e.get("edh_fn") else None
        game_path = Path(e.get("game_fn")) if e.get("game_fn") else None
        
        used_state_check = False
        if _HAS_TEACH_UTILS and edh_path and edh_path.exists() and game_path and game_path.exists():
            try:
                edh_json = read_json(edh_path)
                game_json = read_json(game_path)
                state_changes = edh_json.get("state_changes")
                fin_ep = None
                try: fin_ep = game_json["tasks"][0]["episodes"][0].get("final_state")
                except: fin_ep = None
                
                if state_changes and fin_ep and fin_ep.get("objects"):
                    task_obj = create_task_thor_from_state_diff(state_changes)
                    final_objs = update_objs_with_custom_metadata(fin_ep.get("objects", []), fin_ep.get("custom_object_metadata", {}))
                    progress = task_obj.check_episode_progress(final_objs)
                    if progress and progress.get("success"): kept.append(e)
                    used_state_check = True
            except: used_state_check = False
        
        if used_state_check: continue
        
        if _episode_task_success(e): kept.append(e)
    
    episodes = kept
    # ------------------------------------------

    all_samples = []

    for ep in episodes:
        edh_path = Path(ep["edh_fn"])
        game_path = Path(ep["game_fn"])
        split = game_path.parent.name
        game_id = game_path.stem.replace(".game", "")
        
        images_dir = TEACH_ROOT / "images" / split / game_id
        statediff_index = load_statediff_index(images_dir)
        game_json = read_json(game_path)
        turns = ep.get("turns", [])

        for k, turn in enumerate(turns):
            if turn.get("agent_id") != 1: continue 

            state_before = build_state_before_turn(k, turns, statediff_index, game_json)
            history = build_history_until_turn(turns, k, tools, mappings)
            
            # Get Current Command
            current_cmd = ""
            dialog_candidates = (turn.get("pre_turn_dialogs") or 
                                 turn.get("commander_context") or 
                                 turn.get("history") or [])
            for cand in dialog_candidates:
                if isinstance(cand, dict):
                    role = "Commander" if int(cand.get("agent_id", 0)) == 0 else "Driver"
                    if role == "Commander":
                        current_cmd = cand.get("corrected_utterance") or cand.get("utterance") or cand.get("query") or ""
                        break
            
            past_instructions = get_accumulated_instructions(history)
            full_instruction_context = f"{past_instructions} {current_cmd}".strip()
            
            if current_cmd: latest_instruction_for_prompt = current_cmd
            else:
                last_h = ""
                for h in reversed(history):
                    if h["type"] == "dialogue" and h["role"] == "Commander":
                        last_h = h["utterance"]; break
                latest_instruction_for_prompt = last_h
            
            # MERGE LOGIC (Using .copy and .update)
            merged_objects_map = {
                obj.get("objectId", f"UNK_{i}"): obj.copy() 
                for i, obj in enumerate(state_before["initial_state"].get("objects", []))
            }
            
            if state_before.get("state_diff"):
                diff_objs = state_before["state_diff"].get("objects", {})
                
                def update_map(oid, new_data):
                    if oid not in merged_objects_map:
                        merged_objects_map[oid] = new_data
                    else:
                        merged_objects_map[oid].update(new_data)

                if isinstance(diff_objs, list):
                    for obj in diff_objs:
                        if "objectId" in obj: update_map(obj["objectId"], obj)
                elif isinstance(diff_objs, dict):
                    for oid, obj in diff_objs.items():
                        current_oid = obj.get("objectId", oid)
                        update_map(current_oid, obj)
            
            merged_state = {
                "agents": state_before["state_diff"].get("agents") or state_before["initial_state"].get("agents"),
                "objects": list(merged_objects_map.values())
            }

            # Process
            abstracted_state_text = processor.process_state(merged_state, full_instruction_context)
            
            llm_input = build_llm_input(abstracted_state_text, tools, history, latest_instruction_for_prompt)
            gt_actions = convert_turn_actions_to_tools(turn.get("actions", []), tools, mappings)
            
            def _check_actions_type(acts):
                for a in acts:
                    t = get_tool_schema_by_name(tools, a["tool_name"])
                    if t and t.get("action_type") not in ["Motion", "ObjectInteraction"]: return False
                return True

            turn_last_success = False
            if turn.get("actions"):
                last_agg = turn["actions"][-1]
                if "last_success" in last_agg:
                    turn_last_success = bool(last_agg["last_success"])

            all_samples.append({
                "game_id": ep.get("game_id", game_id),
                "instance_id": ep.get("instance_id"),
                "turn_index": k,
                "prompt": llm_input,
                "answer": {"actions": gt_actions},
                "answer_all_motion_or_objectinteraction": _check_actions_type(gt_actions),
                "turn_last_success": turn_last_success,
                "_split": split
            })

    return all_samples

def main():
    args = parse_args()
    
    print(f"[INFO] Initializing Context Processor (Mode: {args.filter_mode})...")
    processor = ContextProcessor(mode=args.filter_mode, model_name=args.embedding_model)
    
    EPISODE_ROOT = Path(args.episode_root)
    OUTPUT_ROOT = Path(args.output_root)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    tools, mappings = load_tools_and_mappings(TOOLS_JSON)
    
    if args.task_ids is None:
        task_dirs = sorted(EPISODE_ROOT.glob("task_*"))
    else:
        ids = [x.strip() for x in args.task_ids.split(",") if x.strip()]
        task_dirs = [EPISODE_ROOT / f"task_{tid}" for tid in ids]

    print(f"[INFO] Scanning episode files in {len(task_dirs)} task directories...")
    all_episode_files = []
    for task_dir in task_dirs:
        if task_dir.exists():
            files = sorted(task_dir.glob("episode_*.json"))
            all_episode_files.extend(files)
    
    print(f"[INFO] Found {len(all_episode_files)} episodes. Starting generation...")

    for ep_path in tqdm(all_episode_files, desc=f"[{args.filter_mode}] Generating", unit="ep"):
        ep_samples = build_dataset_from_episode(ep_path, tools, mappings, processor)
        if not ep_samples: continue

        task_name = ep_path.parent.name 
        samples_by_split = {}
        for s in ep_samples: samples_by_split.setdefault(s["_split"], []).append(s)
        
        for sp, samples in samples_by_split.items():
            out_dir = OUTPUT_ROOT / sp / task_name
            out_dir.mkdir(parents=True, exist_ok=True)
            write_jsonl(out_dir / f"{ep_path.stem}.jsonl", samples)

if __name__ == "__main__":
    main()