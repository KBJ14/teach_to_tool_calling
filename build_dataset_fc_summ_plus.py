import json
import math
import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
from tqdm import tqdm

# LiteLLM Import
try:
    from litellm import completion
    _HAS_LITELLM = True
except ImportError:
    print("[WARNING] 'litellm' module not found. Summarization will fail. (pip install litellm)")
    _HAS_LITELLM = False

# Try to import SentenceTransformer
try:
    from sentence_transformers import SentenceTransformer, util
    _HAS_SBERT = True
except ImportError:
    _HAS_SBERT = False

# ---------- CONFIG ----------

# High Level Tools 정의 (Navigation 제외됨)
HIGH_LEVEL_TOOL_NAMES = {
    "Pickup", "Place", "Open", "Close",
    "Slice", "Dirty", "Clean", "ToggleOn", "ToggleOff",
    "Fill", "Empty", "Pour", "Break"
}

# 이동 액션 압축용
MOTION_PARAMS = {
    "Forward":       {"x": 0.25, "y": 0, "z": 0, "rot_x": 0, "rot_y": 0, "rot_z": 0},
    "Backward":      {"x": -0.25, "y": 0, "z": 0, "rot_x": 0, "rot_y": 0, "rot_z": 0},
    "Turn Left":     {"x": 0, "y": 0, "z": 0, "rot_x": 0, "rot_y": 0, "rot_z": 90},
    "Turn Right":    {"x": 0, "y": 0, "z": 0, "rot_x": 0, "rot_y": 0, "rot_z": -90},
    "Look Up":       {"x": 0, "y": 0, "z": 0, "rot_x": -30, "rot_y": 0, "rot_z": 0},
    "Look Down":     {"x": 0, "y": 0, "z": 0, "rot_x": 30, "rot_y": 0, "rot_z": 0},
    "Pan Left":      {"x": 0, "y": 0.25, "z": 0, "rot_x": 0, "rot_y": 0, "rot_z": 0},
    "Pan Right":     {"x": 0, "y": -0.25, "z": 0, "rot_x": 0, "rot_y": 0, "rot_z": 0},
    "Move Up":       {"x": 0, "y": 0, "z": 0.25, "rot_x": 0, "rot_y": 0, "rot_z": 0},
    "Move Down":     {"x": 0, "y": 0, "z": -0.25, "rot_x": 0, "rot_y": 0, "rot_z": 0},
    "Stop":          {"x": 0, "y": 0, "z": 0, "rot_x": 0, "rot_y": 0, "rot_z": 0},
}

# ---------- PROMPT TEMPLATE ----------

BASE_PROMPT = """
You are a high-level robot control assistant in a simulated home environment.

The following information describes the current context:

### Environmental Context (Perception & Map):
{initial_state}

### Interaction History Summary (Previous Turns):
{history_summary}

### Recent Interaction (Current Turn):
{history_recent}

Your task:
Based on the latest Commander utterance and the environmental context,
decide the next robot actions by calling the appropriate functions.
"""

# ---------- UTILS ----------

def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f: 
        return json.load(f)

def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

# ---------- SUMMARIZATION HELPER (LiteLLM) ----------

def summarize_history_gpt4o(history_text: str) -> str:
    """
    Summarize the raw history text using LiteLLM (GPT-4o).
    """
    if not history_text.strip():
        return "No previous history."
    
    if not _HAS_LITELLM:
        return "[Error] litellm not installed.\n" + history_text[-500:]

    try:
        # LiteLLM completion call
        response = completion(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "You are a helpful assistant assisting a robot agent. "
                        "Summarize the following chronological interaction history between a Commander and a Driver (Robot). "
                        "Focus strictly on: \n"
                        "1. What actions the robot has successfully performed in the past.\n"
                        "2. What instructions have been completed or attempted.\n"
                        "Keep the summary concise and objective."
                    )
                },
                {"role": "user", "content": history_text}
            ],
            temperature=0.0, 
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[Warning] Summarization failed: {e}")
        return "Summarization failed. Raw text:\n" + history_text[-1000:]


# ---------- CONTEXT PROCESSOR ----------

class ContextProcessor:
    def __init__(self, mode: str, model_name: str):
        self.mode = mode
        self.model = None
        self.NEARBY_THRESHOLD = 1.5

        self.LANDMARKS = {
            "Bed", "Desk", "Dresser", "ArmChair", "SideTable", "CounterTop", 
            "Floor", "Wall", "Door", "Window", "Drawer", "GarbageCan", 
            "Sofa", "Table", "Cabinet", "Shelf", "Microwave", "CoffeeMachine",
            "StoveBurner", "DiningTable", "TVStand", "Safe", "Sink", "Bathtub", "Toilet"
        }

        if "semantic" in mode or "hybrid" in mode:
            if _HAS_SBERT:
                self.model = SentenceTransformer(model_name)

    def _get_relative_coords(self, agent_pos, agent_rot, obj_pos):
        if not agent_pos or not obj_pos:
            return 999.9, 0.0, ""

        dx = obj_pos['x'] - agent_pos['x']
        dz = obj_pos['z'] - agent_pos['z']
        dist = math.sqrt(dx * dx + dz * dz)

        target_angle_global = math.degrees(math.atan2(dx, dz))
        agent_yaw = agent_rot.get('y', 0.0)

        rel_angle = (target_angle_global - agent_yaw) % 360
        if rel_angle > 180:
            rel_angle -= 360

        if -45 <= rel_angle <= 45:
            direction = "Front"
        elif 45 < rel_angle <= 135:
            direction = "Right"
        elif -135 <= rel_angle < -45:
            direction = "Left"
        else:
            direction = "Back"

        return dist, rel_angle, direction

    def process_state(self, state: Dict[str, Any], instruction: str = "") -> str:
        objects = state.get("objects", [])
        agent_data = state.get("agents", [])

        agent_pos = {"x": 0, "y": 0, "z": 0}
        agent_rot = {"x": 0, "y": 0, "z": 0}

        if agent_data and isinstance(agent_data, list):
            for ag in agent_data:
                if ag.get("name") == "agent" or ag.get("agent_id") == 1:
                    agent_pos = ag.get("position", agent_pos)
                    agent_rot = ag.get("rotation", agent_rot)
                    break

        lines = []
        query_embedding = None

        if self.model and instruction:
            query_embedding = self.model.encode(instruction, convert_to_tensor=True)

        for obj in objects:
            obj_type = obj.get("objectType") or obj.get("name", "").split('|')[0]

            if "distance" in obj:
                dist = obj["distance"]
                _, rel_angle, direction = self._get_relative_coords(agent_pos, agent_rot, obj.get("position"))
            else:
                dist, rel_angle, direction = self._get_relative_coords(agent_pos, agent_rot, obj.get("position"))

            is_visible = obj.get("visible", False)
            is_landmark = obj_type in self.LANDMARKS
            is_nearby = dist < self.NEARBY_THRESHOLD

            should_keep = False

            if self.mode == "spatial":
                if dist < 5.0 or is_visible or is_landmark:
                    should_keep = True

            elif self.mode == "semantic":
                if is_landmark or is_visible:
                    should_keep = True
                if self.model and query_embedding is not None:
                    obj_emb = self.model.encode(obj_type, convert_to_tensor=True)
                    score = util.cos_sim(query_embedding, obj_emb).item()
                    if score > 0.3:
                        should_keep = True

            elif self.mode == "hybrid":
                if is_landmark:
                    should_keep = True
                else:
                    semantic_hit = False
                    if self.model and query_embedding is not None:
                        obj_emb = self.model.encode(obj_type, convert_to_tensor=True)
                        score = util.cos_sim(query_embedding, obj_emb).item()
                        if score > 0.35:
                            semantic_hit = True
                    if semantic_hit or is_visible or is_nearby:
                        should_keep = True

            if not should_keep:
                continue

            states = []
            if is_visible:
                states.append("Visible")
            else:
                states.append("Map Knowledge")

            if obj.get("isPickedUp"):
                states.append("Held by Robot")
            if obj.get("isOpen"):
                states.append("Open")
            if obj.get("isToggled"):
                states.append("On")

            parent = obj.get("parentReceptacles")
            loc_str = ""
            if parent:
                p_name = parent[0].split('|')[0]
                loc_str = f"in/on {p_name}"

            desc = [f"{dist:.1f}m", f"{rel_angle:.0f}° {direction}"]
            if loc_str:
                desc.append(loc_str)
            if states:
                desc.append(", ".join(states))

            lines.append(f"- {obj_type} ({', '.join(desc)})")

        if not lines:
            return "(No interactable objects found)"
        return "\n".join(sorted(lines))

# ---------- CORE: HISTORY PARSING ----------

def parse_edh_history(edh_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Reconstructs the full interleaved history (Dialog + Compressed Actions) 
    exactly as in build_dataset_fc_edh.py.
    Returns a sorted list of items.
    """
    # 1. Build timestamp map for dialogs
    interactions = edh_data.get("interactions", [])
    dialog_ts_map = {}
    for it in interactions:
        utt = it.get("utterance")
        if utt:
            dialog_ts_map[utt.strip().lower()] = it.get("time_start", 0)

    timeline = []

    # 2. Add Dialogs
    for speaker, utter in edh_data.get("dialog_history", []):
        key = utter.strip().lower()
        approx_ts = dialog_ts_map.get(key, -1)
        timeline.append({
            "type": "dialog",
            "role": speaker,
            "content": utter,
            "time": approx_ts
        })

    # 3. Add Actions (with Motion Compression)
    action_history = edh_data.get("driver_action_history", [])
    compressed_actions = []
    current_motion = None

    for act in action_history:
        name = act.get("action_name")
        ts = act.get("time_start", 0)
        is_motion = name in MOTION_PARAMS

        if is_motion:
            if current_motion and current_motion["name"] == name:
                base = MOTION_PARAMS[name]
                for k, v in base.items():
                    current_motion["params"][k] += v
                current_motion["count"] += 1
            else:
                if current_motion:
                    compressed_actions.append(current_motion)
                current_motion = {
                    "type": "action",
                    "name": name,
                    "tool_name": "motion_delta",
                    "params": MOTION_PARAMS[name].copy(),
                    "count": 1,
                    "time": ts
                }
        else:
            if current_motion:
                compressed_actions.append(current_motion)
                current_motion = None

            compressed_actions.append({
                "type": "action",
                "name": name,
                "tool_name": name,
                "params": {},
                "time": ts
            })

    if current_motion:
        compressed_actions.append(current_motion)

    # 4. Merge & Sort
    full_history = timeline + compressed_actions
    full_history = sorted(full_history, key=lambda x: x["time"])
    
    return full_history

def items_to_text_block(items: List[Dict[str, Any]]) -> str:
    """Converts a list of history items to a single string block."""
    lines = []
    for item in items:
        if item["time"] < 0:
            if item["type"] == "dialog":
                lines.append(f"{item['role']}: {item['content']}")
            else:
                ajson = {"tool_name": item['tool_name'], "parameters": item['params']}
                lines.append(f"Driver (action): {json.dumps(ajson)}")
        else:
            if item["type"] == "dialog":
                lines.append(f"{item['role']} ({item['time']:.1f}s): {item['content']}")
            else:
                ajson = {"tool_name": item['tool_name'], "parameters": item['params']}
                lines.append(f"Driver (action @ {item['time']:.1f}s): {json.dumps(ajson)}")
    return "\n".join(lines)

def get_game_history_upto(game_data: Dict[str, Any], cutoff_time: float, speaker_map: Dict[str, str]) -> str:
    """
    Fetches interactions from game_data up to cutoff_time.
    Applies simple motion compression and speaker labeling for better summary quality.
    """
    try:
        interactions = game_data["tasks"][0]["episodes"][0]["interactions"]
    except:
        return ""

    # Filter items
    filtered = []
    for it in interactions:
        if it.get("time_start", 0) >= cutoff_time:
            break
        filtered.append(it)
    
    if not filtered:
        return ""

    # Reuse logic for formatting game interactions (Motion Compression)
    # This ensures the summary sees the same high-quality log as the EDH parser.
    # Convert game interactions to "items" similar to EDH format
    formatted_items = []
    current_motion = None

    for it in filtered:
        ts = it.get("time_start", 0)
        utterance = it.get("utterance")
        action_name = it.get("action_name")

        if utterance:
            # Flush motion
            if current_motion:
                formatted_items.append(current_motion)
                current_motion = None
            
            # Identify speaker
            spk = speaker_map.get(utterance.strip(), "Commander") # Default to Commander if not found
            formatted_items.append({
                "type": "dialog",
                "role": spk,
                "content": utterance,
                "time": ts
            })
        
        elif action_name:
            is_motion = action_name in MOTION_PARAMS
            if is_motion:
                if current_motion and current_motion["name"] == action_name:
                    base = MOTION_PARAMS[action_name]
                    for k, v in base.items():
                        current_motion["params"][k] += v
                    current_motion["count"] += 1
                else:
                    if current_motion:
                        formatted_items.append(current_motion)
                    current_motion = {
                        "type": "action",
                        "name": action_name,
                        "tool_name": "motion_delta",
                        "params": MOTION_PARAMS[action_name].copy(),
                        "count": 1,
                        "time": ts
                    }
            else:
                if current_motion:
                    formatted_items.append(current_motion)
                    current_motion = None
                
                formatted_items.append({
                    "type": "action",
                    "name": action_name,
                    "tool_name": action_name,
                    "params": {},
                    "time": ts
                })
    
    if current_motion:
        formatted_items.append(current_motion)

    return items_to_text_block(formatted_items)


def process_edh_file(edh_path: Path, game_root: Path, processor: ContextProcessor):

    # ---- Load EDH
    try:
        edh_data = read_json(edh_path)
    except:
        return None, "EDH Read Failed"

    game_id = edh_data.get("game_id")
    split = edh_path.parent.name

    # ---- Load .game.json
    game_path = game_root / split / f"{game_id}.game.json"
    if not game_path.exists():
        candidates = list(game_root.rglob(f"*{game_id}*.game.json"))
        if candidates:
            game_path = candidates[0]
        else:
            return None, f"Game File Not Found: {game_id}"

    try:
        game_data = read_json(game_path)
    except:
        return None, "Game JSON Read Failed"

    # ==========================================
    # MERGE STATE
    # ==========================================
    try:
        initial_objects = game_data["tasks"][0]["episodes"][0]["initial_state"]["objects"]
        initial_agents  = game_data["tasks"][0]["episodes"][0]["initial_state"]["agents"]

        objects_map = {obj["objectId"]: obj.copy() for obj in initial_objects}

        diff = edh_data.get("init_state_diff", {})
        diff_objects = diff.get("objects", {})
        diff_agents = diff.get("agents", {})

        for oid, changes in diff_objects.items():
            if oid in objects_map:
                objects_map[oid].update(changes)
            else:
                changes["objectId"] = oid
                objects_map[oid] = changes

        current_agents = diff_agents if diff_agents else initial_agents
        merged_state = {"agents": current_agents, "objects": list(objects_map.values())}

    except Exception as e:
        return None, f"State Merge Error: {e}"

    # ==========================================
    # Extract Answer
    # ==========================================
    future_actions = edh_data.get("driver_actions_future", [])
    if not future_actions:
        return None, "No Future Actions"

    next_act = future_actions[0]
    raw_action_name = next_act.get("action_name", "Unknown")

    if raw_action_name not in HIGH_LEVEL_TOOL_NAMES:
        return None, f"Filtered Action: {raw_action_name}"

    answer_json = {
        "actions": [
            {"tool_name": raw_action_name, "parameters": {}}
        ]
    }

    # ==========================================
    # HYBRID HISTORY GENERATION
    # 1. Recent Interaction: From EDH (Last 4 items)
    # 2. Summary: From Game (Everything before recent)
    # ==========================================
    
    # A. Get full history items from EDH (Reliable Source)
    edh_history_items = parse_edh_history(edh_data)
    
    # B. Split Recent (Last 4) vs Past
    # If history is shorter than 4, take all as recent
    recent_count = 4
    if len(edh_history_items) <= recent_count:
        recent_items = edh_history_items
        cutoff_time = 0.0 # No summary needed or very beginning
    else:
        recent_items = edh_history_items[-recent_count:]
        # Determine cutoff time: Start time of the first recent item
        # Use a small epsilon to ensure strict strict inequality in summary filter
        cutoff_time = recent_items[0]["time"]
    
    # C. Generate Recent Text
    recent_text = items_to_text_block(recent_items)
    
    # D. Generate Summary (From Game JSON, up to cutoff_time)
    summary_text = "No previous history."
    if cutoff_time > 0.0:
        # Build Speaker Map for Game JSON processing
        speaker_map = {}
        for speaker, utt in edh_data.get("dialog_history", []):
            speaker_map[utt.strip()] = speaker
            
        game_history_text = get_game_history_upto(game_data, cutoff_time, speaker_map)
        if game_history_text:
            summary_text = summarize_history_gpt4o(game_history_text)

    # ==========================================
    # Build Prompt
    # ==========================================

    last_instruction = "Do the next step."
    for speaker, utt in reversed(edh_data.get("dialog_history", [])):
        if speaker == "Commander":
            last_instruction = utt
            break

    state_text = processor.process_state(merged_state, last_instruction)

    full_prompt = BASE_PROMPT.format(
        initial_state=state_text,
        history_summary=summary_text,
        history_recent=recent_text
    )

    return {
        "sample": {
            "game_id": game_id,
            "instance_id": edh_data.get("instance_id"),
            "prompt": full_prompt,
            "answer": answer_json,
            "turn_last_success": True,
            "answer_all_motion_or_objectinteraction": True
        },
        "meta": {
            "split": split,
            "game_id": game_id,
            "instance_id": edh_data.get("instance_id")
        }
    }, "Success"


# ---------- MAIN SCRIPT ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-root", type=str, required=True)
    parser.add_argument("--game-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--filter-mode", type=str, default="spatial")
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API Key (optional if env var set)")
    parser.add_argument("--filter-file", type=str, default="/home/bjk/tool_learning/teach_to_tool_calling/scripts/selected_500_instance_ids.json", help="Path to JSON file containing allowed instance IDs")

    args = parser.parse_args()

    # Set OpenAI Key for LiteLLM
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    elif _HAS_LITELLM:
        print("[WARNING] OpenAI API Key not found. Please set OPENAI_API_KEY env var or pass --api-key.")
    
    # Load Filter List
    allowed_ids = None
    if args.filter_file:
        print(f"[INFO] Loading filter list from {args.filter_file}...")
        try:
            with open(args.filter_file, "r") as f:
                allowed_ids = set(json.load(f))
            print(f"[INFO] Loaded {len(allowed_ids)} instance IDs to process.")
        except Exception as e:
            print(f"[ERROR] Failed to load filter file: {e}")
            return

    processor = ContextProcessor(args.filter_mode, args.embedding_model)

    print(f"[INFO] Scanning EDH files in {args.episode_root}...")
    
    # 1. Scan All Files
    all_edh_files = sorted(list(Path(args.episode_root).glob("**/*.edh*.json")))
    print(f"[INFO] Total EDH files found: {len(all_edh_files)}")

    # 2. Pre-filter Files (Valid Unseen Only & ID Match)
    target_files = []
    
    for f in all_edh_files:
        # Condition 1: Must be in 'valid_unseen' folder (Check path parts)
        if "valid_unseen" not in f.parts:
            continue
            
        # Condition 2: Must match allowed_ids (if provided)
        if allowed_ids is not None:
            # EDH filenames are like 'instance_id.json' -> remove suffix
            fid = f.name.replace(".json", "")
            if fid not in allowed_ids:
                continue
        
        target_files.append(f)

    print(f"[INFO] Filtered down to {len(target_files)} target files (valid_unseen only + ID match).")

    out_root = Path(args.output_root)
    success_count = 0
    skipped_count = 0

    # 3. Process Only Target Files
    for edh_file in tqdm(target_files, desc="Processing"):
        result, msg = process_edh_file(edh_file, Path(args.game_root), processor)

        if result:
            sample = result["sample"]
            meta = result["meta"]
            save_dir = out_root / meta["split"] / meta["game_id"]
            save_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{meta['instance_id']}.jsonl"
            save_path = save_dir / filename
            write_jsonl(save_path, [sample])
            success_count += 1
        else:
            skipped_count += 1

    print(f"[INFO] Complete! Created: {success_count}, Skipped: {skipped_count}")


if __name__ == "__main__":
    main()