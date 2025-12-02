import json
import math
import argparse
from pathlib import Path
from typing import Any, Dict, List
from tqdm import tqdm

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

### Interaction History (Chronological):
{history}

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

# ---------- CORE: EDH Processing ----------

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
    # MERGE STATE (initial_state + init_state_diff)
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
    # Extract next action (driver_actions_future[0])
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
    # BUILD TRUE INTERLEAVED HISTORY (dialog + action sorted by timestamp)
    # ==========================================

    timeline = []

    # ---- Build timestamp map from EDH 'interactions'
    interactions = edh_data.get("interactions", [])
    dialog_ts_map = {}
    for it in interactions:
        utt = it.get("utterance")
        if utt:
            dialog_ts_map[utt.strip().lower()] = it.get("time_start", 0)

    # ---- 1. Dialog History (with approx timestamps)
    for speaker, utter in edh_data.get("dialog_history", []):
        key = utter.strip().lower()
        approx_ts = dialog_ts_map.get(key, -1)
        timeline.append({
            "type": "dialog",
            "role": speaker,
            "content": utter,
            "time": approx_ts
        })

    # ---- 2. Driver Actions (with motion compression)
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

    # ---- 3. Merge dialog + action → full timeline
    full_history_list = timeline + compressed_actions

    # ---- 4. Sort by timestamp
    full_history_list = sorted(full_history_list, key=lambda x: x["time"])

    # ---- 5. Convert to readable text
    history_lines = []
    for item in full_history_list:
        if item["time"] < 0:   # dialog with no timestamp
            if item["type"] == "dialog":
                history_lines.append(f"{item['role']}: {item['content']}")
            else:
                history_lines.append(f"Driver (action): {json.dumps({'tool_name': item['tool_name'], 'parameters': item['params']})}")
        else:
            if item["type"] == "dialog":
                history_lines.append(f"{item['role']} ({item['time']:.1f}s): {item['content']}")
            else:
                ajson = {"tool_name": item["tool_name"], "parameters": item["params"]}
                history_lines.append(f"Driver (action @ {item['time']:.1f}s): {json.dumps(ajson)}")

    history_text = "\n".join(history_lines[-20:])  # last 20 items

    # ==========================================
    # Build instruction
    # ==========================================

    last_instruction = "Do the next step."
    for speaker, utt in reversed(edh_data.get("dialog_history", [])):
        if speaker == "Commander":
            last_instruction = utt
            break

    state_text = processor.process_state(merged_state, last_instruction)

    full_prompt = BASE_PROMPT.format(
        initial_state=state_text,
        history=history_text
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

    args = parser.parse_args()

    processor = ContextProcessor(args.filter_mode, args.embedding_model)

    print(f"[INFO] Scanning EDH files in {args.episode_root}...")
    all_edh_files = sorted(list(Path(args.episode_root).glob("**/*.edh*.json")))
    print(f"[INFO] Found {len(all_edh_files)} EDH files.")

    out_root = Path(args.output_root)
    success_count = 0
    skipped_count = 0

    for edh_file in tqdm(all_edh_files, desc="Processing"):
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
