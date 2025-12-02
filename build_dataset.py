import json
import math
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional
from tqdm import tqdm

# Try to import SentenceTransformer
try:
    from sentence_transformers import SentenceTransformer, util
    _HAS_SBERT = True
except ImportError:
    _HAS_SBERT = False


# ---------- CONFIG ----------

TOOLS_JSON = Path("dataset/prompts/tools.json")

HIGH_LEVEL_TOOL_NAMES = {
    "Pickup", "Place", "Open", "Close",
    "Slice", "Dirty", "Clean", "ToggleOn", "ToggleOff",
    "Fill", "Empty", "Pour", "Break"
}

# [압축용] 이동 액션 파라미터 정의
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

### Available Tools:
{tool_list}

Your task:
Based on the latest Commander utterance and the environmental context,
decide the next robot actions.

Output format:
Return ONLY a JSON object with "actions" list.
"""

# ---------- UTILS ----------

def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f: return json.load(f)

def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False) + "\n")

def load_tools(tools_path: Path):
    if not tools_path.exists():
        candidate = Path(__file__).parent / "dataset/prompts/tools.json"
        if candidate.exists():
            tools_path = candidate
        else:
            return [], {}

    data = read_json(tools_path)
    tools = [t for t in data["tools"] if t["name"] in HIGH_LEVEL_TOOL_NAMES]
    return tools, data["mappings"]["action_idx_to_tool"]

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
                if dist < 5.0 or is_visible or is_landmark: should_keep = True
            elif self.mode == "semantic":
                if is_landmark or is_visible: should_keep = True
                if self.model and query_embedding is not None:
                    target_name = obj_type
                    obj_emb = self.model.encode(target_name, convert_to_tensor=True)
                    score = util.cos_sim(query_embedding, obj_emb).item()
                    if score > 0.3: should_keep = True
            elif self.mode == "hybrid":
                if is_landmark : should_keep = True
                else:
                    semantic_hit = False
                    if self.model and query_embedding is not None:
                        target_name = obj_type
                        obj_emb = self.model.encode(target_name, convert_to_tensor=True)
                        score = util.cos_sim(query_embedding, obj_emb).item()
                        if score > 0.35: semantic_hit = True
                    if semantic_hit or is_visible or is_nearby: should_keep = True

            if not should_keep: continue

            states = []
            if is_visible: states.append("Visible")
            else: states.append("Map Knowledge")
            if obj.get("isPickedUp"): states.append("Held by Robot")
            if obj.get("isOpen"): states.append("Open")
            if obj.get("isToggled"): states.append("On")

            parent = obj.get("parentReceptacles")
            loc_str = ""
            if parent:
                p_name = parent[0].split('|')[0] 
                loc_str = f"in/on {p_name}"

            desc_parts = [f"{dist:.1f}m", f"{rel_angle:.0f}° {direction}"]
            if loc_str: desc_parts.append(loc_str)
            if states: desc_parts.append(", ".join(states))
            
            lines.append(f"- {obj_type} ({', '.join(desc_parts)})")
        
        if not lines: return "(No interactable objects found)"
        return "\n".join(sorted(lines))

# ---------- CORE LOGIC: EDH FILE PROCESSING ----------

def process_edh_file(edh_path: Path, game_root: Path, tool_list_json: str, processor: ContextProcessor):
    try:
        edh_data = read_json(edh_path)
    except:
        return None, "EDH Read Failed"
        
    game_id = edh_data.get("game_id")
    split = edh_path.parent.name 
    
    # Game JSON 찾기
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

    # --- State Merge ---
    try:
        initial_objects = game_data["tasks"][0]["episodes"][0]["initial_state"]["objects"]
        initial_agents = game_data["tasks"][0]["episodes"][0]["initial_state"]["agents"]
        
        objects_map = {obj["objectId"]: obj.copy() for obj in initial_objects}
        
        edh_diff = edh_data.get("init_state_diff", {})
        diff_objects = edh_diff.get("objects", {})
        diff_agents = edh_diff.get("agents", {})

        for oid, changes in diff_objects.items():
            if oid in objects_map:
                objects_map[oid].update(changes)
            else:
                changes["objectId"] = oid
                objects_map[oid] = changes

        current_agents = diff_agents if diff_agents else initial_agents

        merged_state = {
            "agents": current_agents,
            "objects": list(objects_map.values())
        }
    except Exception as e:
        return None, f"State Merge Error: {e}"

    # --- Answer Extraction ---
    future_actions = edh_data.get("driver_actions_future", [])
    if not future_actions: 
        return None, "No Future Actions"
    
    next_act = future_actions[0]
    raw_action_name = next_act.get("action_name", "Unknown")
    
    # [요구사항 1] 정답 액션 필터링: High Level Tool만 허용
    if raw_action_name not in HIGH_LEVEL_TOOL_NAMES:
        return None, f"Filtered Action: {raw_action_name}"

    answer_json = {
        "actions": [{
            "tool_name": raw_action_name,
            "parameters": {} 
        }]
    }

    # --- [요구사항 2] History Generation & Motion Compression ---
    # Build a timestamp-aware interleaved history (dialog + actions).
    # 1. Build timestamp map from EDH 'interactions' (to approximate dialog times)
    timeline = []
    interactions = edh_data.get("interactions", [])
    dialog_ts_map = {}
    for it in interactions:
        utt = it.get("utterance")
        if utt:
            dialog_ts_map[utt.strip().lower()] = it.get("time_start", 0)

    # 2. Dialog History (with approx timestamps)
    for speaker, utter in edh_data.get("dialog_history", []):
        key = utter.strip().lower()
        approx_ts = dialog_ts_map.get(key, -1)
        timeline.append({
            "type": "dialog",
            "role": speaker,
            "content": utter,
            "time": approx_ts
        })

    # Actions (여기서 압축 로직 적용)
    action_history = edh_data.get("driver_action_history", [])
    compressed_actions = []
    
    current_motion = None # {"name": "Forward", "params": {...}, "count": 1}

    for act in action_history:
        name = act.get("action_name")
        
        # 이동 액션인지 확인
        is_motion = name in MOTION_PARAMS
        
        if is_motion:
            # 같은 이동 액션이 연속되면 합침
            if current_motion and current_motion["name"] == name:
                # 파라미터 누적
                base_params = MOTION_PARAMS[name]
                for k, v in base_params.items():
                    current_motion["params"][k] += v
                current_motion["count"] += 1
            else:
                # 다른 액션이 나오면 이전 것 저장
                if current_motion:
                    compressed_actions.append(current_motion)
                
                # 새로운 모션 시작
                current_motion = {
                    "type": "action",
                    "name": name,
                    "tool_name": "motion_delta",
                    "params": MOTION_PARAMS[name].copy(),
                    "count": 1,
                    "time": act.get("time_start", 0)
                }
        else:
            # 이동 액션이 아니면 (Pickup 등)
            if current_motion:
                compressed_actions.append(current_motion)
                current_motion = None
            
            # 일반 액션 추가
            compressed_actions.append({
                "type": "action",
                "name": name,
                "tool_name": name, # Pickup 등 원래 이름
                "params": {},
                "time": act.get("time_start", 0)
            })
            
    # 마지막 남은 모션 처리
    if current_motion:
        compressed_actions.append(current_motion)

    # Dialog와 Action 합치기 (시간 정보가 있다면 정렬, 없으면 Dialog 먼저)
    # TEACh EDH 파일은 보통 dialog_history가 먼저 오고 그 뒤에 driver_action_history가 시간 순으로 옴
    # 여기서는 단순 리스트 병합 후 텍스트화
    
    # 3. Merge dialog + action → full timeline
    full_history_list = timeline + compressed_actions

    # 4. Sort by timestamp (dialogs without timestamp have time=-1 and will be placed first)
    full_history_list = sorted(full_history_list, key=lambda x: x["time"])
    
    # 텍스트 변환
    history_lines = []
    for item in full_history_list:
        if item["time"] < 0:
            # Dialogs without timestamp
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

    history_text = "\n".join(history_lines[-20:]) # 최근 20개만

    # --- Instruction ---
    last_instruction = "Do the next step."
    for turn in reversed(edh_data.get("dialog_history", [])):
        if turn[0] == "Commander":
            last_instruction = turn[1]
            break
            
    state_text = processor.process_state(merged_state, last_instruction)
    
    full_prompt = BASE_PROMPT.format(
        initial_state=state_text,
        history=history_text,
        tool_list=tool_list_json
    )
    

    return {
        "sample": {
            "game_id": game_id,
            "instance_id": edh_data.get("instance_id"),
            "prompt": full_prompt,
            "answer": answer_json,
            # [요구사항 4] Success 기준: EDH 데이터는 성공한 궤적을 바탕으로 하므로 True
            "turn_last_success": True, 
            "answer_all_motion_or_objectinteraction": True
        },
        "meta": {
            "split": split,
            "game_id": game_id,
            "instance_id": edh_data.get("instance_id")
        }
    }, "Success"


# ---------- MAIN ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-root", type=str, required=True, help="Folder with EDH instances")
    parser.add_argument("--game-root", type=str, required=True, help="Folder with Game files")
    parser.add_argument("--output-root", type=str, required=True)
    
    parser.add_argument("--filter-mode", type=str, default="spatial")
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2")
    
    args = parser.parse_args()

    tools, _ = load_tools(TOOLS_JSON)
    tool_list_json = json.dumps(tools, ensure_ascii=False)
    processor = ContextProcessor(args.filter_mode, args.embedding_model)
    
    print(f"[INFO] Scanning EDH files in {args.episode_root}...")
    all_edh_files = sorted(list(Path(args.episode_root).glob("**/*.edh*.json")))
    print(f"[INFO] Found {len(all_edh_files)} EDH files.")

    out_root = Path(args.output_root)
    
    success_count = 0
    skipped_count = 0
    
    for edh_file in tqdm(all_edh_files, desc="Processing"):
        result, msg = process_edh_file(edh_file, Path(args.game_root), tool_list_json, processor)
        
        if result:
            sample = result["sample"]
            meta = result["meta"]
            
            # [요구사항 3] 저장 구조: output/split/game_id/instance_id.jsonl
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