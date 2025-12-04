#!/usr/bin/env python3
"""
Script: batch_inference_api_hardcoded_final.py
- Uses LiteLLM for API model evaluation (GPT-4o, Claude, etc.).
- Includes Exponential Backoff and Jitter for robust Rate Limit handling.
- API Key is hardcoded (⚠️ USE WITH CAUTION).
- Mirrors directory structure and uses strict JSON parsing.
"""

import argparse
import json
import os
import sys
import re
import asyncio
import random # 👈 1. [추가] 랜덤 임포트 (지터용)
from tqdm.asyncio import tqdm_asyncio
from litellm import acompletion, RateLimitError 

# ************************************************
# ⚠️ 보안 경고: 이 부분을 실제 API 키로 교체해야 합니다! ⚠️
API_KEY = ""
# ************************************************

# --- Configuration ---
API_MODEL = "gpt-4o" 
MAX_CONCURRENT_REQUESTS = 1    # 👈 2. [수정] 동시 요청 수를 보수적으로 낮춤 (10 -> 5)
SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
MAX_RETRIES = 5                # 최대 재시도 횟수
BASE_DELAY = 5                 # 기본 지연 시간 (초)

def log(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def strict_parse_json(text):
    """
    Parses JSON from text, strictly looking for {"actions": [...]}.
    (이전 버전과 동일한 엄격한 파싱 로직)
    """
    markdown_matches = list(re.finditer(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL))
    for match in reversed(markdown_matches):
        try:
            obj = json.loads(match.group(1))
            if "actions" in obj and isinstance(obj["actions"], list):
                return obj
        except: continue
    try:
        obj = json.loads(text)
        if "actions" in obj and isinstance(obj["actions"], list):
            return obj
    except: pass
    end_indices = [m.end() for m in re.finditer(r"\}", text)]
    start_indices = [m.start() for m in re.finditer(r"\{", text)]
    for end in reversed(end_indices):
        for start in reversed(start_indices):
            if start >= end: continue
            candidate = text[start:end]
            try:
                obj = json.loads(candidate)
                if "actions" in obj and isinstance(obj["actions"], list):
                    return obj
            except: continue
    return None

def get_all_jsonl_files(root_dir):
    jsonl_files = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith('.jsonl'):
                jsonl_files.append(os.path.join(root, f))
    return sorted(jsonl_files)

async def process_single_file(input_path, args):
    """
    Reads one file, calls API (with key), applies retry logic, saves result.
    """
    relative_path = os.path.relpath(input_path, args.input_dir)
    output_path = os.path.join(args.output_dir, relative_path)
    
    if os.path.exists(output_path):
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        # Read Input
        with open(input_path, 'r', encoding='utf-8') as fh:
            line = fh.readline()
            if not line.strip(): return
            sample = json.loads(line)

        if 'prompt' not in sample: return
        prompt_content = sample['prompt']
        
        generated_text = ""
        
        # --- [변경] 재시도 루프 시작 ---
        for attempt in range(MAX_RETRIES):
            # 3. [추가] 동시 요청 세마포어 시작
            async with SEMAPHORE:
                try:
                    response = await acompletion(
                        model=args.model_name,
                        messages=[
                            {"role": "system", "content": "You are a robot control assistant. Respond ONLY with a valid JSON object containing the 'actions' list. Do not explain."},
                            {"role": "user", "content": prompt_content}
                        ],
                        temperature=0.0,
                        max_tokens=1024,
                        api_key=API_KEY
                    )
                    generated_text = response.choices[0].message.content
                    break # 성공 시 루프 탈출
                    
                except RateLimitError as e:
                    # 4. [추가] RateLimitError 처리 및 지수 백오프/지터 적용
                    delay = BASE_DELAY * (2 ** attempt) + random.uniform(0, 1) 
                    log(f"\n[RATE LIMIT] File: {os.path.basename(input_path)}. Attempt {attempt+1}/{MAX_RETRIES}. Waiting {delay:.2f}s...")
                    
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(delay)
                    else:
                        # 최종 실패 시 에러 텍스트 저장 후 루프 탈출
                        generated_text = f"API_ERROR: RateLimitError persisted after {MAX_RETRIES} attempts. Details: {str(e)}"
                        break
                
                except Exception as e:
                    # 다른 API 에러(인증, 모델 없음 등)는 재시도 없이 처리
                    generated_text = f"API_ERROR: Non-RateLimit Error: {str(e)}"
                    log(f"[CRITICAL API ERROR] {input_path}: {e}")
                    break
        # --- 재시도 루프 종료 ---


        # Parse
        parsed_json = strict_parse_json(generated_text)

        # Save
        result_entry = {
            "game_id": sample.get("game_id"),
            "instance_id": sample.get("instance_id"),
            "origin_file": input_path,
            "model_name": args.model_name,
            "model_output_raw": generated_text, 
            "prediction": parsed_json,
            "ground_truth": sample.get("answer", {})
        }

        # Write Output
        with open(output_path, 'w', encoding='utf-8') as out_fh:
            out_fh.write(json.dumps(result_entry, ensure_ascii=False) + '\n')

    except Exception as e:
        log(f"[FATAL ERROR] processing {input_path}: {e}")

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', default=API_MODEL, help='Model name for LiteLLM (e.g., gpt-4o, claude-3-5-sonnet)')
    parser.add_argument('--input-dir', required=True, help='Root directory of input dataset')
    parser.add_argument('--output-dir', required=True, help='Root directory where results will be mirrored')
    parser.add_argument('--ids-file', default=None, help='Optional JSON file with list of instance_id strings to process (will filter input files)')
    parser.add_argument('--dry-run', action='store_true', help='If set, only list matched files and do not call API')
    
    args = parser.parse_args()

    # Key Check
    if API_KEY == "YOUR_GPT_4O_API_KEY_HERE":
        log("[FATAL] ⚠️ Please update the API_KEY variable at the top of the script!")
        sys.exit(1)

    input_files = get_all_jsonl_files(args.input_dir)
    log(f'[INFO] Found {len(input_files)} files. Model: {args.model_name}')

    if not input_files:
        return

    # If ids-file provided, filter input_files to only those whose first-line `instance_id` is in the list
    if args.ids_file:
        # Resolve path robustly: try as absolute, cwd-relative, script-relative, or repo-root relative.
        ids_path = args.ids_file
        if not os.path.exists(ids_path):
            # Candidate 1: cwd-relative (probably same as provided already)
            cwd_candidate = os.path.join(os.getcwd(), args.ids_file)
            if os.path.exists(cwd_candidate):
                ids_path = cwd_candidate
            else:
                # Candidate 2: script dir relative
                script_dir_candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.ids_file)
                if os.path.exists(script_dir_candidate):
                    ids_path = script_dir_candidate
                else:
                    # Candidate 3: repo-root relative (one level up from script dir)
                    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    repo_root_candidate = os.path.join(repo_root, args.ids_file)
                    if os.path.exists(repo_root_candidate):
                        ids_path = repo_root_candidate
                    else:
                        raise FileNotFoundError(f"--ids-file not found at: {args.ids_file}. Tried cwd ({cwd_candidate}), script dir ({script_dir_candidate}), and repo root ({repo_root_candidate}). Provide a valid path or use an absolute path.")
        with open(ids_path, 'r', encoding='utf-8') as f:
            ids_list = json.load(f)
        ids_set = set(ids_list)
        log(f'[INFO] Filtering inputs using ids-file {ids_path} ({len(ids_set)} ids)')
        filtered = []
        for p in input_files:
            try:
                with open(p, 'r', encoding='utf-8') as fh:
                    line = fh.readline().strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    inst_id = entry.get('instance_id')
                    if inst_id and inst_id in ids_set:
                        filtered.append(p)
            except Exception:
                continue
        input_files = filtered
        log(f'[INFO] After filtering, {len(input_files)} files will be processed.')

    

    tasks = [process_single_file(f, args) for f in input_files]
    
    await tqdm_asyncio.gather(*tasks, desc=f"API Inference ({args.model_name})")

    # 🔥 여기서 남아 있는 다른 asyncio task(주로 라이브러리 내부)들 정리
    current_task = asyncio.current_task()
    pending = [t for t in asyncio.all_tasks() if t is not current_task and not t.done()]
    for t in pending:
        t.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    log('[INFO] Cleanup done. Exiting main().')

if __name__ == '__main__':
    asyncio.run(main())