#!/usr/bin/env python3
"""
Script: batch_inference_fc.py
- Uses LiteLLM with OpenAI Native Function Calling (Tools API).
- Loads 'description' directly from tools.json for fair comparison.
- Filters out Navigation and low-level tools (motion_delta, etc.).
- Forces tool selection via tool_choice='required'.
- Handles Rate Limits with Exponential Backoff + Jitter.
"""

import argparse
import json
import os
import sys
import asyncio
import random
import re
from tqdm.asyncio import tqdm_asyncio
from litellm import acompletion, RateLimitError 

# ************************************************
# ⚠️ 보안 경고: 실제 API 키로 교체하세요
API_KEY = ""  # <- 여기에 실제 키 넣기
# ************************************************
 
# --- Configuration ---
API_MODEL = "openai/gpt-4o" 
MAX_CONCURRENT_REQUESTS = 5   # Rate Limit에 따라 조절 (에러 많으면 1~2로 낮춤)
SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
MAX_RETRIES = 5
BASE_DELAY = 5

# --- Target Tools Filter ---
# 이 목록에 있는 도구만 tools.json에서 가져와 API에 전달합니다. (Navigation, motion_delta 제외됨)
TARGET_TOOL_NAMES = {
    "Pickup", "Place", "Open", "Close",
    "Slice", "Dirty", "Clean", "ToggleOn", "ToggleOff",
    "Fill", "Empty", "Pour", "Break"
}

safety_settings = [
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    # ... 나머지 카테고리
]

def log(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def load_tools_schema_from_file(json_path):
    """
    tools.json 파일을 읽어서 OpenAI Function Calling Schema로 변환합니다.
    Description을 그대로 유지하여 Prompt 방식과 공정하게 비교합니다.
    """
    if not os.path.exists(json_path):
        log(f"[ERROR] Tools file not found: {json_path}")
        sys.exit(1)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    openai_tools = []
    
    # tools.json의 "tools" 리스트 순회
    for tool in data.get("tools", []):
        tool_name = tool.get("name")
        
        # 1. 타겟 도구인지 확인 (Navigation 등 제외)
        if tool_name in TARGET_TOOL_NAMES:
            # 2. Schema 생성
            schema = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool.get("description", ""), # 👈 핵심: 원본 설명 사용
                    "parameters": {
                        "type": "object",
                        "properties": {}, 
                        "required": []
                    }
                }
            }
            openai_tools.append(schema)
            
    log(f"[INFO] Loaded {len(openai_tools)} tools from {json_path}")
    return openai_tools

def parse_tool_calls(response_message):
    """
    API 응답에서 tool_calls를 추출하여 기존 포맷인 {"actions": [...]}로 변환합니다.
    """
    if not hasattr(response_message, 'tool_calls') or not response_message.tool_calls:
        return None

    actions = []
    for tool_call in response_message.tool_calls:
        function_name = tool_call.function.name
        try:
            # 인자 파싱 (보통 빈 객체 {} 임)
            function_args = json.loads(tool_call.function.arguments)
        except:
            function_args = {}
            
        actions.append({
            "tool_name": function_name,
            "parameters": function_args
        })
    
    if not actions:
        return None
    return {"actions": actions}

def get_all_jsonl_files(root_dir):
    jsonl_files = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith('.jsonl'):
                jsonl_files.append(os.path.join(root, f))
    return sorted(jsonl_files)

async def process_single_file(input_path, args, tools_schema):
    relative_path = os.path.relpath(input_path, args.input_dir)
    output_path = os.path.join(args.output_dir, relative_path)
    
    # 이미 처리된 파일 건너뛰기
    if os.path.exists(output_path):
        return

    # 폴더 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        # 입력 파일 읽기
        with open(input_path, 'r', encoding='utf-8') as fh:
            line = fh.readline()
            if not line.strip():
                return
            sample = json.loads(line)

        if 'prompt' not in sample:
            return
        prompt_content = sample['prompt']
        
        final_prediction = None
        raw_response_content = ""
        
        # 재시도 루프
        for attempt in range(MAX_RETRIES):
            async with SEMAPHORE:
                try:
                    response = await acompletion(
                        model=args.model_name,
                        messages=[
                            {"role": "system", "content": "You are a robot control assistant. Select the correct function based on the context."},
                            {"role": "user", "content": prompt_content}
                        ],
                        temperature=1,
                        max_tokens=1024,      # openai/gpt-4o 는 아직 max_tokens 사용 가능
                        api_key=API_KEY,
                        tools=tools_schema,   # 👈 파일에서 로드한 스키마 사용
                        tool_choice="required",   # 👈 반드시 도구 사용 강제
                    )
                    
                    message = response.choices[0].message
                    raw_response_content = str(message)  # 디버깅용 전체 응답 저장
                    final_prediction = parse_tool_calls(message)
                    break 
                    
                except RateLimitError as e:
                    # 지수 백오프 + 지터
                    delay = BASE_DELAY * (2 ** attempt) + random.uniform(0, 1) 
                    log(f"\n[RATE LIMIT] File: {os.path.basename(input_path)}. Attempt {attempt+1}/{MAX_RETRIES}. Waiting {delay:.2f}s...")
                    
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(delay)
                    else:
                        raw_response_content = f"API_ERROR: RateLimitError persisted. {e}"
                        break
                except Exception as e:
                    raw_response_content = f"API_ERROR: {e}"
                    log(f"[CRITICAL API ERROR] {input_path}: {e}")
                    break

        # 결과 저장 구조
        result_entry = {
            "game_id": sample.get("game_id"),
            "instance_id": sample.get("instance_id"),
            "origin_file": input_path,
            "model_name": args.model_name,
            "model_output_raw": raw_response_content,
            "prediction": final_prediction,
            "ground_truth": sample.get("answer", {})
        }

        # 파일 쓰기
        with open(output_path, 'w', encoding='utf-8') as out_fh:
            out_fh.write(json.dumps(result_entry, ensure_ascii=False) + '\n')

    except Exception as e:
        log(f"[FATAL ERROR] processing {input_path}: {e}")

DEFAULT_IDS_FILE = os.path.join(os.path.dirname(__file__), 'selected_500_instance_ids.json')


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', default=API_MODEL)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--tools-file', required=True, help="Path to tools.json file") # 필수 인자
    parser.add_argument('--ids-file', default=DEFAULT_IDS_FILE,
                        help="Optional: JSON list of instance_ids to process (defaults to selected_500_instance_ids.json in this scripts folder)") 
    
    args = parser.parse_args()

    # ← 원래는 "YOUR_GPT_4O_API_KEY_HERE" 와 비교했는데, 지금 기본값은 "" 이라 항상 통과함
    if not API_KEY:
        log("[FATAL] Please update the API_KEY variable at the top of the script!")
        sys.exit(1)

    # 1. 도구 스키마 로드
    tools_schema = load_tools_schema_from_file(args.tools_file)

    # 2. 파일 목록 수집
    input_files = get_all_jsonl_files(args.input_dir)
    log(f'[INFO] Found {len(input_files)} total files.')

    # 3. (옵션) 특정 ID만 필터링 - 기본으로 scripts/selected_500_instance_ids.json 사용
    if args.ids_file:
        if os.path.exists(args.ids_file):
            with open(args.ids_file, 'r') as f:
                target_ids = set(json.load(f))
            
            filtered_files = []
            for p in input_files:
                try:
                    with open(p, 'r') as fh:
                        line = fh.readline()
                        if not line:
                            continue
                        data = json.loads(line)
                        if data.get("instance_id") in target_ids:
                            filtered_files.append(p)
                except:
                    continue
            input_files = filtered_files
            log(f'[INFO] Filtered to {len(input_files)} files based on ids-file.')
        else:
            log(f'[WARNING] ids-file provided but not found: {args.ids_file}. Processing all files instead.')

    if not input_files:
        log('[INFO] No files to process.')
        return

    log(f'[INFO] Starting FC Inference. Model: {args.model_name}')

    # 4. 작업 시작
    tasks = [process_single_file(f, args, tools_schema) for f in input_files]
    await tqdm_asyncio.gather(*tasks, desc=f"FC Inference ({args.model_name})")

    log('[INFO] All inference tasks finished, cleaning up pending asyncio tasks...')

    # 🔥 여기서 남아 있는 다른 asyncio task(주로 라이브러리 내부)들 정리
    current_task = asyncio.current_task()
    pending = [t for t in asyncio.all_tasks() if t is not current_task and not t.done()]
    for t in pending:
        t.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    log('[INFO] Cleanup done. Exiting main().')


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("[INFO] KeyboardInterrupt received. Exiting.")
        sys.exit(0)
