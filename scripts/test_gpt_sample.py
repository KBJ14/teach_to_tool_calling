#!/usr/bin/env python3
"""
Script: test_gpt4o_sample_hardcoded.py
- Uses the LiteLLM (OpenAI/GPT-4o) API to generate a response.
- API Key is defined directly within the script (NOT RECOMMENDED for production/sharing).
"""

import argparse
import json
import os
import sys
import re
from litellm import completion # API 호출용 라이브러리

# ************************************************
# ⚠️ 보안 경고: 이 부분을 실제 키로 교체해야 합니다! ⚠️
API_KEY = ""
# ************************************************

# --- Configuration ---
DEFAULT_MODEL = "gpt-4o" 
DEFAULT_DATASET_DIR = 'dataset_experiments/hybrid_ours/valid_unseen' 

def log(*args, **kwargs):
    """Print to stderr for logging/debugging."""
    print(*args, file=sys.stderr, **kwargs)

def find_matching_sample(dataset_dir):
    """
    Recursively search for the first valid .jsonl file in the dataset directory.
    """
    log(f"[INFO] Scanning directory: {dataset_dir}")
    
    for root, dirs, files in os.walk(dataset_dir):
        for f in files:
            if f.endswith('.jsonl'):
                fp = os.path.join(root, f)
                try:
                    with open(fp, 'r', encoding='utf-8') as fh:
                        line = fh.readline()
                        if not line.strip(): continue
                        ep = json.loads(line)
                        if 'prompt' in ep and 'answer' in ep:
                            return ep, fp
                except Exception as e:
                    log(f"[WARNING] Error reading {fp}: {e}")
                    continue
    return None, None

def strict_parse_json(text):
    """
    Parses JSON from text, strictly looking for {"actions": [...]}.
    """
    # 1. Markdown Code Block 패턴 추출 (```json ... ```)
    markdown_matches = list(re.finditer(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL))
    for match in reversed(markdown_matches):
        try:
            obj = json.loads(match.group(1))
            if "actions" in obj and isinstance(obj["actions"], list):
                return obj
        except: continue

    # 2. Try raw JSON parsing
    try:
        obj = json.loads(text)
        if "actions" in obj and isinstance(obj["actions"], list):
            return obj
    except: pass

    # 3. Try finding brackets if text is mixed with explanation
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', default=DEFAULT_MODEL, help='API Model name (e.g., gpt-4o)') 
    parser.add_argument('--dataset-dir', default=DEFAULT_DATASET_DIR, help='Root directory to search for .jsonl files')
    parser.add_argument('--max-tokens', type=int, default=1024, help='Max tokens for API response')
    parser.add_argument('--sample-file', default=None, help='Path to a specific jsonl file to test (optional)')
    
    args = parser.parse_args()
    
    # 0. Key Check
    if API_KEY == "YOUR_GPT_4O_API_KEY_HERE":
        log("[FATAL] ⚠️ Please update the API_KEY variable at the top of the script!")
        sys.exit(1)


    # 1. Find Sample
    log('[INFO] Searching dataset for a matching sample...')
    sample, fp = None, None
    
    if args.sample_file:
        if not os.path.isfile(args.sample_file):
            log(f'[ERROR] File not found: {args.sample_file}')
            sys.exit(1)
        fp = args.sample_file
        with open(fp, 'r', encoding='utf-8') as fh:
            line = fh.readline()
            if line:
                sample = json.loads(line)
    else:
        sample, fp = find_matching_sample(args.dataset_dir)

    if not sample:
        log(f'[ERROR] No valid sample found in {args.dataset_dir}')
        sys.exit(1)

    log(f'[INFO] Using sample from: {fp}')
    log(f'[INFO] Model: {args.model_name}')
    
    prompt = sample.get('prompt', '')
    
    # 2. Generate using API (LiteLLM)
    log('[INFO] Generating response via API...')
    
    messages = [
        {"role": "system", "content": "You are a robot control assistant. Respond ONLY with a valid JSON object containing the 'actions' list. Do not explain."},
        {"role": "user", "content": prompt}
    ]
    
    full_response = ""
    try:
        # LiteLLM completion API 호출, api_key 인자로 직접 전달
        response = completion(
            model=args.model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=args.max_tokens,
            api_key=API_KEY # 하드코딩된 키 전달
        )
        full_response = response.choices[0].message.content
        
    except Exception as e:
        log(f'[ERROR] API call failed: {e}')
        log('Check your API_KEY and rate limits.')
        sys.exit(1)

    # 3. Extract & Parse
    content_to_parse = full_response.strip() 
    
    log('--- [API Model Output] ---')
    log(content_to_parse)
    log('--------------------------')

    parsed_json = strict_parse_json(content_to_parse)

    if parsed_json:
        print("\n=== Parsed JSON Action ===")
        print(json.dumps(parsed_json, indent=2))
        
        gt = sample.get("answer", {})
        gt_action = gt.get("actions", [{}])[0].get("tool_name")
        pred_action = parsed_json.get("actions", [{}])[0].get("tool_name")
        
        log(f'\n[Compare] Ground Truth Action: {gt_action}')
        log(f'[Compare] Predicted Action: {pred_action}')
        
        if gt_action == pred_action:
             log('[RESULT] Action match: True ✅')
        else:
             log('[RESULT] Action match: False ❌')
        
        sys.exit(0)
    else:
        log('[ERROR] Failed to parse JSON (missing "actions" list or invalid format).')
        log(f'[RAW RESPONSE] {content_to_parse}')
        sys.exit(1)

if __name__ == '__main__':
    main()