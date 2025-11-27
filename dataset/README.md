Dataset loader and template for TEACh-style episodes

Overview
- This folder includes a lightweight loader (`loader.py`) that converts the episode_data_wo_state style entries into a turn-level dataset for LLM training/inference.

Schema and important fields
- base_prompt: text template "llm, {initial_state}". The loader includes the actual `initial_state` from the matching `.game.json` when available.
- initial_state: the episode's initial scene state (object positions, agents...). Pulled from the referenced game file.
- tool_list: a list of tools (optional) — read from a `tools.json` file if supplied. A sample mapping is in `tools.sample.json`.
- history: chronological dialogue + tool actions before the turn (dialogues as `{type: 'dialogue', text: ...}`, historical actions converted into `{type: 'tool', tool: <name>, params: {}}`).
- actions: canonical list of tool actions for the current turn. Motion actions are merged into `motion` with a single `motion_delta` param. The loader removes sentinel parameters equal to `-1`.
- state_diff: if the turn can be matched to a statediff file, the loader will include its parsed JSON (agent pose + object deltas).
-- answer: the loader returns only `actions` in `answer` (no textual `utterance` field). If the last action in a turn is dialogue the `answer.actions` list is empty (per your requirement); otherwise `answer.actions` lists the canonical tool actions for the turn.

Key behaviors implemented
- Motion merging — consecutive actions containing `pose_delta` are summed into a single `motion` tool entry.
- Parameter cleanup — any parameter with value `-1` is omitted.
- Heuristic fallbacks — if a mapping for an action id is not present in tools.json, the loader tries to guess by raw fields (uid/object interaction/utterance).

How to use
1) Edit or provide your `tools.json` mapping (the loader will auto-detect `teach_to_tool_calling/tools.json` if you don't pass `--tools`; see `tools.sample.json` for a minimal example).
2) Run the loader over an episode file (example below).

Example
```
python -m teach_to_tool_calling.dataset.loader \
    /home/bjk/tool_learning/teach_to_tool_calling/episode_data_wo_state/task_308/episode_0.json \
    --tools /home/bjk/tool_learning/teach_to_tool_calling/dataset/tools.sample.json \
    --images_dir /teach_dataset/images
```

This will emit a newline-delimited JSON file with one record per turn named `<episode_fn>.dataset.jsonl`.

If you'd like, I can now add a small example runner/test that demonstrates converting one episode file into the dataset records and show its output. 
# TEACh dataset extraction: Task -> Episode -> Turn

This module extracts turns from TEACh EDH instance files and compresses consecutive agent actions.

Goals:
- Group dataset into tasks -> episodes -> turns
- Each turn is a block of consecutive agent actions/utterances (driver), separated by Commander messages
- Each turn carries the environment state before the first action of that turn (optionally)
- Consecutive identical driver actions are compressed with a `count` field

Usage:

1. install TEACh dataset under `/teach_dataset` or whatever root path.
   Set `DATA_ROOT` accordingly, or use `--data_root`.

2. Run the extractor (per-task split output; `--out-dir` is required):

```bash
python teach_to_tool_calling/dataset/extract_turns.py \
  --data_root /teach_dataset \
  --out-dir /home/bjk/tool_learning/teach_to_tool_calling/episode_data_wo_state \
  --compress
```

python teach_to_tool_calling/dataset/extract_turns.py \
  --data_root /teach_dataset \
  --out-dir /home/bjk/tool_learning/teach_to_tool_calling/turns_test \
  --limit 10 \
  --compress



- `--compress`: compresses consecutive identical actions within each agent turn
Note: state lookup is no longer performed by default. Use the `game_id` and the first compressed
action timestamp from each turn to lookup state JSONs externally if needed.

3. Output format:
The output file (JSON) is a list of entries, each containing:
- `edh_fn`: path to the EDH instance file
- `game_id`: ID of the game
- `game_fn`: game JSON path (if found)
  - `task_idx`: index of task from EDH. If `task_idx` is missing/unknown the exporter now assigns a small numeric label starting at 0
    in order of first appearance for that `game_id`/`episode_idx` combination; this keeps
    directory names deterministic and easy to reference. Episode indices are handled similarly
    — unknown `episode_idx` values get numbered starting at 0 per-task.
  - `episode_idx`: index of episode. 
- `turns`: list of turns. Each turn contains:
  - `start_interaction_idx`/`end_interaction_idx`: indices into `interactions`
  - `agent_id`: id of agent performing this turn
  - `actions`: compressed list of actions in this turn. Each action has `action_id`, `action_name`, `count`, and `time_starts` (all timestamps for compressed frames)
  - `commander_context`: list of Commander interactions directly preceding this turn
  - `state_fn`: path to state JSON (if found)
  - `state`: this field is not populated by the extractor; fetch states externally by matching `game_id` + timestamp
  - `history`: chronological list of interactions (commander + driver) before this turn; preferred way to access combined dialogue+action context.
  - `pre_turn_dialogs` / `pre_turn_actions`: preserved for backward compatibility.
  - Compressed action objects now include aggregated success fields under `successes` (list of underlying success values)
    and top-level conveniences: `any_success`, `all_success`, and `last_success`. These map to boolean values where possible
    (1 == success, 0 == failure); `None` indicates no explicit success flags were found.

4. To group by task -> episode, run:

```python
from teach_to_tool_calling.dataset.extract_turns import iter_edh_and_extract_all, group_by_task_episode
results = iter_edh_and_extract_all('/path/to/teach_dataset', compress=True, include_state=False)
by_task = group_by_task_episode(results)
```

Notes & tips:
- The script reuses heuristics from `teach_to_tool_calling/dataset/scripts/preview_raw_turn.py` to find associated `game.json` and `state.json` files.
- This extraction is robust for EDH instances; you can limit the number of EDH files with `--limit`.
- By default the script produces a flat list of EDH -> turns; `group_by_task_episode` will produce a nested mapping suitable for training/analysis.
If you want to write this into `teach` repo standard dataset structure, the `group_by_task_episode` result is a good starting point to write per-task JSON files.

State lookup 테스트
-------------------

데이터 추출기에서는 기본적으로 `state` 필드를 채우지 않습니다. `game_id`와 각 턴의 첫 번째 압축 액션 타임스탬프를 사용해 외부에서 상태 파일을 찾을 수 있습니다. 아래는 새로 제공된 헬퍼와 테스트 스크립트 사용 예시입니다.

- 단일 턴 테스트 (CLI):

```bash
./teach_to_tool_calling/scripts/run_test_fetch_one_default.sh
```

이 명령은 지정한 에피소드의 `--idx` 인덱스(기본 0) 턴에 대해 `state` 파일 경로를 찾아 출력하고, 가능하면 JSON 내용을 간단히 미리 보여줍니다.

- 파이썬에서 헬퍼 함수 사용:

```python
from teach_to_tool_calling.state_helpers import load_json, find_state_for_interaction

ep = load_json('/path/to/episode_data_wo_state/some_episode.json')
state_path = find_state_for_interaction(ep, interaction_idx=0, images_dir='/teach_dataset/images')
print(state_path)
```

`find_state_for_interaction`은 기본적으로 `['_state.json', '.state.json', '_state', '.json']` 같은 접미사를 시도합니다. 다른 패턴을 사용하려면 `state_suffixes` 인자를 전달하세요.

원하면 이 README에 더 많은 예시(예: state 내용을 interaction에 직접 삽입하는 방법)를 추가해 드리겠습니다.

Note about state lookup:

 - The script prefers to use the timestamp of the first compressed action (`actions[0].time_starts[0]`) to find a state file recorded immediately before that action.
 - If a time-based match isn't available the script falls back to the `start_interaction_idx` lookup and then to the previous-turn/initial state fallbacks.

Per-task/episode export and dataset stats
---------------------------------------

In addition to writing a single JSON for all EDH instances, the script supports saving outputs per-task/episode and generating basic dataset statistics.

Examples:

```bash
python teach_to_tool_calling/dataset/extract_turns.py \
  --data_root /path/to/teach_dataset \
  --out-dir /tmp/turns_by_task \
  --compress \
  --stats-out /tmp/turns_stats.json
```

- `--out-dir` will produce a directory structure such as:
  - `/tmp/turns_by_task/task_<task_idx>/episode_<episode_idx>.json`
  - Each file is a JSON array of EDH entries belonging to that episode.
- `--stats-out` writes a JSON with counts such as `num_edh`, `num_tasks`, `num_episodes`, `num_turns`, `mean_turns_per_edh`, plus `top_actions` by frequency.

Compression-first and performance
--------------------------------

By default the script compresses consecutive actions first and then performs metadata (state) lookups. This reduces expensive state file lookups when your compression aggregation rule changes the timestamps or reduces the number of action entries to consider.

If you prefer to disable compress-first behavior and look up metadata on raw interactions, set `--no-compress-first`.

Example:

```bash
python teach_to_tool_calling/dataset/extract_turns.py \
  --data_root /path/to/teach_dataset \
  --out-dir /tmp/turns_by_task \
  --compress \
  \
  --no-compress-first
```

Progress bar
------------

For long runs, the script shows a progress bar using `tqdm` while iterating EDH files. If `tqdm` is not installed, the script falls back to a simple loop. To enable the nicer progress bar version, install `tqdm`:

```bash
pip install tqdm
```

When installed, a run will show ETD file progress like:

Processing EDH: 123/1234 [10:24<01:59, 1.95file/s]

If you want to write this into `teach` repo standard dataset structure, the `group_by_task_episode` result is a good starting point to write per-task JSON files.

Reference: the `teach` repository includes a helpful notebook `src/teach/analysis/teach_data_exploration.ipynb` that demonstrates reading EDH instances and investigating dialog acts — you can reuse code snippets from there to inspect results of this extractor interactively.
