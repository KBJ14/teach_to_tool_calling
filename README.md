# TEACh: preview_raw_turn 사용법

이 문서는 `preview_raw_turn.py` 스크립트를 사용해 EDH 인스턴스, 대응 game 파일, 그리고 해당 턴의 state JSON을 하나의 JSON으로 묶어 미리보기(preview)하는 방법을 설명합니다.

## 위치
- 스크립트: `teach_to_tool_calling/dataset/scripts/preview_raw_turn.py`
- 데이터 루트(예): `/teach_dataset` (도메인 루트에 위치)

## 요구사항
- Python 3.7/3.8
- TEACh 데이터가 `--data_root`로 지정한 경로에 풀려 있어야 함

## 기본 사용 예시
- EDH 인스턴스 목록에서 첫 파일의 0번째 턴을 미리보기하고 `/tmp/preview.json`에 저장:

```bash
python /home/bjk/tool_learning/teach_to_tool_calling/dataset/scripts/preview_raw_turn.py \
  --data_root /teach_dataset \
  --edh_index 0 \
  --turn_idx 0 \
  --out /tmp/preview.json
```

- EDH 파일을 직접 지정해서 특정 턴 미리보기:

```bash
python /home/bjk/tool_learning/teach_to_tool_calling/dataset/scripts/preview_raw_turn.py \
  --data_root /teach_dataset \
  --edh_fn /teach_dataset/edh_instances/valid_seen/abcd1234_0.edh0.json \
  --turn_idx 3 \
  --out /tmp/abcd1234_turn3_preview.json
```

## 출력 결과
- 지정한 `--out` 경로에 JSON이 생성됩니다. JSON에는 다음 정보가 포함됩니다:
  - `edh_fn`: 사용한 EDH 인스턴스 경로
  - `game_id`: EDH에서 추출한 game ID
  - `game_fn`: 대응되는 `.game.json` 파일 경로(발견 시)
  - `edh`: EDH 인스턴스 전체 contents
  - `episode_meta`: `task_idx`/`episode_idx` 등 EDH 메타
  - `selected_interaction_idx`: 요청한 턴 인덱스
  - `edh_interaction`: EDH의 해당 인터랙션 디테일
  - `game_interaction_at_same_index`: (가능한 경우) game file에서 같은 인덱스의 interaction
  - `state_fn`: 매칭되는 상태 JSON 파일 경로(발견 시)
  - `state`: state JSON의 내용(발견 시)

## 힌트 & 문제 해결
- 종종 `state_fn`을 찾지 못할 수 있습니다. TEACh 버전에 따라 state/이미지 파일 저장 방식이 달라졌기 때문에 스크립트는 휴리스틱을 사용합니다.
  - 이 경우 `/teach_dataset/images` 또는 `/teach_dataset/images_and_states` 폴더 내의 `game_id`로 시작하는 파일들을 수동으로 검색해 주시고, 필요한 경우 스크립트를 고쳐 `find_state_file_for_turn`의 휴리스틱을 조정할 수 있습니다.
- EDH와 `.game.json` 매핑이 잘못되는 경우에는 EDH metadata(`structured_log_fn`, `game_id`, `episode_idx`)를 확인해서 직접 `--edh_fn`으로 지정하세요.

## 다음 개선 아이디어
- `--include_images` 옵션을 추가해 관련 이미지 파일 목록 또는 썸네일을 JSON에 포함
- 여러 EDH 인스턴스를 한꺼번에 미리보기하는 배치 모드
- 기본적으로 `--out`을 `/tmp/<game_id>_turnX_preview.json` 형태로 자동 생성

## Turn extraction and compression

추가로 `teach_to_tool_calling/dataset/extract_turns.py` 스크립트를 포함했습니다. 이 스크립트는 EDH 인스턴스들을 순회하면서 `task -> episode -> turn` 구조로 데이터를 정리하고, 연속된 같은 동작(`Forward`, `Turn Right` 등)을 압축(compress)할 수 있습니다.

간단한 예시 실행:

```bash
python teach_to_tool_calling/dataset/extract_turns.py \
  --data_root /teach_dataset \
  --out /tmp/turns.json \
  --grouped-out /tmp/turns_grouped.json \
  --compress
```

`--grouped-out` 옵션을 지정하면 task -> episode -> [edh entries] 계층으로 묶어서 내보냅니다.

참고: 추출기 자체는 `state` 정보를 포함하지 않습니다. `game_id`와 turn 액션의 첫 타임스탬프를 사용해서 필요할 경우 별도 스크립트로 state를 가져오실 수 있습니다.

원하면 위 개선 중 하나를 바로 추가해 드릴게요.



# 모든 task_* / episode_* 변환
./run_build_dataset.sh

# 특정 task만
./run_build_dataset.sh --task-ids 0 1