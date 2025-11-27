import os
from teach_to_tool_calling.dataset import loader


def test_iter_episode_to_records_basic():
    here = os.path.dirname(__file__)
    sample = os.path.join(here, "..", "episode_data_wo_state", "task_308", "episode_0.json")
    sample = os.path.abspath(sample)
    assert os.path.exists(sample)

    # prefer the repo-level tools.json (package tools.json) by default
    tools_path = os.path.abspath(os.path.join(here, "..", "tools.json"))
    assert os.path.exists(tools_path)
    records = list(loader.iter_episode_file_to_records(sample, tools_fn=tools_path, images_dir=None))
    # must produce at least one record
    assert len(records) > 0
    # every record should include required keys
    for r in records:
        assert "episode_id" in r
        assert "turn_index" in r
        assert "base_prompt" in r
        assert "actions" in r
        assert "history" in r
        # answer must only include 'actions' and not a textual 'utterance'
        assert "answer" in r
        assert "utterance" not in r["answer"]
        assert isinstance(r["answer"].get("actions"), list)
