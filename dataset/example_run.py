"""Small runner that demonstrates how to use the loader on a sample episode in this repo.
"""
from teach_to_tool_calling.dataset import loader
import os


def demo():
    here = os.path.dirname(__file__)
    sample = os.path.join(here, "..", "episode_data_wo_state", "task_308", "episode_0.json")
    sample = os.path.abspath(sample)
    tools = os.path.abspath(os.path.join(here, "..", "tools.json"))
    out = sample + ".demo.jsonl"
    loader.main([sample, "--tools", tools, "--out", out, "--images_dir", "/teach_dataset/images"])


if __name__ == "__main__":
    demo()
