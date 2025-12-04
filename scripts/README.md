python scripts/test_qwen3_sample.py \
    --sample-file "/home/bjk/tool_learning/teach_to_tool_calling/dataset_experiments/hybrid_ours/train/ccf66d25fbfb95bb_2ea8/ccf66d25fbfb95bb_2ea8.edh0.jsonl"

    python3 scripts/test_gpt_sample.py --dataset-dir /home/bjk/tool_learning/teach_to_tool_calling/dataset_experiments/hybrid_ours/valid_unseen




python3 teach_to_tool_calling/scripts/compare_instance_objects.py \
  --instance-id 85ec788c6b00fb0d_49ba.edh7 \
  --experiments-root teach_to_tool_calling/dataset_experiments_edh \
  --categories spatial,semantic \
  --out teach_to_tool_calling/scripts/compare_85ec788c6b00fb0d_49ba.edh7.json \
  --verbose

