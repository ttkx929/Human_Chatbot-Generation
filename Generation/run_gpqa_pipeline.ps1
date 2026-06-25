# GPQA Diamond multi-turn dialogue pipeline
# Run from Generation/ with venv activated and api.yaml configured.

# Step 1: accept GPQA terms and login
#   huggingface-cli login
#
# Step 2: build seed jsonl (recommended: strong model for first tutor reply)
python data/prepare_gpqa_seed.py --source huggingface --seed-model DeepSeek --limit 5

# Step 3: expand to multi-turn dialogues
python main.py --data gpqa_diamond --inquirer_model DeepSeek --responder_model DeepSeek --max_turns 12 --limit 5

# Step 4: export for assistant SFT (ShareGPT format)
python data/sft_reformat_gpqa.py `
  --input results/gpqa_diamond_DeepSeek_DeepSeek_12.jsonl `
  --output data/gpqa_diamond_sft.json `
  --format sharegpt

# Alternative: Baize-style strings for baize-chatbot finetune.py
python data/sft_reformat_gpqa.py `
  --input results/gpqa_diamond_DeepSeek_DeepSeek_12.jsonl `
  --output data/gpqa_diamond_baize.json `
  --format baize
