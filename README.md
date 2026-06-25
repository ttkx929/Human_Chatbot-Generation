# DialogueForge: LLM Simulation of Human-Chatbot Dialogue

> **Fork note:** This repository is based on [nerchio/Human_Chatbot-Generation](https://github.com/nerchio/Human_Chatbot-Generation). The section below documents **GPQA Diamond extensions** added in this fork. The original DialogueForge documentation follows unchanged.

## GPQA Diamond Extension (This Fork)

This fork extends the original DialogueForge pipeline to generate **multi-turn science tutoring dialogues** from [GPQA Diamond](https://huggingface.co/datasets/Idavidrein/gpqa) (graduate-level MCQ benchmark), for assistant-style SFT data similar in spirit to [Baize](https://github.com/ProjectBaize/baize-chatbot).

### What was added

| Component | Path | Purpose |
|-----------|------|---------|
| GPQA seed builder | `Generation/data/prepare_gpqa_seed.py` | Convert GPQA Diamond → 2-turn seed jsonl |
| Science prompts | `Generation/prompts_gpqa.py` | Inquirer (student) + responder (tutor) prompts |
| Dialogue generation | `Generation/main.py` | `--data gpqa_diamond`, DeepSeek by default |
| Bot-ending fix | `Generation/data/conversation_utils.py`, `fix_dialogue_endings.py` | Ensure dialogues end on a tutor turn |
| SFT export | `Generation/data/sft_reformat_gpqa.py` | ShareGPT / Baize / per-turn assistant formats |
| Pipeline example | `Generation/run_gpqa_pipeline.ps1` | End-to-end command reference |
| Example CSV | `Generation/data/gpqa_diamond_example.csv` | Local test without HuggingFace |

### Quick start

```bash
cd Generation
pip install -r requirements.txt
```

1. **Configure API keys** — create `Generation/api.yaml` with your keys (at minimum `deepseek`; do **not** commit real keys).

2. **HuggingFace access** — accept terms at [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa), then `huggingface-cli login`.

3. **Build seeds** (198 questions, or `--limit 5` for a smoke test):

```bash
python data/prepare_gpqa_seed.py --source huggingface --seed-model DeepSeek
```

4. **Generate multi-turn dialogues**:

```bash
python main.py --data gpqa_diamond --inquirer_model DeepSeek --responder_model DeepSeek --max_turns 12
```

5. **Export for fine-tuning** (ShareGPT format):

```bash
python data/sft_reformat_gpqa.py \
  --input results/gpqa_diamond_DeepSeek_DeepSeek_12.jsonl \
  --output data/gpqa_diamond_sft.json \
  --format sharegpt
```

6. **Fix existing jsonl** that ends on a human turn (optional):

```bash
python data/fix_dialogue_endings.py --input results/your_dialogues.jsonl
```

### Data flow

```
GPQA Diamond → prepare_gpqa_seed.py → gpqa_diamond_seed.jsonl
    → main.py (LangGraph dual-agent) → results/*.jsonl
    → sft_reformat_gpqa.py → gpqa_diamond_sft.json → LoRA / SFT
```

### Notes

- **Models** are loaded lazily in `models.py` — only the model you pass on the CLI needs a valid API key in `api.yaml`.
- **Default LLM** for this fork is **DeepSeek** (`deepseek-chat` via OpenAI-compatible API); see `Generation/setting.yaml`.
- **Do not evaluate** on GPQA Diamond after training on GPQA-generated data (contamination). Use a held-out science benchmark instead.
- Generated artifacts (`results/`, seeds, `gpqa_diamond_sft.json`) are listed in `.gitignore` and are not meant to be committed.

---

## Generation of Dialogues
Please contact Ruizhe Zhu (zhurui@student.ethz.ch) for questions of this part.

To learn more about the generation of dialogues, you can check the following directory:
```bash
cd Generation
```

The generation pipeline is built by `LangGraph`.

### How to run
First, you should install the dependencies by:
```bash
pip install -r requirements.txt
```
or 
```bash
conda env create -f environment.yml
```

Then, you need to set your own API keys in the `api.yaml` file.

After that, you should be able to run the generation experiments. You can refer to `experiment_example.sh` to get more details about the parameters.

Please note if you need more steps to run the fine-tuned models, please see part [Models](#Models).

### Datasets
In the `data` folder we provide the datasets we used.

`arena_model_a_summaries.jsonl` is the arena dataset and `oasst1_en_min_6_turns_summary` is the oasst dataset.

For fine-tuning, we use `sft_reformat.py` to reformat them into finetuning datasets `arena_sft.jsonl` and `oasst1_en_sft.jsonl`.

The two fine-tuning dataset are also public on Hugging Face: [oasst_sft](https://huggingface.co/datasets/SyangZhou/oasst_SFT) and [arena_sft](https://huggingface.co/datasets/SyangZhou/arena_SFT).

### Models
By checking `models.py` and `settings.yaml`, you can find the models and call chains we used in the project.

We have some fine-tuned models in them, and they are all public on Hugging Face. They are:
- [llama-3b-v1](https://huggingface.co/SyangZhou/autotrain-l3b-0520-v1)
- [llama-3b-v2](https://huggingface.co/SyangZhou/autotrain-l3b-0520-v2)
- [llama-8b-v1](https://huggingface.co/SyangZhou/autotrain-l8b-0520-v1)
- [llama-8b-v2](https://huggingface.co/SyangZhou/autotrain-l8b-0520-v2)
- [mistral-v1](https://huggingface.co/SyangZhou/autotrain-m-0520-v1)
- [mistral-v2](https://huggingface.co/SyangZhou/autotrain-m-0520-v2)

You have to start a new inference point on Hugging Face before using them for generation.

## Evaluation of of Generated Dialogues
Please contact Hao Zhu (haozhu2@student.ethz.ch) for questions of this part.

### Evaluation Metrics
All the metrics are defined in `evaluation_metrics.py`. For more details, please check [our paper](https://arxiv.org/abs/2507.15752).

Prompts of the judge LLM are defined in `prompts.py`.  

These two files are provided by Shijing Cai from Calvin Risk AG, Switzerland.  

### How to run
The code to run the UniEval, PairEval, and GTEval process are defined in `uni_eval.py`, `pair_eval.py`, and `gt_eval.py`.  

Before running, you need to change sevaral variables in the files: 
- `ChatGPT4o_api_key`: the OpenAI API key.
- `dialogue_data_folder`: the directory where the dialogue data files to be evaluated reside.
- `saved_result_folder`: the directory where you want to store the Evaluation results. 
- `dialogue_data_file_names`: the dialogue data files you want to evaluate.
- `index_pair_list`: for the PairEval and GTEval, you need to specify the conversations to be compared.  

Then, just execute: 
```bash
python3 *_eval.py
```

### Evaluation results
They are stored in `result_oasst` and `result_arena`.  

And we visualize the results:

- [oasst UniEval (6-turns)](Evaluation/result_summary_oasst/GPT4o_Evaluator/oasst_uni_eval_6_turns.pdf)

- [oasst UniEval (12-turns)](Evaluation/result_summary_oasst/GPT4o_Evaluator/oasst_uni_eval_12_turns.pdf)

- [arena UniEval (6-turns)](Evaluation/result_summary_arena/GPT4o_Evaluator/arena_uni_eval_6_turns.pdf)

- [arena UniEval (12-turns)](Evaluation/result_summary_arena/GPT4o_Evaluator/arena_uni_eval_12_turns.pdf)

- [oasst GTEval (6-turns)](Evaluation/result_summary_oasst/GPT4o_Evaluator/oasst_gt_eval_6_turns.pdf)

- [oasst GTEval (12-turns)](Evaluation/result_summary_oasst/GPT4o_Evaluator/oasst_gt_eval_12_turns.pdf)

- [arena GTEval (6-turns)](Evaluation/result_summary_arena/GPT4o_Evaluator/arena_gt_eval_6_turns.pdf)

- [arena GTEval (12-turns)](Evaluation/result_summary_arena/GPT4o_Evaluator/arena_gt_eval_12_turns.pdf)

## Citation
The paper related to this work is published at the [KDD 2025 workshop on Evaluation and Trustworthiness of Agentic and Generative AI Models](https://kdd-eval-workshop.github.io/genai-evaluation-kdd2025/) and is available on [arXiv](https://arxiv.org/abs/2507.15752).

When using this repository or our fine-tuned models in your work, please cite it as follows:
```
@misc{zhu2025dialogueforgellmsimulationhumanchatbot,
      title={DialogueForge: LLM Simulation of Human-Chatbot Dialogue}, 
      author={Ruizhe Zhu and Hao Zhu and Yaxuan Li and Syang Zhou and Shijing Cai and Malgorzata Lazuka and Elliott Ash},
      year={2025},
      eprint={2507.15752},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.15752}, 
}
```