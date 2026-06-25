"""
Convert GPQA Diamond into DialogueForge seed jsonl.

Usage:
  # HuggingFace (requires: huggingface-cli login + accept dataset terms)
  python data/prepare_gpqa_seed.py --source huggingface

  # Local CSV exported from https://github.com/idavidrein/gpqa
  python data/prepare_gpqa_seed.py --source csv --csv-path path/to/gpqa_diamond.csv

  # Strong-model seed replies (recommended)
  python data/prepare_gpqa_seed.py --source huggingface --seed-model DeepSeek

  # Quick test on first 5 questions
  python data/prepare_gpqa_seed.py --source huggingface --limit 5
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

QUESTION_KEYS = ("Question", "Pre-Revision Question", "question")
CORRECT_KEYS = ("Correct Answer", "Pre-Revision Correct Answer", "correct_answer")
INCORRECT_KEYS = (
    ("Incorrect Answer 1", "Pre-Revision Incorrect Answer 1", "incorrect_answer_1"),
    ("Incorrect Answer 2", "Pre-Revision Incorrect Answer 2", "incorrect_answer_2"),
    ("Incorrect Answer 3", "Pre-Revision Incorrect Answer 3", "incorrect_answer_3"),
)
SUBDOMAIN_KEYS = ("Subdomain", "subdomain", "High-level domain", "high_level_domain")


def _pick(row: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        if key in row and row[key]:
            return str(row[key]).strip()
    return ""


def _normalize_record(row: dict, index: int) -> dict:
    question = _pick(row, QUESTION_KEYS)
    correct = _pick(row, CORRECT_KEYS)
    incorrect = [_pick(row, keys) for keys in INCORRECT_KEYS]
    if not question or not correct or any(not x for x in incorrect):
        raise ValueError(f"Row {index} is missing required GPQA fields: {list(row.keys())}")

    choices = [correct] + incorrect
    rng = random.Random(index)
    shuffled = choices[:]
    rng.shuffle(shuffled)
    labels = ["A", "B", "C", "D"]
    labeled = {label: choice for label, choice in zip(labels, shuffled)}
    correct_label = next(label for label, text in labeled.items() if text == correct)

    subdomain = _pick(row, SUBDOMAIN_KEYS) or "science"

    return {
        "question_id": f"gpqa_diamond_{index}",
        "question": question,
        "correct_answer": correct,
        "correct_label": correct_label,
        "choices": labeled,
        "subdomain": subdomain,
    }


def load_from_huggingface(config: str, limit: int | None) -> list[dict]:
    from datasets import load_dataset

    dataset = load_dataset("Idavidrein/gpqa", config)
    split = "train" if "train" in dataset else list(dataset.keys())[0]
    rows = dataset[split]
    if limit is not None:
        rows = rows.select(range(min(limit, len(rows))))

    records = []
    for i, row in enumerate(rows):
        records.append(_normalize_record(dict(row), i))
    return records


def load_from_csv(csv_path: str, limit: int | None) -> list[dict]:
    records = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            records.append(_normalize_record(row, i))
    return records


def build_task_summary(record: dict) -> str:
    return (
        f"You are working through a graduate-level {record['subdomain']} problem. "
        f"You want to understand the scientific reasoning behind the question, compare plausible "
        f"answer choices, and reach the correct conclusion through step-by-step discussion with a tutor. "
        f"The target problem is about: {record['question'][:500]}"
    )


def build_human_seed(record: dict) -> str:
    choice_lines = "\n".join(
        f"{label}. {text}" for label, text in sorted(record["choices"].items())
    )
    return (
        f"I'm stuck on a challenging {record['subdomain']} question and could use your help thinking it through.\n\n"
        f"{record['question']}\n\n"
        f"Here are the options:\n{choice_lines}\n\n"
        f"Can you walk me through how to approach this step by step?"
    )


def build_bot_seed_template(record: dict) -> str:
    return (
        "Let's break this down carefully.\n\n"
        "First, identify the core concept the question is testing and what quantity or mechanism it asks about. "
        "Then map each option to the underlying principle it implies, and eliminate choices that contradict "
        "basic constraints or units.\n\n"
        "Tell me which part feels most confusing: the setup, the relevant formula or mechanism, "
        "or how to discriminate between the remaining options."
    )


SEED_SYSTEM_PROMPT = """You are an expert science tutor.
Write the opening tutor reply for a graduate-level multiple-choice science problem.
Requirements:
- Be accurate and aligned with the known correct answer.
- Explain how to approach the problem, but do not simply say "the answer is X" in the first sentence.
- Use 150-300 words.
- Sound like a helpful human tutor.
"""


def build_bot_seed_with_llm(record: dict, seed_model_name: str) -> str:
    from langchain.schema import HumanMessage, SystemMessage
    from models import get_model

    llm = get_model(seed_model_name)
    choice_lines = "\n".join(
        f"{label}. {text}" for label, text in sorted(record["choices"].items())
    )
    user_prompt = (
        f"Subdomain: {record['subdomain']}\n\n"
        f"Question:\n{record['question']}\n\n"
        f"Options:\n{choice_lines}\n\n"
        f"Known correct option: {record['correct_label']} ({record['correct_answer']})\n\n"
        f"Write the tutor's first reply to the student."
    )
    response = llm.invoke([SystemMessage(content=SEED_SYSTEM_PROMPT), HumanMessage(content=user_prompt)])
    return response.content.strip()


def record_to_seed(record: dict, seed_model_name: str | None) -> dict:
    human_seed = build_human_seed(record)
    if seed_model_name:
        bot_seed = build_bot_seed_with_llm(record, seed_model_name)
    else:
        bot_seed = build_bot_seed_template(record)

    return {
        "conversation_id": [record["question_id"]],
        "conversation": [
            {"role": "human", "content": human_seed},
            {"role": "bot", "content": bot_seed},
        ],
        "turns": 2,
        "task_summary": build_task_summary(record),
        "metadata": {
            "question_id": record["question_id"],
            "subdomain": record["subdomain"],
            "correct_label": record["correct_label"],
            "correct_answer": record["correct_answer"],
            "choices": record["choices"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare GPQA Diamond seed jsonl for DialogueForge")
    parser.add_argument("--source", choices=("huggingface", "csv"), default="huggingface")
    parser.add_argument("--hf-config", default="gpqa_diamond")
    parser.add_argument("--csv-path", default="")
    parser.add_argument("--output", default="data/gpqa_diamond_seed.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--seed-model",
        default="DeepSeek",
        help="Model from models.py for first tutor reply (DeepSeek/GPT4o). Use 'none' for template-only seed.",
    )
    args = parser.parse_args()

    if args.source == "huggingface":
        records = load_from_huggingface(args.hf_config, args.limit)
    else:
        if not args.csv_path:
            raise ValueError("--csv-path is required when --source csv")
        records = load_from_csv(args.csv_path, args.limit)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seed_model = args.seed_model if args.seed_model.lower() != "none" else None
    with output_path.open("w", encoding="utf-8") as f:
        for i, record in enumerate(records):
            seed = record_to_seed(record, seed_model)
            f.write(json.dumps(seed, ensure_ascii=False) + "\n")
            print(f"Wrote seed {i + 1}/{len(records)}: {record['question_id']}")

    print(f"\nSaved {len(records)} seeds to {output_path}")


if __name__ == "__main__":
    main()
