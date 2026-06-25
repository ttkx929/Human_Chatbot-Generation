"""
Fix existing dialogue jsonl files that end on a human turn.

Usage:
  python data/fix_dialogue_endings.py --input results/gpqa_diamond_DeepSeek_DeepSeek_12.jsonl
  python data/fix_dialogue_endings.py --input results/foo.jsonl --output results/foo_fixed.jsonl --no-llm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.conversation_utils import ends_with_bot, ensure_conversation_ends_with_bot
from prompts_gpqa import GET_GPQA_PROMPT


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Ensure dialogues end with a bot turn")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="", help="Defaults to in-place overwrite of --input")
    parser.add_argument("--responder-model", default="DeepSeek")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Drop trailing human turns instead of generating a closing bot reply",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path
    rows = load_jsonl(input_path)

    responder_llm = None
    if not args.no_llm:
        from models import get_model
        responder_llm = get_model(args.responder_model)

    fixed_count = 0
    for row in rows:
        conversation = row["conversation"]
        if ends_with_bot(conversation):
            continue
        _, _, responder_prompt = GET_GPQA_PROMPT(row.get("task_summary", ""))
        row["conversation"] = ensure_conversation_ends_with_bot(
            conversation,
            responder_prompt,
            responder_llm,
            prefer_closing=not args.no_llm,
        )
        row["turns"] = len(row["conversation"])
        fixed_count += 1

    write_jsonl(output_path, rows)
    print(f"Fixed {fixed_count}/{len(rows)} dialogues -> {output_path}")


if __name__ == "__main__":
    main()
