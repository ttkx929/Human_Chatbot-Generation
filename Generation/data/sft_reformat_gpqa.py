"""
Convert generated GPQA dialogues into fine-tuning datasets.

Supports:
  - baize: Baize-chatbot style causal LM strings
  - sharegpt: [{"role": "user"/"assistant", "content": ...}] per dialogue
  - assistant_turns: one sample per assistant turn with prior history
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.conversation_utils import strip_trailing_human_turns
from data.quality_checks import early_spoiler_in_messages


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def to_sharegpt(conversation: list[dict]) -> dict:
    messages = []
    for turn in conversation:
        role = "user" if turn["role"] == "human" else "assistant"
        messages.append({"role": role, "content": turn["content"]})
    return {"messages": messages}


def to_baize(conversation: list[dict], topic: str) -> dict:
    body = ""
    for turn in conversation:
        if turn["role"] == "human":
            body += "[|Human|] " + turn["content"].strip() + "\n"
        else:
            body += "[|AI|] " + turn["content"].strip() + "\n"
    prompt = "The conversation between human and AI assistant.\n"
    return {"topic": topic, "input": prompt + body}


def to_assistant_turns(conversation: list[dict], topic: str) -> list[dict]:
    samples = []
    history = []
    for turn in conversation:
        if turn["role"] == "human":
            history.append({"role": "user", "content": turn["content"]})
        else:
            messages = history + [{"role": "assistant", "content": turn["content"]}]
            samples.append({"topic": topic, "messages": messages})
            history.append({"role": "assistant", "content": turn["content"]})
    return samples


def attach_quality_metadata(item: dict, conversation: list[dict]) -> None:
    messages = item.get("messages")
    if not messages:
        sharegpt = to_sharegpt(conversation)
        messages = sharegpt["messages"]
    quality = {
        "early_spoiler": early_spoiler_in_messages(messages),
        "turns": len(messages),
    }
    item.setdefault("metadata", {})
    item["metadata"]["quality"] = quality


def convert(
    rows: list[dict],
    fmt: str,
    *,
    drop_trailing_human: bool = True,
    filter_early_spoiler: bool = False,
    tag_quality: bool = True,
) -> list[dict]:
    output = []
    skipped = 0
    for row in rows:
        conversation = list(row["conversation"])
        if drop_trailing_human:
            conversation = strip_trailing_human_turns(conversation)
        topic = row.get("task_summary", row.get("metadata", {}).get("question_id", ""))
        if fmt == "sharegpt":
            item = to_sharegpt(conversation)
            if "metadata" in row:
                item["metadata"] = dict(row["metadata"])
            if tag_quality:
                attach_quality_metadata(item, conversation)
            if filter_early_spoiler and item.get("metadata", {}).get("quality", {}).get("early_spoiler"):
                skipped += 1
                continue
            output.append(item)
        elif fmt == "baize":
            item = to_baize(conversation, topic)
            if tag_quality:
                attach_quality_metadata(item, conversation)
            if filter_early_spoiler and item.get("metadata", {}).get("quality", {}).get("early_spoiler"):
                skipped += 1
                continue
            output.append(item)
        elif fmt == "assistant_turns":
            for item in to_assistant_turns(conversation, topic):
                if "metadata" in row:
                    item["metadata"] = dict(row["metadata"])
                if tag_quality:
                    attach_quality_metadata(item, conversation)
                if filter_early_spoiler and item.get("metadata", {}).get("quality", {}).get("early_spoiler"):
                    skipped += 1
                    continue
                output.append(item)
        else:
            raise ValueError(f"Unknown format: {fmt}")
    if filter_early_spoiler and skipped:
        print(f"Filtered {skipped} samples with early answer spoilers.")
    return output


def main():
    parser = argparse.ArgumentParser(description="Reformat GPQA dialogues for SFT")
    parser.add_argument("--input", required=True, help="Generated dialogue jsonl from main.py")
    parser.add_argument("--output", required=True, help="Output json path or jsonl path")
    parser.add_argument(
        "--format",
        choices=("baize", "sharegpt", "assistant_turns"),
        default="sharegpt",
    )
    parser.add_argument(
        "--keep-trailing-human",
        action="store_true",
        help="Keep dialogues that end on a human turn (not recommended for SFT)",
    )
    parser.add_argument(
        "--filter-early-spoiler",
        action="store_true",
        help="Drop samples whose first assistant reply reveals the answer",
    )
    parser.add_argument(
        "--no-quality-tags",
        action="store_true",
        help="Do not attach metadata.quality tags",
    )
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    converted = convert(
        rows,
        args.format,
        drop_trailing_human=not args.keep_trailing_human,
        filter_early_spoiler=args.filter_early_spoiler,
        tag_quality=not args.no_quality_tags,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".jsonl":
        with output_path.open("w", encoding="utf-8") as f:
            for item in converted:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(converted, f, ensure_ascii=False, indent=2)

    spoiler_count = sum(
        1 for item in converted if item.get("metadata", {}).get("quality", {}).get("early_spoiler")
    )
    print(f"Wrote {len(converted)} samples to {output_path} ({args.format})")
    if not args.no_quality_tags and converted:
        print(f"Early spoiler rate: {spoiler_count}/{len(converted)}")


if __name__ == "__main__":
    main()
