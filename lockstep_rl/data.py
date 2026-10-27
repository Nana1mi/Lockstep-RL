"""Local JSONL task loading and leakage checks; never downloads data."""

import json
from pathlib import Path


def load_tasks(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("task file is empty")
    for row in rows:
        if not all(key in row for key in ("id", "prompt", "answer")) or not row["prompt"].strip():
            raise ValueError("task rows need id, nonempty prompt, and answer")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate task IDs")
    return rows


def validate_splits(train, validation):
    if {r["id"] for r in train} & {r["id"] for r in validation}:
        raise ValueError("train/validation IDs overlap")
    if {r["prompt"].strip() for r in train} & {r["prompt"].strip() for r in validation}:
        raise ValueError("train/validation prompts overlap")


def chat_prompts(tokenizer, tasks):
    return [tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                                          tokenize=True, return_dict=False, add_generation_prompt=True,
                                          enable_thinking=False) for row in tasks]
