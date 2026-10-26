"""Deterministic numeric rewards shared by simple-task and GSM8K evaluation."""

from decimal import Decimal, InvalidOperation
import re


NUMBER = r"[-+]?\d[\d,]*(?:\.\d+)?"
PATTERNS = [re.compile(r"<answer>\s*(" + NUMBER + r")\s*</answer>"),
            re.compile(r"####\s*(" + NUMBER + r")(?=\s*(?:<\|im_end\|>)?\s*$)"),
            re.compile(r"\\boxed\{\s*(" + NUMBER + r")\s*\}")]


def extract_answer(text):
    matches = []
    for pattern in PATTERNS:
        matches.extend((m.start(), m.group(1)) for m in pattern.finditer(text))
    if not matches:
        return None
    value = max(matches, key=lambda match: match[0])[1]
    try:
        return Decimal(value.replace(",", ""))
    except InvalidOperation:
        return None


def numeric_reward(response_text, expected):
    actual = extract_answer(response_text)
    try:
        gold = Decimal(str(expected).replace(",", ""))
    except InvalidOperation as exc:
        raise ValueError("expected answer must be a finite decimal number") from exc
    if not gold.is_finite():
        raise ValueError("expected answer must be finite")
    # Length-truncated outputs use this same rule; missing answers score zero.
    return float(actual is not None and actual == gold)


def simple_tasks(split="train", count=16):
    if split not in ("train", "validation") or count < 1:
        raise ValueError("invalid task split/count")
    offset = 0 if split == "train" else 1000
    return [{"id": f"simple-{split}-{i}",
             "prompt": f"Compute {offset + i + 1} + {i + 2}. Return only <answer>number</answer>.",
             "answer": str(offset + 2 * i + 3)} for i in range(count)]
