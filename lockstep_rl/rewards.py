"""Deterministic numeric rewards shared by simple-task and GSM8K evaluation."""

from decimal import Decimal, InvalidOperation
import re


NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
PATTERNS = [re.compile(r"<answer>\s*([^<]*)\s*</answer>"),
            re.compile(r"####\s*([^\n<]*)(?=\s*(?:<\|im_end\|>)?\s*$)"),
            re.compile(r"\\boxed\{\s*([^{}]*)\s*\}")]


def extract_answer(text):
    matches = []
    for pattern in PATTERNS:
        matches.extend((m.start(), m.group(1)) for m in pattern.finditer(text))
    if not matches:
        return None
    value = max(matches, key=lambda match: match[0])[1].strip()
    if re.fullmatch(NUMBER, value) is None:
        return None
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


def simple_tasks(split="train", count=16, operation="add"):
    if split not in ("train", "validation") or count < 1:
        raise ValueError("invalid task split/count")
    if operation not in ("add", "multiply"):
        raise ValueError("operation must be add or multiply")
    offset = 0 if split == "train" else 1
    tasks = []
    for i in range(count):
        a, b = 11 + 2 * i + offset, 17 + (7 * i) % 23
        answer = a + b if operation == "add" else a * b
        symbol = "+" if operation == "add" else "*"
        tasks.append({"id": f"simple-{split}-{i}",
                      "prompt": f"Compute {a} {symbol} {b}. Return only <answer>number</answer>.",
                      "answer": str(answer)})
    return tasks
