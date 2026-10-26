"""Deterministic numeric rewards shared by simple-task and GSM8K evaluation."""

from decimal import Decimal, InvalidOperation
import re


NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
PATTERNS = [re.compile(r"<answer>\s*([^<]*)\s*</answer>"),
            re.compile(r"####\s*([^\n<]*)(?=\s*(?:<\|im_end\|>)?\s*$)"),
            re.compile(r"\\boxed\{\s*([^{}]*)\s*\}")]


def parse_answer(text, mode="strict"):
    if mode not in ("strict", "lenient"):
        raise ValueError("reward parser mode must be strict or lenient")
    matches = []
    for pattern in PATTERNS:
        matches.extend((m.start(), m.group(1)) for m in pattern.finditer(text))
    if mode == "lenient":
        # Balanced braces handle both {,} thousands separators and text units.
        for match in re.finditer(r"\\boxed\{", text):
            start, depth, end = match.end(), 1, match.end()
            while end < len(text) and depth:
                depth += (text[end] == "{") - (text[end] == "}")
                end += 1
            if depth == 0:
                matches.append((match.start(), text[start:end - 1]))
    if not matches:
        status = "unparseable" if any(marker in text for marker in ("<answer>", "####", r"\boxed{")) else "no_answer"
        return {"parse_status": status, "parsed_answer": None}
    value = max(matches, key=lambda match: match[0])[1].strip()
    if mode == "lenient":
        value = value.replace(r"\$", "").replace("$", "").replace("{,}", ",")
        value = re.sub(r"\\text\{[^{}]*\}", "", value).strip()
        if value.endswith("."):
            value = value[:-1].strip()
    if re.fullmatch(NUMBER, value) is None:
        return {"parse_status": "unparseable", "parsed_answer": None}
    try:
        parsed = Decimal(value.replace(",", ""))
    except InvalidOperation:
        return {"parse_status": "unparseable", "parsed_answer": None}
    return {"parse_status": "ok", "parsed_answer": str(parsed)}


def extract_answer(text, mode="strict"):
    result = parse_answer(text, mode)
    return Decimal(result["parsed_answer"]) if result["parse_status"] == "ok" else None


def reward_details(response_text, expected, mode="strict"):
    parsed = parse_answer(response_text, mode)
    actual = Decimal(parsed["parsed_answer"]) if parsed["parse_status"] == "ok" else None
    try:
        gold = Decimal(str(expected).replace(",", ""))
    except InvalidOperation as exc:
        raise ValueError("expected answer must be a finite decimal number") from exc
    if not gold.is_finite():
        raise ValueError("expected answer must be finite")
    # Length-truncated outputs use this same rule; missing answers score zero.
    return {**parsed, "reward": float(actual is not None and actual == gold)}


def numeric_reward(response_text, expected, mode="strict"):
    return reward_details(response_text, expected, mode)["reward"]


def simple_tasks(split="train", count=16, operation="add"):
    if split not in ("train", "validation") or count < 1:
        raise ValueError("invalid task split/count")
    if operation not in ("add", "multiply"):
        raise ValueError("operation must be add or multiply")
    if count > 32 * 23:
        raise ValueError("simple task pool supports up to 736 unique questions per split")
    offset = 0 if split == "train" else 1
    tasks = []
    for i in range(count):
        a, b = 11 + 2 * (i % 32) + offset, 17 + (7 * i + i // 32) % 23
        answer = a + b if operation == "add" else a * b
        symbol = "+" if operation == "add" else "*"
        tasks.append({"id": f"simple-{split}-{i}",
                      "prompt": f"Compute {a} {symbol} {b}. Return only <answer>number</answer>.",
                      "answer": str(answer)})
    return tasks
