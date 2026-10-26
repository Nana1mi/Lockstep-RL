"""Real local GSM8K questions with token-bounded, solved few-shot context.

No network, model construction, repeated-character filler or synthetic token IDs.
The entire target question and its chat template are preserved. Only the leading
few-shot text may be cropped at a token boundary to reach an exact KV block size.
"""

from lockstep_rl.data import load_tasks, validate_splits


def gsm8k_prompts(tokenizer, *, question_file, fewshot_file, count, prompt_tokens):
    questions, examples = load_tasks(question_file), load_tasks(fewshot_file)
    validate_splits(examples, questions)
    if count < 1 or count > len(questions):
        raise ValueError("diagnostics need num_trajectories distinct real GSM8K questions")
    if not all(row.get("solution", "").strip() for row in examples):
        raise ValueError("few-shot rows must contain the real GSM8K worked solution")
    marker = "LOCKSTEP_CONTENT_8c31b27f"
    wrapper = tokenizer.apply_chat_template([{"role": "user", "content": marker}],
                                             tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
    if wrapper.count(marker) != 1:
        raise ValueError("chat template did not preserve the content marker")
    before, after = wrapper.split(marker)
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    prefix, suffix = encode(before), encode(after)
    context = "\n\n".join(f"Question: {row['prompt']}\nSolution: {row['solution']}" for row in examples)
    context_ids = encode(context + "\n\n")
    result = []
    for k, row in enumerate(questions[:count]):
        target_ids = encode("\n\nNow solve this problem:\n" + row["prompt"])
        needed = prompt_tokens - len(prefix) - len(suffix) - len(target_ids)
        if needed < 0:
            raise ValueError(f"{row['id']}: complete question/template exceeds prompt_tokens; increase budget")
        if needed > len(context_ids):
            raise ValueError("not enough real solved example text; prepare a larger few-shot subset")
        # Keep the end of the example pool (complete solution endings), and all
        # target tokens. Cropping is explicit provenance, never repeated filler.
        filler = context_ids[-needed:] if needed else []
        ids = prefix + filler + target_ids + suffix
        assert len(ids) == prompt_tokens
        result.append({"request_id": f"traj-{k}", "prompt_ids": ids, "task_id": row["id"],
                       "question": row["prompt"], "fewshot_tokens": needed,
                       "fewshot_ids": [example["id"] for example in examples],
                       "fewshot_crop": "left token crop; target and chat wrapper intact"})
    return result
