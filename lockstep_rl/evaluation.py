"""Independent validation using the same answer parser and actual committed actor."""

from lockstep_rl.data import chat_prompts
from lockstep_rl.rewards import numeric_reward


def evaluate(engine, tasks, *, label, max_tokens):
    if not tasks:
        raise ValueError("validation tasks cannot be empty")
    from nanovllm import SamplingParams
    prompts = chat_prompts(engine.tokenizer, tasks)
    ids = [f"eval:{label}:{row['id']}" for row in tasks]
    records = engine.generate(prompts, SamplingParams(max_tokens=max_tokens),
                              request_ids=ids, group_ids=ids,
                              sampling_config={"enable_thinking": False}, use_tqdm=False)
    expected = dict(zip(ids, tasks))
    rows = []
    for record in records:
        task = expected[record["request_id"]]
        text = engine.tokenizer.decode(record["response_ids"], skip_special_tokens=True)
        rows.append({"id": task["id"], "reward": numeric_reward(text, task["answer"]),
                     "response_text": text, "finish_reason": record["finish_reason"],
                     "policy_version": record["policy_version"]})
    return {"count": len(rows), "accuracy": sum(row["reward"] for row in rows) / len(rows), "rollouts": records,
            "truncation_fraction": sum(row["finish_reason"] == "length" for row in rows) / len(rows),
            "rows": rows}
