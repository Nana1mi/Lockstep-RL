"""CPU-only fixed-trajectory numerical statistics and path evidence checks."""

from collections import Counter, defaultdict
import math

import numpy as np


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("distribution requires nonempty finite measurements")
    return {"count": int(values.size), "mean": float(values.mean()), "min": float(values.min()),
            "std": float(values.std()),
            "p95": float(np.quantile(values, .95)), "p99": float(np.quantile(values, .99)),
            "max": float(values.max())}


def compare_trajectories(rollouts, reference, *, threshold=.1, known_error=False):
    if threshold < 0:
        raise ValueError("error threshold must be nonnegative")
    def index(records):
        lookup = {row["request_id"]: row for row in records}
        if len(lookup) != len(records):
            raise ValueError("duplicate request IDs in comparison")
        return lookup
    left, right = index(rollouts), index(reference)
    if set(left) != set(right) or not left:
        raise ValueError("comparison requires identical nonempty request sets")
    delta = []
    rows = []
    for key, sample in left.items():
        other = right[key]
        for field in ("prompt_ids", "response_ids"):
            if sample[field] != other[field]:
                raise ValueError(f"fixed trajectory mismatch: {key}: {field}")
        n = len(sample["response_ids"])
        if len(sample["response_logprobs"]) != n or len(other["response_logprobs"]) != n:
            raise ValueError("scores are not aligned with response tokens")
        difference = np.asarray(other["response_logprobs"], dtype=np.float64) - sample["response_logprobs"]
        if not np.isfinite(difference).all():
            raise ValueError("comparison contains nonfinite logprobs")
        delta.extend(difference.tolist())
        for i, value in enumerate(difference):
            rows.append({"request_id": key, "response_index": i, "delta_logp": float(value),
                         "response_length": n, "prompt_length": len(sample["prompt_ids"])})
    ratio = [math.exp(value) for value in delta]
    grouped = defaultdict(list)
    for row in rows:
        grouped[f"prompt={row['prompt_length']},response={row['response_length']}"].append(abs(row["delta_logp"]))
    return {"known_error_injection": known_error, "definition": "delta_logp = reference - rollout; ratio = exp(delta_logp)",
            "absolute_logprob_error": distribution(np.abs(delta)), "ratio": distribution(ratio),
            "threshold": threshold, "above_threshold_fraction": float(np.mean(np.abs(delta) > threshold)),
            "delta_logp": delta, "rows": rows,
            "grouped_by_length": {key: distribution(values) for key, values in grouped.items()},
            "scope": "measured finite trajectories only; maximum is not a global error bound"}


def path_evidence(trace):
    counts = Counter(row["event"] for row in trace)
    chunks = defaultdict(list)
    partial_prefill = set()
    preempted = set()
    re_prefill = set()
    for row in trace:
        if row["event"] == "preempt":
            preempted.add(row["request_id"])
        if row["event"] == "schedule" and row["is_prefill"]:
            for request in row["requests"]:
                chunks[request["request_id"]].append((request["start"], request["end"]))
                if request["end"] < request.get("context_tokens", request["end"]):
                    partial_prefill.add(request["request_id"])
                if request["request_id"] in preempted:
                    re_prefill.add(request["request_id"])
    executions = [row for row in trace if row["event"] == "execution"]
    return {"event_counts": dict(counts),
            "prefix_reused_blocks": sum(row["blocks"] for row in trace if row["event"] == "prefix_reuse"),
            "prefix_reused_tokens": sum(row["tokens"] for row in trace if row["event"] == "prefix_reuse"),
            "prefill_ranges": dict(chunks), "chunked_request_count": len(partial_prefill),
            "graph_replays": sum(row.get("graph_replay", False) for row in executions),
            "padded_graph_replays": sum(row.get("graph_replay", False) and row["graph_batch"] > row["actual_batch"]
                                        for row in executions),
            "preemptions": counts["preempt"], "re_prefill_requests": sorted(re_prefill)}


def require_path(trace, path):
    evidence = path_evidence(trace)
    key = {"prefix": "prefix_reused_blocks", "chunked": "chunked_request_count",
           "graph": "graph_replays", "graph_padding": "padded_graph_replays",
           "preemption": "re_prefill_requests"}[path]
    if not evidence[key]:
        raise RuntimeError(f"requested path was not triggered: {path}")
    return evidence
