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


def eos_boundary(record):
    """Zero-based first EOS; primary statistics include this token, never its tail."""
    config = record.get("sampling_config", {})
    eos = config.get("eos_token_ids", config.get("eos_token_id", []))
    eos = set(eos if isinstance(eos, (list, tuple)) else [] if eos is None else [eos])
    first = next((index for index, token in enumerate(record["response_ids"]) if token in eos), None)
    length = len(record["response_ids"])
    primary = length if first is None else first + 1
    return {"first_eos_index": first,
            "first_eos_position": None if first is None else len(record["prompt_ids"]) + first,
            "primary_response_tokens": primary, "post_eos_tokens": length - primary}


def _summarize_comparison(rows, threshold):
    by_request, grouped = defaultdict(list), defaultdict(list)
    for row in rows:
        by_request[row["request_id"]].append(row["delta_logp"])
        length = row["primary_response_length"] if row["segment"] == "primary" else row["response_length"] - row["primary_response_length"]
        grouped[f"prompt={row['prompt_length']},response={length}"].append(abs(row["delta_logp"]))
    delta = [row["delta_logp"] for row in rows]
    per_trajectory = [{"request_id": key, "response_tokens": len(values),
                       "max_absolute_logprob_error": max(map(abs, values)),
                       "above_threshold": max(map(abs, values)) > threshold}
                      for key, values in by_request.items()]
    return {"token_count": len(rows),
            "absolute_logprob_error": distribution(np.abs(delta)) if delta else None,
            "ratio": distribution([math.exp(value) for value in delta]) if delta else None,
            "threshold": threshold,
            "above_threshold_fraction": float(np.mean(np.abs(delta) > threshold)) if delta else None,
            "delta_logp": delta, "rows": rows, "trajectory_count": len(per_trajectory),
            "per_trajectory": per_trajectory,
            "trajectories_above_threshold": sum(row["above_threshold"] for row in per_trajectory),
            "grouped_by_length": {key: distribution(values) for key, values in grouped.items()}}


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
    rows, boundaries = [], {}
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
        if not n:
            raise ValueError("each diagnostic trajectory must have response tokens")
        boundary = eos_boundary(sample)
        if boundary != eos_boundary(other):
            raise ValueError("comparison EOS policies/boundaries differ")
        boundaries[key] = boundary
        for i, value in enumerate(difference):
            rows.append({"request_id": key, "response_index": i, "delta_logp": float(value),
                         "token_id": sample["response_ids"][i],
                         "rollout_logprob": float(sample["response_logprobs"][i]),
                         "reference_logprob": float(other["response_logprobs"][i]),
                         "position": len(sample["prompt_ids"]) + i,
                         "response_length": n, "prompt_length": len(sample["prompt_ids"]),
                         "primary_response_length": boundary["primary_response_tokens"],
                         "first_eos_index": boundary["first_eos_index"],
                         "segment": "primary" if i < boundary["primary_response_tokens"] else "post_eos"})
    primary = _summarize_comparison([r for r in rows if r["segment"] == "primary"], threshold)
    tail = _summarize_comparison([r for r in rows if r["segment"] == "post_eos"], threshold)
    return {**primary, "post_eos": tail, "all_rows": rows, "eos_boundaries": boundaries,
            "known_error_injection": known_error, "definition": "delta_logp = reference - rollout; ratio = exp(delta_logp)",
            "primary_scope": "through first EOS inclusive, or whole response if no EOS; tail reported separately",
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
           "preemption": "re_prefill_requests", "preemption_cached": "re_prefill_requests",
           "preemption_recompute": "re_prefill_requests"}[path]
    if not evidence[key]:
        raise RuntimeError(f"requested path was not triggered: {path}")
    if path == "preemption_cached" and evidence["prefix_reused_tokens"] == 0:
        raise RuntimeError("cached preemption must demonstrate actual prefix reuse")
    if path == "preemption_recompute" and evidence["prefix_reused_tokens"] != 0:
        raise RuntimeError("recompute preemption must not reuse prefix KV")
    return evidence
