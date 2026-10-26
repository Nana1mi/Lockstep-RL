"""Validate baseline BF16 attention/KV/graph primitives against FP32 SDPA."""

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F


def reference(q, k, v):
    # Q uses the final positions of K (FlashAttention bottom-right causal mask).
    q_positions = torch.arange(q.shape[0], device=q.device) + k.shape[0] - q.shape[0]
    k_positions = torch.arange(k.shape[0], device=k.device)
    mask = k_positions[None, :] <= q_positions[:, None]
    groups = q.shape[1] // k.shape[1]
    return F.scaled_dot_product_attention(
        q.transpose(0, 1).float(), k.repeat_interleave(groups, dim=1).transpose(0, 1).float(),
        v.repeat_interleave(groups, dim=1).transpose(0, 1).float(), attn_mask=mask,
        dropout_p=0).transpose(0, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args()
    if not args.allow_gpu:
        parser.error("Real GPU experiments require manual --allow-gpu")
    result = {"status": "running", "dtype": "bfloat16", "reference": "FP32 SDPA",
              "atol": 0.02, "rtol": 0.02, "tests": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def check(name, actual, expected):
        error = (actual.float() - expected.float()).abs()
        result["tests"][name] = {"max_abs_error": error.max().item(),
                                  "mean_abs_error": error.mean().item()}
        torch.testing.assert_close(actual.float(), expected.float(), atol=0.02, rtol=0.02)
        result["tests"][name]["status"] = "passed"
    try:
        from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
        from nanovllm.layers.attention import store_kvcache
        torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        device = "cuda"
        q = torch.randn(12, 16, 128, device=device, dtype=torch.bfloat16)
        k = torch.randn(12, 8, 128, device=device, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        cu = torch.tensor([0, 7, 12], device=device, dtype=torch.int32)
        output = flash_attn_varlen_func(q, k, v, cu, cu, 7, 7, causal=True)
        expected = torch.cat([reference(q[:7], k[:7], v[:7]), reference(q[7:], k[7:], v[7:])])
        check("varlen_prefill", output, expected)
        k_cache = torch.zeros(3, 256, 8, 128, device=device, dtype=torch.bfloat16)
        v_cache = torch.zeros_like(k_cache)
        slots = torch.tensor(list(range(512, 519)) + list(range(5)), device=device, dtype=torch.int32)
        store_kvcache(k, v, k_cache, v_cache, slots)
        torch.cuda.synchronize()
        check("triton_key_write", torch.cat([k_cache[2, :7], k_cache[0, :5]]), k)
        check("triton_value_write", torch.cat([v_cache[2, :7], v_cache[0, :5]]), v)
        blocks = torch.tensor([[2], [0]], device=device, dtype=torch.int32)
        cu_q = torch.tensor([0, 3, 5], device=device, dtype=torch.int32)
        q_suffix = torch.cat([q[4:7], q[10:12]])
        output = flash_attn_varlen_func(q_suffix, k_cache, v_cache, cu_q, cu, 3, 7,
                                       causal=True, block_table=blocks)
        expected = torch.cat([reference(q[4:7], k[:7], v[:7]), reference(q[10:12], k[7:], v[7:])])
        check("paged_prefix_prefill", output, expected)
        decode_q = torch.stack([q[6], q[11]]).unsqueeze(1).contiguous()
        lengths = torch.tensor([7, 5], device=device, dtype=torch.int32)
        output = flash_attn_with_kvcache(decode_q, k_cache, v_cache,
                                         cache_seqlens=lengths, block_table=blocks, causal=True)
        expected = torch.stack([reference(q[6:7], k[:7], v[:7]), reference(q[11:12], k[7:], v[7:])])
        check("paged_decode", output, expected)
        # Warmup before capture; change input before replay to detect stale outputs.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                flash_attn_with_kvcache(decode_q, k_cache, v_cache,
                                        cache_seqlens=lengths, block_table=blocks, causal=True)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = flash_attn_with_kvcache(decode_q, k_cache, v_cache,
                                              cache_seqlens=lengths, block_table=blocks, causal=True)
        decode_q.mul_(0.5)
        graph.replay()
        torch.cuda.synchronize()
        expected = torch.stack([reference(q[6:7] * 0.5, k[:7], v[:7]),
                                 reference(q[11:12] * 0.5, k[7:], v[7:])])
        check("cuda_graph_decode_replay", captured, expected)
        result["status"] = "passed"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
