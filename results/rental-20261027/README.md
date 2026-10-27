# L40S readiness evidence — 2026-10-27

Explicitly authorized remote provisioning and minimal GPU verification. Source before runtime fixes:
3f59e33e5340bbb1827023e18db6fd0168b4fb02; fixed upstream bb823b3e06983d71485a8e1f23715ebd87d98ef8.
Baseline source is that commit plus installer einops fix; successful GRPO also includes
explicit return_dict=False in data.chat_prompts and exact_chat_prompt. Each provenance.json
records the actual dirty source diff. No upstream attention backend was changed.

Actual hardware: NVIDIA L40S, driver580.126.09, Anhui; persistent root /root/gpufree-data/lockstep.
Python3.11 / torch2.8.0+cu128 / Transformers5.5.4 / official FA2 2.8.3.post2.
Full wheel SHA is in rental-preflight.json; official model/dataset sources, resolved revisions
and file hashes are in model-manifest.json and dataset-manifest.json. Content is excluded.

Passed:
- results/preflight-01/environment.json and attention.json: CUDA and six attention/kernel/graph gates.
- results/preflight-01/baseline/: all24 workers, short-single32/32 only, three process repeats.
  measurements.csv separates initialization, warmup and steady execution. This is not the full matrix.
- results/readiness-grpo-02/: two true arithmetic GRPO iterations. Summary: two optimizer steps,
  one policy commit, one actor-visible and probe-visible commit, one zero-variance skipped batch.
  Config/metrics.csv/metrics.jsonl/rollouts retain token-aligned raw scores and rewards.
- results/demo-memory-probe-01.json: two AdamW steps, batch16/sequence256, no checkpointing,
  co-resident inferencefraction0.4; peakallocated36920615424B, peakreserved38321258496B.

Failures retained:
- results/readiness-grpo-01/: actual Transformers5 default BatchEncoding input failure, before fix.
- results/bootstrap-01/: installation timeouts, truncatedmetadata recovery, missingeinops and retries.
  No source compilation or silent dependency/attention fallback was used.
- tokenizer-return-type-evidence.txt: CPU reproduction with the actual pinned tokenizer/library.

Exact commands and runtime settings:
- results/preflight-01/manifest.json and baseline/manifest.json
- results/bootstrap-01/*command.json and ops/ runtime provisioning/supervisor scripts
- configs/readiness-grpo.json and results/readiness-grpo-02/config.json
- results/bootstrap-01/environment-info.txt, environment-freeze.txt, final-gpu-state.txt

Full32-trajectory diagnostics, preemption comparisons, weight-sync negativecontrol, long-context
performance and120-iteration learning demonstrations are GPU_PENDING. Small smoke results establish
execution readiness, not improvement or global memory safety. Checkpoint/model/dataset/env/cache
contents are excluded. Existing run IDs must not be overwritten. Stop/release billing in the platform
when finished; stopping test processes or SSH does not stop rental charges.
