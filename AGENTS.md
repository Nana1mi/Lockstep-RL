# Lockstep-RL development

- Project scope and acceptance gates are defined in `PLAN.md`. Keep the fixed
  upstream baseline `bb823b3e06983d71485a8e1f23715ebd87d98ef8` identifiable.
- For library/framework/SDK/API/CLI/cloud details, first resolve the library in
  Context7, then query scoped current documentation (version-specific when given).
  Prefer primary package documentation. General code review does not need a
  library lookup when no library/platform API fact is involved.
- Use Linux / WSL2 for CUDA execution. Windows CUDA availability is not evidence
  that the Linux dependency stack works.
- Delivery target is GPU-validation-ready. Dates in PLAN.md are advisory.
- Never autonomously run CUDA kernels, GPU inference/training, graph capture,
  GPU benchmarks or memory stress. Never automatically download large models or
  datasets. All real GPU validation is manual; mark it `GPU_PENDING` and continue
  independent implementation and CPU tests. Read-only dependency inspection is OK.
- Ordinary tests must remain CPU-only and imports must not initialize CUDA. GPU
  entry points require an explicit `--allow-gpu` flag; download scripts require
  `--allow-download`. Dry runs may not construct models or initialize CUDA.
- Use an independent WSL2 `nano_rl` environment; never modify Windows kg_project
  or existing Linux environments. Provide installation scripts for manual GPU setup.
- Record commands, failures, versions, memory, and limitations in `PROGRESS.md`,
  `IMPLEMENTATION_STATUS.md`, `EXPERIMENTS.md` and `GPU_RUNBOOK.md`. Never mark an
  unexecuted GPU experiment as passed. GPU gates do not block independent coding.
- Keep generated model weights, environments, build caches, and large trajectories
  out of Git. Preserve small machine-readable measurement results.
- Prefer focused correctness checks covering token alignment, gradient behavior,
  cache lifetime, and policy versions over tests mirroring implementation details.
- Do not replace the upstream attention backend silently. Any compatibility
  adaptation needs explicit evidence and separate correctness/performance results.
- Keep user-supplied planning documents intact unless updating a verified task
  checkbox. Project name is Lockstep-RL.

- The user revision dated 2026-10-27 explicitly authorizes PLAN/README/runbook
  updates and prioritizes P0.1-3, then P1.4/6/5, P2.7-8 before rental. No new
  mandatory memory/offload task is part of this revision.
- Current rental target is one L40S 48GB (Ada sm_89), Anhui region, Linux.
  A100-SXM4-80GB and RTX5080/SM120 instructions are historical. Actual GPU name,
  driver >=570 and separate data-disk mount require manual preflight evidence. Pin official FA2 wheel/torch/Python/CUDA/ABI/hash;
  do not silently fall back to source compilation or mix GPU models in contrasts.

- The later pre-rental review authorizes an additional commit-refreshed cached
  FP32 head alongside live conversion. Primary diagnostics include only through
  first EOS inclusive; retain post-EOS data separately. Only environment and
  attention failures hard-gate independent jobs; support --stages retry under a
  fresh run-id. Report sync_seconds as the three update phases, excluding change
  measurement. No real GPU execution/download/rental is authorized by this review.

- The L40S revision keeps the official FA2 wheel URL/hash and Python/torch pins.
  Demo inference fraction is0.4; microbatch16/no checkpointing awaits manual memory
  validation, with microbatch8/accumulation16 as documented OOM fallback.
