# Lockstep-RL progress

## Current objective: GPU-validation-ready code delivery

The user updated the goal on 2026-10-26: implement all required modules, run CPU
checks, and prepare every GPU experiment for manual execution. Do not run CUDA
work or download large models/data autonomously. GPU validation does not block
independent coding. The technical scope remains in `PLAN.md`; dates are advisory.

### Final CPU-verified delivery

- All mandatory A–E source modules and their manual experiment entry points are
  implemented. The detailed requirement/evidence map is docs/acceptance-audit.md.
- Final full suite: **55 CPU tests passed in 112.181s**. Ruff E9/F, compileall,
  Bash syntax, dependency consistency, 16 documentation blocks and six dry plans
  passed. Wheel build and isolated CPU imports passed; 34 library sources match
  the packaged bytes. See results/ for raw records.
- Completed the subprocess-isolated three-way/path matrix, exact original Git
  source benchmark, sampler compute benchmark, prelocked-tolerance A/B recovery
  suite, and independent local evaluation. None was run on a real GPU.
- Added explicit torch.compile control separate from CUDA graph mode, checked
  against versioned official Context7 docs and a CPU backend-counter test.
- Added actual BF16 weight-cast diagnostics, all-clipped zero-gradient records,
  gradient release, early trajectory persistence/failure records and an optional
  actor-effective checkpoint with exact CPU round-trip tests.
- Review fixed partial parameter aliases, incomplete initial checkpoints, global
  dtype/device restoration and preserving unrelated existing process groups on
  rejected initialization. Concurrency tests verify new admission gets the
  successfully committed version.
- First checkpoint: db946ee. Final code/documentation checkpoint is delivered as
  HEAD and identified in the final response. No remote push was performed.
- All real GPU gates remain GPU_PENDING. Continue manually from GPU_RUNBOOK.md;
  FA2/SM120, numerical tolerances, actual memory and learning signal may need fixes
  after those measurements. No throughput or Qwen3 learning claim is made.

The sections below preserve the earlier development history; later delivery
status above supersedes their intermediate remaining-work notes.

### Intermediate implementation history

- A: Integrated FP32 selected-token scores, aligned response masks, EOS/length,
  request/group IDs, policy versions and sampling metadata. Added fixed-token
  replay that does not advance during temporary chunked-prefill samples or lose
  position after preemption. Legacy `step` token tuples remain available;
  `step(return_rollouts=True)` and `generate` expose complete records.
- C: Implemented full TP=1 key/shape/dtype/packed-shard/tied-weight preflight,
  upstream-loader in-place copies, idle-only version commit, logical cache reset,
  and fail-closed broken state. Admission/step/generate/update are serialized by
  an engine RLock. Small CPU tests use actual upstream packed loaders.
- D: Implemented token-normalized clipped GRPO, both frozen anchors, population
  group standard deviation, variable microbatch accumulation and skip behavior.
  A tiny CPU model passed two complete rollout/update/sync/next-rollout cycles.
  Added manual training, numeric rewards, independent validation and guarded
  GSM8K subset preparation. Real Qwen3 training remains GPU_PENDING.
- B/E: Added actual scheduler prefix/chunk/preemption trace events, graph batch
  replay metadata, fixed-trajectory error/ratio statistics and evidence checks.
  HF teacher forcing, same-effective-weight BF16/FP32 loading and saved-result
  plotting are implemented. Path orchestration and sync benchmarks remain.
  Existing attention/memory/benchmark scripts are manual-only.
- CPU evidence: all 36 tests passed in the independent CPU-only nano_rl
  environment (99.620s), including real randomly initialized one-layer Qwen3,
  actual activation-checkpoint invocation, next-token teacher-forcing equivalence,
  exact packed/tied hot-copy vs original safetensors loader and BF16-to-FP32
  effective-weight promotion. `results/cpu-tests.txt` contains the output.
- Ruff E9/F checks, Python compileall, Bash syntax checks and manual GPU installer
  Dry Run passed. No GPU installation or GPU execution occurred in these checks.
- Found and fixed an important integration issue during review: HF gradient
  checkpointing requires training mode. The trainer now keeps training mode and
  disables dropout modules individually, rather than using model.eval().
- All real GPU validations are GPU_PENDING. No Qwen3 inference/training,
  attention smoke, graph capture, or GPU benchmark has been run.

### Environment boundary update

- Stopped the previously started Qwen3 weight download after the user's update;
  its process reported `Terminated`. Partial data remains in ignored `.models/`.
- CUDA 13 compiler/runtime/CRT/NVVM installation completed only inside the old
  project `.venv-linux`. FA2 still has not been built successfully. No further
  GPU dependency work or GPU runs are being started automatically.
- Created an independent `.envs/nano_rl` Python 3.11 environment with CPU-only
  torch 2.11.0+cpu and development dependencies. Exact versions and metadata are
  saved in results/cpu-environment-freeze.txt and results/cpu-environment.json.
  Existing Linux kg_train and Windows
  kg_project environments have not been modified.
- Added opt-in guards to GPU and download scripts and CPU tests checking those
  guards. Environment probe defaults to metadata only; kernel work requires
  manual `--allow-gpu`.
- The old inherited environment failed to import HF Qwen3 because torch CUDA 13
  and torchvision CUDA 12.8 conflict. The independent CPU environment resolves
  this without changing the old environment; its complete suite passed.

### Intermediate next-work notes (completed in final delivery)

1. Finish the manual fixed-trajectory/path matrix and three-way comparison driver,
   with guaranteed trigger checks and raw provenance.
2. Finish A/B hot-update vs fresh-loader/initialization experiments and original
   upstream/logprob-overhead benchmark orchestration.
3. Complete experiment configurations, per-experiment runbook, failure capture,
   metrics/CSV/plot integration, then perform a full requirements audit.

## Historical environment groundwork (before updated GPU boundary)

### 2026-10-26 — environment and training-memory groundwork

- Cloned nano-vllm into the project root, branch `lockstep-rl-dev`, remote
  `upstream`, baseline `bb823b3e06983d71485a8e1f23715ebd87d98ef8`.
- Inspected all six required engine modules and wrote `docs/engine-flow.md`.
- Started Ubuntu WSL2; verified the GPU is visible: RTX 5080 Laptop, 16303 MiB,
  driver 596.49, compute capability 12.0.
- Created `.venv-linux` with read-only access to installed packages from
  `/home/rog/miniconda3/envs/kg_train`. New packages go into the project environment;
  the original environment is not modified. This is not yet a portable lockfile.
- `results/stage0/environment-initial.json`: Python 3.11.15, torch 2.11.0+cu130,
  Triton 3.6.0, Transformers 5.5.4; BF16 GPU GEMM passed. Installed FlashAttention 4
  exposes neither upstream FA2 entrypoint, so the inference backend is unverified.
- First FA2 install attempt failed during metadata generation: no `CUDA_HOME`.
  Existing `nvcc` is 12.8 whereas torch uses CUDA 13. Installed the official NVIDIA
  13.0.88 compiler into the project environment and started a bounded retry.
- Added checkpointed output-head chunk scoring. Five CPU tests passed, including
  full FP32 log-softmax value/gradient equivalence and absence of saved vocabulary
  activations. These tests do not establish real-model GPU memory feasibility.
- Added an executable full-parameter AdamW probe (FP32 master parameters, BF16
  autocast, FP32 scores), including optional inference/KV residency. Not yet run.
- Qwen3-0.6B download is in progress; the downloader resolves and records an exact
  Hub revision before fetching files.

### Remaining manual stage-0 evidence (GPU_PENDING)

- Usable FlashAttention varlen, paged KV decode, Triton writes, and CUDA graphs.
- Actual `example.py` / `bench.py` runs and short/long, single/multiple request paths.
- Full forward/backward/optimizer steps, alone and with real inference/KV residency.
- Measured memory configuration and decision on checkpointing/offload.
- At least three repeats for any reported performance comparison, with warmup,
  initialization, and steady execution recorded separately.
