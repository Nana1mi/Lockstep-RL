# L40S experiment evidence — 2026-10-27

Experimental source: `ce9e3542b3f31475ffa595e3f09b80e3f9190ed0`. Fixed nano-vllm upstream:
`bb823b3e06983d71485a8e1f23715ebd87d98ef8`. This directory is a later publication
of measurements; the experiment did not run at the publication commit.

All13 orchestration jobs returned0. Baseline24 workers, diagnostics14, weight-sync25,
performance96. Both learning-rate demonstrations executed120 iterations each.
This means execution completed; it does not mean every numerical discrepancy is
below threshold or that a single-seed reward change proves robust learning.

- Hardware: one NVIDIA L40S48GB, Anhui, driver580.126.09;24 CPU quota and120GiB container memory.
- Python3.11 / torch2.8.0+cu128 / Transformers5.5.4 / official FA2 2.8.3.post2.
- Completed 2026-10-27T20:35:34.860658+08:00; preflight1173.01s, full9683.30s, combined10856.33s.
- `execution.json`, campaign/full manifests: commands, commit, statuses, actual times and cap overrides.
- `full/diagnostics/*.csv`: token-level primary and post-EOS comparisons. Compact distributions,
  per-trajectory maxima and threshold counts are under `full/diagnostics/summaries/`.
- `full/weight-sync/`: phased timing CSV, reports and negative-control records.
- Training subdirectories: actual config, per-iteration metrics.csv, summaries and figures.
  `training-summary.json` retains initial evaluation metadata and the final CSV row.
- `assets/`: source revisions and content hashes only; no model/dataset contents.
- `runtime/`: actual external orchestration scripts. They retain historical paths/run IDs;
  inspect them before adapting. Source remains pinned; performance parent cap12000s is explicit.

Complete raw trajectories, detailed JSON/logs and server tools are in the
[Release attachment](https://github.com/Nana1mi/Lockstep-RL/releases/tag/l40s-20261027). The raw numerical entries are unchanged.
The public ZIP removes the live SSH host from documentation and omits standalone
derived model/tokenizer/task-list assets. `publication.json` records original and
public archive hashes, exclusions and redactions. No passwords, private keys,
tokens, weights, environments or checkpoints are included.

Post-hoc CPU analysis, figures and limitations are in [REPORT.md](../../analysis/l40s-20261027/REPORT.md).
Original measurements remain unchanged. Multi-seed learning and a matched anchor
comparison remain unvalidated. GSM8K parsing/truncation also needs more evidence:
the last smoke batch had50% parse failures, which is not a dataset-wide estimate.
Original32 trajectories and all raw logs remain available for independent slicing.
