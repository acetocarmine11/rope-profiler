# Changes

## 0.1.0 — initial release

- Monitor raw semantic-stability and positional-sensitivity scores during an
  existing evaluation forward, using sampled pre-RoPE Q/K activations.
- Attach to a Transformers Trainer with one call, or use a standalone context
  manager with optional JSONL/W&B logging and per-evaluation JSON reports.
- Include the paper's fixed top-5% head presets for Qwen3-8B and
  Llama-3.1-8B-Instruct, with selection provenance and explicit overrides.
- Provide repository utilities for head selection by mean or sample variance,
  heatmaps, and native inference frequency-band scaling.
- Document sampling, Gaussian reversal estimation, positional calculations,
  supported execution paths, measured CPU/CUDA evidence, and paper citation.
- Publish the monitor and small preset resource in the wheel; include research
  code, examples, tests, and English guides in the source distribution.
- Prepare immutable candidates and Trusted Publishing workflows that reuse
  identical wheel and source files for TestPyPI and PyPI.

## Internal development snapshots

The following local snapshots preceded the first public release. Their frozen
artifacts are retained outside the public source and package distributions.

## 0.1.0a3 — 2026-10-02, local alpha

- Bundle independently verified paper Common-pair top-5% head presets: Qwen3-8B
  (58 heads) and Llama-3.1-8B-Instruct (52 heads), with selection provenance.
- Auto-select presets only for recognized model IDs; allow explicit presets
  for local checkpoints and all/manual head overrides. Validate model layouts.
- Custom head mappings without explicit layers now select only named layers.
- Add repository research tools for setting-level r_H mean/sample variance,
  head selection and heatmaps, plus a paired-Q/K r_H helper.
- Add repository inference frequency-band scaling on selected query heads,
  using native runtime frequencies and protecting shared GQA K heads.
- Include research code in source archives; keep the core wheel's dependency
  footprint unchanged. Plotting is available through the optional research extra.

No external upload or Git publication was executed.

## 0.1.0a2 — 2026-10-02, local alpha

- Optional per-evaluation JSON report history for Trainer and standalone loops.
- Follow Trainer model replacement, including `model_init`, before evaluation.
- Reject reuse of evaluation/batch contexts to prevent stale budgets and traces.
- Explicit eager/SDPA backend validation for native Llama and Qwen3.
- Prepare immutable local release bundles with source/artifact fingerprints;
  verify existing bundles and refuse conflicting output directories.
- Separate candidate construction from publication, reusing identical package
  files for TestPyPI and PyPI.

No external upload or Git publication was executed.

## 0.1.0a1 — 2026-10-02, local alpha

- Immutable, JSON-recoverable configuration with pickle and dataclass support.
- Explicit causal/region sampling policy, question/context offset helper, and
  optional Trainer batch metadata callback.
- `full_distances()` selects the paper's complete virtual-distance calculation.
- Sparse capture now retains selected query heads and only their required KV heads.
- Export actual frequency grids, amplitude, runtime/configuration, sample/window
  traces and input/mask fingerprints after each successful evaluation.
- Cache statistics include unique cached tensor storage bytes.
- Reject internal attention-mask gaps, nonbinary masks and fractional/reset positions.
- Preserve historical frequency snapshots and clear results after logging failures.
- Single version source, compatibility CI, distribution checks, isolated wheel
  smoke test and manual publication workflow preparation.

No external release was executed. Actual GPU/production-checkpoint overhead,
token-budget error, and sharded execution remain unverified.

## 0.1.0a0 — 2026-10-01, local prototype

Initial same-forward Llama/Qwen3 monitor, Gaussian semantic score, sampled
positional score, Trainer integration, JSONL/W&B logging and tiny CPU validation.
