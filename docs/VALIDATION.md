# Validation and evidence

Validation basis for the initial release 0.1.0, 2026-10-02. The CPU
checks below used the internal 0.1.0a3 snapshot. Release 0.1.0 preserves every
scoring, sampling, capture, integration, and preset source byte from that
snapshot; only the package version and release documentation/metadata changed.
Publication and installation checks for 0.1.0 are recorded separately below.

## Initial release checks

The 0.1.0 source passed all 327 tests locally on Python 3.12.13, torch 2.8.0,
and Transformers 4.56.2. The documented Trainer example was rerun with region
sampling and full virtual distances: both evaluations produced the existing
loss, both raw scores, and separate reports. The scores matched the earlier
tiny-model records below. Release artifacts record exact checksums in their
candidate manifest; GitHub Actions and the index retain the online build and
publication records.

## Paper head presets and selection tools

- The wheel includes `get_head_preset` and `list_head_presets`, with 58 of 1152
  query heads for Qwen3-8B and 52 of 1024 for Llama-3.1-8B-Instruct. Automatic
  resolution requires an exact model identity and matching geometry. Local
  checkpoints can use an explicit preset; custom selection overrides automatic
  resolution.
- The frozen Common-pair selection source has SHA256
  `c57e4424cf21f9f7c57f6d2e9aa979d67a6ee17daf2e66df080ed5e472035a9d`.
  The bundled JSON has SHA256
  `d02c729feb5188362dd6e7523757bbeed8f43fee8a51df2f4b7976a0262f726d`.
  Every selected head matches the formal cached-score inputs and intervention
  lists.
- Means and variances were independently recomputed from 49 frozen settings.
  Feeding all 106624 setting/head rows across both models to the new selection
  tool reproduced the complete top-5% rankings and bundled lists.
- `r_H=U_H/U` is a norm share; its square is an energy share. The tool exports
  means, variances, and selection information. The paper presets use equally
  weighted settings and sample variance with `ddof=1`. Mean-based selection is
  an additional analysis option. The original cutoff used `.05/.10`; the later
  `.06` audit did not change the fixed lists.
- Mean and variance heatmaps for both models were generated in PNG and PDF and
  visually checked. Black boxes mark selected heads; plot axes are one-based
  and code/table indices are zero-based. Raw validation data remain outside the
  package.

Usage and calibration-input requirements are described in the
[head-selection guide](HEAD_SELECTION.md).

## CPU checks

All final suite runs used macOS ARM64 CPU and small randomly initialized
Transformers models. No checkpoints were downloaded for these runs.

| Python | torch | Transformers | Final result |
| --- | --- | --- | --- |
| 3.12.13 | 2.8.0 | 4.56.2 | 327 passed, 9.68 s |
| 3.12.13 | 2.8.0 | 4.51.3 | 326 passed, 1 skipped, 7.95 s |
| 3.10.20 | 2.4.1 | 4.51.3 | 326 passed, 1 skipped, 7.96 s; NumPy 1.26.4 |

The two compatibility environments lacked matplotlib and skipped one plotting
file check. The main environment passed this check with matplotlib 3.11.2.
The 0.1.0a3 suite adds 21 preset, 49 statistics/plotting, and 48 scaling cases to
the preceding 209 checks.

Checks cover the analytic Gaussian score against explicit rotations, the
positional score against a full-distance reference, configuration JSON/pickle
round trips, sparse capture, post-normalization Qwen3 capture, reordered query
heads and native GQA key mapping, unchanged model outputs and forward counts,
padding, prefill capture, exception cleanup, Trainer integration and metadata
callbacks, dynamic frequencies, report/cache counters, and CPU two-process
Gloo aggregation with per-example weights.

Scaling checks cover Llama/Qwen3 eager and SDPA, per-band coefficients, shared-KV
sibling isolation, identity scaling, native cached decoding, frequency updates
before initial mask creation, a fixed mask during later decoding, nonrotary
tails, restoration, and exception cleanup. Scaling hooks take precedence over
monitor hooks, and both context-entry orders capture the effective scaled Q.
CPU overflow checks raise an error and restore the model. Accelerator product
checks are optional and synchronize the device when enabled.

## Training integration and distribution checks

The Trainer example used a randomly initialized two-layer Llama on CPU, trained
for four steps, and ran the existing evaluation every two steps. Region sampling,
full virtual distances, and offline W&B logging were enabled. Existing loss and
both raw scores appeared in the same evaluation record, with separate reports
for each evaluation.

| Step | Eval loss | Semantic score | Positional score |
| --- | --- | --- | --- |
| 2 | 4.15075827 | 0.70621370 | 0.28642347 |
| 4 | 4.15075445 | 0.70635056 | 0.28642385 |

Each evaluation used four examples and sampled 4Q/8K per example. This verifies
the interface and logging path. The tiny random-model changes do not establish
training quality, statistical significance, or production cost. W&B only wrote
local offline files.

Tests also verified binding to the current Trainer model after `model_init`,
preserving subclasses and `compute_metrics`, existing forward counts and
callbacks, independent report histories, and single-use evaluation/batch
contexts. Reports use unique names and become visible only after their complete
JSON is written.

The core wheel was installed into a fresh consumer environment outside the
source tree. Checks verified `site-packages` origin, installed RECORD hashes,
version/metadata agreement, both scores, and configuration recovery. Core
imports did not load Transformers or W&B, and that environment had no NumPy.
Wheel/sdist metadata, Twine format, source identity, and artifact contents passed
local checks. Research tools and examples are included in the source archive;
validation artifacts, environments, logs, and caches are excluded.

The internal 0.1.0a3 candidate remains a historical draft. The release workflow
checks the license and tag, then builds and freezes a 0.1.0 candidate. The upload
workflow verifies the successful build run, source identity, and file checksums,
and reuses those package files without rebuilding them.

## Evidence limits

Task accuracy, training benefits, the paper's full intervention gains, and
sampling error on real evaluation datasets require their corresponding
experiments.

Token/head sampling error and virtual-distance sampling error must be measured
separately. Full virtual distances retain the selected token/head subset and
do not restore all-token statistics. Production NCCL/DDP Trainer, tensor
parallelism, FSDP, fused QKV/RoPE, FlashAttention 2, compilation, CUDA graphs,
vLLM, other hardware, and longer contexts remain outside the executed evidence.

Earlier 0.1.0a1 runs passed 140 baseline checks plus six installation-integrity
cases in each of three CPU environments. The 0.1.0a2 final compatibility suites
passed 209 checks; the main environment passed its combined suite followed by
the final release checks. These records are historical evidence, and their
timings are not used as current-version overhead estimates.

See [compatibility](COMPATIBILITY.md), [sampling](REGIONS.md), and the
[release procedure](RELEASING.md) for supported protocols and installation
requirements.
