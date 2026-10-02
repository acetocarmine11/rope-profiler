# Validation and evidence

Validation basis for the initial release 0.1.0, 2026-10-02. The CPU and CUDA
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

## Real-model CUDA checks

Existing local Qwen3-8B and Llama-3.1-8B-Instruct checkpoints were tested on one
NVIDIA RTX A6000, using Python 3.11.9, torch 2.6.0+cu124, Transformers 4.56.2,
and accelerate 1.3.0. Model computation used BF16 native SDPA, batch size 1,
with gradients and TF32 disabled. Inputs were fixed synthetic token sequences.
The installed wheel's source identity and import origin were checked.

| Model | Explicit paper preset | Local-path default | Exact public-ID automatic selection |
| --- | --- | --- | --- |
| Qwen3-8B | 58 heads | 1152 heads | 58 heads |
| Llama-3.1-8B-Instruct | 52 heads | 1024 heads | 52 heads |

The automatic-selection check explicitly set the configuration's model name to
the public ID. It verifies name resolution; it does not authenticate the
checkpoint's provenance.

Both models were checked at 512 and 2048 tokens:

- Monitoring preserved every full-sequence logit exactly and used one forward
  in both the monitored and unmonitored cases. Both raw scores were finite, with
  zero invalid units.
- Reports matched the model's actual native frequency tensors.
- Identity scaling and scope exit restored baseline behavior and removed hooks.
  A nonidentity intervention verified selected first-layer Q planes, unchanged
  unselected siblings, and unchanged shared K for the same layer input.
- Both monitor/intervention context-entry orders agreed on captured Q/K, scores,
  and logits.
- The scaling helper generated four tokens with query lengths
  `[prompt_length, 1, 1, 1]`, covering prefill and three cached decode steps.
  The monitor checks covered the input forward; these tests do not establish
  monitoring support throughout a generation loop.

Scaling checks used a functional-test threshold, rather than reproducing the
historical paper intervention cutoff. GQA isolation is a local first-layer
claim for the same input; later model activations respond to the intervention.

### CPU/CUDA numerical comparison

| Input | Score dtype | Maximum semantic absolute difference | Maximum positional absolute difference |
| --- | --- | --- | --- |
| Random sparse tensors | float64 | 0 | 1.388e-17 |
| Random sparse tensors | float32 | 2.731e-08 | 2.114e-09 |
| Qwen3-8B, 512 tokens, one sampled head | float64 | 0 | 0 |
| Llama-3.1-8B-Instruct, 512 tokens, one sampled head | float64 | 0 | 1.388e-17 |

The real-model comparison recomputed CPU FP64 results from the monitor's sampled
Q/K, pairs/triplets, and frequencies, using the same scoring algorithm.
Effective counts matched, with `rtol=atol=1e-9`. It checks cross-device numerics
and reuses the score definition and implementation. Sparse activations were
not saved.

## Measured monitoring overhead

The main benchmark used paper presets, 4 sampled query positions, 16 sampled key
positions, 256 virtual distances, and FP64 diagnostic scoring. Each condition
used five alternating-order baseline/monitor pairs with the same input. Both
arms computed logits for all tokens with `use_cache=False`. Warm measurements
reused the same monitor's scoring cache.

Timing includes sampling, hook installation, forward, scoring, aggregation,
normal report construction, and final CUDA synchronization. Model loading, file
writes, and the subsequent sampled-logit audit are outside the timed region.

| Model | Tokens | Baseline median (ms) | Monitor median (ms) | Median paired increase (ms) | Median paired increase (%) |
| --- | --- | --- | --- | --- | --- |
| Qwen3-8B | 512 | 94.214 | 146.877 | 51.683 | 54.81 |
| Qwen3-8B | 2048 | 357.681 | 403.937 | 47.015 | 13.12 |
| Qwen3-8B | 8192 | 1508.302 | 1558.896 | 51.010 | 3.39 |
| Llama-3.1-8B-Instruct | 512 | 88.597 | 130.622 | 42.133 | 47.54 |
| Llama-3.1-8B-Instruct | 2048 | 325.975 | 366.478 | 39.664 | 12.14 |
| Llama-3.1-8B-Instruct | 8192 | 1393.374 | 1440.434 | 47.059 | 3.38 |

Each percentage is calculated per pair as
`100 × (monitored time - baseline time) / baseline time`, then the median is
taken. The median paired difference therefore need not equal the difference
between the two separate medians. These results describe this hardware,
synthetic input, and configuration. Short-context evaluations incurred a
substantial relative cost.

FP32 scoring was also checked under the same sampling and frequencies. Across
these conditions, the maximum aggregate FP32/FP64 absolute differences were
8.141e-9 for the semantic score and 1.692e-8 for the positional score. Valid,
tie, and invalid counts matched. Equal aggregate counts alone do not establish
that every per-unit validity decision is identical.

Cold measurements used a fresh scoring cache after model warmup; first-call
initialization can also contribute to cold time. The internal
`rope_score_seconds` field uses CUDA events around scoring and aggregation and
can include host launch gaps. It is not a sum of pure kernel execution times,
and cannot be added to separately measured report/finalization wall times.

### Memory measurements

For the main conditions, FP64 scoring caches retained about 0.378 MiB of unique
tensor storage; FP32 caches retained about 0.190 MiB. Warm-cache paired peak
allocated-memory differences were about 0.022 MiB for Qwen3 and 0.020 MiB for
Llama. Both arms retained the scoring cache during these paired measurements,
so these differences describe extra transient allocation, rather than the
complete resident cost of monitoring. Pure baseline measurements without the
cache were also recorded.

Sampled Q/K copy counters were consistent across the tested lengths and score
dtypes:

| Model | Cumulative sampled-copy bytes | Peak concurrently retained sampled copies |
| --- | --- | --- |
| Qwen3-8B | 226 KiB | 25 KiB |
| Llama-3.1-8B-Instruct | 168 KiB | 36 KiB |

Layers release their sampled copies after scoring. Cumulative copied bytes are
therefore different from simultaneous retention. These counters, unique cached
tensor storage, PyTorch allocated/reserved memory, and total device usage
measure different quantities.

All-head monitoring at 2048 tokens and full-distance scoring at 512 tokens were
checked once per model. Single observations confirm those configurations run,
but do not support stable latency or variability estimates.

An initial benchmark-only logit-index sampler generated an out-of-range float32
index after the first unmonitored forward, outside timing. It was changed to
integer indexing, independently rechecked on CUDA, and the benchmark was rerun.
That failed observation was excluded from the paired results; the frozen
scoring wheel was unchanged.

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

The real-model CUDA runs used fixed synthetic inputs up to 8192 tokens. They
establish installation, native capture, numerical checks, functional scaling,
cached generation, restoration, and configuration-specific overhead. Task
accuracy, training benefits, the paper's full intervention gains, and sampling
error on real evaluation datasets require their corresponding experiments.

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
