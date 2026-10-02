# Monitoring reference

Detailed integration, sampling, score, and report behavior for RoPE Profiler 0.1.0.

## Attach to an existing Trainer

```python
from rope_profiler import MonitorConfig
from rope_profiler.integrations import attach_rope_monitor

attachment = attach_rope_monitor(
    trainer,
    report_dir="outputs/rope_reports",  # Optional: keep each eval's full report.
    config=MonitorConfig(
        queries=4,
        keys=16,
        distance_samples=256,
        max_examples=32,
        seed=0,
        cache_entries=64,
    ),
)

trainer.train()
# trainer.evaluate() and trainer.predict() are monitored as well.
print(attachment.monitor.last_metrics)
print(attachment.monitor.last_report_path)
attachment.detach()
```

The adapter wraps the instance's existing `prediction_step` and evaluation
loops. Existing Trainer subclasses, `compute_metrics`, data collators, and
model calls continue to run. A subclass must continue to call `prediction_step`
through `evaluation_loop` or `prediction_loop`. Subclasses that implement an
entirely separate evaluation path should use the standalone interface below.

The original evaluation-loop output receives keys such as
`eval_rope_semantic_score` and `eval_rope_positional_score`, before the Trainer's
usual metric logging. Its existing W&B integration therefore records one
combined evaluation record. The adapter creates no run and adds no forward pass.
Multiple evaluation datasets retain their Trainer prefixes, such as
`eval_validation_rope_semantic_score`.

Call `attachment.detach()` to restore the original methods. The attachment can
also be used as a context manager. Detach after an active evaluation finishes.
If Trainer's `model_init` replaces the model, the attachment binds to the
current model before the next evaluation and preserves its sampling settings.
With `report_dir`, each successful eval writes a separate JSON report. Repeated
evaluation at the same step preserves both files. Omitting it creates no report
files. File names include the evaluation prefix, training step and process rank.

To monitor known question/context regions, use
`MonitorConfig(sampling_policy="regions")` and provide a `batch_metadata`
callback. It receives a read-only mapping of the original prediction inputs
and returns `query_mask`, `key_mask`, and optionally `sample_ids` for the
monitor. These fields are never passed to the model. Both masks are required;
overriding the actual model's attention mask or position IDs is rejected.
`token_region_masks` constructs masks from exact input text, question/context
character spans, tokenizer offsets, and a special-token mask. See
[region sampling](https://github.com/acetocarmine11/rope-profiler/blob/v0.1.0/docs/REGIONS.md) for a complete example.

## Standalone evaluation

```python
import torch
from rope_profiler import MonitorConfig, RoPEMonitor
from rope_profiler.loggers import JSONLLogger

monitor = RoPEMonitor(
    model,
    config=MonitorConfig(seed=0),
    logger=JSONLLogger("outputs/rope_metrics.jsonl"),
    report_dir="outputs/rope_reports",  # Optional report history.
)

model.eval()
with torch.inference_mode(), monitor.evaluation(step=training_step, epoch=epoch, prefix="eval"):
    for batch in eval_loader:
        with monitor.batch(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            position_ids=batch.get("position_ids"),
            sample_ids=batch.get("sample_ids"),
        ):
            outputs = model(**model_inputs(batch))
        # Continue your existing loss/accuracy evaluation here.

print(monitor.last_metrics)
print(monitor.last_report_path)
```

Here `model_inputs` is the user's existing selection of model arguments. It
should exclude metadata such as `sample_ids`. If a benchmark identifies the
question and context spans, pass Boolean `query_mask` and `key_mask` tensors
of shape `[batch, sequence]` to `monitor.batch` to sample only those regions.

To log directly to an existing W&B run:

```python
from rope_profiler.loggers import WandbLogger

monitor = RoPEMonitor(model, logger=WandbLogger(run=existing_run))
```

`WandbLogger()` without a run uses `wandb.run` at logging time and fails clearly
if there is no active run. It never calls `wandb.init` or `wandb.login`. An
existing offline run is supported. Standalone W&B metric names use
`eval/rope_semantic_score`; the Trainer adapter uses the Trainer's ordinary
`eval_rope_semantic_score` names. The standalone logger adds `train/global_step`
and defines `eval/*` with that chart axis, once per prefix. W&B advances its own
history index independently, so several training records per optimizer step
do not cause later eval metrics to be discarded. A minimal run object without
`define_metric` can still receive metrics; it must select the training-step
axis itself. The Trainer adapter keeps the Trainer's existing axis handling.
JSONL logging writes only to the requested local file. Non-finite scalar values
are recorded as JSON null.

## Scores

| Metric | Definition |
| --- | --- |
| `rope_reversal_probability` | Gaussian estimate of the chance that the reference winner among two keys loses after a positional shift, averaged over valid sampled triplets. |
| `rope_semantic_score` | `1 - rope_reversal_probability`; higher values indicate greater stability of the sampled content preference. |
| `rope_positional_score` | Mean absolute adjacent-position change in the frozen-Q/K score, divided by the norm of its spectral coefficient vector. Higher values indicate greater local positional response. |

For the semantic score, a triplet contains one query and two distinct causal
keys. The reference winner is determined at relative distance zero. The
Gaussian mean and variance are derived from the captured vectors; there is
no explicit simulation of every positional shift for reversal probability.
Reference ties and undefined cells are excluded and counted.

For the positional score, a pair contains one query and one causal key. With
its captured Q/K fixed, the monitor computes
`mean_m |S(m + 1) - S(m)| / sqrt(sum_n |z_n|^2)` over a deterministic set of
sampled distances. This score uses all rotary dimensions. It is distinct from
the high-frequency coefficient norm share `r_H`. The square of `r_H` is the
corresponding high-frequency energy share. The score uses hypothetical positional
shifts of frozen activations; it does not compare the activations of neighboring
tokens. The exported counts describe which cells contributed.

Scores first average valid cells within each head, then all valid selected
layer/head units within each example with equal weight, and finally valid
examples with equal weight. A layer with more selected heads contributes more
units. Long examples do not gain weight simply by having more tokens. Raw scores
retain their mathematical normalization and do not depend on a reference set
of tasks.

Other outputs include `rope_examples`, `rope_semantic_valid_triplets`,
`rope_semantic_ties`, `rope_positional_valid_pairs`, `rope_invalid_cells`,
`rope_retained_activation_bytes`, `rope_peak_activation_bytes`, and
`rope_score_seconds`. Retained bytes counts the cumulative Q/K sample copies;
peak bytes measures the largest concurrently retained Q/K samples, rather than
the whole model's peak memory. On CUDA, the score timer measures stream intervals surrounding score and
statistics operations, which can include host launch gaps. It is not a sum of
pure kernel execution times. Use paired wall-clock runs to measure total
overhead, including capture, reports, and logging.

## Sampling and initialization

The model performs its ordinary computation for the full input. Token sampling
reduces the monitor's retained activations and score calculation. Projection
hooks gather only selected tokens, detach them from autograd, compute the
scores, and release the retained activations. They do not request attention
matrices or hidden-state histories. Native grouped-query K heads are preserved.
At batch entry, input IDs and sampling metadata may be copied to CPU once to
construct the deterministic plan. The score calculations stay on the model's
device.

Defaults select 4 query tokens and 16 key tokens per example, up to the first
32 examples in each evaluation. The monitor keeps queries with enough causal
prefix and samples keys before the earliest chosen query. This gives the
selected query/key combinations a valid causal interpretation. Padding is
excluded. Short inputs may produce fewer samples or no valid score cells.

The default sample identity is a fingerprint of the effective input tokens.
Explicit stable `sample_ids` are preferable when using a custom loop. Fixed
seed, input tokens, IDs, masks, and model positions give fixed token sampling.
The first-example budget also requires a stable evaluation order. Reusing a
fixed validation subset and tokenizer configuration makes comparisons between
training checkpoints meaningful.

The default `sampling_policy="causal"` selects generic causal tokens. The
paper uses question queries and context keys, excluding special tokens and
overlaps without letters/digits. `regions` plus `token_region_masks` supplies
that eligibility rule. The plugin keeps its own seeded sampler; reproducing
the historical paper token positions requires the original saved selection.
Its 4Q/16K budget alone does not reproduce the paper's chosen head collection
or its original SHA-256 ordering.

The paper's positional calculation enumerates every adjacent virtual distance.
Use `MonitorConfig.full_distances(...)` for that distance calculation, retaining
the chosen token/head subset. The default 256-distance setting estimates the
same average by uniform sampling with replacement; short windows automatically
use all distances. Semantic reversal uses analytic Gaussian moments under
either setting. Computing each score requires no reference-task evaluation.

The default `head_preset="auto"` selects the paper's fixed top 5% for exact
`Qwen/Qwen3-8B` and `meta-llama/Llama-3.1-8B-Instruct` identifiers (their exact
short names are also recognized). Other model identities include all heads.
Local checkpoint paths require an explicit preset. The loaded layer/query-head
layout is checked before applying a preset. Set `head_preset=None` to include
all supported layers and heads.

`heads` maps layer indices to query-head indices and overrides auto selection.
With `layers=None`, only the mapping's layers are selected. With explicit
`layers`, layers omitted from `heads` include all their query heads. Keep a
chosen head/layer subset fixed across checkpoints and label it in reported
results. A head subset now reduces both capture and score calculation: hooks
gather only selected query heads and their required native KV heads.
`score_dtype="float64"` is the default for numerical
verification, while model computation keeps its existing precision.
`cache_entries=64` bounds the cache of frequency/window kernels and sampled
distance phases. Reusing these kernels avoids recomputing them for unchanged
frequencies and sequence lengths. Increasing this bound accommodates more
lengths while retaining more kernel memory; the cache does not retain full
sequence activations.

Initialization inspects the existing model's layers, head dimensions,
query-to-KV mapping, rotary layout, and frequency source. The active frequency
grid is checked against the actual evaluation forward so that a model's RoPE
scaling is represented. Optional `monitor.frequency_bands(M, threshold)` is
available for frequency-band inspection; it does not restrict either score to
the high-frequency dimensions. `monitor.metadata()` reports the captured
configuration.

## Paper head presets and repository research tools

The installation includes two independently verified Common-pair head lists:

| Preset | Total query heads | Selected heads |
| --- | --- | --- |
| `qwen3-8b` | 36 × 32 = 1152 | 58 |
| `llama-3.1-8b-instruct` | 32 × 32 = 1024 | 52 |

The paper ranks each head by the sample variance of its setting-level mean
`r_H` across 49 reference settings, then keeps `ceil(0.05 * total_heads)`.
Both scores use this same fixed list. Indices are zero-based in the API and
packaged JSON; paper figure labels are one-based.

```python
from rope_profiler import MonitorConfig, RoPEMonitor, get_head_preset

preset = get_head_preset("qwen3-8b")
print(preset.heads)          # Exact layer -> query-head list.
print(preset.ranked_heads)   # Descending paper variance order.
monitor = RoPEMonitor(model, MonitorConfig(head_preset="qwen3-8b"))
```

The small preset JSON ships in the wheel. Research utilities live in the source
repository and source distribution, with plotting as an optional dependency:

```bash
pip install -e '.[trainer,research]'
python examples/select_heads.py --demo --output-dir outputs/head_selection
python examples/scale_bands.py
```

`select_heads.py` accepts a CSV/JSON table of already-averaged setting/head
`r_H` values, exports means and sample variances, selects heads by either
statistic, and draws two heatmaps. The demo uses synthetic data. The
`high_frequency_norm_share` helper computes paired Q/K `r_H` for users collecting
their own calibration table. First average valid pairs within each example,
then examples within each setting; do not pool all pairs across settings.

`BandScaleIntervention` scales high/low score coefficients on selected query
heads during native inference and generation. The band is `window * omega >=
threshold`, using actual native RoPE frequencies. It applies the requested
multiplier to post-normalization, pre-RoPE Q coordinates, preserving native K
sharing. Both coordinates of every rotary pair receive the same multiplier.
This realizes the paper's per-band score scaling; the paper's experiments used
symmetric Q/K square-root scaling. The example does not include coefficient-norm
compensation, frozen rotations, task direction selection or intervention search.
CPU products are checked for non-finite values. `validate_finite=True` also
checks products on accelerators, adding a synchronization per selected layer;
the default accelerator path validates that the multiplier fits Q's dtype.
See [head selection and intervention](https://github.com/acetocarmine11/rope-profiler/blob/v0.1.0/docs/HEAD_SELECTION.md)
for table formats, model use and the fixed-window interpretation.

## Reproducible configuration and reports

```python
import json
from rope_profiler import MonitorConfig

config = MonitorConfig(heads={0: [3, 0]}, sampling_policy="regions")
restored = MonitorConfig.from_dict(json.loads(json.dumps(config.to_dict())))
# After a successful eval:
attachment.monitor.save_report("outputs/rope_eval_report.json")
```

Configuration is immutable, supports pickle and `dataclasses.asdict`, and has
explicit JSON round trips. Reports include runtime versions, complete settings,
selected layers/heads, actual frequency snapshots and amplitude, per-example
windows, sampled token positions/pairs/triplets, distance budgets, cache counts,
and cached tensor bytes. Input IDs and text are represented by fingerprints;
caller-supplied sample IDs are kept as supplied. Reports retain the grids used
by each forward even if a dynamic RoPE buffer later changes. Cache bytes measure
unique cached tensor storage and exclude the model's native frequency buffers.
This is distinct from retained Q/K bytes and total device peak memory.
`report_dir` preserves reports from each successful evaluation automatically;
`save_report(path)` exports the latest one to an explicit path. Create a fresh
`monitor.evaluation()` and `monitor.batch(...)` context for each use: contexts
are single use, preventing stale sampling budgets from carrying into later evals.

## Current support and practical limits

The intended first target is ordinary single-process Transformers Llama and
Qwen3 causal language models with separate Q/K projection modules and supported
RoPE layouts. CPU checks use tiny models constructed from config, without
checkpoint downloads. Qwen3 capture is after its Q/K norms. Standard eager and
SDPA paths can keep their original attention backend.
Models using sliding-window attention are rejected during initialization: their
window masks require an adapter that selects only eligible token pairs.

During generated evaluation, only the prompt prefill is monitored. Repeated
decode calls are excluded. The first version does not provide a separate score
for generated tokens, cached-prefix continuations, beam-reordered streams,
vLLM kernels, tensor-parallel head shards, or fused QKV/RoPE implementations.
Such paths require explicit adapters and verification.

CPU two-process scalar reduction has been checked; distributed Trainer,
CUDA/NCCL and sharded execution are unverified. Each process samples the first
`max_examples` of its own eval shard, and scalar statistics are reduced across
processes when PyTorch distributed execution is initialized. This does not
enforce a single global sample budget or remove duplicated sampler rows. Use
the single-process path for a reported reference curve. `torch.compile` and CUDA
graph compatibility are not yet verified; Python hooks may affect their
execution.

The installation range is Python 3.10–3.12, torch 2.x (>=2.4), and
Transformers 4.51–4.56 for the Trainer extra. Installation ranges are distinct
from actually executed checks. See [compatibility evidence](https://github.com/acetocarmine11/rope-profiler/blob/v0.1.0/docs/COMPATIBILITY.md).
Transformers 5, fused QKV, internal attention-mask gaps and packed/reset/fractional
positions require separate adapters or validation.

## Local example and checks

```bash
python examples/train_monitor.py
python examples/train_monitor.py --wandb-offline
python examples/train_monitor.py --regions --full-distances
pytest
```

The example trains a tiny randomly initialized Llama on synthetic tokens and
logs eval scores every two steps. Its default path stays entirely local.
After training, the requested output directory contains readable
`log_history.json`, including both eval records, and `monitor_metadata.json`
with the model and sampling configuration. Use `--output-dir PATH` to choose
their destination.
The offline option explicitly enables the Trainer's normal offline W&B
integration. It demonstrates integration behavior rather than scientific
score quality on a trained model.

The example saves `log_history.json`, `monitor_metadata.json`, and
`monitor_report.json` in its output directory, plus a report for each eval in
`rope_reports/`. Further documentation:

- [Version and compatibility evidence](https://github.com/acetocarmine11/rope-profiler/blob/v0.1.0/docs/COMPATIBILITY.md)
- [Release and upload procedure](https://github.com/acetocarmine11/rope-profiler/blob/v0.1.0/docs/RELEASING.md)
- [Changes](https://github.com/acetocarmine11/rope-profiler/blob/v0.1.0/CHANGELOG.md)

See the [validation report](VALIDATION.md) for the executed CPU compatibility
combinations, installed-wheel checks, training example, and limits of the current
evidence. The corresponding guides are included under `docs/` in the source
archive.
