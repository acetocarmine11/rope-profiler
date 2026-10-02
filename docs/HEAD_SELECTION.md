# Head presets, selection, and inference scaling

The package includes the paper's fixed top-5% Common-pair head lists. The source
repository provides selection by mean or variance, two heatmaps, and basic
frequency-band scaling during inference. The initial public version is 0.1.0.

## Fixed presets included in the package

| Model | Layers × query heads | Top-5% count | Preset name |
| --- | --- | --- | --- |
| Qwen/Qwen3-8B | 36 × 32 | 58 | `qwen3-8b` |
| meta-llama/Llama-3.1-8B-Instruct | 32 × 32 | 52 | `llama-3.1-8b-instruct` |

The lists select heads whose mean high-frequency norm share varies most across
49 reference settings. Within each setting, valid query/key pairs are averaged
within each example, then examples receive equal weight to obtain each head's
setting-level `r_H`. Selection uses the sample variance of these 49 values, with
denominator 48. Heads are ranked by descending variance, breaking ties by
ascending layer and head index. The list retains `ceil(0.05 × total query heads)`.
Both diagnostic scores use the same fixed list across tasks and training
checkpoints.

`r_H = U_H / U` is a norm share in [0, 1]; `r_H²` is the corresponding energy
share. For example, `r_H=0.8` means an energy share of 0.64. A larger mean indicates
more high-frequency content on average. A larger variance indicates more change
between settings. The paper presets use variance; mean-based selection is an
additional analysis option supplied by this repository.

The resource is `src/rope_profiler/data/paper_heads.json`, included in the wheel.
The SHA256 of the original frozen selection file is
`c57e4424cf21f9f7c57f6d2e9aa979d67a6ee17daf2e66df080ed5e472035a9d`.
Every selected head was checked against the formal cached-score inputs and
intervention lists. The variance ranking was independently recomputed from the
49-setting frozen data. Pilot results did not replace the fixed lists.

```python
from rope_profiler import MonitorConfig, RoPEMonitor, get_head_preset

preset = get_head_preset("qwen3-8b")
print(preset.heads)         # {layer: (head, ...)}; all indices are zero-based.
print(preset.ranked_heads)  # Preserves the paper's variance ranking.
print(preset.to_dict())     # Counts, provenance, selection rules, and full list.

# Exact public model IDs are recognized automatically.
monitor = RoPEMonitor(model)
# Reuse the original model's preset for a local training checkpoint.
monitor = RoPEMonitor(model, MonitorConfig(head_preset="qwen3-8b"))
# Include every head.
monitor = RoPEMonitor(model, MonitorConfig(head_preset=None))
# Select custom layers/heads; this overrides automatic selection.
monitor = RoPEMonitor(model, MonitorConfig(heads={0: [2, 5], 3: [1]}))
```

Automatic resolution only matches exact public model IDs, exact short names,
and preset names in the model configuration. Other publishers' similarly named
models, local paths, and matching model dimensions do not imply a preset.
Resolution checks the model type, number of layers, and query heads per layer;
a mismatch raises an error. Explicit `heads` or `layers` overrides automatic
selection. An explicit preset combined with a custom selection raises an error.
When `layers` is specified, a selected layer absent from `heads` includes all its
heads. Reports record the resolved selection and preset provenance so that the
statistical scope can remain fixed across evaluations.

## Select heads and draw heatmaps from calibration data

From the repository root, install the optional plotting dependency and run:

```bash
pip install -e '.[trainer,research]'
python examples/select_heads.py --demo --output-dir outputs/head_selection
python examples/select_heads.py --input calibration.csv --output-dir outputs/my_heads
```

Each CSV row contains a setting, zero-based layer/head indices, and that
setting's already averaged `r_h`:

```csv
task,layer,head,r_h
setting_a,0,0,0.42
setting_a,0,1,0.63
setting_b,0,0,0.38
setting_b,0,1,0.81
```

Each setting must cover the same heads, with exactly one row per
task/layer/head. The script exports head means and sample variances, selections
by each statistic, and two layer × head heatmaps. The default fraction is 5%,
with sample variance `ddof=1`. This variance requires at least two settings.
Missing heads are rejected rather than filled with zeros. The synthetic demo
does not reproduce the paper presets.

Use the exported variance selection with the monitor:

```python
import json
from rope_profiler import MonitorConfig, RoPEMonitor
from research_tools.head_selection import to_head_mapping

with open("outputs/my_heads/head_selection.json") as stream:
    selection = json.load(stream)
heads = to_head_mapping(selection, criterion="variance")
monitor = RoPEMonitor(model, MonitorConfig(heads=heads))
```

If the table contains multiple models, also pass `model="name_from_the_table"`.
Use `criterion="mean"` for selection by mean. Black boxes mark selected heads
in the heatmaps. Plot axes default to the paper's one-based display; CSV, JSON,
and model APIs always use zero-based indices.

The helper `research_tools.head_selection.high_frequency_norm_share` computes
`r_H` from already paired Q/K vectors. Capture these vectors after native Q/K
normalization and before RoPE. For the split-half layout, each pair of matching
coordinates forms a two-dimensional plane; supply an explicit high-frequency
mask. Each plane has coefficient energy `(q_a²+q_b²)(k_a²+k_b²)`, giving the norm
share `sqrt(sum of high-frequency energy / sum of all energy)`. For grouped-query
attention (GQA), use `query_to_kv` to identify each query head's native key head.
Filter invalid pairs, then average using the example/setting hierarchy above.

This helper computes statistics from supplied vectors; it does not load a model
or collect a calibration dataset. It expects vectors whose entire coordinate
range participates in RoPE. For partially rotary models, first slice out the
rotary coordinates. Calibration should cover all candidate heads, keep input
sampling and the frequency partition fixed, and record the actual mask and
window. The bundled historical presets use frozen selection records with
`frame_constant=0.05, epsilon=0.10`, partitioning planes using the actual
frequencies for the reference window. The later `0.06` cutoff audit did not
change these presets. A selection made with a different cutoff should have its
own name and provenance.

## Basic frequency-band scaling during inference

```python
import torch
from rope_profiler import get_head_preset
from research_tools.intervention import BandScaleIntervention

model.eval()
M = input_ids.shape[-1]  # No padding in this example; one shared reference window.
theta = model.config.rope_theta
d_rot = model.model.rotary_emb.inv_freq.numel() * 2
rho = theta ** (-2 / d_rot)
gamma = 2 * 0.05 / ((1 - rho) * 0.10)  # Paper intervention cutoff.

with torch.inference_mode(), BandScaleIntervention(
    model,
    window=M,
    threshold=gamma,
    high_scale=0.5,
    low_scale=1.0,
    heads=get_head_preset("qwen3-8b").heads,
):
    outputs = model.generate(input_ids=input_ids, max_new_tokens=32)
```

A plane is high-frequency when `M × actual native omega >= threshold`. The mask
is established after the first native RoPE forward updates its frequencies. It
remains fixed for the entire intervention context, including later generated
tokens. `window` and `threshold` are explicit parameters for your analysis.
The example uses the paper intervention's `c_int=0.05, epsilon=0.10`. The later
frequency-partition audit used a separate configuration, `c_op=0.06`.

`high_scale=0.5` multiplies the selected heads' high-frequency attention-score
contribution by 0.5. `low_scale=1` preserves the low-frequency contribution.
Setting both scales to 1 preserves ordinary inference. Scales may be zero and
must be nonnegative and finite. `heads=None` applies scaling to all heads;
`heads={}` performs no scaling.

On CPU, actual scaling products are always checked for finite values; overflow
raises an error. On other devices, the default checks that the multiplier is
representable in Q's dtype. Pass `validate_finite=True` to check the product as
well; this adds a device synchronization for each selected layer. The ordinary
monitor does not perform these scaling checks.

The helper scales both Q coordinates in each plane after Q normalization and
before RoPE. Native K/V and cache behavior are preserved. The paper scales Q
and K symmetrically by the square root of the multiplier. At fixed layer inputs,
both operations produce the same per-band score multiplier in exact arithmetic;
finite precision can produce small differences. Q-only scaling can target
individual logical query heads without changing a shared K head used by their
unselected GQA siblings. Downstream layer inputs still change in response to
the intervention.

The helper supports native Transformers Llama/Qwen3 eager and SDPA inference,
prompt prefill, and single-token cached decoding. Call `model.eval()` and disable
gradients before entering the context. Hooks are removed on exit or exception.
An existing KV cache may be continued; the partition uses the first forward's
frequencies in the current intervention context. The model continues to handle
later dynamic-RoPE updates while the selected scaling planes remain fixed.

When combined with `RoPEMonitor`, the scaling hook runs first so the monitor
captures Q as used by attention. Either context-entry order produces the same
capture and scores. This helper supplies the basic scale operation. Norm
compensation, frozen rotations, task-dependent direction selection, two-stage
parameter search, and task-accuracy evaluation require further experiment code.

## Package and repository scope

The wheel provides the training/evaluation monitor, fixed preset JSON, and
preset APIs. Its core dependency is torch. The source repository and sdist
provide `research_tools/` and `examples/`; plotting dependencies are available
through the optional `research` extra. Research tools do not run automatically,
initialize an experiment account, or download a checkpoint.

The small-model examples check the interfaces. Separate tests of real Qwen3-8B
and Llama-3.1-8B-Instruct checkpoints on an RTX A6000 also verified native SDPA
capture, local scaling, cached generation, and restoration. The measured
monitoring overhead depends on context length and configuration. See the
[validation report](VALIDATION.md) for the executed settings and evidence limits.
