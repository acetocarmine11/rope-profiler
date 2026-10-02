# RoPE Profiler

RoPE Profiler is a Python package that adds two RoPE diagnostic scores to existing benchmark evaluations and training-time evaluations. Integrate it with a Hugging Face `Trainer` in a single call, or wrap your own evaluation loop with `RoPEMonitor`. It reuses sampled query and key activations from the model's existing forward pass to measure **semantic stability** and **positional sensitivity**, alongside your usual loss and accuracy metrics, with **zero additional model forward passes**.

This is the official repository for [**RoPE at the End of Its Rope? Theory, Diagnosis, and Mitigation of Long-Context Failures**](https://arxiv.org/abs/2609.39929), by **Yuyang Wu, Yufeng Du, and Hao Peng**.

[Paper](https://arxiv.org/abs/2609.39929) · [Monitoring guide](https://github.com/acetocarmine11/rope-profiler/blob/v0.1.0/docs/MONITORING.md) · [Head selection and intervention](https://github.com/acetocarmine11/rope-profiler/blob/v0.1.0/docs/HEAD_SELECTION.md) · [Citation](#citation)

## Installation

Install RoPE Profiler with:

```bash
pip install rope-profiler
```

This installs the scoring API and the paper's head presets. If your training
environment already has compatible Transformers and Accelerate, this command
also suffices for the Trainer integration.

To install the optional Trainer dependencies as well:

```bash
pip install 'rope-profiler[trainer]'
```

To also install Weights & Biases (W&B):

```bash
pip install 'rope-profiler[trainer,wandb]'
```

The core package requires Python 3.10–3.12 and PyTorch >=2.4,<3. The Trainer extra supports Transformers >=4.51,<4.57. Optional dependencies stay optional; importing the core package does not load a model or initialize a logging account. See the [compatibility guide](https://github.com/acetocarmine11/rope-profiler/blob/v0.1.0/docs/COMPATIBILITY.md) for tested combinations.

## One-call integration with your Trainer

Add this call after constructing your existing `trainer`:

```python
from rope_profiler import attach_rope_monitor

attachment = attach_rope_monitor(trainer)
trainer.train()
```

Every evaluation scheduled by your Trainer now includes:

```text
eval_rope_semantic_score
eval_rope_positional_score
```

The same attachment also monitors `trainer.evaluate()` and `trainer.predict()`. Your Trainer continues to control evaluation scheduling, ordinary metrics, and logging. If it already uses `report_to="wandb"`, the two scores appear in its existing W&B run through the standard Trainer logging path.

To configure sampling and retain a separate report for each evaluation:

```python
from rope_profiler import MonitorConfig, attach_rope_monitor

attachment = attach_rope_monitor(
    trainer,
    config=MonitorConfig(
        queries=4,
        keys=16,
        distance_samples=256,
        max_examples=32,
        seed=0,
    ),
    report_dir="outputs/rope_reports",
)
trainer.train()
print(attachment.monitor.last_metrics)
attachment.detach()
```

`detach()` restores the Trainer's original methods. Custom Trainer subclasses are supported when their evaluation path still calls `prediction_step` through `evaluation_loop` or `prediction_loop`.

The examples below are included in the [source checkout](https://github.com/acetocarmine11/rope-profiler/tree/v0.1.0/examples), rather than the installed wheel. Try the complete CPU training example with a tiny random model; no checkpoint download is needed:

```bash
python examples/train_monitor.py
# Optional: exercise the existing Trainer's W&B integration offline.
python examples/train_monitor.py --wandb-offline
```

## What the scores measure

| Score | Interpretation | Calculation |
| --- | --- | --- |
| `rope_semantic_score` | Higher values indicate a more stable preference between two keys as their virtual relative position changes. | One minus the Gaussian estimate of semantic reversal probability, using one query and two causal keys. |
| `rope_positional_score` | Higher values indicate a stronger local response to a change in relative position. | The mean absolute adjacent-position change in the frozen-Q/K attention score, normalized by the spectral coefficient norm. |

Both scores use the full rotary spectrum. They report raw diagnostic values with their mathematical normalization. Valid observations are averaged within heads, selected heads within examples, and examples within an evaluation. Undefined cells and reference ties are excluded and counted in the report.

By default, the monitor samples 4 query tokens, 16 key tokens, and up to 256 virtual distances for the first 32 examples. Gaussian reversal estimation avoids enumerating positional shifts. Token and head sampling reduce the activations retained for diagnostics while the model processes its ordinary full input.

For comparisons across training checkpoints, keep the validation subset, order, tokenizer, seed, and selected heads fixed. Question/context sampling is available through region masks. `MonitorConfig.full_distances(...)` enumerates all adjacent virtual distances for the selected tokens and heads. See the [monitoring guide](https://github.com/acetocarmine11/rope-profiler/blob/v0.1.0/docs/MONITORING.md) and [region sampling guide](https://github.com/acetocarmine11/rope-profiler/blob/v0.1.0/docs/REGIONS.md) for the exact protocol and report fields.

## Use your own evaluation loop

The standalone interface attaches to an already loaded model:

```python
import torch
from rope_profiler import RoPEMonitor

monitor = RoPEMonitor(model, report_dir="outputs/rope_reports")
model.eval()

with torch.inference_mode(), monitor.evaluation(step=training_step):
    for batch in eval_loader:
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch.get("attention_mask"),
            "position_ids": batch.get("position_ids"),
        }
        with monitor.batch(**inputs):
            outputs = model(**inputs)
        # Continue your existing benchmark evaluation using outputs.

print(monitor.last_metrics)
```

Keep any additional model arguments from your existing evaluation, such as `labels` for loss calculation, in the `model(...)` call. Pass token-position and sampling metadata to `monitor.batch(...)`.

To send standalone metrics to an existing W&B run, pass `logger=WandbLogger(run=existing_run)` from `rope_profiler.loggers`. Standalone metric names include `eval/rope_semantic_score` and `eval/rope_positional_score`. Local JSONL logging is also available.

## Built-in paper head presets

The installed package includes the paper's fixed top-5% head lists:

| Model | Preset | Selected query heads |
| --- | --- | --- |
| Qwen3-8B | `qwen3-8b` | 58 / 1152 |
| Llama-3.1-8B-Instruct | `llama-3.1-8b-instruct` | 52 / 1024 |

These heads were ranked by the sample variance of each head's mean high-frequency norm share, `r_H`, across 49 task/length settings. Both diagnostic scores use the same frozen list. The preset resource includes selection provenance and ships inside the wheel.

```python
from rope_profiler import MonitorConfig, RoPEMonitor, get_head_preset

preset = get_head_preset("qwen3-8b")
print(preset.heads)  # Zero-based layer -> query-head indices.
monitor = RoPEMonitor(model, config=MonitorConfig(head_preset="qwen3-8b"))
```

The default `head_preset="auto"` recognizes the exact paper model identifiers and their short names. Local checkpoint paths require an explicit preset. Other model identities use all supported heads. Use `head_preset=None` for all heads, or supply a custom `heads` mapping. Preset selection validates the loaded model's layout.

## Additional repository tools

The repository and source archive provide head selection, visualization, and basic inference intervention utilities. These utilities are available from a source checkout; the core wheel contains the scoring API and paper presets. Install the optional plotting dependency from the repository root:

```bash
git clone --branch v0.1.0 https://github.com/acetocarmine11/rope-profiler.git
cd rope-profiler
python -m pip install -e '.[trainer,research]'
```

### Select heads and visualize mean/variance heatmaps

Run a self-contained demonstration:

```bash
python examples/select_heads.py --demo --output-dir outputs/head_selection
```

Or provide your own calibration table:

```bash
python examples/select_heads.py --input calibration.csv --output-dir outputs/my_heads
```

The CSV/JSON input contains one already-averaged value per task/length setting and head, with `task`, `layer`, `head`, and `r_h` fields; an optional `model` field separates models. `r_H` denotes the high-frequency coefficient norm divided by the full-spectrum coefficient norm. Indices are zero-based. Each head must have the same setting coverage.

Outputs include `head_statistics.csv`, `head_selection.json`, `selected_heads.json`, and separate mean and variance heatmaps in PNG/PDF. The tool exports selections based on either mean or sample variance; the default fraction is 5%, rounded up. The demo uses synthetic data.

[`research_tools/head_selection.py`](https://github.com/acetocarmine11/rope-profiler/blob/v0.1.0/research_tools/head_selection.py) also provides `high_frequency_norm_share` for paired pre-RoPE Q/K activations. Calibration requires running your model and collecting those activations. Average valid pairs within each example, then examples within each setting before passing the table to the selection script.

### Scale high/low frequency bands during inference

Try the tiny-model example:

```bash
python examples/scale_bands.py --family qwen3 --high-scale 0.5 --low-scale 1.0
```

Apply the same utility to your own supported model from the repository root:

```python
import torch
from rope_profiler import get_head_preset
from research_tools.intervention import BandScaleIntervention

model.eval()
with torch.inference_mode(), BandScaleIntervention(
    model,
    window=8192,
    threshold=2.0,
    high_scale=0.5,
    low_scale=1.0,
    heads=get_head_preset("qwen3-8b").heads,
):
    generated = model.generate(**inputs, max_new_tokens=128)
```

High-frequency planes satisfy `window * omega >= threshold`, using the native frequencies observed on the context's first forward. The mask remains fixed during that context, including cached decode. `high_scale` and `low_scale` multiply the corresponding Q/K score coefficients through post-normalization, pre-RoPE query scaling. Native shared key heads are preserved, and exiting the context removes the hooks. The window, threshold, and scale values above are illustrative settings.

This utility implements basic band scaling. The paper's complete intervention experiments additionally used symmetric Q/K scaling and experiment-specific choices such as coefficient-norm compensation and task direction selection. See the [head selection and intervention guide](https://github.com/acetocarmine11/rope-profiler/blob/v0.1.0/docs/HEAD_SELECTION.md) for the input format, threshold choice, and supported execution paths.

## Compatibility and validation

The reference path is a single-process native Transformers Llama or Qwen3 model with separate Q/K projections, using eager or SDPA attention. Monitoring captures Q/K after normalization and before RoPE, preserves model outputs, and adds no forward passes. During generation, monitoring covers prompt prefill; the intervention utility also supports ordinary cached decode.

The implementation has been checked across three CPU dependency combinations and on the two paper 8B checkpoints with BF16 SDPA on an RTX A6000. The GPU checks cover output preservation, sparse capture, score calculation, band scaling, cached generation, and restoration. Paired latency measurements at 512, 2048, and 8192 tokens are documented in the [validation report](https://github.com/acetocarmine11/rope-profiler/blob/v0.1.0/docs/VALIDATION.md); overhead depends on input length and configuration.

Sliding-window attention, cached-prefix monitoring, and beam-expanded streams are currently rejected. Fused QKV/RoPE, FlashAttention 2, vLLM, `torch.compile`, CUDA graphs, FSDP, tensor parallelism, and production distributed Trainer execution require additional adapters or validation. See [compatibility evidence](https://github.com/acetocarmine11/rope-profiler/blob/v0.1.0/docs/COMPATIBILITY.md).

## Contributing and support

Please open a [GitHub issue](https://github.com/acetocarmine11/rope-profiler/issues) for bugs or feature requests. Include package/runtime versions, your monitor configuration, and a minimal reproduction. Contributions are welcome; see [CONTRIBUTING.md](https://github.com/acetocarmine11/rope-profiler/blob/v0.1.0/CONTRIBUTING.md). Release preparation is documented in [RELEASING.md](https://github.com/acetocarmine11/rope-profiler/blob/v0.1.0/docs/RELEASING.md).

## Citation

If you use RoPE Profiler in your research, please cite our paper:

```bibtex
@article{wu2026rope,
  title         = {{RoPE} at the End of Its Rope? Theory, Diagnosis, and Mitigation of Long-Context Failures},
  author        = {Wu, Yuyang and Du, Yufeng and Peng, Hao},
  journal       = {arXiv preprint arXiv:2609.39929},
  year          = {2026},
  eprint        = {2609.39929},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2609.39929}
}
```

Machine-readable citation metadata is provided in [`CITATION.cff`](https://github.com/acetocarmine11/rope-profiler/blob/v0.1.0/CITATION.cff).
