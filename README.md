# RoPE Profiler

RoPE Profiler is a Python package that adds two RoPE diagnostic scores to existing benchmark evaluations and training-time evaluations. Integrate it with a Hugging Face `Trainer` in a single call, or wrap your own evaluation loop with `RoPEMonitor`. It reuses sampled query and key activations from the model's existing forward pass to measure **semantic stability** and **positional sensitivity**, alongside your usual loss and accuracy metrics, with **zero additional model forward passes**.

This is the official repository for [**RoPE at the End of Its Rope? Theory, Diagnosis, and Mitigation of Long-Context Failures**](https://arxiv.org/abs/2609.39929), by **Yuyang Wu, Yufeng Du, and Hao Peng**.

[Paper](https://arxiv.org/abs/2609.39929) · [PyPI](https://pypi.org/project/rope-profiler/) · [Monitoring guide](https://github.com/acetocarmine11/rope-profiler/blob/main/docs/MONITORING.md) · [Head selection and intervention](https://github.com/acetocarmine11/rope-profiler/blob/main/docs/HEAD_SELECTION.md) · [Citation](#citation)

## Installation

RoPE Profiler is available on [PyPI](https://pypi.org/project/rope-profiler/). Install the package with:

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

Requirements: Python 3.10–3.12 and PyTorch >=2.4,<3. The Trainer extra supports Transformers >=4.51,<4.57.

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

`detach()` restores the Trainer's original methods.

## What the scores measure

| Score | Interpretation | Calculation |
| --- | --- | --- |
| `rope_semantic_score` | Higher values indicate a more stable preference between two keys as their virtual relative position changes. | One minus the Gaussian estimate of semantic reversal probability, using one query and two causal keys. |
| `rope_positional_score` | Higher values indicate a stronger local response to a change in relative position. | The mean absolute adjacent-position change in the frozen-Q/K attention score, normalized by the spectral coefficient norm. |

Both scores use the full rotary spectrum and retain their mathematical normalization. Defaults sample 4 query tokens, 16 key tokens, and up to 256 virtual distances for the first 32 examples. For comparisons across checkpoints, keep the evaluation subset, seed, and selected heads fixed. See the [monitoring guide](https://github.com/acetocarmine11/rope-profiler/blob/main/docs/MONITORING.md) for sampling and report details.

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

The presets select heads by the variance of their mean high-frequency norm share across 49 task/length settings. Exact public model identifiers are recognized automatically; local checkpoint paths require an explicit preset. Both scores use the same fixed list. See the [head selection guide](https://github.com/acetocarmine11/rope-profiler/blob/main/docs/HEAD_SELECTION.md) for configuration.

## Additional repository tools

Clone the repository to use head selection, heatmaps, and inference band scaling:

```bash
git clone https://github.com/acetocarmine11/rope-profiler.git
cd rope-profiler
python -m pip install -e '.[trainer,research]'
```

### Head selection and heatmaps

Select heads by mean or variance and generate both heatmaps from a calibration CSV/JSON table. The input contains one already-averaged `r_h` value per setting and head.

```bash
python examples/select_heads.py --demo --output-dir outputs/head_selection
python examples/select_heads.py --input calibration.csv --output-dir outputs/my_heads
```

### Inference band scaling

Try the tiny-model example:

```bash
python examples/scale_bands.py --family qwen3 --high-scale 0.5 --low-scale 1.0
```

See the [head selection and intervention guide](https://github.com/acetocarmine11/rope-profiler/blob/main/docs/HEAD_SELECTION.md) for details.

## License

RoPE Profiler is released under the [MIT License](https://github.com/acetocarmine11/rope-profiler/blob/main/LICENSE).

## Citation

If you find RoPE Profiler helpful in your research, we would appreciate a citation to our paper:

```bibtex
@misc{wu2026ropeendropetheory,
      title={RoPE at the End of Its Rope? Theory, Diagnosis, and Mitigation of Long-Context Failures},
      author={Yuyang Wu and Yufeng Du and Hao Peng},
      year={2026},
      eprint={2609.39929},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2609.39929},
}
```

Machine-readable citation metadata is provided in [CITATION.cff](https://github.com/acetocarmine11/rope-profiler/blob/main/CITATION.cff).
