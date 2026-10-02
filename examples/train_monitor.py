"""An entirely local tiny training run with RoPE scores after each eval.

Run ``python examples/train_monitor.py`` from the package directory.
Use ``--wandb-offline`` to let the Trainer's standard W&B integration create
an offline run.  No checkpoint download, account login, or online run is used.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments

from rope_profiler import MonitorConfig
from rope_profiler.integrations import attach_rope_monitor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tiny_training"))
    parser.add_argument("--wandb-offline", action="store_true")
    parser.add_argument("--regions", action="store_true", help="use synthetic context/question token regions")
    parser.add_argument("--full-distances", action="store_true", help="enumerate positional distances of the sampled tokens")
    args = parser.parse_args()
    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ.setdefault("WANDB_PROJECT", "rope-profiler-local-demo")
    torch.manual_seed(0)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, max_position_embeddings=64,
        )
    )

    def dataset(count: int):
        rows = []
        for _ in range(count):
            ids = torch.randint(3, 64, (24,))
            rows.append({"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()})
        return rows

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.output_dir),
            use_cpu=True,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            max_steps=4,
            eval_strategy="steps",
            eval_steps=2,
            logging_steps=2,
            save_strategy="no",
            report_to=["wandb"] if args.wandb_offline else [],
            dataloader_pin_memory=False,
            disable_tqdm=True,
        ),
        train_dataset=dataset(8),
        eval_dataset=dataset(4),
    )
    def region_metadata(inputs):
        # Synthetic demo: final four tokens are query candidates, prefix is context.
        # Real tasks should use their native character spans and tokenizer offsets.
        ids = inputs["input_ids"]
        columns = torch.arange(ids.shape[1], device=ids.device)[None, :].expand_as(ids)
        return {"query_mask": columns >= ids.shape[1] - 4,
                "key_mask": columns < ids.shape[1] - 4}

    attachment = attach_rope_monitor(
        trainer,
        config=MonitorConfig(queries=4, keys=8, distance_samples=None if args.full_distances else 64,
                             max_examples=4, seed=0, sampling_policy="regions" if args.regions else "causal"),
        batch_metadata=region_metadata if args.regions else None,
        report_dir=args.output_dir / "rope_reports",
    )
    try:
        trainer.train()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        history_path = args.output_dir / "log_history.json"
        metadata_path = args.output_dir / "monitor_metadata.json"
        history_path.write_text(
            json.dumps(trainer.state.log_history, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        metadata_path.write_text(
            json.dumps(attachment.monitor.metadata(), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        report_path = attachment.monitor.save_report(args.output_dir / "monitor_report.json")
        print("Final evaluation RoPE metrics:", attachment.monitor.last_metrics)
        print("Saved evaluation history:", history_path.resolve())
        print("Saved monitor metadata:", metadata_path.resolve())
        print("Saved monitor report:", report_path.resolve())
        print("Saved per-evaluation reports:", (args.output_dir / "rope_reports").resolve())
    finally:
        attachment.detach()


if __name__ == "__main__":
    main()
