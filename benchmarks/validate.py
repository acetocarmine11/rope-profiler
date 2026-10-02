"""Measure native eval overhead and sampled-distance error without downloads.

Example CPU smoke:
    python benchmarks/validate.py --family qwen3 --length 1024 --output validation/cpu-qwen3.json

Example on an explicitly allocated fleet GPU with existing weights:
    python benchmarks/validate.py --model-path /absolute/checkpoint --device cuda --length 2048

Only the caller's existing eval forward is executed by the monitor. Timing
includes scoring and final aggregation. Synthetic token IDs are diagnostic
fixtures, not benchmark task accuracy or model ability measurements.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

from rope_profiler import MonitorConfig, RoPEMonitor
import rope_profiler.monitor as monitor_module


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def tiny_model(family):
    torch.manual_seed(71)
    config_type, model_type = {
        "llama": (LlamaConfig, LlamaForCausalLM),
        "qwen3": (Qwen3Config, Qwen3ForCausalLM),
    }[family]
    config = config_type(vocab_size=97, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=16,
                         max_position_embeddings=32768, attention_dropout=0.0,
                         pad_token_id=0, bos_token_id=1, eos_token_id=2)
    config._attn_implementation = "sdpa"
    return model_type(config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("llama", "qwen3"), default="llama")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--queries", type=int, default=4)
    parser.add_argument("--keys", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.length, args.repeats, args.batch_size, args.threads, args.queries) < 1 or args.keys < 2:
        parser.error("length/repeats/batch-size/threads must be positive")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("selected CUDA device is not available")
    started = time.perf_counter()
    if args.model_path:
        model = AutoModelForCausalLM.from_pretrained(
            str(args.model_path.resolve()), local_files_only=True,
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            attn_implementation="sdpa", low_cpu_mem_usage=True,
        )
    else:
        model = tiny_model(args.family)
    model = model.to(device).eval()
    load_seconds = time.perf_counter() - started
    generator = torch.Generator().manual_seed(193)
    input_ids = torch.randint(3, model.config.vocab_size,
                              (args.batch_size, args.length), generator=generator).to(device)
    data = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
    sample_ids = [f"fixed-synthetic-{index}" for index in range(args.batch_size)]
    input_sha = hashlib.sha256(input_ids.cpu().numpy().tobytes()).hexdigest()
    count = {"forward": 0}
    hook = model.register_forward_hook(lambda *_: count.__setitem__("forward", count["forward"] + 1))

    def baseline():
        with torch.no_grad():
            return model(**data).logits

    baseline_logits = baseline()
    baseline()  # Warm SDPA and allocator.
    baseline_times = []
    baseline_memory = None
    for _ in range(args.repeats):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        synchronize(device)
        begin = time.perf_counter()
        baseline()
        synchronize(device)
        baseline_times.append(time.perf_counter() - begin)
        if device.type == "cuda":
            baseline_memory = torch.cuda.max_memory_allocated(device)

    variants = {}
    for distance_samples in (256, 512, None):
        config = MonitorConfig(queries=args.queries, keys=args.keys,
                               distance_samples=distance_samples, max_examples=args.batch_size)
        begin = time.perf_counter()
        monitor = RoPEMonitor(model, config=config)
        preload_seconds = time.perf_counter() - begin
        activations = {"bytes_total": 0, "bytes_peak_layer": 0, "score_calls": 0}
        original = monitor_module.score_activations

        def audit(q, k, *score_args, **score_kwargs):
            size = q.numel() * q.element_size() + k.numel() * k.element_size()
            activations["bytes_total"] += size
            activations["bytes_peak_layer"] = max(activations["bytes_peak_layer"], size)
            activations["score_calls"] += 1
            assert not q.requires_grad and not k.requires_grad
            return original(q, k, *score_args, **score_kwargs)

        monitor_module.score_activations = audit

        def evaluate(step):
            activations.update(bytes_total=0, bytes_peak_layer=0, score_calls=0)
            forwards_before = count["forward"]
            with monitor.evaluation(step=step):
                with monitor.batch(**data, sample_ids=sample_ids):
                    with torch.no_grad():
                        logits = model(**data).logits
            assert count["forward"] - forwards_before == 1
            return logits

        try:
            synchronize(device)
            begin = time.perf_counter()
            logits = evaluate(0)
            synchronize(device)
            cold_seconds = time.perf_counter() - begin
            torch.testing.assert_close(logits, baseline_logits, atol=0, rtol=0)
            times = []
            paired_baselines = []
            paired_deltas = []
            peak_memory = None
            scores = []
            for repeat in range(args.repeats):
                def timed(function):
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    synchronize(device)
                    begin = time.perf_counter()
                    function()
                    synchronize(device)
                    duration = time.perf_counter() - begin
                    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
                    return duration, peak
                # Alternate order to reduce CPU-frequency and thermal drift.
                if repeat % 2:
                    elapsed, peak_memory = timed(lambda: evaluate(repeat + 1))
                    base_elapsed, _ = timed(baseline)
                else:
                    base_elapsed, _ = timed(baseline)
                    elapsed, peak_memory = timed(lambda: evaluate(repeat + 1))
                times.append(elapsed)
                paired_baselines.append(base_elapsed)
                paired_deltas.append(elapsed - base_elapsed)
                scores.append({key: monitor.last_metrics[key] for key in
                               ("rope_semantic_score", "rope_positional_score")})
            assert all(row == scores[0] for row in scores), "fixed sampling changed between eval steps"
            key = "exhaustive" if distance_samples is None else str(distance_samples)
            variants[key] = {
                "config": asdict(config), "preload_seconds": preload_seconds,
                "cold_seconds": cold_seconds, "warm_seconds": times,
                "warm_median_seconds": statistics.median(times),
                "paired_baseline_seconds": paired_baselines,
                "paired_baseline_median_seconds": statistics.median(paired_baselines),
                "paired_incremental_seconds": paired_deltas,
                "metrics": monitor.last_metrics, "activation_audit": dict(activations),
                "cache": {"hits": monitor.cache.hits, "misses": monitor.cache.misses,
                          "entries": len(monitor.cache)},
                "cuda_peak_allocated_bytes": peak_memory,
                "logits_exactly_preserved": True, "extra_forwards": 0,
                "fixed_scores_across_steps": True,
            }
        finally:
            monitor_module.score_activations = original
    hook.remove()
    reference = variants["exhaustive"]["metrics"]
    base_median = statistics.median(baseline_times)
    for result in variants.values():
        result["incremental_median_seconds"] = statistics.median(result["paired_incremental_seconds"])
        result["overhead_percent"] = 100 * result["incremental_median_seconds"] / result["paired_baseline_median_seconds"]
        result["absolute_error_vs_exhaustive"] = {
            key: abs(result["metrics"][key] - reference[key])
            for key in ("rope_semantic_score", "rope_positional_score")
        }
    report = {
        "model": str(args.model_path) if args.model_path else f"random-tiny-{args.family}",
        "device": str(device), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "python": platform.python_version(), "torch_threads": torch.get_num_threads(),
        "source_sha256": {
            str(path.relative_to(Path(__file__).resolve().parents[1])):
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__).resolve(),
                         *sorted((Path(__file__).resolve().parents[1] / "src" / "rope_profiler").glob("*.py"))]},
        "model_load_seconds": load_seconds, "input_shape": list(input_ids.shape),
        "theoretical_full_qk_cache_bytes": sum(
            args.batch_size * args.length * (spec.query_heads + spec.kv_heads)
            * spec.head_dim * next(model.parameters()).element_size()
            for spec in monitor.adapter.layers),
        "input_sha256": input_sha, "baseline_seconds": baseline_times,
        "baseline_median_seconds": base_median,
        "baseline_cuda_peak_allocated_bytes": baseline_memory, "variants": variants,
        "limitation": "Synthetic token IDs; timings and errors apply to this input/model/device only.",
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
