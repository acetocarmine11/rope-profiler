"""Scale frequency bands on a tiny random CPU model; no weights are downloaded."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

from research_tools.intervention import BandScaleIntervention


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("llama", "qwen3"), default="qwen3")
    parser.add_argument("--backend", choices=("eager", "sdpa"), default="sdpa")
    parser.add_argument("--window", type=int, default=24, help="Fixed reference window for the high-band rule")
    parser.add_argument("--threshold", type=float, default=2.0, help="High when window * native omega >= threshold")
    parser.add_argument("--high-scale", type=float, default=0.5, help="Multiplier on high-band QK score coefficients")
    parser.add_argument("--low-scale", type=float, default=1.0, help="Multiplier on low-band QK score coefficients")
    parser.add_argument("--validate-finite", action="store_true",
                        help="Check scaled Q products on non-CPU devices with synchronization; CPU always checks")
    args = parser.parse_args()
    config_type, model_type = {"llama": (LlamaConfig, LlamaForCausalLM),
                               "qwen3": (Qwen3Config, Qwen3ForCausalLM)}[args.family]
    torch.manual_seed(47)
    config = config_type(vocab_size=83, hidden_size=32, intermediate_size=48,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=8,
                         max_position_embeddings=128, attention_dropout=0.0,
                         pad_token_id=0, bos_token_id=1, eos_token_id=None)
    config._attn_implementation = args.backend
    model = model_type(config).eval()
    ids = torch.randint(3, config.vocab_size, (1, 24))
    mask = torch.ones_like(ids)
    try:
        intervention = BandScaleIntervention(model, window=args.window, threshold=args.threshold,
            high_scale=args.high_scale, low_scale=args.low_scale, heads={0: [0]},
            validate_finite=args.validate_finite)
        with torch.inference_mode():
            baseline = model(input_ids=ids, attention_mask=mask).logits
            with intervention:
                changed = model(input_ids=ids, attention_mask=mask).logits
                generated = model.generate(input_ids=ids, attention_mask=mask,
                                           max_new_tokens=3, do_sample=False, use_cache=True)
            restored = model(input_ids=ids, attention_mask=mask).logits
        torch.testing.assert_close(restored, baseline, rtol=0, atol=0)
    except (ValueError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps({"random_cpu_model": True, "family": args.family, "backend": args.backend,
        "max_logit_change": float((changed - baseline).abs().max()),
        "generated_token_ids": generated[0, ids.shape[1]:].tolist(),
        "baseline_restored": True, "intervention": intervention.metadata()}, indent=2))


if __name__ == "__main__":
    main()
