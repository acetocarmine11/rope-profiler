"""Select attention heads and plot offline r_H statistics without model inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

# This research helper runs directly from the source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_tools.head_selection import load_head_records, select_heads, synthetic_records, write_selection


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank mean and variance of per-setting Common-pair r_H=U_H/U. Input indices are zero-based; tasks/settings have equal weights.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="CSV or JSON array: task,layer,head,r_h (r_H alias); optional model")
    source.add_argument("--demo", action="store_true", help="Run deterministic synthetic statistics, without loading a model")
    count = parser.add_mutually_exclusive_group()
    count.add_argument("--fraction", type=float, help="Retain ceil(fraction * heads), default 0.05 (top 5%%)")
    count.add_argument("--top-k", type=int, help="Explicit fixed count, e.g. 5 heads rather than 5%%")
    parser.add_argument("--ddof", type=int, choices=(0, 1), default=1, help="1: paper sample variance; 0: population variance")
    parser.add_argument("--output-dir", type=Path, default=Path("head_selection_output"))
    parser.add_argument("--no-plots", action="store_true", help="Write statistics/lists without the optional matplotlib dependency")
    parser.add_argument("--plot-index-base", type=int, choices=(0, 1), default=1, help="Displayed heatmap indices; input/output statistics always zero-based")
    args = parser.parse_args()
    try:
        records = synthetic_records() if args.demo else load_head_records(args.input)
        result = select_heads(records, fraction=args.fraction, top_k=args.top_k, ddof=args.ddof)
        result["provenance"] = {"synthetic": args.demo, "model_inference": False}
        if args.input is not None:
            result["provenance"].update({"input_path": str(args.input),
                "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest()})
        paths = write_selection(result, args.output_dir, plots=not args.no_plots,
                                display_index_base=args.plot_index_base)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps({"synthetic": args.demo, "models": {name: {"heads": model["head_count"],
        "settings": model["task_count"], "selected_per_criterion": model["selected_count"]}
        for name, model in result["models"].items()}, "outputs": paths}, indent=2))


if __name__ == "__main__":
    main()
