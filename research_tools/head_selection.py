"""Offline head selection from per-setting Common-pair r_H means.

r_H is U_H/U, the high-frequency coefficient norm share. Its square is the
energy fraction. The paper selects by sample variance across equally weighted
task/length settings; mean-based selection is an additional descriptive view.
No model, Torch, NumPy, or plotting library is imported for the statistics.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


def load_head_records(path: str | Path) -> list[dict[str, Any]]:
    """Load CSV or a JSON array, with task, layer, head, and r_h/r_H columns.

    Each row is one setting/head mean, already averaged over valid pairs
    within each example and then equally over examples. Indices are zero-based.
    An optional model column keeps different models in separate rankings.
    """
    source = Path(path)
    if source.suffix.lower() == ".csv":
        with source.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ValueError("CSV must have distinct column names")
            return list(reader)
    if source.suffix.lower() == ".json":
        value = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError("JSON input must be an array of setting/head records")
        return value
    raise ValueError("head statistics input must be a .csv or .json file")


def _index(value: Any, name: str) -> int:
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value.strip()):
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative, zero-based integer")
    return value


def _name(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def high_frequency_norm_share(q: Any, k: Any, high_mask: Any,
                              query_to_kv: Any = None) -> Any:
    """Compute Common-pair r_H=U_H/U from paired pre-RoPE Q/K tensors.

    Every last-dimension coordinate must participate in split-half RoPE.
    q/k have matching shapes [..., D] when already paired. For native GQA,
    shapes are [..., Hq, D]/[..., Hkv, D], with query_to_kv mapping the
    penultimate head axis. The output drops D. high_mask is Boolean [D/2].
    Float64 arithmetic and per-vector scale normalization avoid squaring
    overflow. Zero or nonfinite coefficient norms produce NaN. This helper
    does no model computation; its tensor dependency is imported on demand.
    """
    import torch

    if not isinstance(q, torch.Tensor) or not isinstance(k, torch.Tensor):
        raise TypeError("q and k must be Torch tensors")
    if q.ndim < 1 or k.ndim < 1 or not q.is_floating_point() or not k.is_floating_point():
        raise ValueError("q and k must be floating-point vectors")
    if q.device != k.device or q.shape[-1] != k.shape[-1] or q.shape[-1] < 2 or q.shape[-1] % 2:
        raise ValueError("q and k require the same device and a positive even full rotary dimension")
    if query_to_kv is not None:
        if q.ndim < 2 or k.ndim < 2:
            raise ValueError("query_to_kv requires a penultimate head axis")
        if isinstance(query_to_kv, (tuple, list)) and any(isinstance(index, bool) for index in query_to_kv):
            raise ValueError("query_to_kv must contain integer head indices")
        mapping = torch.as_tensor(query_to_kv, device=q.device)
        if mapping.ndim != 1 or mapping.numel() != q.shape[-2] or mapping.dtype not in (torch.int32, torch.int64):
            raise ValueError("query_to_kv must contain one integer index per query head")
        if torch.any((mapping < 0) | (mapping >= k.shape[-2])):
            raise ValueError("query_to_kv index is outside the native KV heads")
        k = k.index_select(-2, mapping.to(dtype=torch.long))
    if q.shape != k.shape:
        raise ValueError("paired Q/K shapes must match after optional native GQA mapping")
    half = q.shape[-1] // 2
    mask = torch.as_tensor(high_mask, device=q.device)
    if mask.dtype != torch.bool or mask.ndim != 1 or mask.numel() != half:
        raise ValueError("high_mask must be Boolean [D/2]; this helper requires all D coordinates to participate in RoPE")
    normalized = []
    for vector in (q, k):
        value = vector.to(dtype=torch.float64)
        scale = value.abs().amax(dim=-1, keepdim=True)
        value = value / torch.where(scale == 0, torch.ones_like(scale), scale)
        normalized.append(value[..., :half].square() + value[..., half:].square())
    energy = normalized[0] * normalized[1]
    total = energy.sum(dim=-1)
    high = energy[..., mask].sum(dim=-1)
    ratio = (high / total).clamp(0, 1).sqrt()
    return torch.where(torch.isfinite(total) & (total > 0), ratio, float("nan"))


def select_heads(records: Iterable[Mapping[str, Any]], *, fraction: float | None = None,
                 top_k: int | None = None, ddof: int = 1) -> dict[str, Any]:
    """Rank mean and between-setting variance, using equal setting weights.

    Default selection retains ceil(0.05 * candidate heads), separately per
    model. Explicit top_k selects at most that many heads. Sorting is descending
    by the requested statistic, then ascending (layer, head). The default
    sample variance (ddof=1) matches the paper; ddof=0 explicitly requests
    population variance. Every head must cover the same reference settings.
    Duplicate setting/head rows are rejected to avoid silently weighting
    examples or pairs instead of settings.
    """
    if isinstance(ddof, bool) or ddof not in (0, 1):
        raise ValueError("ddof must be 0 (population) or 1 (sample)")
    if top_k is not None:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if fraction is not None:
            raise ValueError("fraction and top_k are mutually exclusive")
    else:
        fraction = 0.05 if fraction is None else fraction
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError("fraction must be finite and in (0, 1]")
    grouped: dict[str, dict[tuple[int, int], dict[str, float]]] = {}
    required = {"task", "layer", "head"}
    for row_number, row in enumerate(records, 1):
        if not isinstance(row, Mapping) or not required.issubset(row) or not ({"r_H", "r_h"} & set(row)):
            raise ValueError(f"row {row_number} requires task, layer, head, and r_H=U_H/U (column r_h or r_H); semantic/positional scores cannot replace r_H")
        if "r_H" in row and "r_h" in row:
            raise ValueError("r_h and r_H are aliases; supply exactly one to avoid ambiguity")
        raw_rh = row["r_h"] if "r_h" in row else row["r_H"]
        model = _name(row.get("model", "model"), "model")
        task = _name(row["task"], "task")
        layer, head = _index(row["layer"], "layer"), _index(row["head"], "head")
        if isinstance(raw_rh, bool):
            raise ValueError(f"row {row_number}: r_H must be a finite number in [0, 1]")
        try:
            value = float(raw_rh)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"row {row_number}: r_H must be a finite number in [0, 1]") from exc
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"row {row_number}: r_H must be a finite number in [0, 1]")
        setting_values = grouped.setdefault(model, {}).setdefault((layer, head), {})
        if task in setting_values:
            raise ValueError(f"duplicate setting/head row for {model!r}, {task!r}, layer {layer}, head {head}; supply one already-aggregated setting mean r_H")
        setting_values[task] = value
    if not grouped:
        raise ValueError("at least one setting/head record is required")
    result: dict[str, Any] = {
        "schema_version": 1, "metric": "r_H",
        "metric_definition": "U_H/U; high-frequency coefficient norm share (energy fraction is r_H^2)",
        "input_unit": "setting/head Common-pair mean; valid pairs averaged per example, then equal examples",
        "task_weighting": "equal", "index_base": 0,
        "candidate_universe": "unique layer/head units in input; provide every native head for paper fractions",
        "ddof": ddof, "variance_type": "sample" if ddof else "population",
        "tie_break": "ascending layer, then ascending head",
        "selection": {"fraction": fraction, "requested_top_k": top_k,
                      "count_rule": "min(top_k, H)" if top_k is not None else "ceil(fraction * H)"},
        "paper_criterion": "variance", "mean_criterion": "additional descriptive selection",
        "models": {},
    }
    for model, heads in sorted(grouped.items()):
        tasks = sorted({task for settings in heads.values() for task in settings})
        n = len(tasks)
        if n <= ddof:
            raise ValueError(f"{model!r}: sample variance requires at least two reference settings")
        statistics = []
        for (layer, head), settings in sorted(heads.items()):
            if set(settings) != set(tasks):
                missing = sorted(set(tasks) - set(settings))
                raise ValueError(f"{model!r} layer {layer}, head {head} is missing reference settings {missing}; complete coverage is required")
            values = [settings[task] for task in tasks]
            mean = math.fsum(values) / n
            squared = math.fsum((value - mean) ** 2 for value in values)
            statistics.append({"layer": layer, "head": head, "task_count": n,
                "mean": mean, "variance": squared / (n - ddof)})
        count = min(top_k, len(statistics)) if top_k is not None else math.ceil(fraction * len(statistics))
        rankings = {criterion: sorted(statistics, key=lambda cell: (-cell[criterion], cell["layer"], cell["head"]))
                    for criterion in ("mean", "variance")}
        for criterion, ranking in rankings.items():
            for rank, cell in enumerate(ranking, 1):
                cell[f"rank_{criterion}"] = rank
                cell[f"selected_{criterion}"] = rank <= count
        result["models"][model] = {"tasks": tasks, "task_count": n,
            "head_count": len(statistics), "selected_count": count,
            "statistics": statistics,
            "top_mean": rankings["mean"][:count], "top_variance": rankings["variance"][:count]}
    return result


def to_head_mapping(selection: Mapping[str, Any], model: str | None = None,
                    criterion: str = "variance") -> dict[int, list[int]]:
    """Convert a selected list directly to MonitorConfig(heads=...) indices."""
    if criterion not in ("mean", "variance"):
        raise ValueError("criterion must be mean or variance")
    models = selection["models"]
    if model is None:
        if len(models) != 1:
            raise ValueError("specify model when the selection contains multiple models")
        model = next(iter(models))
    if model not in models:
        raise ValueError(f"unknown model {model!r}")
    mapping: dict[int, list[int]] = {}
    for cell in models[model][f"top_{criterion}"]:
        mapping.setdefault(cell["layer"], []).append(cell["head"])
    return {layer: sorted(heads) for layer, heads in sorted(mapping.items())}


def plot_heatmaps(selection: Mapping[str, Any], output_dir: str | Path, *,
                  display_index_base: int = 1) -> dict[str, dict[str, str]]:
    """Save mean/variance PNG and PDF heatmaps, with selected cells outlined.

    Color scales are shared across models: mean in [0,1], variance in
    [0, maximum observed variance]. Axes display one-based indices by default;
    all input, JSON, and CSV indices remain zero-based.
    """
    if display_index_base not in (0, 1) or isinstance(display_index_base, bool):
        raise ValueError("display_index_base must be 0 or 1")
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.patches import Rectangle
    except ImportError as exc:
        raise RuntimeError("heatmaps require the optional matplotlib plotting dependency; install it or use --no-plots") from exc
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    maximum = max(cell["variance"] for model in selection["models"].values() for cell in model["statistics"])
    paths = {}
    for model_name, model in selection["models"].items():
        cells = model["statistics"]
        layers = sorted({cell["layer"] for cell in cells})
        heads = sorted({cell["head"] for cell in cells})
        layer_positions = {value: i for i, value in enumerate(layers)}
        head_positions = {value: i for i, value in enumerate(heads)}
        slug = re.sub(r"[^A-Za-z0-9_-]+", "-", model_name).strip("-")[:48] or "model"
        slug += "-" + hashlib.sha256(model_name.encode()).hexdigest()[:8]
        paths[model_name] = {}
        for criterion in ("mean", "variance"):
            matrix = [[float("nan")] * len(heads) for _ in layers]
            for cell in cells:
                matrix[layer_positions[cell["layer"]]][head_positions[cell["head"]]] = cell[criterion]
            figure = Figure(figsize=(max(6, min(13, len(heads) * .28)), max(3.5, min(11, len(layers) * .22 + 1.8))), constrained_layout=True)
            FigureCanvasAgg(figure)
            axis = figure.subplots()
            axis.set_facecolor("#dddddd")
            image = axis.imshow(matrix, interpolation="nearest", aspect="auto", cmap="viridis",
                                vmin=0, vmax=1 if criterion == "mean" else max(maximum, 1e-12))
            x_ticks = list(range(0, len(heads), max(1, math.ceil(len(heads) / 32))))
            y_ticks = list(range(0, len(layers), max(1, math.ceil(len(layers) / 40))))
            axis.set_xticks(x_ticks, [heads[i] + display_index_base for i in x_ticks], fontsize=8)
            axis.set_yticks(y_ticks, [layers[i] + display_index_base for i in y_ticks], fontsize=8)
            axis.set_xlabel(f"Query head ({display_index_base}-based display)")
            axis.set_ylabel(f"Layer ({display_index_base}-based display)")
            label = "Mean $r_H$" if criterion == "mean" else f"{selection['variance_type'].capitalize()} variance of setting means $r_H$"
            axis.set_title(f"{model_name}: {label}\nOutlined: {model['selected_count']} of {model['head_count']} heads")
            for cell in model[f"top_{criterion}"]:
                axis.add_patch(Rectangle((head_positions[cell["head"]] - .5,
                                          layer_positions[cell["layer"]] - .5), 1, 1,
                                         fill=False, edgecolor="black", linewidth=1.4))
            figure.colorbar(image, ax=axis, label=label)
            try:
                for extension in ("png", "pdf"):
                    target = directory / f"{slug}-{criterion}_heatmap.{extension}"
                    figure.savefig(target, dpi=200)
                    paths[model_name][f"{criterion}_{extension}"] = str(target)
            finally:
                figure.clear()
    return paths


def write_selection(selection: Mapping[str, Any], output_dir: str | Path, *,
                    plots: bool = True, display_index_base: int = 1) -> dict[str, Any]:
    """Write full statistics, both selected lists, and optional static heatmaps."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    full = directory / "head_selection.json"
    full.write_text(json.dumps(selection, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    selected = directory / "selected_heads.json"
    selected.write_text(json.dumps({name: {criterion: [{"layer": cell["layer"], "head": cell["head"]}
        for cell in model[f"top_{criterion}"]] for criterion in ("mean", "variance")}
        for name, model in selection["models"].items()}, indent=2) + "\n", encoding="utf-8")
    statistics = directory / "head_statistics.csv"
    fields = ["model", "layer", "head", "task_count", "ddof", "mean", "variance",
              "rank_mean", "rank_variance", "selected_mean", "selected_variance"]
    with statistics.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, model in selection["models"].items():
            writer.writerows({"model": name, "ddof": selection["ddof"], **cell}
                             for cell in model["statistics"])
    return {"selection_json": str(full), "selected_heads_json": str(selected),
            "statistics_csv": str(statistics),
            "heatmaps": plot_heatmaps(selection, directory, display_index_base=display_index_base) if plots else {}}


def synthetic_records() -> list[dict[str, Any]]:
    """A deterministic CPU demo; these values describe no trained model."""
    records = []
    for task in range(6):
        for layer in range(2):
            for head in range(20):
                flat = layer * 20 + head
                if flat == 0:
                    value = .95  # Largest mean, zero variance.
                elif flat == 39:
                    value = .1 if task % 2 == 0 else .9  # Largest variance.
                else:
                    value = .2 + .5 * flat / 40 + .03 * (task - 2.5) * flat / 40
                records.append({"model": "synthetic-demo", "task": f"task-{task}",
                                "layer": layer, "head": head, "r_h": value})
    return records


__all__ = ["load_head_records", "select_heads", "high_frequency_norm_share",
           "to_head_mapping", "plot_heatmaps", "write_selection", "synthetic_records"]
