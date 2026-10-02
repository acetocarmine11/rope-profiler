"""Same-pass evaluation monitoring with bounded sparse activation storage."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import re
import tempfile
import time
from typing import Any, Mapping, Sequence
from uuid import uuid4

import torch

from .adapters import ModelAdapter, LayerSpec, projection_view
from .sampling import SamplePlan, input_fingerprint, sample_tokens
from .scores import ScoreCache, score_activations
from .config import MonitorConfig
from .heads import find_head_preset, get_head_preset, validate_head_preset
from ._version import __version__


def _atomic_report_write(target: Path, content: str, *, exclusive: bool = False) -> Path:
    """Make a complete report visible together and preserve existing history."""
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".rope-report-", suffix=".tmp", dir=target.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            destination.write(content)
            destination.flush()
            os.fsync(destination.fileno())
        if exclusive:
            os.link(temporary_path, target)
        else:
            os.replace(temporary_path, target)
    finally:
        temporary_path.unlink(missing_ok=True)
    return target


def _filename_component(value: Any, fallback: str) -> str:
    component = re.sub(r"[^A-Za-z0-9_-]+", "-", str(value)).strip("-")
    return component[:48] or fallback


class RoPEMonitor:
    """Observe eval forward passes on an already-loaded native HF model.

    Use an evaluation context around the loop and one batch context around
    each forward/generate. Only sampled Q/K are retained transiently, then
    reduced into per-example scalar statistics before the next layer runs.
    """

    def __init__(self, model: Any, config: MonitorConfig | None = None, logger: Any = None,
                 *, report_dir: str | Path | None = None) -> None:
        self.model = model
        self.config = config or MonitorConfig()
        self.adapter = ModelAdapter(model)
        self.cache = ScoreCache(max_entries=self.config.cache_entries)
        self.logger = logger
        self.report_dir = Path(report_dir) if report_dir is not None else None
        self.last_metrics: dict[str, float | int] = {}
        self.last_report: dict[str, Any] = {}
        self.last_report_path: Path | None = None
        self.last_plans: tuple[SamplePlan, ...] = ()
        self._session: _Evaluation | None = None
        self._batch: _Batch | None = None
        layers, heads = self.config.layers, self.config.heads
        self.head_preset = None
        if self.config.head_preset == "auto":
            if layers is None and heads is None:
                self.head_preset = find_head_preset(getattr(self.adapter.config, "_name_or_path", None))
        elif self.config.head_preset is not None:
            self.head_preset = get_head_preset(self.config.head_preset)
        if self.head_preset is not None:
            validate_head_preset(self.head_preset, self.adapter)
            heads = self.head_preset.heads
            layers = tuple(heads)
        elif layers is None and heads is not None:
            layers = tuple(heads)
        if layers is None:
            selected = list(range(len(self.adapter.layers)))
        else:
            selected = sorted(set(int(i) if int(i) >= 0 else len(self.adapter.layers) + int(i) for i in layers))
        if not selected or any(i < 0 or i >= len(self.adapter.layers) for i in selected):
            raise ValueError("layers must identify at least one existing layer")
        self.layers = tuple(self.adapter.layers[i] for i in selected)
        if heads is not None:
            unknown = set(heads) - set(selected)
            if unknown:
                raise ValueError(f"head selection names unselected layers: {sorted(unknown)}")
            for spec in self.layers:
                selection = list(heads.get(spec.index, range(spec.query_heads)))
                if not selection or len(set(selection)) != len(selection) or any(h < 0 or h >= spec.query_heads for h in selection):
                    raise ValueError(f"invalid head selection for layer {spec.index}")
        self._query_heads = {s.index: tuple(heads.get(s.index, range(s.query_heads)))
                             if heads is not None else tuple(range(s.query_heads))
                             for s in self.layers}
        self._kv_heads = {s.index: tuple(sorted({h // (s.query_heads // s.kv_heads)
                                               for h in self._query_heads[s.index]}))
                          for s in self.layers}
        self._query_to_kv = {s.index: tuple(self._kv_heads[s.index].index(h // (s.query_heads // s.kv_heads))
                                           for h in self._query_heads[s.index])
                             for s in self.layers}

    @property
    def active(self) -> bool:
        return self._session is not None

    def metadata(self) -> dict[str, Any]:
        return {**self.adapter.metadata(), "package_version": __version__,
                "sampling": "causal-uniform-shared-prefix-v1",
                "sampling_policy": self.config.sampling_policy,
                "queries": self.config.queries, "keys": self.config.keys,
                "seed": self.config.seed, "distance_samples": self.config.distance_samples,
                "max_examples_per_rank": self.config.max_examples,
                "score_dtype": self.config.score_dtype,
                "cache_entries": self.config.cache_entries,
                "selected_layers": [s.index for s in self.layers],
                "selected_heads": {s.index: list(self._query_heads[s.index]) for s in self.layers},
                "head_selection": {
                    "mode": "paper_preset" if self.head_preset else
                            ("custom" if self.config.heads is not None or self.config.layers is not None else "all"),
                    "selected_count": sum(len(value) for value in self._query_heads.values()),
                    "preset": self.head_preset.to_dict() if self.head_preset else None}}

    def save_report(self, path: str | Path) -> Path:
        """Save the last successful evaluation's configuration and sampling trace.

        Reports contain frequency grids and positions, with input/mask hashes
        instead of input token IDs or text. Each distributed rank has its own
        local sampling trace; scalar score metrics can be globally reduced.
        """
        if not self.last_report:
            raise RuntimeError("complete an evaluation before saving its report")
        target = Path(path)
        content = json.dumps(self.last_report, indent=2, allow_nan=False) + "\n"
        return _atomic_report_write(target, content)

    def _save_report_history(self) -> Path | None:
        if self.report_dir is None:
            return None
        evaluation = self.last_report["evaluation"]
        prefix = _filename_component(evaluation["prefix"], "eval")
        step = _filename_component(evaluation["step"], "step")
        rank = evaluation["rank"]
        try:
            content = json.dumps(self.last_report, indent=2, allow_nan=False) + "\n"
            for _ in range(8):
                target = self.report_dir / f"{prefix}-step-{step}-rank-{rank}-{uuid4().hex}.json"
                try:
                    return _atomic_report_write(target, content, exclusive=True)
                except FileExistsError:
                    continue
            raise FileExistsError("could not allocate a unique report filename")
        except (OSError, TypeError, ValueError) as exc:
            raise RuntimeError(f"failed to save RoPE evaluation report in {self.report_dir}: {exc}") from exc

    def frequency_bands(self, window: int, threshold: float) -> dict[int, torch.Tensor]:
        return self.adapter.frequency_bands(window, threshold)

    def evaluation(self, *, step: int = 0, epoch: float | None = None, prefix: str = "eval") -> _Evaluation:
        return _Evaluation(self, step, epoch, prefix)

    def batch(self, input_ids: torch.Tensor, *, attention_mask: torch.Tensor | None = None,
              position_ids: torch.Tensor | None = None, sample_ids: Sequence[str | int] | None = None,
              query_mask: torch.Tensor | None = None, key_mask: torch.Tensor | None = None) -> _Batch:
        if self._session is None:
            raise RuntimeError("batch must run inside monitor.evaluation()")
        return _Batch(self, input_ids, attention_mask, position_ids, sample_ids, query_mask, key_mask)


class _Evaluation:
    def __init__(self, monitor: RoPEMonitor, step: int, epoch: float | None, prefix: str) -> None:
        self.monitor, self.step, self.epoch, self.prefix = monitor, step, epoch, prefix
        self._entered = False
        self.examples: list[torch.Tensor] = []
        self.seen = 0
        self.skipped = 0
        self.retained_bytes = 0
        self.peak_bytes = 0
        self.score_cpu_seconds = 0.0
        self.score_gpu_events: list[tuple[Any, Any]] = []
        self.plans: list[SamplePlan] = []
        self.sampling_records: list[dict[str, Any]] = []
        self.rope_snapshots: dict[tuple[Any, ...], tuple[torch.Tensor, float]] = {}
        self._snapshot_ids: dict[tuple[Any, ...], int] = {}
        self._frequency_anchors: dict[tuple[Any, ...], torch.Tensor] = {}
        self.rope_uses: list[dict[str, Any]] = []

    def __enter__(self) -> _Evaluation:
        if self._entered:
            raise RuntimeError("evaluation context is single-use; create a new monitor.evaluation()")
        if self.monitor._session is not None:
            raise RuntimeError("evaluation contexts cannot be nested")
        self._entered = True
        self.monitor._session = self
        self.monitor.last_metrics = {}
        self.monitor.last_report = {}
        self.monitor.last_report_path = None
        self.monitor.last_plans = ()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            if self.monitor._batch is not None:
                self.monitor._batch._cleanup()
            if exc_type is None:
                self.monitor.last_metrics = self._finalize()
                self.monitor.last_plans = tuple(self.plans)
                self.monitor.last_report = self._report()
                if self.monitor.logger is not None:
                    self.monitor.logger.log(self.monitor.last_metrics, step=self.step,
                                            epoch=self.epoch, prefix=self.prefix)
                self.monitor.last_report_path = self.monitor._save_report_history()
        except BaseException:
            self.monitor.last_metrics = {}
            self.monitor.last_report = {}
            self.monitor.last_report_path = None
            self.monitor.last_plans = ()
            raise
        finally:
            self.examples.clear()
            self.score_gpu_events.clear()
            self.rope_snapshots.clear()
            self._frequency_anchors.clear()
            self.monitor._session = None

    def _report(self) -> dict[str, Any]:
        runtime = {"python": platform.python_version(), "torch": torch.__version__}
        for name in ("transformers", "accelerate", "wandb"):
            try:
                runtime[name] = version(name)
            except PackageNotFoundError:
                runtime[name] = None
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        grids = [{"id": i, "inv_freq": omega.detach().cpu().tolist(), "attention_scaling": scale}
                 for i, (omega, scale) in enumerate(self.rope_snapshots.values())]
        return {"schema_version": 1, "metadata": self.monitor.metadata(),
                "config": self.monitor.config.to_dict(), "runtime": runtime,
                "evaluation": {"step": self.step, "epoch": self.epoch, "prefix": self.prefix,
                               "rank": torch.distributed.get_rank() if distributed else 0,
                               "world_size": torch.distributed.get_world_size() if distributed else 1},
                "metrics": dict(self.monitor.last_metrics), "sampling_trace": self.sampling_records,
                "frequency_grids": grids, "frequency_uses": self.rope_uses,
                "cache": self.monitor.cache.metadata()}

    def _finalize(self) -> dict[str, float | int]:
        # Statistics: semantic sum/count, positional sum/count, examples,
        # valid triplets, ties, valid pairs, invalid cells, skipped examples.
        device = self.examples[0].device if self.examples else next(self.monitor.model.parameters()).device
        totals = torch.zeros(10, dtype=torch.float64, device=device)
        for example in self.examples:
            ssum, scount, psum, pcount = example[:4]
            svalid, pvalid = scount > 0, pcount > 0
            totals[0] += torch.where(svalid, ssum / scount.clamp_min(1), 0)
            totals[1] += svalid
            totals[2] += torch.where(pvalid, psum / pcount.clamp_min(1), 0)
            totals[3] += pvalid
            totals[4] += 1
            totals[5:9] += example[4:8]
        totals[9] = self.skipped
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(totals)
        values = totals.detach().cpu().tolist()
        score_seconds = self.score_cpu_seconds
        for start, end in self.score_gpu_events:
            score_seconds += start.elapsed_time(end) / 1000.0
        metrics: dict[str, float | int] = {
            "rope_examples": int(values[4]), "rope_semantic_valid_examples": int(values[1]),
            "rope_positional_valid_examples": int(values[3]),
            "rope_semantic_valid_triplets": int(values[5]), "rope_semantic_ties": int(values[6]),
            "rope_positional_valid_pairs": int(values[7]), "rope_invalid_cells": int(values[8]),
            "rope_skipped_examples": int(values[9]),
            "rope_retained_activation_bytes": self.retained_bytes,
            "rope_peak_activation_bytes": self.peak_bytes,
            "rope_score_seconds": score_seconds,
        }
        if values[1]:
            metrics["rope_semantic_score"] = values[0] / values[1]
            metrics["rope_reversal_probability"] = 1.0 - metrics["rope_semantic_score"]
        if values[3]:
            metrics["rope_positional_score"] = values[2] / values[3]
        return metrics


class _Batch:
    def __init__(self, monitor: RoPEMonitor, input_ids: torch.Tensor, attention_mask: torch.Tensor | None,
                 position_ids: torch.Tensor | None, sample_ids: Sequence[str | int] | None,
                 query_mask: torch.Tensor | None, key_mask: torch.Tensor | None) -> None:
        self.monitor = monitor
        self._evaluation = monitor._session
        self._entered = False
        self.input_ids, self.attention_mask, self.position_ids = input_ids, attention_mask, position_ids
        self.sample_ids, self.query_mask, self.key_mask = sample_ids, query_mask, key_mask
        self._handles: list[Any] = []
        self._slots: dict[int, dict[str, torch.Tensor]] = {}
        self._done: set[int] = set()
        self.plans: list[SamplePlan] = []
        self.rows: list[int] = []
        self._statistics: list[torch.Tensor] = []
        self._indices: dict[tuple[str, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}
        self._head_indices: dict[tuple[int, str, torch.device], torch.Tensor] = {}
        self._sampling_records: list[dict[str, Any]] = []
        self._bytes_live = 0

    def __enter__(self) -> _Batch:
        if self._entered:
            raise RuntimeError("batch context is single-use; create a new monitor.batch()")
        if self.monitor._batch is not None:
            raise RuntimeError("batch contexts cannot be nested")
        if self.monitor._session is None or self.monitor._session is not self._evaluation:
            raise RuntimeError("batch context must run inside the evaluation that created it")
        self._entered = True
        if self.input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch,sequence]")
        if self.input_ids.dtype not in {torch.int32, torch.int64}:
            raise ValueError("input_ids must contain integer token IDs")
        if self.monitor.config.sampling_policy == "regions" and (self.query_mask is None or self.key_mask is None):
            raise ValueError("regions sampling requires both query_mask and key_mask")
        shape = tuple(self.input_ids.shape)
        for value in (self.attention_mask, self.position_ids, self.query_mask, self.key_mask):
            if value is not None and tuple(value.shape) != shape:
                raise ValueError("sampling masks/position_ids must have the input_ids shape")
        if self.sample_ids is not None and len(self.sample_ids) != shape[0]:
            raise ValueError("sample_ids must contain one ID per batch row")
        session = self.monitor._session
        assert session is not None
        budget = self.monitor.config.max_examples
        if budget is not None and session.seen >= budget:
            self.monitor._batch = self
            return self
        row_budget = shape[0] if budget is None else min(shape[0], budget - session.seen)
        tokens = self.input_ids[:row_budget].detach().cpu().tolist()
        raw_valid = self.attention_mask[:row_budget].detach().cpu().tolist() if self.attention_mask is not None else [[True] * shape[1] for _ in tokens]
        if any(value not in (0, 1) for row in raw_valid for value in row):
            raise ValueError("attention_mask must contain Boolean or binary values")
        valid = [[bool(value) for value in row] for row in raw_valid]
        optional = [v[:row_budget].detach().cpu().tolist() if v is not None else None
                    for v in (self.position_ids, self.query_mask, self.key_mask)]
        for mask in optional[1:]:
            if mask is not None and any(value not in (0, 1) for row in mask for value in row):
                raise ValueError("query_mask and key_mask must contain Boolean or binary values")
        for row, token in enumerate(tokens):
            if budget is not None and session.seen >= budget:
                break
            session.seen += 1
            plan = sample_tokens(token, valid[row], queries=self.monitor.config.queries,
                                 keys=self.monitor.config.keys, seed=self.monitor.config.seed,
                                 sample_id=str(self.sample_ids[row]) if self.sample_ids is not None else None,
                                 position_ids=optional[0][row] if optional[0] is not None else None,
                                 query_mask=optional[1][row] if optional[1] is not None else None,
                                 key_mask=optional[2][row] if optional[2] is not None else None)
            if plan is None:
                session.skipped += 1
                continue
            self.rows.append(row)
            self.plans.append(plan)
            effective = [i for i, active in enumerate(valid[row]) if active]
            mask_record = {"query": [bool(optional[1][row][i]) for i in effective] if optional[1] is not None else None,
                           "key": [bool(optional[2][row][i]) for i in effective] if optional[2] is not None else None,
                           "positions": [optional[0][row][i] for i in effective] if optional[0] is not None else None}
            self._sampling_records.append({**asdict(plan),
                "input_sha256": input_fingerprint([token[i] for i in effective]),
                "sampling_metadata_sha256": hashlib.sha256(json.dumps(mask_record, sort_keys=True).encode()).hexdigest(),
                "positional_distance_samples": min(self.monitor.config.distance_samples or (plan.window - 1), plan.window - 1),
                "positional_distance_sampling": "exhaustive" if self.monitor.config.distance_samples is None
                    or self.monitor.config.distance_samples >= plan.window - 1 else "uniform-with-replacement",
                "query_mask_supplied": self.query_mask is not None, "key_mask_supplied": self.key_mask is not None})
        self.monitor._batch = self
        try:
            if self.plans:
                for spec in self.monitor.layers:
                    self._slots[spec.index] = {}
                    self._handles.append(spec.query_module.register_forward_hook(self._hook(spec, "q")))
                    self._handles.append(spec.key_module.register_forward_hook(self._hook(spec, "k")))
        except BaseException:
            self._cleanup()
            raise
        return self

    def _hook(self, spec: LayerSpec, kind: str):
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            if spec.index in self._done:
                return  # Subsequent decode steps do not replace prompt states.
            if self.monitor.model.training:
                raise RuntimeError("monitor captures only eval-mode forwards; call model.eval()")
            value = projection_view(output, spec, kind)
            if tuple(value.shape[:2]) != tuple(self.input_ids.shape):
                raise ValueError("beam-expanded/chunked/cached-prefix inputs need a dedicated capture adapter")
            index_key = (kind, value.device)
            if index_key not in self._indices:
                positions = [p.query_positions if kind == "q" else p.key_positions for p in self.plans]
                width = max(map(len, positions))
                indices = [list(p) + [p[0]] * (width - len(p)) for p in positions]
                self._indices[index_key] = (torch.tensor(self.rows, device=value.device)[:, None],
                                           torch.tensor(indices, device=value.device))
            rows, positions = self._indices[index_key]
            head_key = (spec.index, kind, value.device)
            if head_key not in self._head_indices:
                heads = self.monitor._query_heads[spec.index] if kind == "q" else self.monitor._kv_heads[spec.index]
                self._head_indices[head_key] = torch.tensor(heads, device=value.device)[None, None, :]
            selected = value.detach()[rows[..., None], positions[..., None], self._head_indices[head_key]].permute(0, 2, 1, 3).contiguous()
            slot = self._slots[spec.index]
            if kind in slot:
                raise RuntimeError("Q/K module invoked twice before a complete capture")
            slot[kind] = selected
            nbytes = selected.numel() * selected.element_size()
            self._bytes_live += nbytes
            session = self.monitor._session
            assert session is not None
            session.retained_bytes += nbytes
            session.peak_bytes = max(session.peak_bytes, self._bytes_live)
            if "q" in slot and "k" in slot:
                self._consume(spec, slot)
                self._bytes_live -= sum(v.numel() * v.element_size() for v in slot.values())
                slot.clear()
                self._done.add(spec.index)
        return capture

    def _consume(self, spec: LayerSpec, slot: dict[str, torch.Tensor]) -> None:
        session = self.monitor._session
        assert session is not None
        omega, rope_scale = spec.active_rope()
        try:
            frequency_version: Any = omega._version
        except RuntimeError:
            frequency_version = object()  # Preserve each untracked grid independently.
        grid_key = (id(omega), frequency_version, str(omega.device), rope_scale)
        if grid_key not in session.rope_snapshots:
            session._snapshot_ids[grid_key] = len(session.rope_snapshots)
            session.rope_snapshots[grid_key] = (omega.detach().clone(), rope_scale)
            session._frequency_anchors[grid_key] = omega
        session.rope_uses.append({"layer": spec.index, "grid_id": session._snapshot_ids[grid_key],
                                 "sample_indices": list(range(len(session.plans), len(session.plans) + len(self.plans)))})
        q = slot["q"]
        mapping = self.monitor._query_to_kv[spec.index]
        start_cpu = time.perf_counter()
        events = None
        if q.device.type == "cuda":
            with torch.cuda.device(q.device):
                events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                events[0].record()
        for index, plan in enumerate(self.plans):
            result = score_activations(q[index, :, :len(plan.query_positions)],
                                      slot["k"][index, :, :len(plan.key_positions)], mapping,
                                      plan.pair_indices, plan.triplet_indices, omega, plan.window,
                                      layout=spec.layout, rotary_dim=spec.rotary_dim,
                                      attention_scale=spec.attention_scale, rope_scale=rope_scale,
                                      distance_samples=self.monitor.config.distance_samples,
                                      seed=self.monitor.config.seed, dtype=self.monitor.config.score_dtype,
                                      cache=self.monitor.cache)
            semantic, positional = result.semantic_by_head, result.positional_by_head
            svalid, pvalid = torch.isfinite(semantic), torch.isfinite(positional)
            stats = torch.stack((torch.where(svalid, semantic, 0).sum(), svalid.sum(),
                                 torch.where(pvalid, positional, 0).sum(), pvalid.sum(),
                                 result.semantic_valid_triplets, result.semantic_ties,
                                 result.positional_valid_pairs, result.invalid_cells)).to(dtype=torch.float64)
            if len(self._statistics) <= index:
                self._statistics.append(stats)
            else:
                self._statistics[index] += stats.to(self._statistics[index].device)
        if events is not None:
            events[1].record()
            session.score_gpu_events.append(events)
        else:
            session.score_cpu_seconds += time.perf_counter() - start_cpu

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            if exc_type is None and self.plans:
                missing = {s.index for s in self.monitor.layers} - self._done
                if missing:
                    raise RuntimeError(f"eval forward did not capture layers {sorted(missing)}")
                session = self.monitor._session
                assert session is not None
                session.examples.extend(self._statistics)
                session.plans.extend(self.plans)
                session.sampling_records.extend(self._sampling_records)
        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._slots.clear()
        self._indices.clear()
        self._head_indices.clear()
        self._statistics = []
        self.monitor._batch = None
