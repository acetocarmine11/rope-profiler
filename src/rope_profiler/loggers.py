"""Optional scalar logging; no logger starts an online service or a new run."""

from __future__ import annotations

import importlib
import json
import math
from numbers import Real
from pathlib import Path
from threading import Lock
from typing import Any, Mapping, Protocol


class MetricLogger(Protocol):
    def log(
        self,
        metrics: Mapping[str, Any],
        *,
        step: int,
        epoch: float | None = None,
        prefix: str = "eval",
    ) -> None: ...


def _scalars(metrics: Mapping[str, Any]) -> dict[str, int | float | bool | None]:
    result: dict[str, int | float | bool | None] = {}
    for name, raw in metrics.items():
        if not isinstance(name, str):
            raise TypeError("Metric names must be strings.")
        value = raw
        if value is not None and not isinstance(value, Real) and hasattr(value, "item"):
            value = value.item()
        if value is None:
            result[name] = None
        elif isinstance(value, bool):
            result[name] = value
        elif isinstance(value, int):
            result[name] = value
        elif isinstance(value, Real):
            number = float(value)
            result[name] = number if math.isfinite(number) else None
        else:
            raise TypeError(f"Metric {name!r} must be a numeric scalar, got {type(value).__name__}.")
    return result


class JSONLLogger:
    """Append one local JSON record per evaluation, with the original metric names."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = Lock()

    def log(
        self,
        metrics: Mapping[str, Any],
        *,
        step: int,
        epoch: float | None = None,
        prefix: str = "eval",
    ) -> None:
        record = {
            "step": int(step),
            "epoch": None if epoch is None else float(epoch),
            "prefix": prefix,
            "metrics": _scalars(metrics),
        }
        line = json.dumps(record, allow_nan=False, sort_keys=True)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as destination:
                destination.write(line + "\n")


class WandbLogger:
    """Log to an existing W&B run, including an existing offline run.

    ``run=None`` resolves ``wandb.run`` when logging.  This class never calls
    ``wandb.init`` or ``wandb.login`` and does not change the run's settings.
    A supplied object only needs a compatible ``log`` method, allowing tests
    and callers to avoid importing W&B entirely.
    """

    def __init__(self, run: Any | None = None) -> None:
        self.run = run
        self._metric_run: Any | None = None
        self._defined_prefixes: set[str] = set()

    def log(
        self,
        metrics: Mapping[str, Any],
        *,
        step: int,
        epoch: float | None = None,
        prefix: str = "eval",
    ) -> None:
        run = self.run
        if run is None:
            try:
                wandb = importlib.import_module("wandb")
            except ImportError as exc:
                raise RuntimeError("Install the wandb extra, or pass an existing run object.") from exc
            run = wandb.run
        if run is None:
            raise RuntimeError("WandbLogger requires an existing active W&B run.")
        prefix = prefix.strip("/")
        if run is not self._metric_run:
            self._metric_run = run
            self._defined_prefixes.clear()
        define_metric = getattr(run, "define_metric", None)
        if prefix not in self._defined_prefixes and callable(define_metric):
            pattern = f"{prefix}/*" if prefix else "*"
            define_metric(pattern, step_metric="train/global_step")
            self._defined_prefixes.add(prefix)
        payload = {
            (f"{prefix}/{name}" if prefix else name): value
            for name, value in _scalars(metrics).items()
        }
        # The optimizer step is a chart axis, rather than W&B's history index.
        # An existing run may already have logged several rows per train step.
        payload["train/global_step"] = int(step)
        if epoch is not None:
            payload["epoch"] = float(epoch)
        run.log(payload)


__all__ = ["MetricLogger", "JSONLLogger", "WandbLogger"]
