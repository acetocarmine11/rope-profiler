"""Attach monitoring to an existing Transformers Trainer without another pass."""

from __future__ import annotations

import inspect
from functools import wraps
from pathlib import Path
from types import MappingProxyType, MethodType
from typing import Any, Callable, Mapping

from ..monitor import MonitorConfig, RoPEMonitor


_MISSING = object()
_ATTACHMENT_ATTRIBUTE = "_rope_profiler_attachment"
_BATCH_METADATA_FIELDS = frozenset({"query_mask", "key_mask", "sample_ids"})
BatchMetadataCallback = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def _argument(
    original: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    name: str,
    default: Any = None,
) -> Any:
    if name in kwargs:
        return kwargs[name]
    try:
        signature = inspect.signature(original)
        bound = signature.bind_partial(*args, **kwargs)
        if name in bound.arguments:
            return bound.arguments[name]
        parameter = signature.parameters.get(name)
        if parameter is not None and parameter.default is not inspect.Parameter.empty:
            return parameter.default
    except (TypeError, ValueError):
        pass
    return default


class TrainerMonitorAttachment:
    """A reversible adapter that preserves the Trainer's existing implementations.

    The original ``prediction_step`` still performs all model computation.
    Evaluation-loop output metrics are extended before the Trainer's original
    logging path runs, so its W&B integration receives one combined record.
    """

    def __init__(
        self,
        trainer: Any,
        *,
        config: MonitorConfig | None = None,
        batch_metadata: BatchMetadataCallback | None = None,
        report_dir: str | Path | None = None,
    ) -> None:
        if batch_metadata is not None and not callable(batch_metadata):
            raise TypeError("batch_metadata must be callable or None.")
        if getattr(trainer, _ATTACHMENT_ATTRIBUTE, None) is not None:
            raise ValueError("This Trainer already has a RoPE monitor attachment.")
        if not callable(getattr(trainer, "prediction_step", None)):
            raise TypeError("The Trainer must expose prediction_step().")
        loop_names = tuple(
            name
            for name in ("evaluation_loop", "prediction_loop")
            if callable(getattr(trainer, name, None))
        )
        if not loop_names:
            raise TypeError("The Trainer must expose evaluation_loop() or prediction_loop().")
        self.trainer = trainer
        self.batch_metadata = batch_metadata
        self.config = config or MonitorConfig()
        self.report_dir = Path(report_dir) if report_dir is not None else None
        self.monitor = self._new_monitor()
        self._original_instance_values: dict[str, Any] = {}
        self._wrappers: dict[str, Any] = {}
        self._loop_depth = 0
        self._active = False
        self._detached = False
        self._install("prediction_step", self._prediction_wrapper)
        for name in loop_names:
            self._install(name, self._loop_wrapper)
        setattr(trainer, _ATTACHMENT_ATTRIBUTE, self)

    def _new_monitor(self) -> RoPEMonitor:
        options = {"report_dir": self.report_dir} if self.report_dir is not None else {}
        return RoPEMonitor(self.trainer.model, config=self.config, **options)

    def _install(self, name: str, factory: Callable[..., Any]) -> None:
        original = getattr(self.trainer, name)
        self._original_instance_values[name] = vars(self.trainer).get(name, _MISSING)
        wrapper = MethodType(factory(original), self.trainer)
        self._wrappers[name] = wrapper
        setattr(self.trainer, name, wrapper)

    def _prediction_wrapper(self, original: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(getattr(original, "__func__", original))
        def prediction_step(_trainer: Any, *args: Any, **kwargs: Any) -> Any:
            if not self._active:
                return original(*args, **kwargs)
            inputs = _argument(original, args, kwargs, "inputs")
            if inputs is None and len(args) > 1:
                # A common subclass forwards ``*args, **kwargs`` to the base
                # implementation, hiding the named signature from inspection.
                inputs = args[1]
            if not isinstance(inputs, Mapping) or inputs.get("input_ids") is None:
                raise ValueError("RoPE monitoring requires prediction_step inputs with input_ids.")
            metadata: dict[str, Any] = {}
            if self.batch_metadata is not None:
                supplied = self.batch_metadata(MappingProxyType(dict(inputs)))
                if not isinstance(supplied, Mapping):
                    raise TypeError("batch_metadata must return a Mapping.")
                unknown = set(supplied) - _BATCH_METADATA_FIELDS
                if unknown:
                    names = ", ".join(sorted(map(repr, unknown)))
                    raise ValueError(
                        "batch_metadata only accepts query_mask, key_mask, and sample_ids; "
                        f"unexpected fields: {names}. Model attention_mask and position_ids "
                        "are taken unchanged from the original inputs."
                    )
                metadata.update(supplied)
            with self.monitor.batch(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                position_ids=inputs.get("position_ids"),
                **metadata,
            ):
                return original(*args, **kwargs)

        return prediction_step

    def _loop_wrapper(self, original: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(getattr(original, "__func__", original))
        def evaluation_loop(_trainer: Any, *args: Any, **kwargs: Any) -> Any:
            # Some subclasses implement one loop through the other.  The outer
            # call owns the evaluation session and adds its metrics once.
            if self._loop_depth:
                return original(*args, **kwargs)
            if self.trainer.model is not self.monitor.model:
                self.monitor = self._new_monitor()
            prefix = _argument(original, args, kwargs, "metric_key_prefix", "eval")
            state = getattr(self.trainer, "state", None)
            step = int(getattr(state, "global_step", 0))
            epoch = getattr(state, "epoch", None)
            self._loop_depth += 1
            try:
                with self.monitor.evaluation(step=step, epoch=epoch, prefix=prefix):
                    self._active = True
                    try:
                        output = original(*args, **kwargs)
                    finally:
                        self._active = False
                metrics = getattr(output, "metrics", None)
                if not isinstance(metrics, dict):
                    raise TypeError("The Trainer evaluation loop must return an output with a metrics dict.")
                metrics.update(
                    {f"{prefix}_{name}": value for name, value in self.monitor.last_metrics.items()}
                )
                return output
            finally:
                self._loop_depth -= 1

        return evaluation_loop

    def detach(self) -> None:
        """Restore the original methods; repeated detach calls are harmless."""
        if self._detached:
            return
        if self._loop_depth:
            raise RuntimeError("Detach the RoPE monitor after the current evaluation finishes.")
        for name, original_value in self._original_instance_values.items():
            # Leave a method deliberately replaced by the user after attach.
            if vars(self.trainer).get(name, _MISSING) is not self._wrappers[name]:
                continue
            if original_value is _MISSING:
                delattr(self.trainer, name)
            else:
                setattr(self.trainer, name, original_value)
        if getattr(self.trainer, _ATTACHMENT_ATTRIBUTE, None) is self:
            delattr(self.trainer, _ATTACHMENT_ATTRIBUTE)
        self._detached = True

    def __enter__(self) -> "TrainerMonitorAttachment":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.detach()


def attach_rope_monitor(
    trainer: Any,
    *,
    config: MonitorConfig | None = None,
    batch_metadata: BatchMetadataCallback | None = None,
    report_dir: str | Path | None = None,
) -> TrainerMonitorAttachment:
    """Monitor eval/predict passes of a Trainer, including existing subclasses.

    The subclass must continue to use its ``prediction_step`` and one of the
    standard evaluation loops.  No model is loaded, no forward is added, and
    no logger or W&B run is created.  Call ``attachment.detach()`` to restore
    the Trainer's original methods.

    ``batch_metadata(inputs)`` optionally returns ``query_mask``, ``key_mask``,
    and/or ``sample_ids`` for the monitor only.  It receives a read-only mapping;
    tensor values must also be treated as read-only.  It cannot replace the
    model's ``attention_mask`` or ``position_ids``, preserving the positions and
    validity actually used by the forward.  Region sampling requires both
    query and key masks.  No metadata is added to the model's inputs.
    """
    return TrainerMonitorAttachment(trainer, config=config, batch_metadata=batch_metadata,
                                    report_dir=report_dir)


__all__ = ["TrainerMonitorAttachment", "attach_rope_monitor"]
