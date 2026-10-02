from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from rope_profiler.loggers import JSONLLogger, WandbLogger


def test_jsonl_logger_appends_raw_metrics_without_creating_file_early(tmp_path):
    path = tmp_path / "nested" / "metrics.jsonl"
    logger = JSONLLogger(path)
    assert not path.exists()
    logger.log({"rope_semantic_score": 0.75, "rope_examples": 4}, step=8, epoch=1.0)
    logger.log({"rope_semantic_score": float("nan")}, step=16, prefix="eval_test")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[0] == {
        "step": 8,
        "epoch": 1.0,
        "prefix": "eval",
        "metrics": {"rope_semantic_score": 0.75, "rope_examples": 4},
    }
    assert records[1]["metrics"]["rope_semantic_score"] is None
    assert records[1]["prefix"] == "eval_test"


def test_wandb_logger_reuses_supplied_run_without_import(monkeypatch):
    records = []
    run = SimpleNamespace(log=lambda payload, **kwargs: records.append((payload, kwargs)))
    monkeypatch.setattr(
        "rope_profiler.loggers.importlib.import_module",
        lambda _: pytest.fail("A supplied run must avoid importing W&B."),
    )
    WandbLogger(run).log({"rope_semantic_score": 0.8}, step=7, epoch=2.5)
    assert records == [({"eval/rope_semantic_score": 0.8, "train/global_step": 7, "epoch": 2.5}, {})]


def test_wandb_logger_uses_existing_active_run_without_init_or_login(monkeypatch):
    records = []
    fake_wandb = SimpleNamespace(
        run=SimpleNamespace(log=lambda payload, **kwargs: records.append((payload, kwargs))),
        init=lambda *args, **kwargs: pytest.fail("Logger must not initialize a run."),
        login=lambda *args, **kwargs: pytest.fail("Logger must not log in."),
    )
    monkeypatch.setattr("rope_profiler.loggers.importlib.import_module", lambda _: fake_wandb)
    logger = WandbLogger()
    logger.log({"rope_examples": 3}, step=4, prefix="validation/")
    assert records == [({"validation/rope_examples": 3, "train/global_step": 4}, {})]
    fake_wandb.run = None
    with pytest.raises(RuntimeError, match="existing active"):
        logger.log({"rope_examples": 3}, step=5)


def test_loggers_reject_non_scalar_values(tmp_path):
    with pytest.raises(TypeError, match="numeric scalar"):
        JSONLLogger(tmp_path / "metrics.jsonl").log({"bad": [1, 2]}, step=0)


def test_wandb_logger_defines_each_prefix_axis_once_and_tracks_current_run(monkeypatch):
    axes, records = [], []
    run = SimpleNamespace(
        define_metric=lambda pattern, **kwargs: axes.append((pattern, kwargs)),
        log=lambda payload, **kwargs: records.append((payload, kwargs)),
    )
    fake_wandb = SimpleNamespace(run=run)
    monkeypatch.setattr("rope_profiler.loggers.importlib.import_module", lambda _: fake_wandb)
    logger = WandbLogger()
    logger.log({"rope_semantic_score": 0.5}, step=0)
    logger.log({"rope_semantic_score": 0.6}, step=0)
    logger.log({"rope_semantic_score": 0.7}, step=1, prefix="validation")
    assert axes == [
        ("eval/*", {"step_metric": "train/global_step"}),
        ("validation/*", {"step_metric": "train/global_step"}),
    ]
    assert all(kwargs == {} for _, kwargs in records)
    assert [payload["train/global_step"] for payload, _ in records] == [0, 0, 1]
    next_axes = []
    fake_wandb.run = SimpleNamespace(
        define_metric=lambda pattern, **kwargs: next_axes.append((pattern, kwargs)),
        log=lambda *args, **kwargs: None,
    )
    logger.log({"rope_semantic_score": 0.5}, step=0)
    assert next_axes == [("eval/*", {"step_metric": "train/global_step"})]


def test_wandb_offline_run_logging(tmp_path, monkeypatch):
    wandb = pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_MODE", "offline")
    with wandb.init(mode="offline", project="rope-profiler-test", dir=str(tmp_path)) as run:
        for index in range(5):
            run.log({"train/loss": 1.0 / (index + 1)})
        axes = []
        original_define_metric = run.define_metric

        def record_axis(pattern, **kwargs):
            axes.append((pattern, kwargs))
            return original_define_metric(pattern, **kwargs)

        monkeypatch.setattr(run, "define_metric", record_axis)
        WandbLogger(run).log({"rope_semantic_score": 0.6, "rope_examples": 2}, step=0)
        # Only the test creates the explicit offline run; the logger reuses it.
        assert run.settings.mode == "offline"
        assert axes == [("eval/*", {"step_metric": "train/global_step"})]
        assert run.summary["eval/rope_semantic_score"] == 0.6
        assert run.summary["eval/rope_examples"] == 2
        assert run.summary["train/global_step"] == 0
