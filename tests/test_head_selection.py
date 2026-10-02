from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from research_tools.head_selection import (high_frequency_norm_share, load_head_records, plot_heatmaps,
                                         select_heads, synthetic_records, to_head_mapping, write_selection)


ROOT = Path(__file__).resolve().parents[1]


def records():
    return [{"task": f"setting-{task}", "layer": 0, "head": head, "r_H": value,
             "example_count": [100, 1, 1][task]}
            for head, values in [(0, [.9, .9, .9]), (1, [0., .5, 1.])]
            for task, value in enumerate(values)]


def test_mean_and_paper_sample_variance_select_different_heads():
    result = select_heads(records(), top_k=1)
    model = result["models"]["model"]
    constant, changing = model["statistics"]
    assert constant["mean"] == pytest.approx(.9)
    assert constant["variance"] == pytest.approx(0.)
    assert changing["mean"] == pytest.approx(.5)
    assert changing["variance"] == pytest.approx(.25)
    assert not {"population_variance", "sample_variance"}.intersection(changing)
    assert model["top_mean"][0]["head"] == 0
    assert model["top_variance"][0]["head"] == 1
    assert result["ddof"] == 1 and result["task_weighting"] == "equal"
    # Example counts above deliberately differ; every setting still has weight 1/3.
    assert result["metric"] == "r_H"


def test_population_variance_is_explicit_and_rh_is_not_squared():
    result = select_heads(records(), top_k=1, ddof=0)
    model = result["models"]["model"]
    assert result["variance_type"] == "population"
    assert model["statistics"][1]["variance"] == pytest.approx(1 / 6)
    assert model["statistics"][1]["mean"] == pytest.approx(.5)
    assert model["statistics"][1]["mean"] != pytest.approx((0 + .25 + 1) / 3)


def test_lowercase_rh_alias_and_ambiguous_columns():
    lower = [{"r_h": row["r_H"], **{key: value for key, value in row.items() if key != "r_H"}}
             for row in records()]
    assert select_heads(lower) == select_heads(records())
    with pytest.raises(ValueError, match="aliases"):
        select_heads([{**records()[0], "r_h": .2}])


def test_norm_share_uses_split_half_coefficients_and_not_energy_fraction():
    import torch
    q, k = torch.tensor([3., 4., 0., 0.]), torch.tensor([1., 1., 0., 0.])
    result = high_frequency_norm_share(q, k, [True, False])
    assert result.item() == pytest.approx(.6)
    assert result.square().item() == pytest.approx(.36)
    assert high_frequency_norm_share(q, k, [True, True]).item() == pytest.approx(1.)
    assert high_frequency_norm_share(q, k, [False, False]).item() == 0.


def test_norm_share_native_gqa_mapping_and_undefined_coefficients():
    import torch
    q = torch.tensor([[3., 4., 0., 0.]]).repeat(2, 4, 1)
    k = torch.tensor([[1., 1., 0., 0.], [1., 2., 0., 0.]]).repeat(2, 1, 1)
    q[0, 0] = 0
    result = high_frequency_norm_share(q, k, [True, False], [0, 0, 1, 1])
    assert result.shape == (2, 4) and torch.isnan(result[0, 0])
    assert result[0, 1].item() == pytest.approx(.6)
    assert result[0, 2].item() == pytest.approx(3 / (73 ** .5))
    assert result[1, 0].item() == pytest.approx(.6)


def test_norm_share_is_stable_for_extreme_finite_scales():
    import torch
    q = torch.tensor([3e200, 4e200, 0., 0.], dtype=torch.float64)
    k = torch.tensor([1e200, 1e200, 0., 0.], dtype=torch.float64)
    assert high_frequency_norm_share(q, k, [True, False]).item() == pytest.approx(.6)
    q[0] = float("inf")
    assert torch.isnan(high_frequency_norm_share(q, k, [True, False]))


@pytest.mark.parametrize("mask", [[1, 0], [True], [True, False, True]])
def test_norm_share_requires_full_rotary_boolean_mask(mask):
    import torch
    q = torch.ones(4)
    with pytest.raises(ValueError, match="high_mask"):
        high_frequency_norm_share(q, q, mask)


@pytest.mark.parametrize("mapping", [[0], [True, 0], [0, 2], [0., 1.]])
def test_norm_share_rejects_invalid_gqa_mapping(mapping):
    import torch
    with pytest.raises(ValueError, match="query_to_kv"):
        high_frequency_norm_share(torch.ones(2, 4), torch.ones(2, 4), [True, False], mapping)


@pytest.mark.parametrize("head_count,expected", [(1024, 52), (1152, 58), (3, 1)])
def test_paper_fraction_ceil_and_layer_major_tie_break(head_count, expected):
    rows = [{"task": task, "layer": flat // 32, "head": flat % 32, "r_H": .5}
            for flat in reversed(range(head_count)) for task in ("A", "B")]
    result = select_heads(rows)
    model = result["models"]["model"]
    assert model["selected_count"] == expected
    for criterion in ("mean", "variance"):
        assert [(row["layer"], row["head"]) for row in model[f"top_{criterion}"]] == [
            (flat // 32, flat % 32) for flat in range(expected)]


def test_models_are_never_combined_and_input_order_does_not_matter():
    first = [{**row, "model": "A"} for row in records()]
    second = [{"model": "B", "task": task, "layer": 5, "head": 7, "r_H": value}
              for task, value in [("X", .2), ("Y", .8)]]
    result = select_heads(first + second)
    assert result == select_heads(reversed(first + second))
    assert result["models"]["A"]["task_count"] == 3
    assert result["models"]["B"]["task_count"] == 2
    assert result["models"]["B"]["statistics"][0]["mean"] == pytest.approx(.5)
    assert result["models"]["B"]["statistics"][0]["variance"] == pytest.approx(.18)


def test_missing_settings_and_duplicates_are_rejected():
    with pytest.raises(ValueError, match="missing reference settings"):
        select_heads(records()[:-1])
    with pytest.raises(ValueError, match="duplicate setting/head"):
        select_heads(records() + [records()[0]])


def test_requires_rh_instead_of_monitor_scores():
    with pytest.raises(ValueError, match="semantic/positional scores cannot replace r_H"):
        select_heads([{"task": "A", "layer": 0, "head": 0,
                       "rope_semantic_score": .7, "rope_positional_score": .1}])


@pytest.mark.parametrize("value", [True, None, float("nan"), float("inf"), -.1, 1.1, "bad"])
def test_invalid_rh_is_rejected(value):
    rows = records()
    rows[0]["r_H"] = value
    with pytest.raises(ValueError, match="r_H must be a finite number"):
        select_heads(rows)


@pytest.mark.parametrize("field,value", [("layer", -1), ("head", True), ("head", 1.0),
                                         ("task", ""), ("model", None)])
def test_invalid_identity_is_rejected(field, value):
    rows = records()
    rows[0][field] = value
    with pytest.raises(ValueError):
        select_heads(rows)


@pytest.mark.parametrize("kwargs", [{"fraction": 0}, {"fraction": 2}, {"fraction": True},
                                    {"fraction": float("nan")}, {"top_k": 0},
                                    {"top_k": True}, {"top_k": 1, "fraction": .5},
                                    {"ddof": 2}, {"ddof": True}])
def test_invalid_selection_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        select_heads(records(), **kwargs)


def test_single_setting_has_only_population_variance():
    rows = [{"task": "A", "layer": 0, "head": 0, "r_H": .3}]
    with pytest.raises(ValueError, match="at least two reference settings"):
        select_heads(rows)
    cell = select_heads(rows, ddof=0)["models"]["model"]["statistics"][0]
    assert cell["variance"] == 0
    assert select_heads(records(), top_k=10)["models"]["model"]["selected_count"] == 2


def test_selected_heads_convert_directly_to_monitor_config():
    from rope_profiler import MonitorConfig
    result = select_heads(records(), top_k=1)
    assert to_head_mapping(result) == {0: [1]}
    assert to_head_mapping(result, criterion="mean") == {0: [0]}
    config = MonitorConfig(heads=to_head_mapping(result))
    assert config.heads == {0: (1,)}
    multi = select_heads([{**row, "model": name} for name in ("A", "B") for row in records()])
    with pytest.raises(ValueError, match="specify model"):
        to_head_mapping(multi)
    assert to_head_mapping(multi, model="B") == {0: [1]}
    with pytest.raises(ValueError, match="unknown model"):
        to_head_mapping(result, model="other")
    with pytest.raises(ValueError, match="criterion"):
        to_head_mapping(result, criterion="score")


def test_csv_and_json_inputs_have_identical_statistics(tmp_path):
    json_path = tmp_path / "input.json"
    json_path.write_text(json.dumps(records()))
    csv_path = tmp_path / "input.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records()[0]))
        writer.writeheader()
        writer.writerows(records())
    assert select_heads(load_head_records(json_path)) == select_heads(load_head_records(csv_path))


def test_loader_rejects_ambiguous_input(tmp_path):
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("task,layer,head,r_H,r_H\nA,0,0,.2,.7\n")
    with pytest.raises(ValueError, match="distinct column names"):
        load_head_records(csv_path)
    json_path = tmp_path / "input.json"
    json_path.write_text('{"rope_semantic_score": 0.8}')
    with pytest.raises(ValueError, match="JSON input must be an array"):
        load_head_records(json_path)


def test_statistics_only_output_and_lazy_dependency_import(tmp_path):
    result = select_heads(records(), top_k=1)
    outputs = write_selection(result, tmp_path, plots=False)
    assert json.loads(Path(outputs["selection_json"]).read_text()) == result
    selected = json.loads(Path(outputs["selected_heads_json"]).read_text())
    assert selected["model"] == {"mean": [{"layer": 0, "head": 0}],
                                  "variance": [{"layer": 0, "head": 1}]}
    assert outputs["heatmaps"] == {}
    with Path(outputs["statistics_csv"]).open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 2
    check = subprocess.run([sys.executable, "-c", "import sys; from research_tools.head_selection import select_heads; assert not {'torch','numpy','matplotlib'} & set(sys.modules)"],
                           cwd=ROOT, capture_output=True, text=True)
    assert check.returncode == 0, check.stderr


def test_synthetic_cli_runs_without_model_or_plot_dependencies(tmp_path):
    run = subprocess.run([sys.executable, str(ROOT / "examples/select_heads.py"),
        "--demo", "--no-plots", "--output-dir", str(tmp_path)],
        cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    summary = json.loads(run.stdout)
    assert summary["synthetic"] and summary["models"]["synthetic-demo"]["selected_per_criterion"] == 2
    saved = json.loads((tmp_path / "head_selection.json").read_text())
    assert saved["provenance"] == {"synthetic": True, "model_inference": False}
    assert saved["models"]["synthetic-demo"]["top_mean"][0]["head"] == 0
    assert saved["models"]["synthetic-demo"]["top_variance"][0]["head"] == 19


def test_cli_rejects_a_score_report(tmp_path):
    source = tmp_path / "wrong.json"
    source.write_text(json.dumps([{"task": "A", "layer": 0, "head": 0, "rope_semantic_score": .7}]))
    output = tmp_path / "output"
    run = subprocess.run([sys.executable, str(ROOT / "examples/select_heads.py"),
        "--input", str(source), "--output-dir", str(output), "--no-plots"],
        capture_output=True, text=True)
    assert run.returncode == 2 and "r_H=U_H/U" in run.stderr
    assert not output.exists()


def test_heatmaps_export_png_pdf_and_model_paths_are_safe(tmp_path):
    pytest.importorskip("matplotlib")
    rows = [{**row, "model": "../demo"} for row in synthetic_records()]
    rows += [{**row, "model": "..\\demo"} for row in synthetic_records()]
    result = select_heads(rows)
    paths = plot_heatmaps(result, tmp_path)
    all_paths = [Path(path) for model in paths.values() for path in model.values()]
    assert len(all_paths) == len(set(all_paths)) == 8
    assert all(path.parent == tmp_path and path.stat().st_size > 1000 for path in all_paths)
    assert all(path.read_bytes().startswith(b"%PDF") for path in all_paths if path.suffix == ".pdf")
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in all_paths if path.suffix == ".png")


def test_heatmap_display_index_validation(tmp_path):
    with pytest.raises(ValueError, match="display_index_base"):
        plot_heatmaps(select_heads(records()), tmp_path, display_index_base=2)
