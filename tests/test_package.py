"""Validate experiment coverage, CLI plans, and statistical aggregation."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_inventory_and_complete_measurements():
    a = module("reproduce", "analyze.py")
    specs, groups = a.load()
    assert len(specs) == 27
    assert sum(map(len, groups.values())) == 89719
    assert {s["task"] for s in specs} == {
        "humaneval",
        "mbpp",
        "gsm8k",
        "math500",
        "writingbench",
    }
    for s in specs:
        if s["family"] == "setdlm":
            assert s["temperature"] == 0 and s["cap"] == 1024
        elif s["task"] in ("humaneval", "mbpp", "writingbench"):
            assert s["cap"] == 512
    for h, correct in [(16, 902), (32, 878)]:
        rows = groups[f"setdlm_{h}_LTR"]
        for method in ["Linear", "Quadratic", "Packed"]:
            rs = [r for r in rows if r["method"] == method]
            assert sum(r["correct"] for r in rs) == correct
            assert all(r["nfe"] == r["decode_nfe"] + 1 for r in rs)


def test_plans_are_read_only_and_use_configured_caps():
    result = subprocess.run(
        [sys.executable, str(ROOT / "experiments/nemotron.py"), "--policy", "rejection"],
        capture_output=True,
        text=True,
        check=True,
        cwd="/tmp",
    )
    p = json.loads(result.stdout)
    assert p["execute"] is False and len(p["cells"]) == 10
    assert {c["task"]: c["repeats"] for c in p["cells"]} == {
        "humaneval": 20,
        "mbpp": 7,
        "gsm8k": 1,
        "math500": 1,
        "writingbench": 1,
    }
    for c in p["cells"]:
        assert c["cap"] == (1024 if c["task"] in ("gsm8k", "math500") else 512)


def test_seed_contract_and_no_unsupported_tasks():
    n = module("runner", "experiments/nemotron.py")
    assert n.sample_seed("3B", "humaneval", "rejection", 0, 0) == 202609160000
    assert n.sample_seed("8B", "mbpp", "rejection", 6, 377) == 202610266377
    assert n.sample_seed("3B", "writingbench", "greedy", 0, 10) == 20260927
    assert n.cap_for("gsm8k", "lora") == 512
    result = subprocess.run(
        [sys.executable, str(ROOT / "experiments/nemotron.py"), "--tasks", "unsupported"],
        capture_output=True,
    )
    assert result.returncode != 0


def test_ratio_of_totals_and_prompt_cluster_bootstrap():
    a = module("stats", "analyze.py")
    spec = dict(
        id="synthetic",
        family="rejection",
        scale="3B",
        task="humaneval",
        temperature=1,
        order="LTR",
        n=2,
        repeats=2,
        cap=512,
    )
    rows = []
    for i in range(2):
        for rep in range(2):
            for method, speed in [("Linear", 1), ("Quadratic", 2), ("Packed", 3)]:
                rows.append(
                    dict(
                        index=i,
                        replicate=rep,
                        method=method,
                        tokens=10 * (i + 1),
                        wall_s=(i + 1) ** 2 / speed,
                        nfe=2,
                        correct=i,
                        model_s=None,
                        stopped_on_eos=True,
                    )
                )
    summary = a.summarize(spec, rows)
    assert summary[0]["tokens_per_s"] == 6
    pair = next(
        r
        for r in a.comparisons(spec, rows, 100)
        if r["numerator"] == "Packed" and r["denominator"] == "Linear"
    )
    assert abs(pair["throughput_ratio"] - 3) < 1e-12
    assert abs(pair["throughput_ratio_low"] - 3) < 1e-12
    assert pair["score_delta_pp"] == 0


def test_expected_metrics_cover_every_primary_cell():
    check = module("manuscript", "analyze.py")
    rows = check.printed_rows()
    assert len(rows) == 66
    assert sum(len(r) - 3 for r in rows) == 252


@pytest.mark.parametrize("extra,widths", [([], [8, 16, 32]), (["--widths", "32", "8", "32"], [32, 8])])
def test_width_ablation_plan_is_read_only(extra, widths, tmp_path):
    output = tmp_path / "unused"
    result = subprocess.run(
        [sys.executable, str(ROOT / "experiments/nemotron.py"), "--suite", "width",
         "--output", str(output), *extra],
        capture_output=True, text=True, check=True, cwd="/tmp",
    )
    plan = json.loads(result.stdout)
    assert plan["widths"] == widths
    assert plan["prompts"] == 10 and plan["cap"] == 512
    assert plan["execute"] is False
    assert not output.exists()


def test_width_ablation_single_size_keeps_configured_row_counts():
    result = subprocess.run(
        [sys.executable, str(ROOT / "experiments/nemotron.py"),
         "--suite", "width", "--width", "16"],
        capture_output=True, text=True, check=True,
    )
    plan = json.loads(result.stdout)
    assert plan["cells"] == [dict(scale="8B", task="gsm8k", cap=512, prompts=10, repeats=1)]
    assert [v["block_length"] for v in plan["variants"]] == [16, 16, 16]
    assert [v["max_uncached_rows_per_forward"] for v in plan["variants"]] == [16, 272, 32]


def test_width_ablation_dispatches_selected_sizes(monkeypatch, tmp_path):
    runner = module("width_runner", "experiments/nemotron.py")
    calls = []
    monkeypatch.setattr(sys, "argv", [
        "nemotron.py", "--suite", "width", "--widths", "8", "32", "8",
        "--num-samples", "2", "--repeats", "3", "--output", str(tmp_path), "--execute",
    ])
    monkeypatch.setattr(runner.subprocess, "run", lambda cmd, **kwargs: calls.append((cmd, kwargs)))
    runner.main()
    assert len(calls) == 2
    for (command, kwargs), width in zip(calls, [8, 32]):
        assert command[command.index("--width") + 1] == str(width)
        assert command[command.index("--output") + 1] == str(tmp_path / f"B{width}")
        assert command[command.index("--num-samples") + 1] == "2"
        assert command[command.index("--repeats") + 1] == "3"
        assert "--execute" in command and kwargs["check"] is True


def test_rebuilt_metric_check_rejects_changed_values(tmp_path):
    analysis = module("metric_check", "analyze.py")
    rows = [dict(row) for row in analysis.printed_rows()]
    analysis.write_csv(tmp_path / "all_measurements.csv", rows)
    analysis.check_metrics(tmp_path)
    assert json.loads((tmp_path / "point_estimate_check.json").read_text())["passed"]
    rows[0]["mean_wall_s"] = "999.000"
    analysis.write_csv(tmp_path / "all_measurements.csv", rows)
    with pytest.raises(SystemExit) as error:
        analysis.check_metrics(tmp_path)
    assert error.value.code == 1
    report = json.loads((tmp_path / "point_estimate_check.json").read_text())
    assert not report["passed"] and len(report["failures"]) == 1
