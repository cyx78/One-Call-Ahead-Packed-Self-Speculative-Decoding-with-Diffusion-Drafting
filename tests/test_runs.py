import json
import math
from types import SimpleNamespace
import pytest
from common import (
    append_row,
    bind_protocol,
    check_context,
    check_coverage,
    output_path,
    rows_at,
    run_lock,
    validate_measurement,
    write_json,
)
import common
import nemotron


def sample(index=0, variant="linear"):
    return dict(
        index=index,
        replicate=0,
        variant=variant,
        walltime_s=1.0,
        nfe=3,
        generated_tokens=5,
        correct=True,
    )


def test_resume_recovers_only_uncommitted_tail(tmp_path):
    path = tmp_path / "raw.jsonl"
    append_row(path, sample())
    with path.open("ab") as handle:
        handle.write(b'{"index": 1, "var')
    assert rows_at(path, recover_tail=True) == [sample()]
    append_row(path, sample(1))
    assert rows_at(path) == [sample(), sample(1)]


def test_resume_keeps_complete_record_without_newline(tmp_path):
    path = tmp_path / "raw.jsonl"
    path.write_text(json.dumps(sample()))
    assert rows_at(path, recover_tail=True) == [sample()]
    append_row(path, sample(1))
    assert len(rows_at(path)) == 2


def test_resume_rejects_corruption_before_last_record(tmp_path):
    path = tmp_path / "raw.jsonl"
    content = b"{broken}\n" + json.dumps(sample()).encode() + b"\n"
    path.write_bytes(content)
    with pytest.raises(ValueError, match="line 1"):
        rows_at(path, recover_tail=True)
    assert path.read_bytes() == content


def test_lock_prevents_concurrent_writer_and_releases(tmp_path):
    with run_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another process"):
            with run_lock(tmp_path):
                pass
    with run_lock(tmp_path):
        pass


@pytest.mark.parametrize("rows", [[sample(), sample()], [sample(3)], []])
def test_grading_requires_complete_unique_coverage(rows):
    with pytest.raises(ValueError):
        check_coverage(rows, 1, 1, ["linear"], complete=True)


def test_protocol_changes_are_rejected(tmp_path):
    bind_protocol(tmp_path, {"width": 8})
    bind_protocol(tmp_path, {"width": 8})
    with pytest.raises(ValueError, match="Protocol changed"):
        bind_protocol(tmp_path, {"width": 16})
    assert json.loads((tmp_path / "protocol.json").read_text()) == {"width": 8}


@pytest.mark.parametrize("value", [math.nan, math.inf, -1, 0, True])
def test_nonfinite_or_invalid_timing_is_rejected(value):
    row = sample()
    row["walltime_s"] = value
    with pytest.raises(ValueError, match="walltime_s"):
        validate_measurement(row, 32)


def test_atomic_json_keeps_existing_file_on_serialization_failure(tmp_path):
    path = tmp_path / "status.json"
    write_json(path, {"status": "running"})
    with pytest.raises(ValueError):
        write_json(path, {"invalid": math.nan})
    assert json.loads(path.read_text()) == {"status": "running"}


@pytest.mark.parametrize(
    "value", ["", "reference", "src", "configs", "outputs"]
)
def test_output_directory_cannot_overwrite_package(value):
    with pytest.raises(ValueError):
        output_path(common.ROOT / value)


def test_output_directory_resolves_symlinks(tmp_path):
    link = tmp_path / "link"
    link.symlink_to(common.ROOT / "reference", target_is_directory=True)
    with pytest.raises(ValueError):
        output_path(link / "run")


def test_context_check_reserves_speculative_tail():
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=64))
    check_context(model, 10, 48, 7)
    with pytest.raises(ValueError, match="context"):
        check_context(model, 10, 48, 8)


def test_grade_does_not_mark_partial_output_complete(tmp_path, monkeypatch):
    args = SimpleNamespace(
        output=tmp_path, num_samples=2, repeats=1, policy="greedy", suite="main"
    )
    directory = tmp_path / "3B/gsm8k"
    directory.mkdir(parents=True)
    append_row(directory / "raw.jsonl", sample())
    monkeypatch.setattr(nemotron, "load_task", lambda task: [{}, {}])
    with pytest.raises(ValueError, match="Incomplete generation"):
        nemotron.grade_and_merge(args, "3B", "gsm8k")
    assert not (directory / "status.json").exists()


def test_completed_math_rows_are_aggregated(tmp_path, monkeypatch):
    args = SimpleNamespace(
        output=tmp_path, num_samples=1, repeats=1, policy="greedy", suite="main"
    )
    directory = tmp_path / "3B/gsm8k"
    directory.mkdir(parents=True)
    for variant in nemotron.VARIANTS:
        append_row(directory / "raw.jsonl", sample(variant=variant["id"]))
    monkeypatch.setattr(nemotron, "load_task", lambda task: [{}])
    nemotron.grade_and_merge(args, "3B", "gsm8k")
    summary = json.loads((directory / "summary.json").read_text())
    assert all(row["score_pct"] == 100 for row in summary)
    assert json.loads((directory / "status.json").read_text())["status"] == "complete"


def test_sandbox_has_explicit_mounts_and_no_network(monkeypatch, tmp_path):
    import grading_sandbox

    monkeypatch.setattr(grading_sandbox.shutil, "which", lambda name: "/usr/bin/bwrap")
    command = grading_sandbox.command(
        ["python", "-c", "pass"],
        readonly=[tmp_path / "input"],
        writable=[tmp_path / "output"],
    )
    assert "--unshare-all" in command and "--clearenv" in command
    mounts = [
        command[index + 1]
        for index, value in enumerate(command)
        if value in ("--bind", "--ro-bind")
    ]
    assert "/" not in mounts and "/home" not in mounts
    assert "--bind" in command


def test_sandbox_failure_is_not_a_failed_model_answer(monkeypatch):
    import grading_sandbox

    monkeypatch.setattr(grading_sandbox.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="bubblewrap"):
        grading_sandbox.preflight()
