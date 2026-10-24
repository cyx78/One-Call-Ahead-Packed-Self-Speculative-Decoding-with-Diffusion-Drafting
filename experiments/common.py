"""Atomic output files, exclusive run locks, and result validation."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
import signal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def output_path(value):
    path = Path(value).expanduser().resolve()
    if path == ROOT or path in ROOT.parents:
        raise ValueError("Choose a dedicated output directory")
    if ROOT in path.parents and ROOT / "outputs" not in path.parents:
        raise ValueError("Output within the package must be below outputs/")
    return path


@contextmanager
def run_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another process is using {directory}") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, allow_nan=False) + "\n"
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def rows_at(path, *, recover_tail=False):
    if not path.exists():
        return []
    content = path.read_bytes()
    lines = content.splitlines(keepends=True)
    result, offset = [], 0
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            if recover_tail and index == len(lines) - 1 and not line.endswith(b"\n"):
                with path.open("r+b") as handle:
                    handle.truncate(offset)
                    handle.flush()
                    os.fsync(handle.fileno())
                break
            raise ValueError(
                f"Invalid JSON record at line {index + 1} in {path.name}"
            ) from error
        if not isinstance(row, dict):
            raise ValueError(f"Expected an object at line {index + 1}")
        result.append(row)
        offset += len(line)
    if (
        recover_tail
        and result
        and offset == len(content)
        and not content.endswith(b"\n")
    ):
        with path.open("ab") as handle:
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    return result


def append_row(path, row):
    content = json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"
    with path.open("a") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def write_rows(path, rows):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    paths = [
        path
        for folder in ("src", "experiments", "configs")
        for path in (ROOT / folder).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    return {str(path.relative_to(ROOT)): file_sha(path) for path in sorted(paths)}


def bind_protocol(directory, protocol):
    path = directory / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Protocol changed; choose a new output directory")
    write_json(path, protocol)


def check_coverage(rows, prompts, repeats, variants, *, complete=False):
    expected = {
        (rep, index, variant)
        for rep in range(repeats)
        for index in range(prompts)
        for variant in variants
    }
    keys = [(row["replicate"], row["index"], row["variant"]) for row in rows]
    if any(
        type(row["index"]) is not int or type(row["replicate"]) is not int
        for row in rows
    ):
        raise ValueError("Response indices must be integers")
    actual = set(keys)
    if len(keys) != len(actual) or not actual <= expected:
        raise ValueError("Duplicate or incompatible response records")
    if complete and actual != expected:
        raise ValueError(
            f"Incomplete generation: {len(actual)}/{len(expected)} responses"
        )
    return actual, expected


def validate_measurement(row, cap):
    for key in ("walltime_s", "nfe", "generated_tokens"):
        value = row.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"Invalid {key}")
    if type(row["nfe"]) is not int or type(row["generated_tokens"]) is not int:
        raise ValueError("Token and call counts must be integers")
    if row["generated_tokens"] > cap:
        raise ValueError("Output exceeds the token cap")
    if "output_token_ids" in row:
        tokens = row["output_token_ids"]
        if len(tokens) != row["generated_tokens"] or any(
            type(token) is not int or token < 0 for token in tokens
        ):
            raise ValueError("Output tokens disagree with the recorded count")
    if row.get("stopped_on_eos") is False and row["generated_tokens"] != cap:
        raise ValueError("Output ended before EOS or the token cap")
    if row.get("correct") is not None and type(row["correct"]) is not bool:
        raise ValueError("Correctness must be boolean or null")


def check_context(model, prompt_length, limit, width):
    config = model.config
    maximum = getattr(config, "max_position_embeddings", None)
    if maximum is None:
        maximum = getattr(
            getattr(config, "text_config", None), "max_position_embeddings", None
        )
    if maximum is not None and prompt_length + limit + width - 1 > maximum:
        raise ValueError(
            "Prompt, output cap, and speculative window exceed the model context"
        )
    if prompt_length < 1:
        raise ValueError("Prompt must contain at least one token")


SAMPLE_TIMEOUT_S = 600


@contextmanager
def sample_timeout(seconds=SAMPLE_TIMEOUT_S):
    def expired(signum, frame):
        raise TimeoutError(
            "Per-response resource guard; interrupted generation is not a completed result"
        )

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
