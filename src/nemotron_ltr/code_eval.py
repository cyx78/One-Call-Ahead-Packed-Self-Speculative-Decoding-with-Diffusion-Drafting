"Extract code candidates and score HumanEval in a subprocess."

from __future__ import annotations
import ast
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
BLOCKED_IMPORTS = {
    "asyncio",
    "builtins",
    "ctypes",
    "glob",
    "http",
    "importlib",
    "multiprocessing",
    "os",
    "pathlib",
    "pickle",
    "requests",
    "resource",
    "shutil",
    "socket",
    "subprocess",
    "tempfile",
    "threading",
    "urllib",
}
BLOCKED_CALLS = {"breakpoint", "input", "open", "__import__"}
BLOCKED_ATTRIBUTES = {
    "chmod",
    "connect",
    "kill",
    "makedirs",
    "mkdir",
    "open",
    "popen",
    "removedirs",
    "rename",
    "rmdir",
    "rmtree",
    "send",
    "system",
    "unlink",
    "write_text",
    "write_bytes",
}
FENCE_RE = re.compile("```(?:python|py)?\\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def _candidate_bodies(output: str) -> list[str]:
    fenced = [value.strip() for value in FENCE_RE.findall(output)]
    raw = output.strip()
    candidates = fenced + ([raw] if raw else [])
    unique: list[str] = []
    seen: set[str] = set()
    for value in candidates:
        value = re.sub("^\\s*(?:python|py)\\s*\\n", "", value, flags=re.IGNORECASE)
        value = value.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        if value and value not in seen:
            unique.append(value)
            seen.add(value)
    return unique


def _policy_check(source: str) -> tuple[bool, str | None]:
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return (False, f"syntax_error:{error.msg}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] in BLOCKED_IMPORTS:
                    return (False, f"blocked_import:{alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".", 1)[0]
            if node.level or module in BLOCKED_IMPORTS:
                return (False, f"blocked_import:{node.module}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALLS:
                return (False, f"blocked_call:{node.func.id}")
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in BLOCKED_ATTRIBUTES
            ):
                return (False, f"blocked_attribute:{node.func.attr}")
    return (True, None)


def _candidate_sources(task: str, output: str, row: dict[str, Any]) -> list[str]:
    bodies = _candidate_bodies(output)
    if task != "humaneval":
        return bodies
    dataset = _dataset_row(task, int(row["index"]))
    prompt = str(dataset["prompt"])
    entry = str(dataset["entry_point"])
    result: list[str] = []
    for body in bodies:
        if re.search(f"\\bdef\\s+{re.escape(entry)}\\s*\\(", body):
            result.append(body)
        result.append(prompt + body)
    return list(dict.fromkeys(result))


_DATASETS: dict[str, Any] = {}


def _dataset_row(task: str, index: int) -> dict[str, Any]:
    return dict(_dataset(task)[index])


def _worker_payload(
    task: str, candidate: str, row: dict[str, Any], *, challenge: bool
) -> dict[str, Any]:
    source = _dataset_row(task, int(row["index"]))
    if task != "humaneval" or challenge:
        raise ValueError("the local sandbox is used only for HumanEval base tests")
    return {
        "kind": "humaneval",
        "candidate": candidate,
        "tests": source["test"],
        "entry_point": source["entry_point"],
    }


def _run_worker(payload):
    from grading_sandbox import command

    with tempfile.TemporaryDirectory(prefix="code_eval_") as directory:
        payload_path = Path(directory) / "payload.json"
        payload_path.write_text(json.dumps(payload))
        worker = BASE_DIR / "code_eval_worker.py"
        # Mount only the worker and its input in addition to the Python runtime.
        argv = command(
            [sys.executable, "-I", str(worker), str(payload_path)],
            readonly=[worker, payload_path],
        )
        # Place the input outside the sandbox's private /tmp mount.
        for index, value in enumerate(argv):
            if value == "--ro-bind" and argv[index + 1] == str(payload_path):
                argv[index + 2] = "/payload.json"
        argv[-1] = "/payload.json"
        try:
            completed = subprocess.run(
                argv, capture_output=True, text=True, timeout=12, check=False
            )
        except subprocess.TimeoutExpired:
            return dict(passed=False, outcome="timeout", error_type="Timeout")
    if completed.returncode in (124, 137, -9, -24):
        return dict(passed=False, outcome="timeout", error_type="Timeout")
    if completed.returncode != 0:
        return dict(
            passed=False,
            outcome="sandbox_error",
            error_type=f"returncode_{completed.returncode}",
            stderr=completed.stderr[-1000:],
        )
    try:
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        if type(result.get("passed")) is not bool or not isinstance(
            result.get("outcome"), str
        ):
            raise ValueError("Invalid result schema")
        return result
    except (IndexError, ValueError, AttributeError):
        return dict(
            passed=False,
            outcome="sandbox_error",
            error_type="invalid_worker_output",
            stderr=completed.stderr[-1000:],
        )


def _score_code_row(task: str, row: dict[str, Any]) -> dict[str, Any]:
    if task != "humaneval":
        raise ValueError("MBPP must be scored by the official EvalPlus harness")
    attempts: list[dict[str, Any]] = []
    chosen: str | None = None
    final: dict[str, Any] | None = None
    for candidate in _candidate_sources(task, str(row["output_text"]), row):
        (allowed, policy_reason) = _policy_check(candidate)
        digest = hashlib.sha256(candidate.encode()).hexdigest()
        if not allowed:
            attempts.append(
                {
                    "candidate_sha256": digest,
                    "outcome": "policy_rejected",
                    "reason": policy_reason,
                }
            )
            continue
        result = _run_worker(_worker_payload(task, candidate, row, challenge=False))
        result["candidate_sha256"] = digest
        attempts.append(result)
        if bool(result.get("passed")):
            chosen = candidate
            final = result
            break
        if final is None or final.get("outcome") in {"policy_rejected", "syntax_error"}:
            final = result
    if final is None:
        final = {
            "passed": False,
            "outcome": "no_executable_candidate",
            "error_type": None,
        }
    return {
        "passed": bool(final.get("passed")),
        "outcome": str(final.get("outcome")),
        "error_type": final.get("error_type"),
        "selected_candidate_sha256": hashlib.sha256(chosen.encode()).hexdigest()
        if chosen is not None
        else None,
        "attempts": attempts,
        "sandbox": "bubblewrap_readonly_runtime+isolated_namespaces+restricted_builtins+rlimits",
    }


def _mbpp_candidate(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for candidate in _candidate_bodies(str(row["output_text"])):
        digest = hashlib.sha256(candidate.encode()).hexdigest()
        (allowed, reason) = _policy_check(candidate)
        attempts.append(
            {
                "candidate_sha256": digest,
                "outcome": "selected" if allowed else "policy_rejected",
                "reason": reason,
            }
        )
        if allowed:
            return (
                candidate,
                {
                    "selected_candidate_sha256": digest,
                    "candidate_extraction": "first_fenced_else_raw_policy_allowed_v1",
                    "attempts": attempts,
                },
            )
    return (
        "pass\n",
        {
            "selected_candidate_sha256": None,
            "candidate_extraction": "no_valid_candidate_fallback_noop_v1",
            "attempts": attempts,
        },
    )


_CONFIG: dict[str, Any] = {}


def _dataset(task):
    if task not in _DATASETS:
        from .data import prepared_rows

        _DATASETS[task] = prepared_rows(task)
    return _DATASETS[task]
