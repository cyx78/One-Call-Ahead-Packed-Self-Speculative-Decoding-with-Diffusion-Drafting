"Evaluate MBPP base tests with the EvalPlus correctness checker."

from __future__ import annotations
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

_ORACLE_CACHE: dict[tuple[str, bool], tuple[dict[str, Any], str, dict[str, Any]]] = {}


def _base_oracle(
    dataset_version: str, noextreme: bool
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Build the official base oracle once, without constructing plus outputs."""
    key = (dataset_version, noextreme)
    if key in _ORACLE_CACHE:
        return _ORACLE_CACHE[key]
    from evalplus.data import get_mbpp_plus, get_mbpp_plus_hash
    from evalplus.evaluate import MBPP_OUTPUT_NOT_NONE_TASKS, trusted_exec

    problems = get_mbpp_plus(version=dataset_version, noextreme=noextreme)
    dataset_hash = get_mbpp_plus_hash(version=dataset_version, noextreme=noextreme)
    if (
        dataset_version != "v0.2.0"
        or not noextreme
        or dataset_hash != "85d8d7a406c0686e80353e676189bfee"
        or len(problems) != 378
    ):
        raise ValueError("MBPP base grading requires the pinned 378-task release")
    expected: dict[str, Any] = {}
    for task_id, problem in problems.items():
        (outputs, runtimes) = trusted_exec(
            str(problem["prompt"]) + str(problem["canonical_solution"]),
            problem["base_input"],
            problem["entry_point"],
            record_time=True,
            output_not_none=problem["entry_point"] in MBPP_OUTPUT_NOT_NONE_TASKS,
        )
        expected[task_id] = {"base": outputs, "base_time": runtimes}
    _ORACLE_CACHE[key] = (problems, dataset_hash, expected)
    return _ORACLE_CACHE[key]


def evaluate_mbpp_base(
    samples: list[dict[str, str]],
    *,
    dataset_version: str,
    noextreme: bool,
    parallel_workers: int,
) -> dict[str, Any]:
    from evalplus.evaluate import check_correctness

    (problems, dataset_hash, expected) = _base_oracle(dataset_version, noextreme)
    task_ids = [str(sample["task_id"]) for sample in samples]
    if len(task_ids) != len(set(task_ids)) or set(task_ids) != set(problems):
        raise RuntimeError("EvalPlus MBPP samples must cover every task exactly once")
    completed: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
        futures = {}
        for sample in samples:
            task_id = str(sample["task_id"])
            future = executor.submit(
                check_correctness,
                "mbpp",
                0,
                problems[task_id],
                str(sample["solution"]),
                expected[task_id],
                True,
                False,
                task_id,
                1.0,
                4.0,
            )
            futures[future] = task_id
        for future in as_completed(futures):
            task_id = futures[future]
            result = future.result()
            (base_status, details) = result["base"]
            completed[task_id] = {
                "task_id": task_id,
                "solution_sha256": hashlib.sha256(
                    str(result["solution"]).encode()
                ).hexdigest(),
                "base_status": str(base_status),
                "base_test_details": [bool(value) for value in details],
                "base_fail_indices": [
                    index for (index, passed) in enumerate(details) if not passed
                ],
            }
    if set(completed) != set(problems):
        raise RuntimeError("EvalPlus MBPP evaluator lost task results")
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "official_harness": "evalplus.evaluate.check_correctness",
        "dataset": "mbpp",
        "dataset_version": dataset_version,
        "dataset_md5": dataset_hash,
        "noextreme_release": noextreme,
        "base_only": True,
        "samples": len(samples),
        "eval": {task_id: [completed[task_id]] for task_id in sorted(completed)},
    }
