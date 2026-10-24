"Pinned dataset releases and zero-shot chat prompts."

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
DATASET_CACHE = BASE_DIR.parents[1] / ".cache" / "datasets"


@dataclass(frozen=True)
class TaskSpec:
    key: str
    label: str
    category: str
    dataset_id: str
    dataset_config: str | None
    revision: str
    split: str
    num_samples: int
    max_new_tokens: int
    metric: str
    prompt_protocol: str


TASK_SPECS = {
    "humaneval": TaskSpec(
        key="humaneval",
        label="HumanEval",
        category="Code",
        dataset_id="openai/openai_humaneval",
        dataset_config=None,
        revision="7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
        split="test",
        num_samples=164,
        max_new_tokens=512,
        metric="pass_at_1",
        prompt_protocol="zero_shot_complete_function_chat",
    ),
    "mbpp": TaskSpec(
        key="mbpp",
        label="MBPP",
        category="Code",
        dataset_id="evalplus/mbppplus",
        dataset_config=None,
        revision="b2d74c91837c3f2a20c1299ae98133cbe7cfa077",
        split="test",
        num_samples=378,
        max_new_tokens=512,
        metric="pass_at_1_on_original_base_tests",
        prompt_protocol="evalplus_mbpp378_zero_shot_visible_base_tests_chat",
    ),
    "gsm8k": TaskSpec(
        key="gsm8k",
        label="GSM8K",
        category="Mathematics",
        dataset_id="openai/gsm8k",
        dataset_config="main",
        revision="740312add88f781978c0658806c59bc2815b9866",
        split="test",
        num_samples=1319,
        max_new_tokens=512,
        metric="exact_match",
        prompt_protocol="boxed_numeric_zero_shot_chat",
    ),
    "math500": TaskSpec(
        key="math500",
        label="MATH-500",
        category="Mathematics",
        dataset_id="HuggingFaceH4/MATH-500",
        dataset_config=None,
        revision="6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        split="test",
        num_samples=500,
        max_new_tokens=512,
        metric="boxed_exact_match",
        prompt_protocol="boxed_answer_zero_shot_chat",
    ),
}


def _load_hf_dataset(spec: TaskSpec) -> Any:
    from datasets import load_dataset

    return load_dataset(
        spec.dataset_id,
        spec.dataset_config,
        revision=spec.revision,
        split=spec.split,
        cache_dir=str(DATASET_CACHE),
    )


def messages_for_row(task_name: str, row: dict[str, Any]) -> list[dict[str, str]]:
    if task_name == "humaneval":
        content = (
            "Complete the Python function below. Return only executable Python code, preferably in one ```python``` block. Do not add tests or an explanation.\n\n```python\n"
            + str(row["prompt"])
            + "\n```"
        )
        return [{"role": "user", "content": content}]
    if task_name == "mbpp":
        tests = "\n".join((str(value) for value in row["test_list"]))
        content = (
            "Write a Python function that solves the task below. Return only executable Python code, preferably in one ```python``` block. Your function must use the interface demonstrated by the tests.\n\nTask:\n"
            + str(row["prompt"])
            + "\n\nTests:\n"
            + tests
        )
        return [{"role": "user", "content": content}]
    if task_name == "gsm8k":
        content = (
            "Solve the following math problem. Put the final numerical answer inside \\boxed{} at the very end.\n\n"
            + str(row["question"])
        )
        return [{"role": "user", "content": content}]
    if task_name == "math500":
        content = (
            "Solve the following math problem. Put the final answer inside \\boxed{} at the very end.\n\n"
            + str(row["problem"])
        )
        return [{"role": "user", "content": content}]
    raise ValueError(f"unknown task {task_name!r}")


def prepared_rows(task_name):
    if task_name not in TASK_SPECS:
        raise ValueError(f"Unknown task: {task_name}")
    spec = TASK_SPECS[task_name]
    rows = [dict(row) for row in _load_hf_dataset(spec)]
    if len(rows) != spec.num_samples:
        raise ValueError(f"Unexpected dataset size for {task_name}: {len(rows)}")
    return rows
