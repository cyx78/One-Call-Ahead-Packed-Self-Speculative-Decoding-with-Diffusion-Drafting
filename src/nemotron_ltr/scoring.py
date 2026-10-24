"Answer extraction, numeric comparison, and task metadata."

from __future__ import annotations
import hashlib
import re
from typing import Any
import torch

NUMBER_RE = re.compile("-?\\d+(?:\\.\\d+)?")


def _prompt_ids(
    tokenizer: Any, messages: list[dict[str, str]], device: torch.device
) -> tuple[torch.LongTensor, str]:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    return (ids, digest)


def _last_boxed(text: str) -> str | None:
    marker = "\\boxed{"
    results: list[str] = []
    start = 0
    while True:
        location = text.find(marker, start)
        if location < 0:
            break
        index = location + len(marker)
        depth = 1
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        if depth == 0:
            results.append(text[location + len(marker) : index - 1].strip())
        start = location + len(marker)
    return results[-1] if results else None


def _numbers_equal(left: str | None, right: str) -> bool:
    if left is None:
        return False
    try:
        return abs(float(left.replace(",", "")) - float(right.replace(",", ""))) < 1e-06
    except (TypeError, ValueError):
        return left.strip() == right.strip()


def _score_output(task_name: str, text: str, row: dict[str, Any]) -> dict[str, Any]:
    if task_name == "gsm8k":
        gold = str(row["answer"]).split("####")[-1].strip().replace(",", "")
        prediction = _last_boxed(text)
        if prediction is None:
            numbers = NUMBER_RE.findall(text.replace(",", ""))
            prediction = numbers[-1] if numbers else None
        return {
            "gold_answer": gold,
            "prediction": prediction,
            "correct": _numbers_equal(prediction, gold),
        }
    if task_name == "math500":
        gold = str(row["answer"])
        prediction = _last_boxed(text)
        normalize = lambda value: re.sub(
            "\\s+",
            "",
            value.strip().strip("$").replace("\\left", "").replace("\\right", ""),
        )
        correct = prediction is not None and (
            normalize(prediction) == normalize(gold) or _numbers_equal(prediction, gold)
        )
        return {"gold_answer": gold, "prediction": prediction, "correct": correct}
    if task_name == "humaneval":
        return {
            "gold_answer": str(row["task_id"]),
            "prediction": None,
            "correct": None,
            "score_status": "pending_sandboxed_code_execution",
        }
    if task_name == "mbpp":
        return {
            "gold_answer": str(row["task_id"]),
            "prediction": None,
            "correct": None,
            "score_status": "pending_sandboxed_code_execution",
        }
    raise ValueError(task_name)


def _row_metadata(task_name: str, row: dict[str, Any]) -> dict[str, Any]:
    if task_name == "humaneval":
        return {"task_id": row["task_id"], "entry_point": row["entry_point"]}
    if task_name == "mbpp":
        return {"task_id": int(row["task_id"])}
    if task_name == "math500":
        return {
            "task_id": row["unique_id"],
            "subject": row["subject"],
            "level": int(row["level"]),
        }
    return {}


BOXED_RE = re.compile("\\\\boxed\\{([^{}]+)\\}")


def _extract_prediction(text: str) -> str | None:
    boxed = BOXED_RE.findall(text)
    if boxed:
        return boxed[-1].strip()
    numbers = NUMBER_RE.findall(text.replace(",", ""))
    return numbers[-1] if numbers else None


def _numbers_equal_plain(left, right):
    if left is None:
        return False
    try:
        return abs(float(left) - float(right)) < 1e-6
    except (TypeError, ValueError):
        return left.strip() == right.strip()


def score_output(scale, task, text, source):
    if scale == "8B" and task == "gsm8k":
        gold = str(source["answer"]).split("####")[-1].strip().replace(",", "")
        prediction = _extract_prediction(text)
        return dict(
            gold_answer=gold,
            prediction=prediction,
            correct=_numbers_equal_plain(prediction, gold),
        )
    return _score_output(task, text, source)
