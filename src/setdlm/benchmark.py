"GSM8K prompts, answer extraction, and native SetDLM generation."

from __future__ import annotations
import re
from typing import Any
import torch


def _encode(
    tokenizer: Any, model: Any, prompt: str, *, target_length: int
) -> torch.LongTensor:
    device = next(model.parameters()).device
    ids = encode_prompt(
        tokenizer, model, prompt, max_new_tokens=target_length, device=device
    )
    if ids is None:
        return torch.empty((1, 0), dtype=torch.long, device=device)
    return ids


def _gsm_prompt(tokenizer: Any, question: str) -> str:
    instruction = (
        "Please reason step by step, and put your final answer within $\\boxed{}$. "
    )
    content = instruction + question
    if tokenizer.chat_template:
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": content}], enable_thinking=True, **kwargs
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": content}], **kwargs
            )
    return content


def _normalize_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    return answer.strip().replace(",", "").replace(" ", "")


def _gold_answer(answer: str) -> str | None:
    match = re.search("####\\s*([-+0-9,./]+)", answer)
    return _normalize_answer(match.group(1)) if match else None


def _predicted_answer(text: str) -> str | None:
    boxed = re.findall("(?:\\\\boxed|boxed)\\{([^{}]+)\\}", text)
    if boxed:
        return _normalize_answer(boxed[-1])
    hashes = re.findall("####\\s*([-+0-9,./]+)", text)
    return _normalize_answer(hashes[-1]) if hashes else None


def _trim_generation(text: str, tokenizer: Any) -> str:
    stops = ["<|im_end|>", "<|eot_id|>"]
    if tokenizer.eos_token:
        stops.append(tokenizer.eos_token)
    for stop in stops:
        text = text.split(stop)[0]
    return text


import time


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_memory_gib(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / 1024**3)


def _peak_reserved_memory_gib(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_reserved(device) / 1024**3)


def _normalized_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    return (
        answer.strip()
        .replace(",", "")
        .replace(" ", "")
        .replace("\\$", "")
        .replace("$", "")
    )


def _decode_completion(
    tokenizer: Any, target_ids: torch.LongTensor
) -> tuple[str, str, str | None]:
    raw = tokenizer.decode(target_ids[0], skip_special_tokens=False)
    completion = _trim_generation(raw, tokenizer)
    predicted = _normalized_answer(_predicted_answer(completion))
    return (raw, completion, predicted)


def _native_decode(
    model: Any,
    tokenizer: Any,
    prompt_ids: torch.LongTensor,
    *,
    target_length: int,
    window_size: int,
    seed: int,
    max_uncached_rows_per_forward: int = 128,
    record_trace: bool = False,
    trace_top_k: int = 0,
    stop_token_ids: tuple[int, ...] = (),
    stop_sequences: tuple[tuple[int, ...], ...] = (),
) -> tuple[torch.LongTensor, dict[str, Any]]:
    """Run native confidence-threshold generation."""
    seed_everything(seed)
    device = prompt_ids.device
    config = build_generation_config(
        model,
        tokenizer,
        "confidence_parallel_0.9",
        max_new_tokens=target_length,
        max_window_size=window_size,
        do_sample=False,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    generation_trace: list[dict[str, Any]] | None = [] if record_trace else None
    _sync(device)
    started = time.perf_counter()
    output = model.generate(
        inputs=prompt_ids,
        generation_config=config,
        max_new_tokens=target_length,
        return_dict_in_generate=True,
        device=str(device),
        tokenizer=tokenizer,
        generation_stop_token_ids=stop_token_ids,
        generation_stop_sequences=stop_sequences,
        max_uncached_decode_rows_per_forward=max_uncached_rows_per_forward,
        generation_trace=generation_trace,
        generation_trace_top_k=trace_top_k,
    )
    _sync(device)
    latency_s = time.perf_counter() - started
    target_ids = output.sequences[:, prompt_ids.shape[-1] :]
    generated_tokens_per_step = [
        int(value) for value in output.generated_tokens_per_step
    ]
    decode_nfe = int(output.num_steps)
    termination_position = getattr(output, "termination_position", None)
    termination_token_id = getattr(output, "termination_token_id", None)
    termination_detected_at_nfe = getattr(output, "termination_detected_at_step", None)
    uncached_rows_per_forward = [
        int(value) for value in output.uncached_rows_per_forward
    ]
    uncached_row_components_per_forward = [
        {str(component): int(value) for (component, value) in components.items()}
        for components in output.uncached_row_components_per_forward
    ]
    max_uncached_rows = int(output.max_uncached_rows)
    return (
        target_ids,
        {
            "latency_s": float(latency_s),
            "decode_nfe": decode_nfe,
            "setup_forward_calls": 1,
            "total_forward_calls": decode_nfe + 1,
            "parallelism_factor": float(output.parallelism_factor),
            "generated_tokens_per_step": generated_tokens_per_step,
            "uncached_rows_per_forward": uncached_rows_per_forward,
            "uncached_row_components_per_forward": uncached_row_components_per_forward,
            "max_uncached_rows": max_uncached_rows,
            "uncached_row_cap": int(max_uncached_rows_per_forward),
            "peak_memory_gib": _peak_memory_gib(device),
            "peak_reserved_memory_gib": _peak_reserved_memory_gib(device),
            "converged": bool(
                target_ids.shape[-1] == target_length
                or termination_position is not None
            ),
            "canvas_capacity": int(target_length),
            "termination_position": int(termination_position)
            if termination_position is not None
            else None,
            "termination_token_id": int(termination_token_id)
            if termination_token_id is not None
            else None,
            "termination_detected_at_nfe": int(termination_detected_at_nfe)
            if termination_detected_at_nfe is not None
            else None,
            "stopped_early": bool(termination_position is not None),
            "kv_cache": True,
            "cache_rule": "native accepted-set cache; prompt and irreversible accepted sets are retained",
            "window_size": int(window_size),
            "confidence_threshold": 0.9,
            "sampling_strategy": "predict_and_noise",
            "do_sample": False,
            "trace": generation_trace or [],
        },
    )


def seed_everything(seed):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_generation_config(
    model, tokenizer, preset, *, max_new_tokens, max_window_size, do_sample=False
):
    from setdlm.model.diffusion_config import SetDiffusionGenerationConfig

    return SetDiffusionGenerationConfig(
        max_new_tokens=max_new_tokens,
        num_steps=max_new_tokens,
        block_size=int(model.config.length),
        max_window_size=max_window_size,
        do_sample=do_sample,
        use_cache=True,
        noise_removal=False,
        align_inputs_to_blocks=False,
        nucleus_p=1.0,
        pad_token_id=tokenizer.mask_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        sampling_strategy="predict_and_noise",
        first_hitting=False,
        confidence_based_noising=True,
        confidence_margin_based_noising=False,
        confidence_threshold=0.9,
        token_selection_policy="confidence",
        use_first_hitting_order_in_decode=False,
    )


def encode_prompt(
    tokenizer: Any,
    model: Any,
    prompt: str,
    *,
    max_new_tokens: int,
    device: torch.device,
) -> torch.LongTensor | None:
    if not prompt:
        return None
    input_ids = tokenizer(
        prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids
    if getattr(model.config, "keep_clean_bos", False):
        bos_id = tokenizer.bos_token_id
        if bos_id is not None and (input_ids.numel() == 0 or input_ids[0, 0] != bos_id):
            input_ids = torch.cat(
                [torch.tensor([[bos_id]], dtype=input_ids.dtype), input_ids], dim=-1
            )
    return input_ids.to(device)
