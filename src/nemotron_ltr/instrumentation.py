"""Model timing, acceptance diagnostics, and compact result summaries."""

from collections import Counter, defaultdict
import time
import torch
from . import sampling as sampling


def histogram(values):
    return {str(k): v for (k, v) in sorted(Counter((int(x) for x in values)).items())}


def decode(model, prompt, eos, method, width, limit):
    original_forward = model.encoder.forward
    original_reject = sampling.reject
    accepted = []
    calls = []
    had_instance_forward = "forward" in model.encoder.__dict__

    def forward(*args, **kwargs):
        role = (
            "prefill"
            if not calls
            else "proposal"
            if not kwargs.get("use_cache", True)
            else "verify"
            if method == "linear"
            else "verify_and_next_proposal"
        )
        if prompt.device.type == "cuda":
            torch.cuda.synchronize(prompt.device)
        start = time.perf_counter()
        result = original_forward(*args, **kwargs)
        if prompt.device.type == "cuda":
            torch.cuda.synchronize(prompt.device)
        elapsed = time.perf_counter() - start
        calls.append((role, elapsed, int(kwargs["input_ids"].shape[1])))
        return result

    def reject(p, q, candidates):
        (result, n) = original_reject(p, q, candidates)
        accepted.append(n)
        return (result, n)

    model.encoder.forward = forward
    try:
        result = sampling.decode(
            model, prompt, eos, method, width, limit, rejection_fn=reject
        )
    finally:
        if had_instance_forward:
            model.encoder.forward = original_forward
        else:
            del model.encoder.forward
    assert len(calls) == result["nfe"] and len(accepted) == result["rounds"]
    emitted = [n + 1 for n in accepted]
    if emitted:
        emitted[-1] = result["generated_tokens"] - 1 - sum(emitted[:-1])
        assert 1 <= emitted[-1] <= accepted[-1] + 1
    else:
        assert result["generated_tokens"] == 1
    role_times = defaultdict(float)
    for role, elapsed, _ in calls:
        role_times[role] += elapsed
    result.update(
        model_call_work_walltime_s=sum((t for (_, t, _) in calls)),
        prefill_walltime_s=calls[0][1],
        model_call_walltime_s_by_role=dict(role_times),
        model_call_count_by_role=dict(Counter((r for (r, _, _) in calls))),
        encoder_rows_histogram=histogram((n for (_, _, n) in calls)),
        emitted_tokens_per_round_histogram=histogram(emitted),
        accepted_draft_tokens_raw_histogram=histogram(accepted),
        accepted_draft_tokens_emitted_histogram=histogram(
            (min(n, m) for (n, m) in zip(accepted, emitted))
        ),
        initial_tokens_outside_rounds=1,
        histogram_semantics="Raw acceptance counts exclude correction/bonus, before EOS/cap clipping. Emitted-per-round counts include the correction/bonus if retained. Initial causal seed is outside rounds.",
    )
    return result


def compact(row):
    keep = {
        "output_token_ids",
        "model_call_walltime_s_by_role",
        "model_call_count_by_role",
        "code_eval",
    }
    result = {
        k: v
        for (k, v) in row.items()
        if not isinstance(v, (list, dict)) or k in keep or k.endswith("_histogram")
    }
    if "accepted_lengths" in row:
        result["emitted_tokens_per_round_histogram"] = {
            str(k): v for (k, v) in sorted(Counter(row["accepted_lengths"]).items())
        }
        result["initial_tokens_outside_rounds"] = row["generated_tokens"] - sum(
            row["accepted_lengths"]
        )
        result["stable_prefix_tokens_clipped_histogram"] = {
            str(k): v
            for (k, v) in sorted(Counter(row["stable_prefix_lengths"]).items())
        }
        raw = [
            int(
                x.get(
                    "raw_accepted_length",
                    x.get("raw_selected_accepted_length", x["accepted_length"]),
                )
            )
            - 1
            for x in row["iteration_records"]
            if x.get("phase") == "decode" and "accepted_length" in x
        ]
        result["accepted_draft_tokens_raw_histogram"] = {
            str(k): v for (k, v) in sorted(Counter(raw).items())
        }
        result["histogram_semantics"] = (
            "Emitted counts include correction or bonus tokens and are clipped at EOS or the output cap. Raw draft acceptance precedes clipping. Bootstrap output is counted separately."
        )
    return result


def _aggregate_call_times(roles: list[str], times: list[float]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for role, duration in zip(roles, times, strict=True):
        grouped[role].append(float(duration))
    return {role: float(sum(values)) for (role, values) in sorted(grouped.items())}
