"""Greedy linear, quadratic, and packed self-speculative decoding."""

from __future__ import annotations
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from typing import Any
import torch
from .lora_runtime import require_lora, row_selective_lora
from .runtime_attention import (
    bootstrap_visibility,
    build_next_packed_candidate,
    extract_quadratic_roots,
    packed_position_ids,
    packed_visibility,
    quadratic_position_ids,
    quadratic_visibility,
    runtime_attention,
    select_quadratic_step,
)
from .selection import select_clean_first_ltr_branch


@dataclass
class StrictLTRDecodeResult:
    output_ids: torch.LongTensor
    nfe: int
    prompt_prefill_rows: int
    uncached_rows_per_forward: list[int]
    model_call_roles: list[str]
    model_call_rows: list[int]
    model_call_walltimes_s: list[float]
    prefill_walltime_s: float
    iteration_records: list[dict[str, Any]]
    accepted_lengths: list[int]
    stable_prefix_lengths: list[int]
    branch_accepted_lengths: list[list[int]]
    branch_candidate_shared_prefix_lengths: list[int]
    branch_shared_accepted_prefix_lengths: list[int]
    winner_branches: list[str]
    branch_selection_counts: dict[str, int]
    stopped_on_eos: bool
    max_uncached_rows: int
    block_length: int
    strict_ltr: bool = True
    trajectory_trace: list[dict[str, Any]] = field(default_factory=list)
    model_call_lora_rows: list[int] = field(default_factory=list)

    def diagnostics(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("output_ids")
        return payload


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _set_diffusion_lm(model: Any, enabled: bool) -> None:
    for layer in model.encoder.layers:
        attention = getattr(layer, "self_attn", None)
        if attention is not None and hasattr(attention, "diffusion_lm"):
            attention.diffusion_lm = enabled


def _toggle_adapters(model: Any, enabled: bool) -> None:
    for module in model.modules():
        if hasattr(module, "_disable_adapters"):
            module._disable_adapters = not enabled


def _crop_cache(cache: Any, maximum_length: int) -> None:
    method = getattr(cache, "crop", None)
    if method is None:
        raise TypeError("cache does not support crop")
    method(int(maximum_length))


def _record_call(
    *, rows: int, role: str, row_cap: int, rows_per_forward: list[int], roles: list[str]
) -> None:
    if rows <= 0:
        raise AssertionError("decode forward must contain at least one query row")
    if rows > row_cap:
        raise AssertionError(
            f"decode forward exceeds uncached-row cap: rows={rows}, cap={row_cap}"
        )
    rows_per_forward.append(int(rows))
    roles.append(role)


def _argmax_tokens(logits: torch.Tensor) -> torch.LongTensor:
    return logits.argmax(dim=-1).to(torch.long)


@torch.inference_mode()
def _strict_ltr_decode(
    model: Any,
    prompt_ids: torch.LongTensor,
    *,
    max_new_tokens: int,
    block_length: int,
    threshold: float,
    eos_token_id: int | None,
    max_uncached_rows_per_forward: int,
    record_model_call_walltimes: bool,
    record_trajectory: bool = False,
    record_iteration_diagnostics: bool = True,
    use_draft_lora: bool = False,
) -> StrictLTRDecodeResult:
    if (
        prompt_ids.ndim != 2
        or int(prompt_ids.shape[0]) != 1
        or prompt_ids.shape[1] == 0
    ):
        raise ValueError(
            "strict-LTR decoding requires a nonempty prompt and batch size one"
        )
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if block_length < 2:
        raise ValueError("block_length must be at least two")
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    if record_trajectory and (not record_iteration_diagnostics):
        raise ValueError("trajectory recording requires iteration diagnostics")
    if 1 * block_length > max_uncached_rows_per_forward:
        raise ValueError("configured block length exceeds the uncached-row cap")
    if use_draft_lora:
        require_lora(model)
    device = prompt_ids.device
    token_mask_id = int(model.config.mask_token_id)
    if eos_token_id is None:
        eos_token_id = getattr(model.config, "eos_token_id", None)
    rows_per_forward: list[int] = []
    roles: list[str] = []
    model_call_rows: list[int] = []
    model_call_lora_rows: list[int] = []
    call_walltimes: list[float] = []
    accepted_lengths: list[int] = []
    stable_prefix_lengths: list[int] = []
    branch_accepted_lengths: list[list[int]] = []
    branch_candidate_shared_prefix_lengths: list[int] = []
    branch_shared_accepted_prefix_lengths: list[int] = []
    winner_branches: list[str] = []
    branch_counts = {"clean": 0}
    trajectory_trace: list[dict[str, Any]] = []
    iteration_records: list[dict[str, Any]] = []

    def timed_encoder_call(
        role: str, rows: int, *, lora_rows: int = 0, **kwargs: Any
    ) -> Any:
        _record_call(
            rows=rows,
            role=role,
            row_cap=max_uncached_rows_per_forward,
            rows_per_forward=rows_per_forward,
            roles=roles,
        )
        model_call_rows.append(int(rows))
        model_call_lora_rows.append(int(lora_rows))
        if record_model_call_walltimes:
            _sync(device)
            started = time.perf_counter()
        result = model.encoder(**kwargs)
        if record_model_call_walltimes:
            _sync(device)
            call_walltimes.append(float(time.perf_counter() - started))
        return result

    _set_diffusion_lm(model, False)
    _toggle_adapters(model, False)
    dynamic_cache_cls = _dynamic_cache_class()
    if record_model_call_walltimes:
        _sync(device)
        prefill_started = time.perf_counter()
    roles.append("prompt_prefill")
    model_call_rows.append(int(prompt_ids.numel()))
    model_call_lora_rows.append(0)
    enc_out = model.encoder(
        input_ids=prompt_ids,
        past_key_values=dynamic_cache_cls(),
        use_cache=True,
        use_causal_mask=True,
    )
    if record_model_call_walltimes:
        _sync(device)
        prefill_walltime = float(time.perf_counter() - prefill_started)
        call_walltimes.append(prefill_walltime)
    else:
        prefill_walltime = 0.0
    past_key_values = enc_out.past_key_values
    last_logit = model.diffusion_head(enc_out.last_hidden_state[:, -1:, :]).squeeze(1)
    next_token = _argmax_tokens(last_logit).reshape(1, 1)
    nfe = 1
    if eos_token_id is not None and int(next_token.item()) == int(eos_token_id):
        output_ids = torch.cat((prompt_ids, next_token), dim=1)
        return StrictLTRDecodeResult(
            output_ids=output_ids,
            nfe=nfe,
            prompt_prefill_rows=int(prompt_ids.numel()),
            uncached_rows_per_forward=rows_per_forward,
            model_call_roles=roles,
            model_call_rows=model_call_rows,
            model_call_walltimes_s=call_walltimes,
            prefill_walltime_s=prefill_walltime,
            iteration_records=iteration_records,
            accepted_lengths=accepted_lengths,
            stable_prefix_lengths=stable_prefix_lengths,
            branch_accepted_lengths=branch_accepted_lengths,
            branch_candidate_shared_prefix_lengths=branch_candidate_shared_prefix_lengths,
            branch_shared_accepted_prefix_lengths=branch_shared_accepted_prefix_lengths,
            winner_branches=winner_branches,
            branch_selection_counts=branch_counts,
            stopped_on_eos=True,
            max_uncached_rows=0,
            block_length=block_length,
            model_call_lora_rows=model_call_lora_rows,
        )
    generated = [next_token]
    total_generated = 1
    stopped_on_eos = False
    while total_generated < max_new_tokens:
        if record_iteration_diagnostics and record_model_call_walltimes:
            _sync(device)
            iteration_started = time.perf_counter()
        call_start = len(call_walltimes) if record_iteration_diagnostics else 0
        role_start = len(roles) if record_iteration_diagnostics else 0
        generated_before = total_generated
        cache_length = int(past_key_values.get_seq_length())
        sequence_before_ids = (
            torch.cat(generated, dim=1)[0].detach().cpu().tolist()
            if record_trajectory
            else []
        )
        clean = torch.full(
            (1, block_length), token_mask_id, dtype=torch.long, device=device
        )
        clean[:, 0] = next_token
        _set_diffusion_lm(model, True)
        _toggle_adapters(model, use_draft_lora)
        while bool((clean == token_mask_id).any()):
            enc_out = timed_encoder_call(
                role="fresh_diffusion_draft",
                rows=block_length,
                lora_rows=block_length if use_draft_lora else 0,
                input_ids=clean,
                past_key_values=past_key_values,
                use_cache=False,
            )
            nfe += 1
            draft_logits = model.diffusion_head(enc_out.last_hidden_state)
            draft_tokens = _argmax_tokens(draft_logits)
            is_mask = clean == token_mask_id
            if threshold > 0.0:
                probabilities = torch.softmax(draft_logits, dim=-1)
                confidence = probabilities.gather(
                    dim=-1, index=draft_tokens.unsqueeze(-1)
                ).squeeze(-1)
                confidence = torch.where(is_mask, confidence, -torch.inf)
                unmask = confidence >= threshold
                if not bool(unmask.any()):
                    best = int(confidence.reshape(-1).argmax().item())
                    unmask = torch.zeros_like(is_mask)
                    unmask.reshape(-1)[best] = True
                clean[unmask] = draft_tokens[unmask]
            else:
                clean[is_mask] = draft_tokens[is_mask]
        candidates = clean
        branch_sources = [["seed", *["fresh_diffusion"] * (block_length - 1)]]
        _set_diffusion_lm(model, False)
        _toggle_adapters(model, False)
        verifier_cache = _identity_cache(past_key_values, 1)
        enc_out = timed_encoder_call(
            role="causal_verify_clean",
            rows=1 * block_length,
            input_ids=candidates,
            past_key_values=verifier_cache,
            use_cache=True,
            use_causal_mask=True,
        )
        nfe += 1
        verified_cache = enc_out.past_key_values
        verifier_ids = _argmax_tokens(model.diffusion_head(enc_out.last_hidden_state))
        selection = select_clean_first_ltr_branch(candidates, verifier_ids)
        per_branch_accepted = [int(value) for value in selection.accepted_lengths]
        candidate_shared_prefix = block_length
        shared_accepted_prefix = min(candidate_shared_prefix, *per_branch_accepted)
        winner = int(selection.winner)
        winner_name = "clean" if winner == 0 else "clean"
        raw_accepted = int(selection.accepted_lengths[winner])
        stable_prefix = int(selection.stable_prefix_lengths[winner])
        remaining = max_new_tokens - total_generated
        accepted = min(raw_accepted, remaining)
        winner_verifier = verifier_ids[winner : winner + 1]
        accepted_tokens = winner_verifier[:, :accepted]
        _crop_cache(verified_cache, cache_length + accepted)
        past_key_values = verified_cache
        eos_offset: int | None = None
        if eos_token_id is not None:
            eos_positions = (accepted_tokens[0] == int(eos_token_id)).nonzero(
                as_tuple=True
            )[0]
            if int(eos_positions.numel()) > 0:
                eos_offset = int(eos_positions[0].item())
                accepted_tokens = accepted_tokens[:, : eos_offset + 1]
                accepted = eos_offset + 1
                _crop_cache(past_key_values, cache_length + accepted)
                stopped_on_eos = True
        generated.append(accepted_tokens)
        total_generated += accepted
        next_token = winner_verifier[:, accepted - 1 : accepted]
        winner_verifier.detach().clone()
        accepted_lengths.append(accepted)
        if record_iteration_diagnostics:
            stable_prefix_lengths.append(min(stable_prefix, max(accepted - 1, 0)))
            branch_accepted_lengths.append(per_branch_accepted)
            branch_candidate_shared_prefix_lengths.append(candidate_shared_prefix)
            branch_shared_accepted_prefix_lengths.append(shared_accepted_prefix)
        winner_branches.append(winner_name)
        branch_counts[winner_name] += 1
        if record_iteration_diagnostics:
            if record_model_call_walltimes:
                _sync(device)
                iteration_walltime = float(time.perf_counter() - iteration_started)
            else:
                iteration_walltime = 0.0
            iteration_call_times = call_walltimes[call_start:]
            iteration_model_time = float(sum(iteration_call_times))
            iteration_records.append(
                {
                    "iteration": len(iteration_records) + 1,
                    "phase": "decode",
                    "walltime_s": iteration_walltime,
                    "model_call_walltime_s": iteration_model_time,
                    "other_work_walltime_s": max(
                        0.0, iteration_walltime - iteration_model_time
                    ),
                    "model_call_roles": roles[role_start:],
                    "model_call_rows": model_call_rows[role_start:],
                    "model_call_lora_rows": model_call_lora_rows[role_start:],
                    "model_call_walltimes_s": iteration_call_times,
                    "accepted_length": int(accepted),
                    "stable_prefix_length": min(stable_prefix, max(accepted - 1, 0)),
                    "generated_tokens_before": int(generated_before),
                    "generated_tokens_after": int(total_generated),
                    "cache_length_before": int(cache_length),
                    "cache_length_after": int(cache_length + accepted),
                }
            )
        if record_trajectory:
            branch_names = ["clean"]
            branches = []
            for branch_index, branch_name in enumerate(branch_names):
                candidate_row = candidates[branch_index].detach().cpu().tolist()
                verifier_row = verifier_ids[branch_index].detach().cpu().tolist()
                matches = [
                    int(verifier_row[index]) == int(candidate_row[index + 1])
                    for index in range(block_length - 1)
                ]
                branches.append(
                    {
                        "name": branch_name,
                        "candidate_ids": candidate_row,
                        "candidate_sources": branch_sources[branch_index],
                        "verifier_ids": verifier_row,
                        "ltr_matches": matches,
                        "stable_prefix_length": int(
                            selection.stable_prefix_lengths[branch_index]
                        ),
                        "accepted_length": int(per_branch_accepted[branch_index]),
                        "selected": branch_index == winner,
                    }
                )
            sequence_after_ids = torch.cat(generated, dim=1)[0].detach().cpu().tolist()
            trajectory_trace.append(
                {
                    "iteration": len(trajectory_trace) + 1,
                    "proposal_start": len(sequence_before_ids) - 1,
                    "cache_generated_tokens_before": cache_length
                    - int(prompt_ids.shape[1]),
                    "cache_generated_tokens_after": cache_length
                    - int(prompt_ids.shape[1])
                    + accepted,
                    "sequence_before_ids": sequence_before_ids,
                    "sequence_after_ids": sequence_after_ids,
                    "seed_id": int(sequence_before_ids[-1]),
                    "branches": branches,
                    "winner": winner_name,
                    "tie": False,
                    "accepted_length": int(accepted),
                    "raw_selected_accepted_length": int(raw_accepted),
                    "accepted_token_ids": accepted_tokens[0].detach().cpu().tolist(),
                    "next_seed_id": int(next_token.item()),
                    "draft_rows": int(block_length),
                    "verify_rows": int(1 * block_length),
                    "draft_attention": "bidirectional",
                    "verify_attention": "causal",
                    "stopped_on_eos": bool(stopped_on_eos),
                }
            )
        if stopped_on_eos:
            break
    all_generated = torch.cat(generated, dim=1)[:, :max_new_tokens]
    output_ids = torch.cat((prompt_ids, all_generated), dim=1)
    return StrictLTRDecodeResult(
        output_ids=output_ids,
        nfe=nfe,
        prompt_prefill_rows=int(prompt_ids.numel()),
        uncached_rows_per_forward=rows_per_forward,
        model_call_roles=roles,
        model_call_rows=model_call_rows,
        model_call_walltimes_s=call_walltimes,
        prefill_walltime_s=prefill_walltime,
        iteration_records=iteration_records,
        accepted_lengths=accepted_lengths,
        stable_prefix_lengths=stable_prefix_lengths,
        branch_accepted_lengths=branch_accepted_lengths,
        branch_candidate_shared_prefix_lengths=branch_candidate_shared_prefix_lengths,
        branch_shared_accepted_prefix_lengths=branch_shared_accepted_prefix_lengths,
        winner_branches=winner_branches,
        branch_selection_counts=branch_counts,
        stopped_on_eos=stopped_on_eos,
        max_uncached_rows=max(rows_per_forward, default=0),
        block_length=block_length,
        trajectory_trace=trajectory_trace,
        model_call_lora_rows=model_call_lora_rows,
    )


def _dynamic_cache_class() -> type[Any]:
    try:
        from transformers import DynamicCache
    except ImportError as error:
        raise RuntimeError(
            "Nemotron decoding requires Transformers >=5.0 in its isolated environment"
        ) from error
    return DynamicCache


@torch.inference_mode()
def _strict_ltr_packed_or_quadratic_decode(
    model: Any,
    prompt_ids: torch.LongTensor,
    *,
    quadratic: bool,
    max_new_tokens: int,
    block_length: int,
    threshold: float,
    eos_token_id: int | None,
    max_uncached_rows_per_forward: int,
    record_model_call_walltimes: bool,
    record_trajectory: bool = False,
    use_draft_lora: bool = False,
) -> StrictLTRDecodeResult:
    """One-call packed or NVIDIA TiDAR-style quadratic self-speculation."""
    if (
        prompt_ids.ndim != 2
        or int(prompt_ids.shape[0]) != 1
        or prompt_ids.shape[1] == 0
    ):
        raise ValueError(
            "strict-LTR decoding requires a nonempty prompt and batch size one"
        )
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if block_length < 2:
        raise ValueError("block_length must be at least two")
    if threshold != 0.0:
        raise ValueError("packed and quadratic comparison requires threshold zero")
    semantic_rows = block_length * (block_length + 1) if quadratic else 2 * block_length
    if semantic_rows > max_uncached_rows_per_forward:
        raise ValueError(
            f"configured layout exceeds the uncached-row cap: rows={semantic_rows}, cap={max_uncached_rows_per_forward}"
        )
    if use_draft_lora:
        require_lora(model)
    device = prompt_ids.device
    prompt_length = int(prompt_ids.shape[1])
    token_mask_id = int(model.config.mask_token_id)
    if eos_token_id is None:
        eos_token_id = getattr(model.config, "eos_token_id", None)
    method_name = "quadratic" if quadratic else "packed"
    rows_per_forward: list[int] = []
    roles: list[str] = []
    model_call_rows: list[int] = []
    model_call_lora_rows: list[int] = []
    call_walltimes: list[float] = []
    accepted_lengths: list[int] = []
    stable_prefix_lengths: list[int] = []
    iteration_records: list[dict[str, Any]] = []
    trajectory_trace: list[dict[str, Any]] = []

    def timed_encoder_call(
        *,
        role: str,
        semantic_row_count: int,
        actual_row_count: int,
        mask: torch.BoolTensor,
        lora_row_mask: torch.BoolTensor | None = None,
        **kwargs: Any,
    ) -> Any:
        _record_call(
            rows=semantic_row_count,
            role=role,
            row_cap=max_uncached_rows_per_forward,
            rows_per_forward=rows_per_forward,
            roles=roles,
        )
        model_call_rows.append(int(actual_row_count))
        lora_rows = int(lora_row_mask.sum().item()) if lora_row_mask is not None else 0
        model_call_lora_rows.append(lora_rows)
        if record_model_call_walltimes:
            _sync(device)
            call_started = time.perf_counter()
        lora_context = (
            row_selective_lora(model, lora_row_mask)
            if lora_row_mask is not None
            else nullcontext()
        )
        with runtime_attention(model, mask.to(device)), lora_context:
            output = model.encoder(**kwargs)
        if record_model_call_walltimes:
            _sync(device)
            call_walltimes.append(float(time.perf_counter() - call_started))
        return output

    if record_model_call_walltimes:
        _sync(device)
        bootstrap_started = time.perf_counter()
    bootstrap_call_start = len(call_walltimes)
    bootstrap_role_start = len(roles)
    _set_diffusion_lm(model, False)
    _toggle_adapters(model, False)
    bootstrap_ids = torch.full(
        (1, prompt_length + block_length),
        token_mask_id,
        dtype=torch.long,
        device=device,
    )
    bootstrap_ids[:, :prompt_length] = prompt_ids
    bootstrap_lora_mask = None
    if use_draft_lora:
        bootstrap_lora_mask = torch.zeros(
            prompt_length + block_length, dtype=torch.bool, device=device
        )
        bootstrap_lora_mask[prompt_length:] = True
    bootstrap = timed_encoder_call(
        role=f"prompt_prefill_and_{method_name}_bootstrap",
        semantic_row_count=block_length,
        actual_row_count=prompt_length + block_length,
        mask=bootstrap_visibility(prompt_length, block_length),
        lora_row_mask=bootstrap_lora_mask,
        input_ids=bootstrap_ids,
        past_key_values=_dynamic_cache_class()(),
        use_cache=True,
    )
    bootstrap_logits = model.diffusion_head(bootstrap.last_hidden_state)
    first_token = _argmax_tokens(bootstrap_logits[:, prompt_length - 1 : prompt_length])
    current_draft = _argmax_tokens(bootstrap_logits[:, prompt_length:])
    current_draft[:, 0:1] = first_token
    past_key_values = bootstrap.past_key_values
    _crop_cache(past_key_values, prompt_length)
    if record_model_call_walltimes:
        _sync(device)
        bootstrap_walltime = float(time.perf_counter() - bootstrap_started)
    else:
        bootstrap_walltime = 0.0
    bootstrap_call_times = call_walltimes[bootstrap_call_start:]
    bootstrap_model_time = float(sum(bootstrap_call_times))
    iteration_records.append(
        {
            "iteration": 0,
            "phase": "bootstrap",
            "walltime_s": bootstrap_walltime,
            "model_call_walltime_s": bootstrap_model_time,
            "other_work_walltime_s": max(
                0.0, bootstrap_walltime - bootstrap_model_time
            ),
            "model_call_roles": roles[bootstrap_role_start:],
            "model_call_rows": model_call_rows[bootstrap_role_start:],
            "model_call_lora_rows": model_call_lora_rows[bootstrap_role_start:],
            "model_call_walltimes_s": bootstrap_call_times,
            "semantic_draft_rows": int(block_length),
            "root_conditioned": bool(False),
            "prompt_rows": int(prompt_length),
            "generated_tokens_before": 0,
            "generated_tokens_after": 0 if quadratic else 1,
            "cache_length_before": 0,
            "cache_length_after": int(prompt_length),
        }
    )
    nfe = 1
    stopped_on_eos = False
    if quadratic:
        generated: list[torch.LongTensor] = []
        total_generated = 0
    else:
        generated = [first_token]
        total_generated = 1
        if eos_token_id is not None and int(first_token.item()) == int(eos_token_id):
            stopped_on_eos = True
    while total_generated < max_new_tokens and (not stopped_on_eos):
        if record_model_call_walltimes:
            _sync(device)
            iteration_started = time.perf_counter()
        call_start = len(call_walltimes)
        role_start = len(roles)
        cache_length = int(past_key_values.get_seq_length())
        generated_before = total_generated
        draft_before = current_draft.detach().clone()
        if quadratic:
            roots = current_draft.reshape(1, block_length, 1)
            masks = torch.full(
                (1, block_length, block_length),
                token_mask_id,
                dtype=torch.long,
                device=device,
            )
            query_ids = torch.cat((roots, masks), dim=2).reshape(1, -1)
            position_ids = quadratic_position_ids(cache_length, block_length, device)
            visibility = quadratic_visibility(cache_length, block_length)
            role = "quadratic_verify_and_all_position_proposals"
            lora_row_mask = None
            if use_draft_lora:
                lora_row_mask = torch.ones(
                    block_length, block_length + 1, dtype=torch.bool, device=device
                )
                lora_row_mask[:, 0] = False
                lora_row_mask = lora_row_mask.reshape(-1)
        else:
            proposal_query = torch.full_like(current_draft, token_mask_id)
            proposal_query[:, 0:1] = current_draft[:, 0:1]
            query_ids = torch.cat(
                (current_draft[:, :block_length], proposal_query), dim=1
            )
            position_ids = packed_position_ids(cache_length, block_length, device)
            visibility = packed_visibility(cache_length, block_length)
            role = "packed_verify_and_next_proposal"
            lora_row_mask = None
            if use_draft_lora:
                lora_row_mask = torch.zeros(
                    block_length + block_length, dtype=torch.bool, device=device
                )
                lora_row_mask[block_length:] = True
        _set_diffusion_lm(model, False)
        _toggle_adapters(model, False)
        output = timed_encoder_call(
            role=role,
            semantic_row_count=int(query_ids.shape[1]),
            actual_row_count=int(query_ids.shape[1]),
            mask=visibility,
            lora_row_mask=lora_row_mask,
            input_ids=query_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        nfe += 1
        predictions = _argmax_tokens(model.diffusion_head(output.last_hidden_state))
        if quadratic:
            structured = predictions.reshape(1, block_length, block_length + 1)
            (raw_accepted, committed, next_draft) = select_quadratic_step(
                current_draft, structured
            )
            stable_prefix = raw_accepted - 1
            extract_quadratic_roots(past_key_values, cache_length, block_length)
            remaining = max_new_tokens - total_generated
            accepted = min(raw_accepted, remaining)
            accepted_tokens = committed[:, :accepted]
            _crop_cache(past_key_values, cache_length + accepted)
            current_draft = next_draft
        else:
            verifier_ids = predictions[:, :block_length]
            proposal_ids = predictions[:, block_length:]
            stable_prefix = 0
            for rank in range(block_length - 1):
                candidate_token = int(current_draft[0, rank + 1].item())
                if candidate_token == token_mask_id:
                    break
                if int(verifier_ids[0, rank].item()) != candidate_token:
                    break
                stable_prefix += 1
            raw_accepted = stable_prefix + 1
            remaining = max_new_tokens - total_generated
            accepted = min(raw_accepted, remaining)
            accepted_tokens = verifier_ids[:, :accepted]
            _crop_cache(past_key_values, cache_length + accepted)
            current_draft = build_next_packed_candidate(
                verifier_ids,
                proposal_ids,
                accepted_length=accepted,
                block_length=block_length,
                mask_token_id=token_mask_id,
            )
            max(1, block_length - accepted)
        eos_offset: int | None = None
        if eos_token_id is not None:
            eos_positions = (accepted_tokens[0] == int(eos_token_id)).nonzero(
                as_tuple=True
            )[0]
            if int(eos_positions.numel()) > 0:
                eos_offset = int(eos_positions[0].item())
                accepted_tokens = accepted_tokens[:, : eos_offset + 1]
                accepted = eos_offset + 1
                _crop_cache(past_key_values, cache_length + accepted)
                stopped_on_eos = True
        generated.append(accepted_tokens)
        total_generated += accepted
        accepted_lengths.append(int(accepted))
        stable_prefix_lengths.append(min(stable_prefix, max(accepted - 1, 0)))
        if record_model_call_walltimes:
            _sync(device)
            iteration_walltime = float(time.perf_counter() - iteration_started)
        else:
            iteration_walltime = 0.0
        iteration_call_times = call_walltimes[call_start:]
        iteration_model_time = float(sum(iteration_call_times))
        iteration_records.append(
            {
                "iteration": len(accepted_lengths),
                "phase": "decode",
                "walltime_s": iteration_walltime,
                "model_call_walltime_s": iteration_model_time,
                "other_work_walltime_s": max(
                    0.0, iteration_walltime - iteration_model_time
                ),
                "model_call_roles": roles[role_start:],
                "model_call_rows": model_call_rows[role_start:],
                "model_call_lora_rows": model_call_lora_rows[role_start:],
                "model_call_walltimes_s": iteration_call_times,
                "accepted_length": int(accepted),
                "raw_accepted_length": int(raw_accepted),
                "stable_prefix_length": min(stable_prefix, max(accepted - 1, 0)),
                "generated_tokens_before": int(generated_before),
                "generated_tokens_after": int(total_generated),
                "cache_length_before": int(cache_length),
                "cache_length_after": int(cache_length + accepted),
                "proposal_width": int(block_length),
                "verifier_rows": int(block_length if quadratic else block_length),
                "unused_verifier_rows_omitted": int(
                    0 if quadratic else block_length - block_length
                ),
                "unresolved_input_masks": int(
                    (draft_before == token_mask_id).sum().item()
                ),
            }
        )
        if record_trajectory:
            trajectory_trace.append(
                {
                    "iteration": len(accepted_lengths),
                    "method": method_name,
                    "candidate_ids": draft_before[0].detach().cpu().tolist(),
                    "accepted_token_ids": accepted_tokens[0].detach().cpu().tolist(),
                    "accepted_length": int(accepted),
                    "raw_accepted_length": int(raw_accepted),
                    "stable_prefix_length": int(stable_prefix),
                    "next_candidate_ids": current_draft[0].detach().cpu().tolist(),
                    "cache_length_before": int(cache_length),
                    "cache_length_after": int(cache_length + accepted),
                    "query_rows": int(query_ids.shape[1]),
                    "stopped_on_eos": bool(stopped_on_eos),
                }
            )
    all_generated = (
        torch.cat(generated, dim=1)[:, :max_new_tokens]
        if generated
        else torch.empty((1, 0), dtype=torch.long, device=device)
    )
    return StrictLTRDecodeResult(
        output_ids=torch.cat((prompt_ids, all_generated), dim=1),
        nfe=nfe,
        prompt_prefill_rows=prompt_length,
        uncached_rows_per_forward=rows_per_forward,
        model_call_roles=roles,
        model_call_rows=model_call_rows,
        model_call_walltimes_s=call_walltimes,
        prefill_walltime_s=bootstrap_walltime,
        iteration_records=iteration_records,
        accepted_lengths=accepted_lengths,
        stable_prefix_lengths=stable_prefix_lengths,
        branch_accepted_lengths=[[value] for value in accepted_lengths],
        branch_candidate_shared_prefix_lengths=[block_length] * len(accepted_lengths),
        branch_shared_accepted_prefix_lengths=accepted_lengths.copy(),
        winner_branches=[method_name] * len(accepted_lengths),
        branch_selection_counts={method_name: len(accepted_lengths)},
        stopped_on_eos=stopped_on_eos,
        max_uncached_rows=max(rows_per_forward, default=0),
        block_length=block_length,
        trajectory_trace=trajectory_trace,
        model_call_lora_rows=model_call_lora_rows,
    )


def strict_ltr_self_speculative(
    model: Any, prompt_ids: torch.LongTensor, **kwargs: Any
) -> StrictLTRDecodeResult:
    """Linear self-speculation with causal verification."""
    return _strict_ltr_decode(model, prompt_ids, **kwargs)


def strict_ltr_packed_self_speculative(
    model: Any, prompt_ids: torch.LongTensor, **kwargs: Any
) -> StrictLTRDecodeResult:
    """O(B)-row pipelined verifier/proposal with strict AR-only commitment."""
    return _strict_ltr_packed_or_quadratic_decode(
        model, prompt_ids, quadratic=False, **kwargs
    )


def strict_ltr_quadratic_self_speculative(
    model: Any, prompt_ids: torch.LongTensor, **kwargs: Any
) -> StrictLTRDecodeResult:
    """NVIDIA TiDAR B(B+1)-row all-acceptance-position self-speculation."""
    return _strict_ltr_packed_or_quadratic_decode(
        model, prompt_ids, quadratic=True, **kwargs
    )


def _identity_cache(cache, repeats):
    return cache
