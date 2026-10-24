from __future__ import annotations
import sys
from contextlib import contextmanager
from types import MethodType
from typing import Any, Iterator
import torch
import torch.nn.functional as F


def bootstrap_visibility(prompt_length: int, block_length: int) -> torch.BoolTensor:
    """Causal prompt plus a fully bidirectional draft appended in one call."""
    if prompt_length <= 0 or block_length <= 0:
        raise ValueError("prompt and block lengths must be positive")
    total = prompt_length + block_length
    q = torch.arange(total)[:, None]
    kv = torch.arange(total)[None, :]
    prompt_causal = (q < prompt_length) & (kv <= q)
    draft_full = q >= prompt_length
    return prompt_causal | draft_full


def packed_visibility(prefix_length: int, block_length: int) -> torch.BoolTensor:
    """Causal verifier rows see the prefix and earlier verifier rows. Proposal rows see the prefix and proposal lane."""
    if prefix_length <= 0 or block_length <= 0:
        raise ValueError("prefix and block lengths must be positive")
    query_length = block_length + block_length
    key_length = prefix_length + query_length
    mask = torch.zeros((query_length, key_length), dtype=torch.bool)
    mask[:, :prefix_length] = True
    verifier = torch.tril(torch.ones((block_length, block_length), dtype=torch.bool))
    proposal_start = prefix_length + block_length
    mask[:block_length, prefix_length:proposal_start] = verifier
    mask[block_length:, proposal_start:] = True
    return mask


def quadratic_visibility(prefix_length: int, block_length: int) -> torch.BoolTensor:
    """NVIDIA TiDAR visibility in physical ``[prefix | expanded groups]`` order.

    Each of B groups contains ``[causal root | B diffusion queries]``.  Every
    expanded query sees the prefix and causal roots through its group.  The B
    diffusion queries additionally see their complete group bidirectionally.
    """
    if prefix_length <= 0 or block_length <= 0:
        raise ValueError("prefix and block lengths must be positive")
    group_width = block_length + 1
    query_length = block_length * group_width
    key_length = prefix_length + query_length
    mask = torch.zeros((query_length, key_length), dtype=torch.bool)
    mask[:, :prefix_length] = True
    for query in range(query_length):
        group = query // group_width
        offset = query % group_width
        for earlier_group in range(group + 1):
            mask[query, prefix_length + earlier_group * group_width] = True
        if offset != 0:
            start = prefix_length + group * group_width
            mask[query, start : start + group_width] = True
    return mask


def packed_position_ids(
    prefix_length: int, block_length: int, device: torch.device
) -> torch.LongTensor:
    proposal_positions = torch.arange(
        prefix_length, prefix_length + block_length, device=device, dtype=torch.long
    )
    return torch.cat(
        (proposal_positions[:block_length], proposal_positions), dim=0
    ).unsqueeze(0)


def quadratic_position_ids(
    prefix_length: int, block_length: int, device: torch.device
) -> torch.LongTensor:
    base = torch.arange(
        prefix_length, prefix_length + block_length + 1, device=device, dtype=torch.long
    )
    offsets = torch.arange(block_length, device=device, dtype=torch.long)[:, None]
    return (base[None, :] + offsets).reshape(1, -1)


def build_next_packed_candidate(
    verifier_ids: torch.LongTensor,
    proposal_ids: torch.LongTensor,
    *,
    accepted_length: int,
    block_length: int,
    mask_token_id: int,
) -> torch.LongTensor:
    """Shift a pipelined proposal after committing AR-certified verifier rows."""
    if accepted_length < 1 or accepted_length > block_length:
        raise ValueError("accepted_length must lie in [1, block_length]")
    if proposal_ids.shape != (1, block_length):
        raise ValueError("proposal must have shape [1, block_length]")
    if (
        verifier_ids.ndim != 2
        or int(verifier_ids.shape[0]) != 1
        or int(verifier_ids.shape[1]) < accepted_length
        or (int(verifier_ids.shape[1]) > block_length)
    ):
        raise ValueError(
            "verifier must have shape [1, V] with accepted_length <= V <= block_length"
        )
    next_seed = verifier_ids[:, accepted_length - 1 : accepted_length]
    overlap = proposal_ids[:, accepted_length + 1 :]
    tail_length = block_length - int(next_seed.shape[1]) - int(overlap.shape[1])
    tail = torch.full(
        (1, tail_length),
        int(mask_token_id),
        dtype=torch.long,
        device=verifier_ids.device,
    )
    return torch.cat((next_seed, overlap, tail), dim=1)


def select_quadratic_step(
    current_draft: torch.LongTensor, predictions: torch.LongTensor
) -> tuple[int, torch.LongTensor, torch.LongTensor]:
    """Apply NVIDIA's strict-LTR quadratic acceptance/group selection rule."""
    if current_draft.ndim != 2 or int(current_draft.shape[0]) != 1:
        raise ValueError("current_draft must have shape [1, B]")
    block_length = int(current_draft.shape[1])
    if predictions.shape != (1, block_length, block_length + 1):
        raise ValueError("predictions must have shape [1, B, B+1]")
    accepted = 1
    next_draft = predictions[:, 0, 1:].clone()
    while accepted < block_length:
        verifier = int(predictions[0, accepted - 1, 0].item())
        candidate = int(current_draft[0, accepted].item())
        if verifier != candidate:
            break
        next_draft = predictions[:, accepted, 1:].clone()
        accepted += 1
    correction = predictions[:, accepted - 1, 0:1]
    next_draft[:, 0:1] = correction
    committed = current_draft[:, :accepted]
    return (accepted, committed, next_draft)


def extract_quadratic_roots(cache: Any, prefix_length: int, block_length: int) -> None:
    """Keep prefix K/V plus the first (causal) row of every expanded group."""
    group_width = block_length + 1
    layers = getattr(cache, "layers", None)
    if layers is None:
        raise TypeError("quadratic cache extraction requires a DynamicCache")
    for layer in layers:
        keys = layer.keys
        values = layer.values
        roots_k = keys[:, :, prefix_length::group_width, :][:, :, :block_length, :]
        roots_v = values[:, :, prefix_length::group_width, :][:, :, :block_length, :]
        layer.keys = torch.cat((keys[:, :, :prefix_length, :], roots_k), dim=2)
        layer.values = torch.cat((values[:, :, :prefix_length, :], roots_v), dim=2)


def _runtime_attention_forward(
    attention: Any,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Any = None,
    cache_position: torch.LongTensor | None = None,
    use_cache: bool = False,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    mask = getattr(attention, "_nemotron_runtime_mask", None)
    if mask is None:
        return attention._nemotron_original_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=use_cache,
            **kwargs,
        )
    module = sys.modules[type(attention).__module__]
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, attention.head_dim)
    query_states = attention.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = attention.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = attention.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    (cos, sin) = position_embeddings
    (query_states, key_states) = module.apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )
    query_states = query_states * module._get_llama_4_attn_scale(
        cache_position,
        attention.config.rope_parameters.get("llama_4_scaling_beta"),
        attention.config.rope_parameters.get("original_max_position_embeddings"),
    ).to(query_states.dtype)
    if past_key_values is not None:
        if use_cache:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            (key_states, value_states) = past_key_values.update(
                key_states, value_states, attention.layer_idx, cache_kwargs
            )
        else:
            layer_cache = past_key_values.layers[attention.layer_idx]
            key_states = torch.cat((layer_cache.keys, key_states), dim=-2)
            value_states = torch.cat((layer_cache.values, value_states), dim=-2)
    key_states = module.repeat_kv(key_states, attention.num_key_value_groups)
    value_states = module.repeat_kv(value_states, attention.num_key_value_groups)
    if tuple(mask.shape) != (int(query_states.shape[-2]), int(key_states.shape[-2])):
        raise AssertionError(
            f"runtime mask shape does not match attention tensors: mask={tuple(mask.shape)}, q={query_states.shape[-2]}, kv={key_states.shape[-2]}"
        )
    attn_output = F.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=mask.to(query_states.device),
        dropout_p=0.0,
        is_causal=False,
        scale=attention.scaling,
    )
    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    return (attention.o_proj(attn_output), None)


def install_runtime_attention(model: Any) -> None:
    """Install the inert wrapper once; ordinary calls retain native behavior."""
    for layer in model.encoder.layers:
        attention = layer.self_attn
        if hasattr(attention, "_nemotron_original_forward"):
            continue
        attention._nemotron_original_forward = attention.forward
        attention._nemotron_runtime_mask = None
        attention.forward = MethodType(_runtime_attention_forward, attention)


@contextmanager
def runtime_attention(model: Any, mask: torch.BoolTensor) -> Iterator[None]:
    """Enable one custom visibility matrix for all attention layers."""
    install_runtime_attention(model)
    for layer in model.encoder.layers:
        layer.self_attn._nemotron_runtime_mask = mask
    try:
        yield
    finally:
        for layer in model.encoder.layers:
            layer.self_attn._nemotron_runtime_mask = None
