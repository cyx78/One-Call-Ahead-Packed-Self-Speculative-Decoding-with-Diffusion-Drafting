"SetDLM self-speculation with cached semantic positions."

from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import Any, Literal
import torch
from setdlm.model.base import DenoiserInput

OrderPolicy = Literal["ltr", "confidence"]
ParallelRefreshExecutionMode = Literal["packed", "quadratic"]
DEFAULT_MAX_UNCACHED_ROWS_PER_FORWARD = 128


@dataclass
class DecodeResult:
    """Output and measurements from one cached sliding-window decode."""

    target_ids: torch.LongTensor
    decode_nfe: int
    setup_forward_calls: int
    latency_s: float
    peak_memory_gib: float
    peak_reserved_memory_gib: float
    generated_tokens_per_step: list[int]
    trace: list[dict[str, Any]]
    converged: bool
    canvas_capacity: int
    termination_position: int | None = None
    termination_token_id: int | None = None
    termination_detected_at_nfe: int | None = None
    branch_selection_counts: dict[str, int] = field(default_factory=dict)
    uncached_rows_per_forward: list[int] = field(default_factory=list)
    uncached_row_components_per_forward: list[dict[str, int]] = field(
        default_factory=list
    )
    model_call_walltimes_s: list[float] = field(default_factory=list)
    model_call_roles: list[str] = field(default_factory=list)
    uncached_row_cap: int = DEFAULT_MAX_UNCACHED_ROWS_PER_FORWARD
    decoder_diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def total_forward_calls(self) -> int:
        return self.setup_forward_calls + self.decode_nfe

    @property
    def parallelism_factor(self) -> float:
        if not self.generated_tokens_per_step:
            return 0.0
        return float(
            sum(self.generated_tokens_per_step) / len(self.generated_tokens_per_step)
        )

    @property
    def max_uncached_rows(self) -> int:
        """Maximum realized uncached rows in any decode forward."""
        return max(self.uncached_rows_per_forward, default=0)


def _record_uncached_forward_rows(
    packed_ids: torch.LongTensor,
    *,
    components: dict[str, int],
    row_cap: int,
    rows_per_forward: list[int],
    components_per_forward: list[dict[str, int]],
) -> int:
    """Assert and record the realized width of one packed decode forward.

    ``packed_ids.numel()`` is the runtime count sent to the backbone, while
    ``components`` independently describes the pending materializations and
    packed lanes that produced it.  The prompt-cache setup forward does not
    call this helper and is therefore excluded from the decode-row cap.
    """
    if packed_ids.ndim != 2 or int(packed_ids.shape[0]) <= 0:
        raise ValueError("decode inputs must have a positive batch dimension")
    if row_cap <= 0:
        raise ValueError("uncached decode row cap must be positive")
    normalized_components = {
        str(name): int(count) for (name, count) in components.items()
    }
    if not normalized_components:
        raise ValueError("uncached row components must not be empty")
    if any((count < 0 for count in normalized_components.values())):
        raise ValueError("uncached row component counts must be nonnegative")
    realized_rows = int(packed_ids.numel())
    component_rows = sum(normalized_components.values())
    if component_rows != realized_rows:
        raise RuntimeError(
            f"Uncached row components do not match the realized packed input: components={component_rows}, realized={realized_rows}."
        )
    if realized_rows > int(row_cap):
        raise AssertionError(
            f"Decode forward exceeds the uncached-row cap: realized={realized_rows}, cap={int(row_cap)}, components={normalized_components}."
        )
    rows_per_forward.append(realized_rows)
    components_per_forward.append(normalized_components)
    return realized_rows


def _validate_uncached_row_diagnostics(
    *,
    decode_nfe: int,
    rows_per_forward: list[int],
    components_per_forward: list[dict[str, int]],
) -> None:
    """Require one realized row record for every completed decode forward."""
    if not len(rows_per_forward) == len(components_per_forward) == int(decode_nfe):
        raise RuntimeError(
            "Uncached-row diagnostics must contain exactly one record per decode forward."
        )


def stable_permutation_prefix_length(
    old_ids: torch.LongTensor, new_ids: torch.LongTensor, *, mask_token_id: int
) -> int:
    """Return the length of the unchanged non-mask prefix in permutation order."""
    if old_ids.ndim != 1 or new_ids.ndim != 1:
        raise ValueError("old_ids and new_ids must be one-dimensional")
    if old_ids.shape != new_ids.shape:
        raise ValueError("old_ids and new_ids must have the same shape")
    agreed = (old_ids == new_ids) & (old_ids != int(mask_token_id))
    if bool(agreed.all()):
        return int(agreed.numel())
    return int((~agreed).nonzero(as_tuple=False)[0].item())


def build_parallel_refresh_attention_mask(
    *, cache_len: int, pending_count: int, old_ids: torch.LongTensor, mask_token_id: int
) -> torch.BoolTensor:
    """Build the three-lane mask used by parallel clean refresh.

    Columns and rows are
    ``[persistent cache | pending | old | verifier MASK | clean MASK]``.
    ``pending`` contains an already accepted verifier correction whose token
    KV is materialized by this forward. Verifier queries see pending tokens
    and the known old-sketch prefix. Clean queries see persistent and pending
    accepted tokens, but no old-sketch token and no sibling query.
    """
    if old_ids.ndim != 1:
        raise ValueError("old_ids must be one-dimensional")
    if cache_len < 0 or pending_count < 0:
        raise ValueError("cache_len and pending_count must be nonnegative")
    device = old_ids.device
    active_len = int(old_ids.numel())
    local_len = pending_count + 3 * active_len
    visible = torch.zeros(
        (local_len, cache_len + local_len), dtype=torch.bool, device=device
    )
    if cache_len:
        visible[:, :cache_len] = True
    pending_start = cache_len
    old_start = pending_start + pending_count
    verifier_start = old_start + active_len
    clean_start = verifier_start + active_len
    for rank in range(pending_count):
        visible[rank, pending_start : pending_start + rank + 1] = True
    known = old_ids != int(mask_token_id)
    local_offsets = torch.arange(active_len, device=device)
    for rank in range(active_len):
        old_row = pending_count + rank
        verifier_row = pending_count + active_len + rank
        clean_row = pending_count + 2 * active_len + rank
        if pending_count:
            visible[old_row : clean_row + 1 : active_len, pending_start:old_start] = (
                True
            )
        visible[old_row, old_start : old_start + active_len] = known & (
            local_offsets <= rank
        )
        visible[old_row, old_start + rank] = True
        visible[verifier_row, old_start : old_start + active_len] = known & (
            local_offsets < rank
        )
        visible[verifier_row, verifier_start + rank] = True
        visible[clean_row, clean_start + rank] = True
    return visible


def build_quadratic_parallel_refresh_attention_mask(
    *, cache_len: int, pending_count: int, old_ids: torch.LongTensor, mask_token_id: int
) -> torch.BoolTensor:
    """Build SetDLM Quadratic-SS visibility.

    Rows and columns are ``[cache | pending | old | verifier | proposals]``.
    The final region contains ``m`` isolated proposal lanes of length ``m``.
    Proposal lane ``g`` represents the state in which exactly the first ``g``
    old tokens matched verification.  It may see those matched old rows, but
    not the verifier correction at rank ``g`` (which is unknown until after
    the forward), other proposal lanes, or sibling queries in its own lane.

    SetDLM predicts a token at the MASK row for that semantic position, so the
    shared old lane cannot double as the verifier as it can in next-token
    TiDAR/Nemotron.  The resulting local width is therefore ``p + 2m + m^2``.
    """
    if old_ids.ndim != 1:
        raise ValueError("old_ids must be one-dimensional")
    if cache_len < 0 or pending_count < 0:
        raise ValueError("cache_len and pending_count must be nonnegative")
    device = old_ids.device
    active_len = int(old_ids.numel())
    if active_len <= 0:
        raise ValueError("quadratic active lane must be nonempty")
    proposal_rows = active_len * active_len
    local_len = pending_count + 2 * active_len + proposal_rows
    visible = torch.zeros(
        (local_len, cache_len + local_len), dtype=torch.bool, device=device
    )
    if cache_len:
        visible[:, :cache_len] = True
    pending_start = cache_len
    old_start = pending_start + pending_count
    verifier_start = old_start + active_len
    proposal_start = verifier_start + active_len
    for rank in range(pending_count):
        visible[rank, pending_start : pending_start + rank + 1] = True
    known = old_ids != int(mask_token_id)
    offsets = torch.arange(active_len, device=device)
    for rank in range(active_len):
        old_row = pending_count + rank
        verifier_row = pending_count + active_len + rank
        if pending_count:
            visible[old_row, pending_start:old_start] = True
            visible[verifier_row, pending_start:old_start] = True
        visible[old_row, old_start:verifier_start] = known & (offsets <= rank)
        visible[old_row, old_start + rank] = True
        visible[verifier_row, old_start:verifier_start] = known & (offsets < rank)
        visible[verifier_row, verifier_start + rank] = True
    for matching_prefix in range(active_len):
        matched_context = known & (offsets < matching_prefix)
        lane_start = proposal_start + matching_prefix * active_len
        for rank in range(active_len):
            proposal_row = pending_count + 2 * active_len
            proposal_row += matching_prefix * active_len + rank
            if pending_count:
                visible[proposal_row, pending_start:old_start] = True
            visible[proposal_row, old_start:verifier_start] = matched_context
            visible[proposal_row, lane_start + rank] = True
    return visible


def build_multi_branch_draft_attention_mask(
    *, cache_len: int, pending_count: int, clean_length: int, device: torch.device
) -> torch.BoolTensor:
    """Build split ``[pending | clean-query]`` attention visibility.

    Pending accepted tokens are materialized causally.  Each isolated clean
    query sees persistent cache, all pending tokens, and only its own MASK.
    """
    if cache_len < 0 or pending_count < 0 or clean_length <= 0:
        raise ValueError("split draft dimensions are invalid")
    local_len = pending_count + clean_length
    visible = torch.zeros(
        (local_len, cache_len + local_len), dtype=torch.bool, device=device
    )
    if cache_len:
        visible[:, :cache_len] = True
    pending_start = cache_len
    clean_start = pending_start + pending_count
    for rank in range(pending_count):
        visible[rank, pending_start : pending_start + rank + 1] = True
    for rank in range(clean_length):
        row = pending_count + rank
        if pending_count:
            visible[row, pending_start:clean_start] = True
        visible[row, clean_start + rank] = True
    return visible


def build_multi_branch_verification_attention_mask(
    *, cache_len: int, candidate_ids: torch.LongTensor, mask_token_id: int
) -> torch.BoolTensor:
    """Build split isolated ``[(candidate, verifier) * B]`` visibility.

    The cache already contains any pending accepted tokens materialized by
    the preceding draft call.  Each branch is otherwise identical to its
    pair in :func:`build_multi_branch_attention_mask`.
    """
    if candidate_ids.ndim != 2:
        raise ValueError("candidate_ids must have shape [branches, active]")
    if cache_len < 0:
        raise ValueError("cache_len must be nonnegative")
    (branch_count, active_len) = candidate_ids.shape
    if branch_count <= 0 or active_len <= 0:
        raise ValueError("candidate_ids must contain at least one token")
    device = candidate_ids.device
    local_len = 2 * branch_count * active_len
    visible = torch.zeros(
        (local_len, cache_len + local_len), dtype=torch.bool, device=device
    )
    if cache_len:
        visible[:, :cache_len] = True
    offsets = torch.arange(active_len, device=device)
    for branch_index in range(branch_count):
        old_start = cache_len + 2 * branch_index * active_len
        verifier_start = old_start + active_len
        known = candidate_ids[branch_index] != int(mask_token_id)
        for rank in range(active_len):
            old_row = 2 * branch_index * active_len + rank
            verifier_row = old_row + active_len
            visible[old_row, old_start : old_start + active_len] = known & (
                offsets <= rank
            )
            visible[old_row, old_start + rank] = True
            visible[verifier_row, old_start : old_start + active_len] = known & (
                offsets < rank
            )
            visible[verifier_row, verifier_start + rank] = True
    return visible


def select_multi_branch_step(
    candidate_ids: torch.LongTensor,
    verifier_ids: torch.LongTensor,
    *,
    mask_token_id: int,
) -> tuple[int, list[int], list[int]]:
    """Select the first branch with maximum stable-prefix acceptance."""
    if candidate_ids.ndim != 2 or verifier_ids.ndim != 2:
        raise ValueError("candidate and verifier IDs must be two-dimensional")
    if candidate_ids.shape != verifier_ids.shape:
        raise ValueError("candidate and verifier IDs must have equal shapes")
    active_len = int(candidate_ids.shape[1])
    prefixes = [
        stable_permutation_prefix_length(
            candidate_ids[index], verifier_ids[index], mask_token_id=mask_token_id
        )
        for index in range(int(candidate_ids.shape[0]))
    ]
    accepted = [min(active_len, prefix + 1) for prefix in prefixes]
    winner = max(range(len(accepted)), key=accepted.__getitem__)
    return (winner, prefixes, accepted)


def select_parallel_refresh_step(
    old_ids: torch.LongTensor,
    verifier_ids: torch.LongTensor,
    clean_ids: torch.LongTensor,
    *,
    mask_token_id: int,
) -> tuple[int, int, bool, torch.LongTensor, list[str]]:
    """Accept the matching prefix and one correction, then refresh the remaining draft."""
    if old_ids.shape != verifier_ids.shape or old_ids.shape != clean_ids.shape:
        raise ValueError("old, verifier, and clean IDs must have equal shapes")
    if old_ids.ndim != 1:
        raise ValueError("proposal IDs must be one-dimensional")
    matching_prefix = stable_permutation_prefix_length(
        old_ids, verifier_ids, mask_token_id=mask_token_id
    )
    active_len = int(old_ids.numel())
    accepted_count = min(active_len, matching_prefix + 1)
    selected_ids = clean_ids.clone()
    if accepted_count:
        selected_ids[:accepted_count] = verifier_ids[:accepted_count]
    selected_sources = [
        "verifier" if rank < accepted_count else "clean" for rank in range(active_len)
    ]
    return (matching_prefix, accepted_count, True, selected_ids, selected_sources)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_memory_gib(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / 1073741824)


def _peak_reserved_memory_gib(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_reserved(device) / 1073741824)


def _cache_length(past_key_values: Any) -> int:
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    if len(past_key_values) == 0:
        return 0
    return int(past_key_values[0][0].shape[-2])


def _make_semantic_cache_room(
    model: Any,
    past_key_values: Any,
    semantic_position_ids: torch.LongTensor,
    *,
    local_length: int,
) -> tuple[Any, torch.LongTensor, list[int]]:
    """Keep the newest semantic cache entries within model context capacity."""
    if semantic_position_ids.ndim != 1:
        raise ValueError("semantic_position_ids must be one-dimensional")
    if local_length <= 0:
        raise ValueError("local_length must be positive")
    capacity = int(model.config.length)
    max_cache_length = capacity - int(local_length)
    if max_cache_length <= 0:
        raise ValueError("Packed local lanes leave no room for persistent KV context.")
    cache_length = _cache_length(past_key_values)
    if cache_length != int(semantic_position_ids.numel()):
        raise RuntimeError("KV cache and semantic-position ledger disagree.")
    if cache_length <= max_cache_length:
        return (past_key_values, semantic_position_ids, [])
    semantic_order = semantic_position_ids.argsort(stable=True)
    keep_indices = semantic_order[-max_cache_length:].sort().values
    keep_mask = torch.zeros(
        cache_length, dtype=torch.bool, device=semantic_position_ids.device
    )
    keep_mask[keep_indices] = True
    evicted = semantic_position_ids[~keep_mask].detach().cpu().tolist()
    past_key_values = model._select_kv_cache_positions(past_key_values, keep_indices)
    semantic_position_ids = semantic_position_ids.index_select(0, keep_indices)
    return (
        past_key_values,
        semantic_position_ids,
        [int(position) for position in evicted],
    )


def _clear_evicted_target_cache_flags(
    persistent_cached: torch.BoolTensor,
    evicted_semantic_positions: list[int],
    *,
    prompt_length: int,
) -> None:
    target_positions = [
        position - prompt_length
        for position in evicted_semantic_positions
        if prompt_length <= position < prompt_length + int(persistent_cached.numel())
    ]
    if target_positions:
        persistent_cached[
            torch.tensor(
                target_positions, dtype=torch.long, device=persistent_cached.device
            )
        ] = False


def earliest_verified_stop_sequence(
    token_ids: torch.LongTensor,
    verified: torch.BoolTensor,
    *,
    stop_sequences: tuple[tuple[int, ...], ...],
    current_stop_position: int | None = None,
) -> tuple[int | None, int | None]:
    """Return the earliest verified stop-sequence end position."""
    if token_ids.ndim != 1 or verified.ndim != 1:
        raise ValueError("token_ids and verified must be one-dimensional")
    if token_ids.shape != verified.shape:
        raise ValueError("token_ids and verified must have the same shape")
    if not stop_sequences:
        if current_stop_position is None:
            return (None, None)
        return (current_stop_position, int(token_ids[current_stop_position].item()))
    discovered_positions: list[int] = []
    target_length = int(token_ids.numel())
    for sequence in stop_sequences:
        if not sequence:
            raise ValueError("stop sequences must be nonempty")
        sequence_length = len(sequence)
        candidate_count = target_length - sequence_length + 1
        if candidate_count <= 0:
            continue
        matches = torch.ones(candidate_count, dtype=torch.bool, device=token_ids.device)
        for offset, token_id in enumerate(sequence):
            matches &= token_ids[offset : offset + candidate_count] == int(token_id)
            matches &= verified[offset : offset + candidate_count]
        starts = matches.nonzero(as_tuple=False).flatten()
        if starts.numel():
            discovered_positions.append(int(starts.min().item()) + sequence_length - 1)
    discovered_position = min(discovered_positions) if discovered_positions else None
    if current_stop_position is None:
        resolved_position = discovered_position
    elif discovered_position is None:
        resolved_position = current_stop_position
    else:
        resolved_position = min(current_stop_position, discovered_position)
    if resolved_position is None:
        return (None, None)
    return (resolved_position, int(token_ids[resolved_position].item()))


def _crop_cache_after_semantic_position(
    model: Any,
    past_key_values: Any,
    semantic_position_ids: torch.LongTensor,
    *,
    final_semantic_position: int,
) -> tuple[Any, torch.LongTensor, list[int]]:
    """Drop cached future tokens after an accepted response terminator."""
    keep_mask = semantic_position_ids <= int(final_semantic_position)
    if bool(keep_mask.all()):
        return (past_key_values, semantic_position_ids, [])
    keep_indices = keep_mask.nonzero(as_tuple=False).flatten()
    evicted = semantic_position_ids[~keep_mask].detach().cpu().tolist()
    past_key_values = model._select_kv_cache_positions(past_key_values, keep_indices)
    semantic_position_ids = semantic_position_ids.index_select(0, keep_indices)
    return (
        past_key_values,
        semantic_position_ids,
        [int(position) for position in evicted],
    )


def _truncate_target_state_after_stop(
    *,
    model: Any,
    past_key_values: Any,
    cache_semantic_positions: torch.LongTensor,
    canvas: torch.LongTensor,
    confidence: torch.FloatTensor,
    verified: torch.BoolTensor,
    persistent_cached: torch.BoolTensor,
    prompt_length: int,
    stop_position: int,
) -> tuple[Any, torch.LongTensor, list[int]]:
    """Remove state that must not condition the retained response prefix."""
    if stop_position + 1 < int(canvas.numel()):
        canvas[stop_position + 1 :] = int(model.mask_token_id)
        confidence[stop_position + 1 :] = float("nan")
        verified[stop_position + 1 :] = False
        persistent_cached[stop_position + 1 :] = False
    return _crop_cache_after_semantic_position(
        model,
        past_key_values,
        cache_semantic_positions,
        final_semantic_position=prompt_length + stop_position,
    )


def _rank_active_positions(
    *,
    policy: OrderPolicy,
    active_positions: list[int],
    canvas: torch.LongTensor,
    confidence: torch.FloatTensor,
    mask_token_id: int,
    window_left: int,
) -> tuple[list[int], dict[int, float], dict[int, float]]:
    if policy == "ltr":
        return (sorted(active_positions), {}, {})
    if policy != "confidence":
        raise ValueError(f"Unknown order policy: {policy}")
    known = [
        position
        for position in active_positions
        if int(canvas[position].item()) != int(mask_token_id)
        and math.isfinite(float(confidence[position].item()))
    ]
    unknown = sorted(set(active_positions) - set(known))
    if not known:
        return (sorted(active_positions), {}, {})
    prior_by_position: dict[int, float] = {}
    score_by_position = {
        position: math.log(max(float(confidence[position].item()), 1e-12)) + 0.0
        for position in known
    }
    ranked_known = sorted(
        known, key=lambda position: (-score_by_position[position], position)
    )
    return (ranked_known + unknown, score_by_position, prior_by_position)


def sliding_window_active_positions(
    verified: torch.BoolTensor, *, window_size: int
) -> tuple[int, int, list[int]]:
    """Return the unresolved positions in the next semantic window.

    The left edge is the first unresolved output position. The right edge is
    recomputed after every promotion, so newly reachable positions enter the
    active set immediately instead of waiting for a fixed block to finish.
    """
    if verified.ndim != 1:
        raise ValueError("verified must be one-dimensional")
    if verified.dtype != torch.bool:
        raise TypeError("verified must be a boolean tensor")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    unresolved = (~verified).nonzero(as_tuple=False).flatten()
    target_length = int(verified.numel())
    if unresolved.numel() == 0:
        return (target_length, target_length, [])
    window_left = int(unresolved[0].item())
    window_right = min(window_left + int(window_size), target_length)
    active = unresolved[(unresolved >= window_left) & (unresolved < window_right)]
    return (
        window_left,
        window_right,
        [int(position) for position in active.detach().cpu().tolist()],
    )


@torch.inference_mode()
def packed_or_quadratic_decode(
    model: Any,
    prompt_ids: torch.LongTensor,
    *,
    target_length: int = 256,
    window_size: int = 16,
    order_policy: OrderPolicy | None = None,
    execution_mode: ParallelRefreshExecutionMode = "packed",
    record_model_call_walltimes: bool = False,
    record_trace: bool = False,
    trace_top_k: int = 0,
    max_decode_nfe: int | None = None,
    max_uncached_rows_per_forward: int = DEFAULT_MAX_UNCACHED_ROWS_PER_FORWARD,
    stop_token_ids: tuple[int, ...] = (),
    stop_sequences: tuple[tuple[int, ...], ...] = (),
) -> DecodeResult:
    """Verify the current draft and compute the next proposal in one packed or quadratic call."""
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or prompt_ids.shape[1] == 0:
        raise ValueError("Decoding requires a nonempty prompt and batch size one.")
    if target_length <= 0 or window_size <= 0:
        raise ValueError("target_length and window_size must be positive")
    if prompt_ids.shape[-1] >= int(model.config.length):
        raise ValueError("prompt must leave room for packed decode queries")
    if order_policy is None:
        order_policy = "ltr"
    if order_policy not in ("ltr", "confidence"):
        raise ValueError(f"Unknown order policy: {order_policy}")
    if execution_mode not in ("packed", "quadratic"):
        raise ValueError(f"Unknown parallel-refresh execution: {execution_mode}")
    if trace_top_k < 0:
        raise ValueError("trace_top_k must be nonnegative")
    if max_uncached_rows_per_forward <= 0:
        raise ValueError("max_uncached_rows_per_forward must be positive")
    if execution_mode == "quadratic":
        maximum_active_len = min(int(target_length), int(window_size))
        maximum_quadratic_rows = (
            1 + 2 * maximum_active_len + maximum_active_len * maximum_active_len
        )
        if maximum_quadratic_rows > max_uncached_rows_per_forward:
            raise ValueError(
                f"quadratic self-speculation can require 1+2m+m^2={maximum_quadratic_rows} uncached rows at m={maximum_active_len}, exceeding the configured cap {max_uncached_rows_per_forward}"
            )
    device = prompt_ids.device
    mask_token_id = int(model.mask_token_id)
    prompt_length = int(prompt_ids.shape[-1])
    if max_decode_nfe is None:
        max_decode_nfe = 4 * target_length
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    started = time.perf_counter()
    cache = model._new_generation_cache(batch_size=1, device=device)
    prompt_position_ids = torch.arange(prompt_length, dtype=torch.long, device=device)[
        None, :
    ]
    cache = model.update_cache(
        prompt_ids, cache=cache, position_ids=prompt_position_ids
    )
    past_key_values = cache["past_key_values"]
    if _cache_length(past_key_values) != prompt_length:
        raise RuntimeError("Prompt cache length does not match prompt length.")
    cache_semantic_positions = torch.arange(
        prompt_length, dtype=torch.long, device=device
    )
    canvas = torch.full(
        (target_length,), mask_token_id, dtype=torch.long, device=device
    )
    confidence = torch.full(
        (target_length,), float("nan"), dtype=torch.float32, device=device
    )
    verified = torch.zeros(target_length, dtype=torch.bool, device=device)
    persistent_cached = torch.zeros(target_length, dtype=torch.bool, device=device)
    pending_positions: list[int] = []
    fixed_order: list[int] | None = None
    fixed_scores: dict[int, float] = {}
    fixed_prior: dict[int, float] = {}
    generated_tokens_per_step: list[int] = []
    trace: list[dict[str, Any]] = []
    uncached_rows_per_forward: list[int] = []
    uncached_row_components_per_forward: list[dict[str, int]] = []
    model_call_walltimes_s: list[float] = []
    model_call_roles: list[str] = []
    decode_nfe = 0
    max_pending_count = 0
    stop_position: int | None = None
    stop_token_id: int | None = None
    stop_detected_at_nfe: int | None = None
    resolved_stop_sequences = stop_sequences + tuple(
        ((token_id,) for token_id in stop_token_ids)
    )
    while True:
        decode_limit = stop_position + 1 if stop_position is not None else target_length
        if bool(verified[:decode_limit].all()):
            break
        if decode_nfe >= max_decode_nfe:
            break
        (window_left, window_right, active_positions) = sliding_window_active_positions(
            verified[:decode_limit], window_size=window_size
        )
        if not active_positions:
            raise RuntimeError("The active sliding window is unexpectedly empty.")
        if order_policy == "confidence":
            if fixed_order is None:
                (fixed_order, fixed_scores, fixed_prior) = _rank_active_positions(
                    policy=order_policy,
                    active_positions=active_positions,
                    canvas=canvas,
                    confidence=confidence,
                    mask_token_id=mask_token_id,
                    window_left=window_left,
                )
            elif set(fixed_order) != set(active_positions):
                raise RuntimeError(
                    "Active positions changed while the verification order was frozen."
                )
        else:
            fixed_order = sorted(active_positions)
            fixed_scores = {}
            fixed_prior = {}
        order = list(fixed_order)
        order_tensor = torch.tensor(order, dtype=torch.long, device=device)
        pending_before = list(pending_positions)
        pending_tensor = torch.tensor(pending_before, dtype=torch.long, device=device)
        pending_ids = (
            canvas.index_select(0, pending_tensor)
            if pending_before
            else torch.empty(0, dtype=torch.long, device=device)
        )
        old_ids = canvas.index_select(0, order_tensor).clone()
        input_canvas_ids = canvas.detach().cpu().tolist()
        verified_mask_before = verified.detach().cpu().tolist()
        persistent_mask_before = persistent_cached.detach().cpu().tolist()
        active_len = int(order_tensor.numel())
        pending_count = len(pending_before)
        max_pending_count = max(max_pending_count, pending_count)
        masks = torch.full_like(old_ids, mask_token_id)
        semantic_positions = prompt_length + order_tensor
        pending_semantic_positions = prompt_length + pending_tensor
        if execution_mode == "quadratic":
            quadratic_proposal_ids = masks.repeat(active_len)
            packed_ids = torch.cat(
                (pending_ids, old_ids, masks, quadratic_proposal_ids), dim=0
            )[None, :]
            packed_position_ids = torch.cat(
                (
                    pending_semantic_positions,
                    semantic_positions,
                    semantic_positions,
                    semantic_positions.repeat(active_len),
                ),
                dim=0,
            )[None, :]
        else:
            packed_ids = torch.cat((pending_ids, old_ids, masks, masks), dim=0)[None, :]
            packed_position_ids = torch.cat(
                (
                    pending_semantic_positions,
                    semantic_positions,
                    semantic_positions,
                    semantic_positions,
                ),
                dim=0,
            )[None, :]
        (past_key_values, cache_semantic_positions, evicted_semantic_positions) = (
            _make_semantic_cache_room(
                model,
                past_key_values,
                cache_semantic_positions,
                local_length=int(packed_ids.shape[-1]),
            )
        )
        _clear_evicted_target_cache_flags(
            persistent_cached, evicted_semantic_positions, prompt_length=prompt_length
        )
        cache_len = _cache_length(past_key_values)
        if execution_mode == "packed":
            visible = build_parallel_refresh_attention_mask(
                cache_len=cache_len,
                pending_count=pending_count,
                old_ids=old_ids,
                mask_token_id=mask_token_id,
            )
            additive_mask = model._preprocess_attention_mask(
                visible[None, None, :, :], dtype=torch.float
            )
            denoiser_input = DenoiserInput(
                xt=packed_ids,
                attention_mask=additive_mask,
                past_key_values=past_key_values,
                backbone_kwargs={
                    "position_ids": packed_position_ids,
                    "return_updated_cache": True,
                    "use_cache": True,
                },
            )
            row_components = {
                "pending": int(pending_ids.numel()),
                "old_lane": int(old_ids.numel()),
                "verifier_lane": int(masks.numel()),
                "clean_lane": int(masks.numel()),
            }
            call_role = "packed_self_speculative_draft_verification"
        elif execution_mode == "quadratic":
            visible = build_quadratic_parallel_refresh_attention_mask(
                cache_len=cache_len,
                pending_count=pending_count,
                old_ids=old_ids,
                mask_token_id=mask_token_id,
            )
            denoiser_input = DenoiserInput(
                xt=packed_ids,
                attention_mask=model._preprocess_attention_mask(
                    visible[None, None, :, :], dtype=torch.float
                ),
                past_key_values=past_key_values,
                backbone_kwargs={
                    "position_ids": packed_position_ids,
                    "return_updated_cache": True,
                    "use_cache": True,
                },
            )
            row_components = {
                "pending": int(pending_ids.numel()),
                "old_lane": active_len,
                "verifier_lane": active_len,
                "all_acceptance_position_proposal_lanes": active_len * active_len,
            }
            call_role = "quadratic_self_speculative_draft_verification"
        _record_uncached_forward_rows(
            denoiser_input.xt,
            components=row_components,
            row_cap=max_uncached_rows_per_forward,
            rows_per_forward=uncached_rows_per_forward,
            components_per_forward=uncached_row_components_per_forward,
        )
        if record_model_call_walltimes:
            _sync(device)
            call_started = time.perf_counter()
        backbone_output = model._backbone_forward(denoiser_input)
        if record_model_call_walltimes:
            _sync(device)
            model_call_walltimes_s.append(float(time.perf_counter() - call_started))
            model_call_roles.append(call_role)
        decode_nfe += 1
        updated_cache = backbone_output.past_key_values
        if updated_cache is None:
            raise RuntimeError("Backbone did not return an updated KV cache.")
        if execution_mode == "quadratic":
            local_row_count = 2 * active_len + active_len * active_len
        else:
            local_lane_count = 3 if execution_mode == "packed" else 2
            local_row_count = local_lane_count * active_len
        expected_extended_length = cache_len + pending_count + local_row_count
        if _cache_length(updated_cache) != expected_extended_length:
            raise RuntimeError(
                "Parallel-refresh forward produced an unexpected KV cache length."
            )
        log_probs = model._forward(backbone_output.logits, denoiser_input)
        verifier_start = pending_count + active_len
        quadratic_group_log_probs = None
        quadratic_group_index = None
        if execution_mode == "packed":
            clean_start = verifier_start + active_len
            verifier_log_probs = log_probs[:, verifier_start:clean_start, :]
            clean_log_probs = log_probs[:, clean_start : clean_start + active_len, :]
        elif execution_mode == "quadratic":
            proposal_start = verifier_start + active_len
            verifier_log_probs = log_probs[:, verifier_start:proposal_start, :]
            quadratic_group_log_probs = log_probs[
                :, proposal_start : proposal_start + active_len * active_len, :
            ].reshape(1, active_len, active_len, -1)
            provisional_verifier_ids = verifier_log_probs.argmax(dim=-1)[0]
            provisional_matching_prefix = stable_permutation_prefix_length(
                old_ids, provisional_verifier_ids, mask_token_id=mask_token_id
            )
            quadratic_group_index = min(provisional_matching_prefix, active_len - 1)
            clean_log_probs = quadratic_group_log_probs[:, quadratic_group_index, :, :]
        else:
            verifier_log_probs = log_probs[
                0:1, verifier_start : verifier_start + active_len, :
            ]
            clean_log_probs = log_probs[
                1:2, pending_count : pending_count + active_len, :
            ]
        verifier_ids = verifier_log_probs.argmax(dim=-1)[0]
        clean_ids = clean_log_probs.argmax(dim=-1)[0]
        verifier_confidence = (
            verifier_log_probs.gather(dim=-1, index=verifier_ids[None, :, None])
            .squeeze(0)
            .squeeze(-1)
            .exp()
            .to(torch.float32)
        )
        clean_confidence = (
            clean_log_probs.gather(dim=-1, index=clean_ids[None, :, None])
            .squeeze(0)
            .squeeze(-1)
            .exp()
            .to(torch.float32)
        )
        (
            matching_prefix,
            accepted_count,
            restart_used,
            selected_ids,
            selected_sources,
        ) = select_parallel_refresh_step(
            old_ids, verifier_ids, clean_ids, mask_token_id=mask_token_id
        )
        if quadratic_group_index is not None and quadratic_group_index != min(
            matching_prefix, active_len - 1
        ):
            raise RuntimeError("quadratic proposal-group selection drifted")
        clean_source_ranks = [
            rank for (rank, source) in enumerate(selected_sources) if source == "clean"
        ]
        selected_confidence = verifier_confidence.clone()
        if clean_source_ranks:
            clean_source_tensor = torch.tensor(
                clean_source_ranks, dtype=torch.long, device=device
            )
            selected_confidence[clean_source_tensor] = clean_confidence[
                clean_source_tensor
            ]
        selected_top_ids = None
        selected_top_probabilities = None
        selected_residual_probability = None
        if record_trace and trace_top_k > 0:
            resolved_top_k = min(trace_top_k, verifier_log_probs.shape[-1])
            (verifier_top_log_probabilities, verifier_top_ids) = torch.topk(
                verifier_log_probs, k=resolved_top_k, dim=-1
            )
            (clean_top_log_probabilities, clean_top_ids) = torch.topk(
                clean_log_probs, k=resolved_top_k, dim=-1
            )
            verifier_top_probabilities = verifier_top_log_probabilities.exp()
            clean_top_probabilities = clean_top_log_probabilities.exp()
            verifier_residual = (
                1.0 - verifier_top_probabilities.sum(dim=-1)
            ).clamp_min(0.0)
            clean_residual = (1.0 - clean_top_probabilities.sum(dim=-1)).clamp_min(0.0)
            selected_top_ids = verifier_top_ids.clone()
            selected_top_probabilities = verifier_top_probabilities.clone()
            selected_residual_probability = verifier_residual.clone()
            if clean_source_ranks:
                selected_top_ids[:, clean_source_tensor] = clean_top_ids[
                    :, clean_source_tensor
                ]
                selected_top_probabilities[:, clean_source_tensor] = (
                    clean_top_probabilities[:, clean_source_tensor]
                )
                selected_residual_probability[:, clean_source_tensor] = clean_residual[
                    :, clean_source_tensor
                ]
        canvas.scatter_(0, order_tensor, selected_ids)
        confidence.scatter_(0, order_tensor, selected_confidence)
        newly_verified = order[:accepted_count]
        verified_before_step = verified.clone()
        if newly_verified:
            verified[torch.tensor(newly_verified, dtype=torch.long, device=device)] = (
                True
            )
        promotable_old_count = matching_prefix
        retained_length = cache_len + pending_count + promotable_old_count
        updated_cache.crop(retained_length)
        past_key_values = updated_cache
        retained_semantic_positions = torch.cat(
            (pending_semantic_positions, semantic_positions[:promotable_old_count])
        )
        cache_semantic_positions = torch.cat(
            (cache_semantic_positions, retained_semantic_positions)
        )
        if int(cache_semantic_positions.numel()) != _cache_length(past_key_values):
            raise RuntimeError(
                "Retained KV cache and semantic-position ledger disagree."
            )
        if pending_before:
            persistent_cached[pending_tensor] = True
        if promotable_old_count:
            persistent_cached[order_tensor[:promotable_old_count]] = True
        pending_positions = []
        correction_position = None
        if accepted_count > matching_prefix:
            correction_position = order[matching_prefix]
            pending_positions.append(correction_position)
        previous_stop_position = stop_position
        (stop_position, stop_token_id) = earliest_verified_stop_sequence(
            canvas,
            verified,
            stop_sequences=resolved_stop_sequences,
            current_stop_position=stop_position,
        )
        stop_cache_evictions: list[int] = []
        if stop_position is not None:
            if previous_stop_position != stop_position:
                stop_detected_at_nfe = decode_nfe
            (past_key_values, cache_semantic_positions, stop_cache_evictions) = (
                _truncate_target_state_after_stop(
                    model=model,
                    past_key_values=past_key_values,
                    cache_semantic_positions=cache_semantic_positions,
                    canvas=canvas,
                    confidence=confidence,
                    verified=verified,
                    persistent_cached=persistent_cached,
                    prompt_length=prompt_length,
                    stop_position=stop_position,
                )
            )
            pending_positions = [
                position for position in pending_positions if position <= stop_position
            ]
            if correction_position is not None and correction_position > stop_position:
                correction_position = None
        retained_acceptance_count = int((~verified_before_step & verified).sum().item())
        generated_tokens_per_step.append(retained_acceptance_count)
        if record_trace:
            trace.append(
                {
                    "step": decode_nfe,
                    "input_canvas_ids": input_canvas_ids,
                    "verified_mask_before": verified_mask_before,
                    "persistent_cached_mask_before": persistent_mask_before,
                    "pending_positions_before": pending_before,
                    "window_left": window_left,
                    "window_right": window_right,
                    "active_positions": active_positions,
                    "permutation": order,
                    "old_ids": old_ids.detach().cpu().tolist(),
                    "proposal_ids": selected_ids.detach().cpu().tolist(),
                    "proposal_confidence": selected_confidence.detach().cpu().tolist(),
                    "proposal_source": selected_sources,
                    "verification_proposal_ids": verifier_ids.detach().cpu().tolist(),
                    "verification_confidence": verifier_confidence.detach()
                    .cpu()
                    .tolist(),
                    "clean_proposal_ids": clean_ids.detach().cpu().tolist(),
                    "clean_confidence": clean_confidence.detach().cpu().tolist(),
                    "quadratic_proposal_group_index": quadratic_group_index,
                    "top_token_ids": selected_top_ids[0].detach().cpu().tolist()
                    if selected_top_ids is not None
                    else [],
                    "top_probabilities": selected_top_probabilities[0]
                    .detach()
                    .float()
                    .cpu()
                    .tolist()
                    if selected_top_probabilities is not None
                    else [],
                    "residual_probability": selected_residual_probability[0]
                    .detach()
                    .float()
                    .cpu()
                    .tolist()
                    if selected_residual_probability is not None
                    else [],
                    "order_score": {
                        str(position): float(value)
                        for (position, value) in fixed_scores.items()
                    },
                    "training_next_probability": {
                        str(position): float(value)
                        for (position, value) in fixed_prior.items()
                    },
                    "stable_prefix_length": matching_prefix,
                    "accepted_count": accepted_count,
                    "total_accepted_count": retained_acceptance_count,
                    **{},
                    **{},
                    "restart_used": True,
                    "correction_position": correction_position,
                    "newly_cached_positions": newly_verified,
                    "persistent_cached_positions": [
                        position
                        for position in pending_before + order[:promotable_old_count]
                    ],
                    "pending_positions_after": pending_positions,
                    "cache_length": _cache_length(past_key_values),
                    "cache_semantic_positions": cache_semantic_positions.detach()
                    .cpu()
                    .tolist(),
                    "evicted_semantic_positions": evicted_semantic_positions
                    + stop_cache_evictions,
                    "termination_position": stop_position,
                    "termination_token_id": stop_token_id,
                    "canvas_ids": canvas.detach().cpu().tolist(),
                    "cached_mask": verified.detach().cpu().tolist(),
                    "persistent_cached_mask": persistent_cached.detach().cpu().tolist(),
                }
            )
        fixed_order = None
    _sync(device)
    latency_s = time.perf_counter() - started
    final_length = stop_position + 1 if stop_position is not None else target_length
    converged = bool(verified[:final_length].all())
    _validate_uncached_row_diagnostics(
        decode_nfe=decode_nfe,
        rows_per_forward=uncached_rows_per_forward,
        components_per_forward=uncached_row_components_per_forward,
    )
    if record_model_call_walltimes and (
        not len(model_call_walltimes_s) == len(model_call_roles) == decode_nfe
    ):
        raise RuntimeError("Model-call timing diagnostics must match decode forwards.")
    return DecodeResult(
        target_ids=canvas[None, :final_length],
        decode_nfe=decode_nfe,
        setup_forward_calls=1,
        latency_s=float(latency_s),
        peak_memory_gib=_peak_memory_gib(device),
        peak_reserved_memory_gib=_peak_reserved_memory_gib(device),
        generated_tokens_per_step=generated_tokens_per_step,
        trace=trace,
        converged=converged,
        canvas_capacity=target_length,
        termination_position=stop_position,
        termination_token_id=stop_token_id,
        termination_detected_at_nfe=stop_detected_at_nfe,
        uncached_rows_per_forward=uncached_rows_per_forward,
        uncached_row_components_per_forward=uncached_row_components_per_forward,
        model_call_walltimes_s=model_call_walltimes_s,
        model_call_roles=model_call_roles,
        uncached_row_cap=int(max_uncached_rows_per_forward),
        decoder_diagnostics={
            **{},
            **(
                {
                    "parallel_refresh_execution_mode": execution_mode,
                    "quadratic_proposal_groups": window_size,
                    "maximum_quadratic_rows_per_forward": 1
                    + 2 * window_size
                    + window_size * window_size,
                    "quadratic_correction_conditioning": "next_forward_pending",
                }
                if execution_mode == "quadratic"
                else {}
            ),
            **{},
        },
    )


@torch.inference_mode()
def linear_decode(
    model: Any,
    prompt_ids: torch.LongTensor,
    *,
    target_length: int = 256,
    window_size: int = 16,
    order_policy: OrderPolicy = "confidence",
    record_model_call_walltimes: bool = False,
    record_trace: bool = False,
    trace_top_k: int = 0,
    max_decode_nfe: int | None = None,
    max_uncached_rows_per_forward: int = DEFAULT_MAX_UNCACHED_ROWS_PER_FORWARD,
    stop_token_ids: tuple[int, ...] = (),
    stop_sequences: tuple[tuple[int, ...], ...] = (),
) -> DecodeResult:
    """Compute a fresh proposal, then verify it in a separate forward call."""
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or prompt_ids.shape[1] == 0:
        raise ValueError("Decoding requires a nonempty prompt and batch size one.")
    if target_length <= 0 or window_size <= 0:
        raise ValueError("target_length and window_size must be positive")
    if prompt_ids.shape[-1] >= int(model.config.length):
        raise ValueError("prompt must leave room for packed decode queries")
    if order_policy not in ("ltr", "confidence"):
        raise ValueError(f"Unknown order policy: {order_policy}")
    if trace_top_k < 0:
        raise ValueError("trace_top_k must be nonnegative")
    if max_uncached_rows_per_forward <= 0:
        raise ValueError("max_uncached_rows_per_forward must be positive")
    device = prompt_ids.device
    mask_token_id = int(model.mask_token_id)
    prompt_length = int(prompt_ids.shape[-1])
    if max_decode_nfe is None:
        max_decode_nfe = 4 * target_length
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    started = time.perf_counter()
    cache = model._new_generation_cache(batch_size=1, device=device)
    prompt_position_ids = torch.arange(prompt_length, dtype=torch.long, device=device)[
        None, :
    ]
    cache = model.update_cache(
        prompt_ids, cache=cache, position_ids=prompt_position_ids
    )
    past_key_values = cache["past_key_values"]
    if _cache_length(past_key_values) != prompt_length:
        raise RuntimeError("Prompt cache length does not match prompt length.")
    cache_semantic_positions = torch.arange(
        prompt_length, dtype=torch.long, device=device
    )
    used_canvas = torch.full(
        (target_length,), mask_token_id, dtype=torch.long, device=device
    )
    clean_canvas = used_canvas.clone()
    confidence = torch.full(
        (target_length,), float("nan"), dtype=torch.float32, device=device
    )
    verified = torch.zeros(target_length, dtype=torch.bool, device=device)
    persistent_cached = torch.zeros(target_length, dtype=torch.bool, device=device)
    pending_positions: list[int] = []
    generated_tokens_per_step: list[int] = []
    trace: list[dict[str, Any]] = []
    uncached_rows_per_forward: list[int] = []
    uncached_row_components_per_forward: list[dict[str, int]] = []
    model_call_walltimes_s: list[float] = []
    model_call_roles: list[str] = []
    decode_nfe = 0
    stop_position: int | None = None
    stop_token_id: int | None = None
    stop_detected_at_nfe: int | None = None
    branch_selection_counts: dict[str, int] = {}
    resolved_stop_sequences = stop_sequences + tuple(
        ((token_id,) for token_id in stop_token_ids)
    )
    while True:
        decode_limit = stop_position + 1 if stop_position is not None else target_length
        if bool(verified[:decode_limit].all()):
            break
        if decode_nfe >= max_decode_nfe:
            break
        (window_left, window_right, active_positions) = sliding_window_active_positions(
            verified[:decode_limit], window_size=window_size
        )
        if not active_positions:
            raise RuntimeError("The active sliding window is unexpectedly empty.")
        (order, order_scores, order_prior) = _rank_active_positions(
            policy=order_policy,
            active_positions=active_positions,
            canvas=used_canvas,
            confidence=confidence,
            mask_token_id=mask_token_id,
            window_left=window_left,
        )
        order_tensor = torch.tensor(order, dtype=torch.long, device=device)
        active_tensor = torch.tensor(active_positions, dtype=torch.long, device=device)
        semantic_index = {
            position: index for (index, position) in enumerate(active_positions)
        }
        order_to_semantic = torch.tensor(
            [semantic_index[position] for position in order],
            dtype=torch.long,
            device=device,
        )
        clean_semantic = clean_canvas.index_select(0, active_tensor)
        candidate_values = {"clean": clean_semantic}
        candidate_names = list(_single_candidate_names("clean_only_fresh"))
        candidate_semantic = [
            (name, candidate_values[name]) for name in candidate_names
        ]
        candidate_ids = torch.stack(
            [ids.index_select(0, order_to_semantic) for (_, ids) in candidate_semantic],
            dim=0,
        )
        (branch_count, active_len) = candidate_ids.shape
        pending_before = list(pending_positions)
        pending_tensor = torch.tensor(pending_before, dtype=torch.long, device=device)
        pending_ids = (
            used_canvas.index_select(0, pending_tensor)
            if pending_before
            else torch.empty(0, dtype=torch.long, device=device)
        )
        pending_count = len(pending_before)
        masks = torch.full(
            (active_len,), mask_token_id, dtype=torch.long, device=device
        )
        semantic_positions = prompt_length + order_tensor
        combined_local_length = pending_count + 3 * active_len
        if pending_count > 1 or combined_local_length > 49:
            raise AssertionError(
                f"Clean-only split exceeds p+3m<=49: realized={combined_local_length}, pending={pending_count}, active={active_len}."
            )
        (past_key_values, cache_semantic_positions, evicted_semantic_positions) = (
            _make_semantic_cache_room(
                model,
                past_key_values,
                cache_semantic_positions,
                local_length=combined_local_length,
            )
        )
        _clear_evicted_target_cache_flags(
            persistent_cached, evicted_semantic_positions, prompt_length=prompt_length
        )
        cache_len = _cache_length(past_key_values)
        call_count = 2
        if decode_nfe + call_count > max_decode_nfe:
            break
        draft_ids = torch.cat((pending_ids, masks))[None, :]
        draft_position_ids = torch.cat(
            (prompt_length + pending_tensor, semantic_positions)
        )[None, :]
        if int(draft_ids.numel()) > 17:
            raise AssertionError("Split draft exceeds p+m<=17")
        draft_visible = build_multi_branch_draft_attention_mask(
            cache_len=cache_len,
            pending_count=pending_count,
            clean_length=active_len,
            device=device,
        )
        draft_input = DenoiserInput(
            xt=draft_ids,
            attention_mask=model._preprocess_attention_mask(
                draft_visible[None, None, :, :], dtype=torch.float
            ),
            past_key_values=past_key_values,
            backbone_kwargs={
                "position_ids": draft_position_ids,
                "return_updated_cache": True,
                "use_cache": True,
            },
        )
        _record_uncached_forward_rows(
            draft_input.xt,
            components={
                "pending": int(pending_ids.numel()),
                "clean_draft_lane": int(masks.numel()),
            },
            row_cap=max_uncached_rows_per_forward,
            rows_per_forward=uncached_rows_per_forward,
            components_per_forward=uncached_row_components_per_forward,
        )
        if record_model_call_walltimes:
            _sync(device)
            call_started = time.perf_counter()
        draft_output = model._backbone_forward(draft_input)
        if record_model_call_walltimes:
            _sync(device)
            model_call_walltimes_s.append(float(time.perf_counter() - call_started))
            model_call_roles.append("clean_draft")
        decode_nfe += 1
        draft_updated_cache = draft_output.past_key_values
        if draft_updated_cache is None:
            raise RuntimeError("Draft forward did not return a KV cache.")
        if _cache_length(draft_updated_cache) != cache_len + pending_count + active_len:
            raise RuntimeError("Split draft produced an unexpected KV length.")
        draft_log_probs = model._forward(draft_output.logits, draft_input)
        clean_log_probs = draft_log_probs[
            0, pending_count : pending_count + active_len, :
        ]
        clean_ids = clean_log_probs.argmax(dim=-1)
        clean_branch_index = candidate_names.index("clean")
        candidate_ids = candidate_ids.clone()
        candidate_ids[clean_branch_index] = clean_ids
        verification_cache = model._select_kv_cache_positions(
            draft_updated_cache,
            torch.arange(cache_len + pending_count, dtype=torch.long, device=device),
        )
        verification_parts: list[torch.Tensor] = []
        verification_position_parts: list[torch.Tensor] = []
        for branch_index in range(1):
            verification_parts.extend((candidate_ids[branch_index], masks))
            verification_position_parts.extend((semantic_positions, semantic_positions))
        verification_ids = torch.cat(verification_parts)[None, :]
        verification_position_ids = torch.cat(verification_position_parts)[None, :]
        if int(verification_ids.numel()) > 64:
            raise AssertionError("Split verification exceeds 4m<=64")
        if int(verification_ids.numel()) > 32:
            raise AssertionError("Clean-only verification exceeds 2m<=32")
        verification_visible = build_multi_branch_verification_attention_mask(
            cache_len=cache_len + pending_count,
            candidate_ids=candidate_ids,
            mask_token_id=mask_token_id,
        )
        verification_input = DenoiserInput(
            xt=verification_ids,
            attention_mask=model._preprocess_attention_mask(
                verification_visible[None, None, :, :], dtype=torch.float
            ),
            past_key_values=verification_cache,
            backbone_kwargs={
                "position_ids": verification_position_ids,
                "return_updated_cache": True,
                "use_cache": True,
            },
        )
        verification_components: dict[str, int] = {}
        for branch_index, candidate_name in enumerate(candidate_names):
            verification_components[
                f"candidate_{branch_index}_{candidate_name}_lane"
            ] = int(candidate_ids[branch_index].numel())
            verification_components[
                f"verifier_{branch_index}_{candidate_name}_lane"
            ] = int(masks.numel())
        _record_uncached_forward_rows(
            verification_input.xt,
            components=verification_components,
            row_cap=max_uncached_rows_per_forward,
            rows_per_forward=uncached_rows_per_forward,
            components_per_forward=uncached_row_components_per_forward,
        )
        if record_model_call_walltimes:
            _sync(device)
            call_started = time.perf_counter()
        verification_output = model._backbone_forward(verification_input)
        if record_model_call_walltimes:
            _sync(device)
            model_call_walltimes_s.append(float(time.perf_counter() - call_started))
            model_call_roles.append("branch_verification")
        decode_nfe += 1
        updated_cache = verification_output.past_key_values
        expected_cache_len = cache_len + pending_count + 2 * active_len
        if updated_cache is None or _cache_length(updated_cache) != expected_cache_len:
            raise RuntimeError("Split verification produced an unexpected KV length.")
        verification_log_probs = model._forward(
            verification_output.logits, verification_input
        )
        verifier_log_probs = torch.stack(
            [
                verification_log_probs[
                    0,
                    2 * branch_index * active_len + active_len : 2
                    * (branch_index + 1)
                    * active_len,
                    :,
                ]
                for branch_index in range(1)
            ],
            dim=0,
        )
        verifier_ids = verifier_log_probs.argmax(dim=-1)
        clean_ids = clean_log_probs.argmax(dim=-1)
        verifier_confidence = (
            verifier_log_probs.gather(dim=-1, index=verifier_ids[..., None])
            .squeeze(-1)
            .exp()
            .to(torch.float32)
        )
        clean_confidence = (
            clean_log_probs.gather(dim=-1, index=clean_ids[..., None])
            .squeeze(-1)
            .exp()
            .to(torch.float32)
        )
        (winner, stable_prefixes, accepted_counts) = select_multi_branch_step(
            candidate_ids, verifier_ids, mask_token_id=mask_token_id
        )
        winner_name = candidate_names[winner]
        branch_selection_counts[winner_name] = (
            branch_selection_counts.get(winner_name, 0) + 1
        )
        stable_prefix = stable_prefixes[winner]
        accepted_count = accepted_counts[winner]
        winner_ids = verifier_ids[winner]
        winner_confidence = verifier_confidence[winner]
        selected_ids = winner_ids
        selected_confidence = winner_confidence
        selected_log_probs = verifier_log_probs[winner]
        selected_sources = [winner_name] * active_len
        selected_top_ids = None
        selected_top_probabilities = None
        selected_residual_probability = None
        if record_trace and trace_top_k > 0:
            resolved_top_k = min(trace_top_k, int(verifier_log_probs.shape[-1]))
            (top_log_probabilities, selected_top_ids) = torch.topk(
                selected_log_probs, k=resolved_top_k, dim=-1
            )
            selected_top_probabilities = top_log_probabilities.exp()
            selected_residual_probability = (
                1.0 - selected_top_probabilities.sum(dim=-1)
            ).clamp_min(0.0)
        input_canvas_ids = used_canvas.detach().cpu().tolist()
        verified_mask_before = verified.detach().cpu().tolist()
        persistent_mask_before = persistent_cached.detach().cpu().tolist()
        used_canvas.scatter_(0, order_tensor, selected_ids)
        confidence.scatter_(0, order_tensor, selected_confidence)
        clean_canvas.scatter_(0, order_tensor, clean_ids)
        newly_verified = order[:accepted_count]
        verified_before_step = verified.clone()
        if newly_verified:
            newly_verified_tensor = torch.tensor(
                newly_verified, dtype=torch.long, device=device
            )
            verified[newly_verified_tensor] = True
            accepted_ids = used_canvas.index_select(0, newly_verified_tensor)
            clean_canvas.scatter_(0, newly_verified_tensor, accepted_ids)
        winner_old_start = cache_len + pending_count + 2 * winner * active_len
        retained_indices = [
            torch.arange(cache_len + pending_count, dtype=torch.long, device=device)
        ]
        if stable_prefix:
            retained_indices.append(
                winner_old_start
                + torch.arange(stable_prefix, dtype=torch.long, device=device)
            )
        past_key_values = model._select_kv_cache_positions(
            updated_cache, torch.cat(retained_indices)
        )
        pending_semantic_positions = prompt_length + pending_tensor
        retained_semantic_positions = torch.cat(
            (pending_semantic_positions, semantic_positions[:stable_prefix])
        )
        cache_semantic_positions = torch.cat(
            (cache_semantic_positions, retained_semantic_positions)
        )
        if int(cache_semantic_positions.numel()) != _cache_length(past_key_values):
            raise RuntimeError(
                "Retained KV cache and semantic-position ledger disagree."
            )
        if pending_before:
            persistent_cached[pending_tensor] = True
        if stable_prefix:
            persistent_cached[order_tensor[:stable_prefix]] = True
        pending_positions = []
        correction_position = None
        if accepted_count > stable_prefix:
            correction_position = order[stable_prefix]
            pending_positions.append(correction_position)
        previous_stop_position = stop_position
        (stop_position, stop_token_id) = earliest_verified_stop_sequence(
            used_canvas,
            verified,
            stop_sequences=resolved_stop_sequences,
            current_stop_position=stop_position,
        )
        stop_cache_evictions: list[int] = []
        if stop_position is not None:
            if previous_stop_position != stop_position:
                stop_detected_at_nfe = decode_nfe
            (past_key_values, cache_semantic_positions, stop_cache_evictions) = (
                _truncate_target_state_after_stop(
                    model=model,
                    past_key_values=past_key_values,
                    cache_semantic_positions=cache_semantic_positions,
                    canvas=used_canvas,
                    confidence=confidence,
                    verified=verified,
                    persistent_cached=persistent_cached,
                    prompt_length=prompt_length,
                    stop_position=stop_position,
                )
            )
            pending_positions = [
                position for position in pending_positions if position <= stop_position
            ]
            if correction_position is not None and correction_position > stop_position:
                correction_position = None
        retained_acceptance_count = int((~verified_before_step & verified).sum().item())
        generated_tokens_per_step.append(retained_acceptance_count)
        if record_trace:
            branch_diagnostics = []
            for branch_index, branch_name in enumerate(candidate_names):
                branch_diagnostics.append(
                    {
                        "name": branch_name,
                        "candidate_ids": candidate_ids[branch_index]
                        .detach()
                        .cpu()
                        .tolist(),
                        "verifier_ids": verifier_ids[branch_index]
                        .detach()
                        .cpu()
                        .tolist(),
                        "verifier_confidence": verifier_confidence[branch_index]
                        .detach()
                        .cpu()
                        .tolist(),
                        "stable_prefix_length": stable_prefixes[branch_index],
                        "accepted_count": accepted_counts[branch_index],
                        "won": branch_index == winner,
                    }
                )
            trace.append(
                {
                    "step": decode_nfe,
                    "input_canvas_ids": input_canvas_ids,
                    "verified_mask_before": verified_mask_before,
                    "persistent_cached_mask_before": persistent_mask_before,
                    "pending_positions_before": pending_before,
                    "window_left": window_left,
                    "window_right": window_right,
                    "active_positions": active_positions,
                    "permutation": order,
                    "old_ids": candidate_ids[winner].detach().cpu().tolist(),
                    "proposal_ids": selected_ids.detach().cpu().tolist(),
                    "proposal_confidence": selected_confidence.detach().cpu().tolist(),
                    "proposal_source": selected_sources,
                    "verification_proposal_ids": winner_ids.detach().cpu().tolist(),
                    "verification_confidence": winner_confidence.detach()
                    .cpu()
                    .tolist(),
                    "clean_proposal_ids": clean_ids.detach().cpu().tolist(),
                    "clean_confidence": clean_confidence.detach().cpu().tolist(),
                    "top_token_ids": selected_top_ids.detach().cpu().tolist()
                    if selected_top_ids is not None
                    else [],
                    "top_probabilities": selected_top_probabilities.detach()
                    .float()
                    .cpu()
                    .tolist()
                    if selected_top_probabilities is not None
                    else [],
                    "residual_probability": selected_residual_probability.detach()
                    .float()
                    .cpu()
                    .tolist()
                    if selected_residual_probability is not None
                    else [],
                    "order_score": {
                        str(position): float(value)
                        for (position, value) in order_scores.items()
                    },
                    "training_next_probability": {
                        str(position): float(value)
                        for (position, value) in order_prior.items()
                    },
                    "stable_prefix_length": stable_prefix,
                    "accepted_count": accepted_count,
                    **{},
                    "restart_used": True,
                    "winner_branch": winner_name,
                    "candidate_count": 1,
                    "execution_mode": "separate_draft_verification",
                    "candidate_branches": branch_diagnostics,
                    "correction_position": correction_position,
                    "newly_cached_positions": newly_verified,
                    "persistent_cached_positions": [
                        position for position in pending_before + order[:stable_prefix]
                    ],
                    "pending_positions_after": pending_positions,
                    "cache_length": _cache_length(past_key_values),
                    "cache_semantic_positions": cache_semantic_positions.detach()
                    .cpu()
                    .tolist(),
                    "evicted_semantic_positions": evicted_semantic_positions
                    + stop_cache_evictions,
                    "termination_position": stop_position,
                    "termination_token_id": stop_token_id,
                    "canvas_ids": used_canvas.detach().cpu().tolist(),
                    "cached_mask": verified.detach().cpu().tolist(),
                    "persistent_cached_mask": persistent_cached.detach().cpu().tolist(),
                }
            )
    _sync(device)
    latency_s = time.perf_counter() - started
    final_length = stop_position + 1 if stop_position is not None else target_length
    converged = bool(verified[:final_length].all())
    _validate_uncached_row_diagnostics(
        decode_nfe=decode_nfe,
        rows_per_forward=uncached_rows_per_forward,
        components_per_forward=uncached_row_components_per_forward,
    )
    if record_model_call_walltimes and (
        not len(model_call_walltimes_s) == len(model_call_roles) == decode_nfe
    ):
        raise RuntimeError("Model-call timing diagnostics must match decode forwards.")
    return DecodeResult(
        target_ids=used_canvas[None, :final_length],
        decode_nfe=decode_nfe,
        setup_forward_calls=1,
        latency_s=float(latency_s),
        peak_memory_gib=_peak_memory_gib(device),
        peak_reserved_memory_gib=_peak_reserved_memory_gib(device),
        generated_tokens_per_step=generated_tokens_per_step,
        trace=trace,
        converged=converged,
        canvas_capacity=target_length,
        termination_position=stop_position,
        termination_token_id=stop_token_id,
        termination_detected_at_nfe=stop_detected_at_nfe,
        branch_selection_counts=branch_selection_counts,
        uncached_rows_per_forward=uncached_rows_per_forward,
        uncached_row_components_per_forward=uncached_row_components_per_forward,
        model_call_walltimes_s=model_call_walltimes_s,
        model_call_roles=model_call_roles,
        uncached_row_cap=int(max_uncached_rows_per_forward),
        decoder_diagnostics={
            **{"multi_branch_execution_mode": "separate_draft_verification"},
            **{},
            **{},
        },
    )


def _single_candidate_names(*args, **kwargs):
    return ["clean"]
