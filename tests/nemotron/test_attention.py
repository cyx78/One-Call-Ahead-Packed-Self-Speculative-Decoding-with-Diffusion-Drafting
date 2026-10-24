from __future__ import annotations
from types import SimpleNamespace
import torch
from nemotron_ltr.runtime_attention import (
    bootstrap_visibility,
    build_next_packed_candidate,
    extract_quadratic_roots,
    packed_visibility,
    quadratic_visibility,
    select_quadratic_step,
)


def test_bootstrap_visibility_is_causal_prompt_and_full_draft() -> None:
    assert bootstrap_visibility(2, 2).tolist() == [
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, True],
        [True, True, True, True],
    ]


def test_packed_visibility_isolates_verifier_and_proposal_lanes() -> None:
    assert packed_visibility(2, 2).tolist() == [
        [True, True, True, False, False, False],
        [True, True, True, True, False, False],
        [True, True, False, False, True, True],
        [True, True, False, False, True, True],
    ]


def test_quadratic_visibility_matches_root_and_group_contract() -> None:
    assert quadratic_visibility(1, 2).tolist() == [
        [True, True, False, False, False, False, False],
        [True, True, True, True, False, False, False],
        [True, True, True, True, False, False, False],
        [True, True, False, False, True, False, False],
        [True, True, False, False, True, True, True],
        [True, True, False, False, True, True, True],
    ]


def test_quadratic_visibility_exactly_translates_nvidia_logical_key_order() -> None:
    prefix_length = 7
    for block_length in range(2, 17):
        draft_length = block_length * (block_length + 1)
        query = torch.arange(draft_length)[:, None]
        logical_key = torch.arange(draft_length + prefix_length)[None, :]
        root = (
            (logical_key % (block_length + 1) == 0)
            & (logical_key < draft_length)
            & (query >= logical_key)
        )
        same_group = (
            query // (block_length + 1) == logical_key // (block_length + 1)
        ) & (query % (block_length + 1) != 0)
        nvidia_logical = root | same_group | (logical_key >= draft_length)
        nvidia_physical = torch.cat(
            (nvidia_logical[:, draft_length:], nvidia_logical[:, :draft_length]), dim=1
        )
        assert torch.equal(
            quadratic_visibility(prefix_length, block_length), nvidia_physical
        )


def test_packed_recurrence_preserves_only_position_aligned_overlap() -> None:
    next_draft = build_next_packed_candidate(
        torch.tensor([[10, 11, 12, 13]]),
        torch.tensor([[20, 21, 22, 23]]),
        accepted_length=2,
        block_length=4,
        mask_token_id=99,
    )
    assert next_draft.tolist() == [[11, 23, 99, 99]]


def test_packed_recurrence_retains_correction_after_full_width_acceptance() -> None:
    next_draft = build_next_packed_candidate(
        torch.tensor([[10, 11, 12, 13]]),
        torch.tensor([[20, 21, 22, 23]]),
        accepted_length=4,
        block_length=4,
        mask_token_id=99,
    )
    assert next_draft.tolist() == [[13, 99, 99, 99]]


def test_quadratic_selection_uses_group_at_observed_acceptance_position() -> None:
    current = torch.tensor([[1, 2, 3]])
    predictions = torch.tensor([[[2, 20, 21, 22], [99, 30, 31, 32], [100, 40, 41, 42]]])
    (accepted, committed, next_draft) = select_quadratic_step(current, predictions)
    assert accepted == 2
    assert committed.tolist() == [[1, 2]]
    assert next_draft.tolist() == [[99, 31, 32]]


def test_quadratic_cache_extraction_keeps_prefix_and_group_roots() -> None:
    keys = torch.arange(7).reshape(1, 1, 7, 1)
    values = keys + 100
    cache = SimpleNamespace(
        layers=[SimpleNamespace(keys=keys.clone(), values=values.clone())]
    )
    extract_quadratic_roots(cache, prefix_length=1, block_length=2)
    assert cache.layers[0].keys.flatten().tolist() == [0, 1, 4]
    assert cache.layers[0].values.flatten().tolist() == [100, 101, 104]
