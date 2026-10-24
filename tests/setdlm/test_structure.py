from __future__ import annotations
import pytest
import torch
from setdlm.decoders import (
    _clear_evicted_target_cache_flags,
    _make_semantic_cache_room,
    _record_uncached_forward_rows,
    build_parallel_refresh_attention_mask,
    build_quadratic_parallel_refresh_attention_mask,
    earliest_verified_stop_sequence,
    sliding_window_active_positions,
    stable_permutation_prefix_length,
)


def test_uncached_row_accounting_counts_all_batch_elements() -> None:
    rows: list[int] = []
    component_rows: list[dict[str, int]] = []
    batched_ids = torch.zeros((3, 5), dtype=torch.long)
    components = {
        "pending_materialization_all_batches": 3,
        "candidate_lanes": 4,
        "verifier_lanes": 4,
        "clean_and_filler_lanes": 4,
    }
    realized = _record_uncached_forward_rows(
        batched_ids,
        components=components,
        row_cap=15,
        rows_per_forward=rows,
        components_per_forward=component_rows,
    )
    assert realized == 15
    assert rows == [15]
    assert component_rows == [components]


def test_uncached_row_accounting_rejects_cap_and_component_mismatch() -> None:
    rows: list[int] = []
    component_rows: list[dict[str, int]] = []
    packed_ids = torch.zeros((1, 9), dtype=torch.long)
    with pytest.raises(AssertionError, match="realized=9, cap=8"):
        _record_uncached_forward_rows(
            packed_ids,
            components={"pending": 1, "packed_lanes": 8},
            row_cap=8,
            rows_per_forward=rows,
            components_per_forward=component_rows,
        )
    assert rows == []
    assert component_rows == []
    with pytest.raises(RuntimeError, match="do not match"):
        _record_uncached_forward_rows(
            packed_ids,
            components={"pending": 1, "packed_lanes": 7},
            row_cap=9,
            rows_per_forward=rows,
            components_per_forward=component_rows,
        )


class _FakeCache:
    def __init__(self, positions: list[int]) -> None:
        self.positions = torch.tensor(positions, dtype=torch.long)

    def get_seq_length(self) -> int:
        return int(self.positions.numel())


class _FakeRollingModel:
    class Config:
        length = 8

    config = Config()

    @staticmethod
    def _select_kv_cache_positions(
        cache: _FakeCache, keep_indices: torch.LongTensor
    ) -> _FakeCache:
        cache.positions = cache.positions.index_select(0, keep_indices)
        return cache


def test_semantic_cache_eviction_keeps_newest_positions() -> None:
    cache = _FakeCache([0, 8, 3, 12, 6])
    semantic_positions = cache.positions.clone()
    (cache, ledger, evicted) = _make_semantic_cache_room(
        _FakeRollingModel(), cache, semantic_positions, local_length=5
    )
    assert cache.positions.tolist() == [8, 12, 6]
    assert ledger.tolist() == [8, 12, 6]
    assert evicted == [0, 3]


def test_evicted_target_positions_clear_only_physical_cache_flags() -> None:
    persistent = torch.tensor([True, True, True, True])
    _clear_evicted_target_cache_flags(persistent, [0, 4, 6, 99], prompt_length=4)
    assert persistent.tolist() == [False, True, False, True]


def test_verified_multitoken_stop_sequence_uses_final_position() -> None:
    tokens = torch.tensor([10, 40, 41, 42, 20])
    verified = torch.tensor([True, True, True, False, True])
    (position, token_id) = earliest_verified_stop_sequence(
        tokens, verified, stop_sequences=((40, 41, 42),)
    )
    assert position is None
    assert token_id is None
    verified[3] = True
    (position, token_id) = earliest_verified_stop_sequence(
        tokens, verified, stop_sequences=((40, 41, 42),)
    )
    assert (position, token_id) == (3, 42)


def test_stable_prefix_requires_non_mask_contiguous_agreement() -> None:
    old = torch.tensor([4, 5, 6, 7])
    new = torch.tensor([4, 5, 9, 7])
    assert stable_permutation_prefix_length(old, new, mask_token_id=99) == 2
    old_with_mask = torch.tensor([4, 99, 6])
    new_with_mask = torch.tensor([4, 99, 6])
    assert (
        stable_permutation_prefix_length(old_with_mask, new_with_mask, mask_token_id=99)
        == 1
    )


def test_parallel_refresh_mask_separates_clean_queries() -> None:
    mask = build_parallel_refresh_attention_mask(
        cache_len=2,
        pending_count=1,
        old_ids=torch.tensor([10, 99, 12]),
        mask_token_id=99,
    )
    assert mask.shape == (10, 12)
    assert bool(mask[:, :2].all())
    assert bool(mask[1, 3])
    assert not bool(mask[1, 6])
    assert mask[0].tolist() == [
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    assert mask[6].tolist() == [
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    assert mask[9].tolist() == [
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
    ]


def test_quadratic_refresh_mask_selectively_exposes_matched_prefixes() -> None:
    mask = build_quadratic_parallel_refresh_attention_mask(
        cache_len=1, pending_count=1, old_ids=torch.tensor([10, 11]), mask_token_id=99
    )
    assert mask.shape == (9, 10)
    assert bool(mask[:, 0].all())
    assert mask[5].tolist() == [
        True,
        True,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    assert mask[8].tolist() == [
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
    ]


def test_quadratic_refresh_smax16_row_formula() -> None:
    mask = build_quadratic_parallel_refresh_attention_mask(
        cache_len=5, pending_count=1, old_ids=torch.arange(8), mask_token_id=99
    )
    assert mask.shape == (81, 86)
    assert 1 + 2 * 8 + 8 * 8 == 81


def test_sliding_window_advances_without_waiting_for_fixed_block() -> None:
    verified = torch.zeros(24, dtype=torch.bool)
    (left, right, active) = sliding_window_active_positions(verified, window_size=16)
    assert (left, right) == (0, 16)
    assert active == list(range(16))
    verified[:3] = True
    (left, right, active) = sliding_window_active_positions(verified, window_size=16)
    assert (left, right) == (3, 19)
    assert active == list(range(3, 19))
    verified[7] = True
    (left, right, active) = sliding_window_active_positions(verified, window_size=16)
    assert (left, right) == (3, 19)
    assert active == [*range(3, 7), *range(8, 19)]
