from __future__ import annotations
import torch
from nemotron_ltr.selection import select_clean_first_ltr_branch, strict_ltr_acceptance


def test_strict_ltr_acceptance_stops_at_first_mismatch() -> None:
    candidate = torch.tensor([10, 20, 30, 40, 50])
    verifier = torch.tensor([20, 99, 40, 50, 60])
    (stable, accepted) = strict_ltr_acceptance(candidate, verifier)
    assert stable == 1
    assert accepted == 2


def test_full_match_accepts_whole_block() -> None:
    candidate = torch.tensor([10, 20, 30, 40])
    verifier = torch.tensor([20, 30, 40, 50])
    assert strict_ltr_acceptance(candidate, verifier) == (3, 4)


def test_branch_selection_uses_clean_first_ties() -> None:
    candidates = torch.tensor([[10, 20, 30, 40], [10, 20, 31, 41]])
    verifiers = torch.tensor([[20, 99, 98, 97], [20, 88, 87, 86]])
    selection = select_clean_first_ltr_branch(candidates, verifiers)
    assert selection.accepted_lengths == (2, 2)
    assert selection.winner == 0
