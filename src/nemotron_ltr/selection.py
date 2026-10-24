from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class BranchSelection:
    """Clean-first selection result for one strict-LTR verification step."""

    winner: int
    stable_prefix_lengths: tuple[int, ...]
    accepted_lengths: tuple[int, ...]


def strict_ltr_acceptance(
    candidate_ids: torch.LongTensor, verifier_ids: torch.LongTensor
) -> tuple[int, int]:
    """Return consecutive draft matches and matches-plus-one acceptance.

    ``candidate_ids[j + 1]`` is the speculative token predicted for the next
    semantic position after candidate rank ``j``. ``verifier_ids[j]`` is the
    causal AR prediction for exactly that position. Only consecutive equality
    from the left is dependency-closed. The first disagreement contributes one
    guaranteed AR correction, as does the final verifier position when the
    whole candidate tail matches.
    """
    if candidate_ids.ndim != 1 or verifier_ids.ndim != 1:
        raise ValueError("candidate and verifier IDs must be one-dimensional")
    if candidate_ids.shape != verifier_ids.shape:
        raise ValueError("candidate and verifier IDs must have equal shapes")
    length = int(candidate_ids.numel())
    if length <= 0:
        raise ValueError("candidate and verifier must be nonempty")
    stable = 0
    for rank in range(length - 1):
        if int(candidate_ids[rank + 1].item()) != int(verifier_ids[rank].item()):
            break
        stable += 1
    return (stable, min(length, stable + 1))


def select_clean_first_ltr_branch(
    candidate_ids: torch.LongTensor, verifier_ids: torch.LongTensor
) -> BranchSelection:
    """Choose the first branch with maximum strict-LTR accepted length.

    Branch zero is the fresh clean branch. Python's ``max`` keeps the first
    occurrence of an equal key, giving the required clean-first tie behavior.
    """
    if candidate_ids.ndim != 2 or verifier_ids.ndim != 2:
        raise ValueError("candidate and verifier batches must be two-dimensional")
    if candidate_ids.shape != verifier_ids.shape:
        raise ValueError("candidate and verifier batches must have equal shapes")
    if int(candidate_ids.shape[0]) <= 0:
        raise ValueError("at least one candidate branch is required")
    stats = [
        strict_ltr_acceptance(candidate_ids[index], verifier_ids[index])
        for index in range(int(candidate_ids.shape[0]))
    ]
    stable = tuple((item[0] for item in stats))
    accepted = tuple((item[1] for item in stats))
    winner = max(range(len(accepted)), key=accepted.__getitem__)
    return BranchSelection(
        winner=winner, stable_prefix_lengths=stable, accepted_lengths=accepted
    )
