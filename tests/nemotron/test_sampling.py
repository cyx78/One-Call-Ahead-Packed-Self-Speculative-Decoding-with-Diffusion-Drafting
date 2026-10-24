from fractions import Fraction as F
from unittest.mock import patch
import torch
from nemotron_ltr.sampling import reject
from nemotron_ltr.sampling import draw


def test_equal_distributions_accept_all_and_use_target_bonus():
    q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    p = torch.cat((q, torch.tensor([[[1.0, 0.0]]])), dim=1)
    (segment, n) = reject(p, q, torch.tensor([[0, 1]]))
    assert n == 2 and segment.tolist() == [[0, 1, 0]]


def test_disjoint_support_rejects_first_and_draws_positive_residual():
    p = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
    q = torch.tensor([[[1.0, 0.0]]])
    (segment, n) = reject(p, q, torch.tensor([[0]]))
    assert n == 0 and segment.tolist() == [[1]]


def test_empty_carried_suffix_is_an_ordinary_target_draw():
    p = torch.tensor([[[0.0, 1.0]]])
    (segment, n) = reject(p, p[:, :0], torch.empty((1, 0), dtype=torch.long))
    assert n == 0 and segment.tolist() == [[1]]


def test_rejected_suffix_is_never_emitted():
    q = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]])
    p = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    (segment, n) = reject(p, q, torch.tensor([[0, 1, 0]]))
    assert n == 1 and segment.tolist() == [[0, 0]]


def test_actual_residual_is_normalized_positive_p_minus_q():
    p = torch.tensor([[[0.1, 0.3, 0.6], [0.2, 0.3, 0.5]]])
    q = torch.tensor([[[0.8, 0.15, 0.05]]])
    with patch(
        "nemotron_ltr.sampling.torch.rand_like", return_value=torch.tensor([[0.9]])
    ):
        with patch("nemotron_ltr.sampling.draw", wraps=draw) as mocked:
            (_, n) = reject(p, q, torch.tensor([[0]]))
    assert n == 0
    expected = torch.tensor([[[0.0, 0.15 / 0.7, 0.55 / 0.7]]])
    torch.testing.assert_close(mocked.call_args.args[0], expected)


def test_exact_rejection_law_for_arbitrary_stale_distributions():
    for p in [(F(1, 5), F(3, 10), F(1, 2)), (F(0), F(1), F(0))]:
        for q in [(F(4, 5), F(1, 10), F(1, 10)), p, (F(1), F(0), F(0))]:
            accepted = tuple((min(x, y) for (x, y) in zip(p, q)))
            rejection_mass = 1 - sum(accepted)
            residual = tuple((max(x - y, 0) for (x, y) in zip(p, q)))
            result = tuple(
                (
                    accepted[i]
                    + (
                        rejection_mass * residual[i] / sum(residual)
                        if rejection_mass
                        else 0
                    )
                    for i in range(3)
                )
            )
            assert result == p


def test_packed_suffix_offsets_include_correction_and_root():
    for width in (2, 8, 32):
        for advance in range(1, width + 1):
            c = 91
            positions = list(range(c, c + width))
            surviving = positions[advance + 1 :]
            assert surviving == list(range(c + advance + 1, c + width))
            assert len(surviving) == max(0, width - advance - 1)


def test_quadratic_selected_group_discards_correction_position():
    for width in (2, 8, 32):
        c = 91
        positions = [
            c + group + offset for group in range(width) for offset in range(width + 1)
        ]
        for n in range(width):
            start = n * (width + 1) + 2
            assert positions[start : start + width - 1] == list(
                range(c + n + 2, c + n + width + 1)
            )
