import pytest
import torch
from setdlm.decoders import linear_decode, packed_or_quadratic_decode
from .toy_model import ToyModel


@pytest.mark.parametrize("method", ["linear", "packed", "quadratic"])
@pytest.mark.parametrize("order", ["ltr", "confidence"])
@pytest.mark.parametrize("width", [2, 4, 8])
def test_full_decode_preserves_positions_and_call_counts(method, order, width):
    kwargs = {} if method == "linear" else {"execution_mode": method}
    decoder = linear_decode if method == "linear" else packed_or_quadratic_decode
    result = decoder(
        ToyModel(),
        torch.tensor([[1, 2]]),
        target_length=17,
        window_size=width,
        order_policy=order,
        record_trace=True,
        record_model_call_walltimes=True,
        **kwargs,
    )
    assert result.target_ids.tolist() == [list(range(3, 20))]
    assert result.converged
    assert result.total_forward_calls == result.decode_nfe + 1
    assert len(result.uncached_rows_per_forward) == result.decode_nfe
    assert len(result.model_call_walltimes_s) == result.decode_nfe
    assert sum(result.generated_tokens_per_step) == 17
    assert result.max_uncached_rows <= 1 + width * (width + 2)


@pytest.mark.parametrize("method", ["linear", "packed", "quadratic"])
@pytest.mark.parametrize("order", ["ltr", "confidence"])
def test_stop_sequence_is_retained_and_later_positions_are_discarded(method, order):
    kwargs = {} if method == "linear" else {"execution_mode": method}
    decoder = linear_decode if method == "linear" else packed_or_quadratic_decode
    result = decoder(
        ToyModel(),
        torch.tensor([[1, 2]]),
        target_length=20,
        window_size=4,
        order_policy=order,
        stop_sequences=((7, 8),),
        record_trace=True,
        **kwargs,
    )
    assert result.target_ids.tolist() == [[3, 4, 5, 6, 7, 8]]
    assert result.termination_position == 5
    assert result.converged


@pytest.mark.parametrize("method", ["linear", "packed", "quadratic"])
def test_context_eviction_preserves_semantic_positions(method):
    model = ToyModel()
    model.config = type("Config", (), {"length": 24})()
    kwargs = {} if method == "linear" else {"execution_mode": method}
    decoder = linear_decode if method == "linear" else packed_or_quadratic_decode
    result = decoder(
        model,
        torch.tensor([[1, 2]]),
        target_length=41,
        window_size=3,
        order_policy="ltr",
        record_trace=True,
        **kwargs,
    )
    assert result.target_ids.tolist() == [list(range(3, 44))]
    assert any(row["evicted_semantic_positions"] for row in result.trace)
