import pytest
import torch
from nemotron_ltr import decoders, instrumentation, sampling
from nemotron_ltr.lora_runtime import row_selective_lora
from .toy_model import ToyModel, autoregressive
from .toy_model import FakeModel

METHODS = [
    decoders.strict_ltr_self_speculative,
    decoders.strict_ltr_quadratic_self_speculative,
    decoders.strict_ltr_packed_self_speculative,
]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("width", [2, 4, 8])
@pytest.mark.parametrize("limit", [1, 7, 19])
def test_greedy_matches_full_causal_recomputation(method, width, limit):
    prompt = torch.tensor([[1, 2, 3]])
    expected = autoregressive(prompt, limit)
    result = method(
        ToyModel(),
        prompt,
        max_new_tokens=limit,
        block_length=width,
        threshold=0.0,
        eos_token_id=None,
        max_uncached_rows_per_forward=width * (width + 1),
        record_model_call_walltimes=False,
        record_trajectory=True,
    )
    assert torch.equal(result.output_ids, expected)
    assert result.nfe == len(result.model_call_roles)
    assert len(result.output_ids[0]) == len(prompt[0]) + limit
    assert result.max_uncached_rows <= width * (width + 1)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("eos_offset", [0, 5])
def test_greedy_stops_at_first_eos(method, eos_offset):
    prompt = torch.tensor([[1, 2, 3]])
    eos = autoregressive(prompt, 12)[0, prompt.shape[1] + eos_offset].item()
    expected = autoregressive(prompt, 12, eos)
    result = method(
        ToyModel(),
        prompt,
        max_new_tokens=12,
        block_length=4,
        threshold=0.0,
        eos_token_id=eos,
        max_uncached_rows_per_forward=20,
        record_model_call_walltimes=False,
    )
    assert torch.equal(result.output_ids, expected)
    assert result.stopped_on_eos


@pytest.mark.parametrize("method", ["linear", "packed", "quadratic"])
@pytest.mark.parametrize("width", [2, 4, 8])
def test_sampling_cache_and_proposals_match_recomputation(method, width):
    torch.manual_seed(83)
    result = sampling.decode(
        ToyModel(), torch.tensor([[1, 2, 3]]), set(), method, width, 23, audit=True
    )
    assert result["generated_tokens"] == 23
    assert result["nfe"] == 1 + len(result["decode_rows_per_forward"])
    assert result["rounds"] == len(result["audit"])
    assert all(row["target_vs_full_recompute_tv"] < 1e-6 for row in result["audit"])


@pytest.mark.parametrize("method", ["linear", "packed", "quadratic"])
def test_instrumentation_preserves_random_stream_and_restores_forward(method):
    prompt = torch.tensor([[1, 2, 3]])
    torch.manual_seed(93)
    expected = sampling.decode(ToyModel(), prompt, set(), method, 4, 17)
    model = ToyModel()
    original = model.encoder.forward.__func__
    torch.manual_seed(93)
    result = instrumentation.decode(model, prompt, set(), method, 4, 17)
    assert result["output_ids"] == expected["output_ids"]
    assert result["nfe"] == expected["nfe"]
    assert model.encoder.forward.__func__ is original
    assert (
        sum(int(k) * v for k, v in result["emitted_tokens_per_round_histogram"].items())
        + 1
        == 17
    )


def test_lora_changes_only_selected_rows_and_restores_after_error():
    model = FakeModel()
    values = torch.arange(9).float().reshape(1, 3, 3)
    base = model(values)
    model.projection._disable_adapters = False
    adapted = model(values)
    model.projection._disable_adapters = True
    mask = torch.tensor([False, True, False])
    with pytest.raises(RuntimeError, match="deliberate"):
        with row_selective_lora(model, mask):
            torch.testing.assert_close(
                model(values), torch.where(mask[None, :, None], adapted, base)
            )
            raise RuntimeError("deliberate")
    torch.testing.assert_close(model(values), base)


@pytest.mark.parametrize("method", METHODS)
def test_empty_prompt_is_rejected(method):
    with pytest.raises(ValueError, match="prompt"):
        method(
            ToyModel(),
            torch.empty((1, 0), dtype=torch.long),
            max_new_tokens=8,
            block_length=4,
            threshold=0.0,
            eos_token_id=None,
            max_uncached_rows_per_forward=20,
            record_model_call_walltimes=False,
        )
