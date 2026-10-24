import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, Qwen3Config
from setdlm.model.bd3lm import BD3LMConfig
from setdlm.model.setdlm import SetDLM
from setdlm.benchmark import _native_decode
from setdlm.loading import _restore_flattened_hf_target, _restore_flattened_hf_config_targets
from setdlm.decoders import linear_decode, packed_or_quadratic_decode


@pytest.fixture
def model_and_tokenizer(tmp_path):
    torch.manual_seed(17)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
        eos_token_id=1,
        bos_token_id=0,
    )
    config.save_pretrained(tmp_path)
    vocabulary = {str(index): index for index in range(32)}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocabulary, unk_token="30")),
        bos_token="0",
        eos_token="1",
        pad_token="2",
        mask_token="31",
        unk_token="30",
    )
    config = BD3LMConfig(
        length=64,
        block_size=64,
        attn_backend="sdpa",
        setdlm_fht_cache_order=True,
        backbone_config=dict(
            _target_="backbone_automodel.AutoModelFromPreTrained",
            automodel_cls="AutoModelForCausalLM",
            pretrained_model_name_or_path=str(tmp_path),
            num_layers=2,
            reinit_model=True,
        ),
        noise_config=dict(
            _target_="src.noise_schedule.noise_schedules.StaggeredNoise",
            length=64,
            block_size=64,
            desired_block_size=2,
            max_block_size=4,
        ),
        tokenization_config=dict(
            vocab_size=32,
            mask_token_id=31,
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=2,
            pad_vocab_size_multiple=1,
        ),
    )
    _restore_flattened_hf_config_targets(config)
    return SetDLM(config, tokenizer=tokenizer).eval(), tokenizer


@pytest.mark.parametrize("method", ["linear", "packed", "quadratic"])
def test_qwen_backbone_cached_decode_matches_single_position_reference(
    model_and_tokenizer, method
):
    model, _ = model_and_tokenizer
    prompt = torch.tensor([[4, 5, 6]])
    expected = linear_decode(
        model, prompt, target_length=8, window_size=1, order_policy="ltr"
    )
    kwargs = {} if method == "linear" else {"execution_mode": method}
    decoder = linear_decode if method == "linear" else packed_or_quadratic_decode
    actual = decoder(
        model, prompt, target_length=8, window_size=3, order_policy="ltr", **kwargs
    )
    assert torch.equal(actual.target_ids, expected.target_ids)
    assert actual.converged


def test_native_generation_accounts_for_every_emitted_token(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    tokens, metrics = _native_decode(
        model,
        tokenizer,
        torch.tensor([[4, 5, 6]]),
        target_length=8,
        window_size=2,
        seed=17,
    )
    assert tokens.shape == (1, 8)
    assert not bool((tokens == model.mask_token_id).any())
    assert sum(metrics["generated_tokens_per_step"]) == 8
    assert metrics["total_forward_calls"] == metrics["decode_nfe"] + 1


@pytest.mark.parametrize("target,expected", [
    ("src.backbone.automodel.AutoModelFromPreTrained", "setdlm.model.automodel.AutoModelFromPreTrained"),
    ("backbone_automodel.AutoModelFromPreTrained", "setdlm.model.automodel.AutoModelFromPreTrained"),
    ("src.noise_schedule.noise_schedules.StaggeredNoise", "setdlm.model.noise_schedules.StaggeredNoise"),
    ("noise_schedule_noise_schedules.StaggeredNoise", "setdlm.model.noise_schedules.StaggeredNoise"),
    ("src.noise_schedule.StaggeredNoise", "setdlm.model.noise_schedules.StaggeredNoise"),
    ("denoiser_bd3lm.BD3LMConfig", "setdlm.model.bd3lm.BD3LMConfig"),
    ("scripts.utils.maybe_add_missing_special_tokens", "setdlm.model.tokenizer.maybe_add_missing_special_tokens"),
    ("setdlm.model.base.DenoiserInput", "setdlm.model.base.DenoiserInput"),
    ("transformers.AutoTokenizer", "transformers.AutoTokenizer"),
])
def test_checkpoint_targets_resolve_after_reorganization(target, expected):
    from hydra.utils import get_object
    resolved = _restore_flattened_hf_target(target)
    assert resolved == expected
    assert get_object(resolved) is not None
