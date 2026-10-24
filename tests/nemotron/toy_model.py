"""A one-layer decoder with explicit visibility and real DynamicCache storage."""

from types import SimpleNamespace
import torch
from torch import nn
from transformers import DynamicCache


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_lm = False

    def forward(self, hidden_states, **kwargs):
        return hidden_states, None


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Module()])
        self.layers[0].self_attn = Attention()

    def forward(
        self,
        input_ids,
        past_key_values=None,
        use_cache=False,
        use_causal_mask=False,
        position_ids=None,
        **kwargs,
    ):
        cache = past_key_values if past_key_values is not None else DynamicCache()
        prefix = cache.get_seq_length()
        rows = input_ids.shape[1]
        if position_ids is None:
            position_ids = torch.arange(prefix, prefix + rows)[None, :]
        local = input_ids.float()[:, None, :, None]
        previous = cache.layers[0].keys if prefix else local[:, :, :0]
        keys = torch.cat((previous, local), dim=2)
        mask = getattr(self.layers[0].self_attn, "_nemotron_runtime_mask", None)
        if mask is None:
            mask = torch.ones((rows, prefix + rows), dtype=torch.bool)
            if use_causal_mask or not self.layers[0].self_attn.diffusion_lm:
                mask[:, prefix:] = torch.ones((rows, rows), dtype=torch.bool).tril()
        assert mask.shape == (rows, prefix + rows)
        values = keys[0, 0, :, 0].clone()
        values[values == 16] = 0
        context = (mask * values[None, :]).sum(-1)[None, :]
        position = position_ids + (input_ids != 16)
        vocabulary = torch.arange(16).float()
        logits = torch.cos(
            context[..., None] * 0.7 + position[..., None] * 0.3 + vocabulary * 0.9
        )
        logits = torch.cat(
            (logits, torch.full((*logits.shape[:-1], 1), -100.0)), dim=-1
        )
        if use_cache:
            cache.update(local, local, 0)
        return SimpleNamespace(last_hidden_state=logits, past_key_values=cache)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.config = SimpleNamespace(
            mask_token_id=16, eos_token_id=None, max_position_embeddings=1024
        )

    @staticmethod
    def diffusion_head(hidden):
        return hidden


def autoregressive(prompt, limit, eos=None):
    model = ToyModel()
    sequence = prompt.clone()
    for _ in range(limit):
        logits = model.encoder(
            sequence, use_cache=False, use_causal_mask=True
        ).last_hidden_state
        token = logits[:, -1:].argmax(-1)
        sequence = torch.cat((sequence, token), dim=1)
        if eos is not None and token.item() == eos:
            break
    return sequence


class FakeLoraLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_layer = nn.Linear(3, 2, bias=False)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(3, 1, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(1, 2, bias=False)})
        self.lora_dropout = nn.ModuleDict({"default": nn.Identity()})
        self.scaling = {"default": 2.0}
        self.active_adapters = ["default"]
        self.lora_variant: dict[str, object] = {}
        self._disable_adapters = True
        self.merged = False

    @property
    def disable_adapters(self) -> bool:
        return self._disable_adapters

    @staticmethod
    def _cast_input_dtype(value: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return value.to(dtype)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(value)
        if not self.disable_adapters:
            result = result + 2.0 * self.lora_B["default"](
                self.lora_A["default"](value)
            )
        return result


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = FakeLoraLinear()
        with torch.no_grad():
            self.projection.base_layer.weight.copy_(
                torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            )
            self.projection.lora_A["default"].weight.copy_(
                torch.tensor([[1.0, 1.0, 1.0]])
            )
            self.projection.lora_B["default"].weight.copy_(
                torch.tensor([[1.0], [-1.0]])
            )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.projection(value)
