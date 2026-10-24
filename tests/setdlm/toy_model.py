"Deterministic position-based denoiser for cache and stopping tests."

from __future__ import annotations
from types import SimpleNamespace
import torch


class ToyCache:
    def __init__(self, positions: torch.LongTensor) -> None:
        self.positions = positions.clone()

    def get_seq_length(self) -> int:
        return int(self.positions.numel())

    def crop(self, length: int) -> None:
        self.positions = self.positions[:length]


class ToyModel:
    class Config:
        length = 128

    config = Config()
    mask_token_id = 99

    @staticmethod
    def _new_generation_cache(*, batch_size: int, device: torch.device) -> None:
        assert batch_size == 1
        assert device.type == "cpu"
        return None

    @staticmethod
    def update_cache(
        prompt_ids: torch.LongTensor, *, cache: None, position_ids: torch.LongTensor
    ) -> dict[str, ToyCache]:
        assert prompt_ids.shape == position_ids.shape
        assert cache is None
        return {"past_key_values": ToyCache(position_ids[0])}

    @staticmethod
    def _preprocess_attention_mask(
        mask: torch.BoolTensor, *, dtype: torch.dtype
    ) -> torch.Tensor:
        return mask.to(dtype=dtype)

    @staticmethod
    def _select_kv_cache_positions(
        cache: ToyCache, keep_indices: torch.LongTensor
    ) -> ToyCache:
        cache.positions = cache.positions.index_select(0, keep_indices)
        return cache

    @staticmethod
    def _backbone_forward(denoiser_input: object) -> SimpleNamespace:
        position_ids = denoiser_input.backbone_kwargs["position_ids"]
        local_positions = position_ids[0]
        prior_cache = denoiser_input.past_key_values
        updated_cache = ToyCache(torch.cat((prior_cache.positions, local_positions)))
        vocab_size = 128
        logits = torch.full((1, int(local_positions.numel()), vocab_size), -100.0)
        top_ids = local_positions.remainder(50) + 1
        logits[0, torch.arange(local_positions.numel()), top_ids] = 0.0
        return SimpleNamespace(logits=logits, past_key_values=updated_cache)

    @staticmethod
    def _forward(
        logits: torch.FloatTensor, denoiser_input: object
    ) -> torch.FloatTensor:
        assert logits.shape[1] == denoiser_input.xt.shape[1]
        return logits
