import inspect
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, masking_utils
from nemotron_ltr.loading import _load_model


def test_compatibility_wrapper_preserves_mask_factory_signature(monkeypatch, tmp_path):
    originals = {
        name: getattr(masking_utils, name)
        for name in ("create_causal_mask", "create_sliding_window_causal_mask")
    }
    for name, original in originals.items():
        monkeypatch.setattr(masking_utils, name, original)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1, dtype=torch.bfloat16))

        def to(self, device):
            return self

    loaded = []

    def load(model_id, **kwargs):
        loaded.append((model_id, kwargs))
        return Model()

    monkeypatch.setattr(AutoModel, "from_pretrained", load)
    monkeypatch.setattr(
        AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object()
    )
    config = dict(
        model=dict(
            id="nvidia/Nemotron-Labs-Diffusion-3B",
            revision="pinned",
            cache_dir=str(tmp_path),
        ),
        generation=dict(proposal_only_lora=False),
    )
    _load_model(config)
    _load_model(config)
    for name, original in originals.items():
        assert inspect.signature(getattr(masking_utils, name)) == inspect.signature(
            original
        )
    assert all(
        kwargs["dtype"] == torch.bfloat16 and kwargs["revision"] == "pinned"
        for _, kwargs in loaded
    )
