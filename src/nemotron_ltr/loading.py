"Load BF16 Nemotron checkpoints and optional proposal adapters."

from __future__ import annotations
from functools import wraps
from pathlib import Path
from typing import Any
import torch


def _load_model(config: dict[str, Any]) -> tuple[Any, Any, torch.device]:
    from transformers import AutoModel, AutoTokenizer

    cache_dir = _relative_path(config["model"]["cache_dir"])
    model_id = str(config["model"]["id"])
    model_revision = str(config["model"]["revision"])
    if model_id == "nvidia/Nemotron-Labs-Diffusion-3B":
        from transformers import masking_utils

        if not hasattr(masking_utils, "sdpa_mask_older_torch"):
            masking_utils.sdpa_mask_older_torch = masking_utils.sdpa_mask

        def legacy_mask_factory(function: Any) -> Any:
            if getattr(function, "_nemotron_3b_legacy_keywords", False):
                return function

            @wraps(function)
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                if "input_embeds" in kwargs and "inputs_embeds" not in kwargs:
                    kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
                kwargs.pop("cache_position", None)
                return function(*args, **kwargs)

            wrapped._nemotron_3b_legacy_keywords = True
            return wrapped

        masking_utils.create_causal_mask = legacy_mask_factory(
            masking_utils.create_causal_mask
        )
        masking_utils.create_sliding_window_causal_mask = legacy_mask_factory(
            masking_utils.create_sliding_window_causal_mask
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=model_revision, cache_dir=cache_dir, trust_remote_code=True
    )
    base = AutoModel.from_pretrained(
        model_id,
        revision=model_revision,
        cache_dir=cache_dir,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    ).to("cuda")
    if bool(config["generation"]["proposal_only_lora"]):
        from peft import PeftModel

        wrapped = PeftModel.from_pretrained(
            base, _relative_path(config["model"]["adapter_path"]), is_trainable=False
        ).eval()
        model = wrapped.model
    else:
        model = base
    model.eval()
    device = next(model.parameters()).device
    return (model, tokenizer, device)


def _relative_path(value):
    return Path(value).expanduser().resolve()
