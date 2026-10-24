"Restore checkpoint configuration targets and load FP32 SetDLM."

import json
from pathlib import Path
from typing import Any
from omegaconf import DictConfig, OmegaConf


_CHECKPOINT_MODULES = {
    "src.backbone.automodel": "setdlm.model.automodel",
    "src.backbone.custom_modeling_qwen3": "setdlm.model.custom_modeling_qwen3",
    "src.custom_transformers.generation": "setdlm.model.generation",
    "src.denoiser.base": "setdlm.model.base",
    "src.denoiser.bd3lm": "setdlm.model.bd3lm",
    "src.denoiser.diffusion_config": "setdlm.model.diffusion_config",
    "src.denoiser.mdlm": "setdlm.model.mdlm",
    "src.denoiser.setdlm": "setdlm.model.setdlm",
    "src.noise_schedule.noise_schedules": "setdlm.model.noise_schedules",
    "scripts.utils": "setdlm.model.tokenizer",
    "src.noise_schedule": "setdlm.model.noise_schedules",
}


def _restore_flattened_hf_target(target: str) -> str:
    """Map checkpoint module names to the local SetDLM model package."""
    for prefix in ("backbone", "noise_schedule", "denoiser", "custom_transformers"):
        if target.startswith(prefix + "_"):
            target = "src." + prefix + "." + target[len(prefix) + 1:]
            break
    for old, new in _CHECKPOINT_MODULES.items():
        if target == old or target.startswith(old + "."):
            return new + target[len(old):]
    return target


def _restore_flattened_hf_config_targets(config: Any) -> None:
    for attr in ("backbone_config", "noise_config", "tokenization_config"):
        value = getattr(config, attr, None)
        if isinstance(value, dict) and isinstance(value.get("_target_"), str):
            value["_target_"] = _restore_flattened_hf_target(value["_target_"])


def _hydrate_project_hf_config(config: Any, snapshot_path: str) -> None:
    config_path = Path(snapshot_path) / "config.json"
    try:
        raw_config = json.loads(config_path.read_text())
    except Exception:
        raw_config = {}
    tokenization_config = getattr(config, "tokenization_config", None)
    if isinstance(tokenization_config, DictConfig):
        tokenization_config = OmegaConf.to_container(tokenization_config, resolve=False)
    if not isinstance(tokenization_config, dict):
        tokenization_config = {}
    for key in (
        "vocab_size",
        "mask_token_id",
        "pad_token_id",
        "bos_token_id",
        "eos_token_id",
        "pad_vocab_size_multiple",
    ):
        value = tokenization_config.get(key, raw_config.get(key))
        if getattr(config, key, None) is None and value is not None:
            setattr(config, key, value)
    if getattr(config, "tokenizer_name", None) is None:
        config.tokenizer_name = snapshot_path


def load_model(horizon, cache_dir, device="cuda"):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from setdlm.model.tokenizer import maybe_add_missing_special_tokens
    from setdlm.model.bd3lm import BD3LMConfig
    from setdlm.model.setdlm import SetDLM

    revisions = {
        16: "555271c1ef014ecd78a5c11d1e8849a6fc721388",
        32: "fe334b030407c145e76fd3f64472fc0fc1235a86",
    }
    model_id = f"kuleshov-group/setdlm-gsm8k-smax{horizon}"
    snapshot = snapshot_download(
        model_id, revision=revisions[horizon], cache_dir=str(cache_dir)
    )
    tokenizer = maybe_add_missing_special_tokens(
        AutoTokenizer.from_pretrained(snapshot, trust_remote_code=True)
    )
    config = BD3LMConfig.from_pretrained(snapshot)
    _hydrate_project_hf_config(config, snapshot)
    _restore_flattened_hf_config_targets(config)
    config.attn_backend = "sdpa"
    config.backbone_config["revision"] = "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
    config.backbone_config["cache_dir"] = str(cache_dir)
    config.setdlm_fht_cache_order = True
    config.noise_config["max_block_size"] = horizon
    model = (
        SetDLM.from_pretrained(
            snapshot, config=config, tokenizer=tokenizer, torch_dtype=torch.float32
        )
        .to(device)
        .eval()
    )
    return (model, tokenizer, next(model.parameters()).device)
