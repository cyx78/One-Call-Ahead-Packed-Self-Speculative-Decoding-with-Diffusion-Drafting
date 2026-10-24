from __future__ import annotations
from types import SimpleNamespace
import pytest
import torch
from .toy_model import FakeModel
from nemotron_ltr.lora_runtime import describe_lora, require_lora, row_selective_lora


def test_lora_inventory_counts_targets_names_and_weights() -> None:
    inventory = describe_lora(FakeModel())
    assert inventory.target_module_count == 1
    assert inventory.adapter_names == ("default",)
    assert inventory.parameter_count == 5
    assert inventory.target_module_names == ("projection",)


def test_row_selective_lora_requires_globally_disabled_adapter() -> None:
    model = FakeModel()
    model.projection._disable_adapters = False
    with pytest.raises(RuntimeError, match="globally disabled"):
        with row_selective_lora(model, torch.tensor([True])):
            pass


def test_require_lora_rejects_plain_model() -> None:
    with pytest.raises(RuntimeError, match="no attached LoRA modules"):
        require_lora(SimpleNamespace(named_modules=lambda: []))
