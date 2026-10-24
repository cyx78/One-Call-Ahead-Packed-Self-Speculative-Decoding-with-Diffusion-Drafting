"Apply proposal-only LoRA deltas with dense row masking."

from __future__ import annotations
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Iterator
import torch


@dataclass(frozen=True)
class LoraInventory:
    target_module_count: int
    adapter_names: tuple[str, ...]
    parameter_count: int
    target_module_names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["adapter_names"] = list(self.adapter_names)
        payload["target_module_names"] = list(self.target_module_names)
        return payload


def _lora_modules(model: Any) -> tuple[tuple[str, Any], ...]:
    modules: list[tuple[str, Any]] = []
    for name, module in model.named_modules():
        if (
            hasattr(module, "base_layer")
            and hasattr(module, "lora_A")
            and hasattr(module, "lora_B")
            and hasattr(module, "scaling")
            and hasattr(module, "active_adapters")
        ):
            modules.append((name, module))
    return tuple(modules)


def describe_lora(model: Any) -> LoraInventory:
    """Return a serializable inventory of LoRA modules attached to ``model``."""
    modules = _lora_modules(model)
    adapter_names: set[str] = set()
    parameters: dict[int, torch.nn.Parameter] = {}
    for _, module in modules:
        adapter_names.update((str(value) for value in module.lora_A.keys()))
        for parameter in module.lora_A.parameters():
            parameters[id(parameter)] = parameter
        for parameter in module.lora_B.parameters():
            parameters[id(parameter)] = parameter
    return LoraInventory(
        target_module_count=len(modules),
        adapter_names=tuple(sorted(adapter_names)),
        parameter_count=sum((int(value.numel()) for value in parameters.values())),
        target_module_names=tuple((name for (name, _) in modules)),
    )


def require_lora(model: Any) -> LoraInventory:
    inventory = describe_lora(model)
    if inventory.target_module_count == 0:
        raise RuntimeError(
            "proposal LoRA was requested, but the model has no attached LoRA modules"
        )
    if not inventory.adapter_names:
        raise RuntimeError("attached LoRA modules expose no adapter weights")
    return inventory


class _RowSelectiveLoraController:
    def __init__(self, model: Any) -> None:
        named_modules = _lora_modules(model)
        if not named_modules:
            raise RuntimeError(
                "row-selective LoRA requires an attached PEFT LoRA adapter"
            )
        self.named_modules = named_modules
        self.row_mask: torch.BoolTensor | None = None
        self.handles = [
            module.register_forward_hook(self._post_forward, with_kwargs=True)
            for (_, module) in named_modules
        ]

    def activate(self, row_mask: torch.BoolTensor) -> None:
        if self.row_mask is not None:
            raise RuntimeError("row-selective LoRA contexts cannot be nested")
        if row_mask.ndim not in (1, 2):
            raise ValueError("LoRA row mask must have shape [Q] or [batch, Q]")
        if row_mask.dtype != torch.bool:
            raise TypeError("LoRA row mask must be Boolean")
        for name, module in self.named_modules:
            if bool(getattr(module, "merged", False)):
                raise RuntimeError(f"LoRA module {name!r} is merged")
            if not bool(getattr(module, "disable_adapters", False)):
                raise RuntimeError(
                    f"LoRA module {name!r} must be globally disabled before row-selective activation"
                )
            variants = getattr(module, "lora_variant", {})
            for adapter in module.active_adapters:
                if adapter in variants:
                    raise RuntimeError(
                        f"row-selective runtime supports vanilla LoRA only; module {name!r}, adapter {adapter!r} uses a variant"
                    )
        self.row_mask = row_mask

    def deactivate(self) -> None:
        self.row_mask = None

    @staticmethod
    def _expanded_mask(
        inputs: torch.Tensor, row_mask: torch.BoolTensor
    ) -> torch.BoolTensor:
        """Broadcast a query-row mask over every leading input dimension."""
        query_rows = int(inputs.shape[-2])
        leading_shape = tuple((int(value) for value in inputs.shape[:-1]))
        if row_mask.ndim == 1:
            view_shape = (1,) * (inputs.ndim - 2) + (query_rows,)
            return row_mask.reshape(view_shape).expand(leading_shape)
        if inputs.ndim < 3:
            raise ValueError("a batched LoRA row mask requires batched inputs")
        batch = int(inputs.shape[0])
        selected = row_mask
        if int(selected.shape[0]) == 1 and batch != 1:
            selected = selected.expand(batch, query_rows)
        view_shape = (batch,) + (1,) * (inputs.ndim - 3) + (query_rows,)
        return selected.reshape(view_shape).expand(leading_shape)

    def _adapter_delta(
        self, module: Any, inputs: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        """Apply every active vanilla adapter with PEFT-compatible casting."""
        adapted_output = output
        for adapter in module.active_adapters:
            if adapter not in module.lora_A:
                continue
            lora_a = module.lora_A[adapter]
            lora_b = module.lora_B[adapter]
            dropout = module.lora_dropout[adapter]
            cast_inputs = module._cast_input_dtype(inputs, lora_a.weight.dtype)
            adapter_delta = lora_b(lora_a(dropout(cast_inputs)))
            adapter_delta = adapter_delta * module.scaling[adapter]
            adapted_output = adapted_output + adapter_delta
        return adapted_output.to(output.dtype)

    def _post_forward(
        self, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any
    ) -> Any:
        row_mask = self.row_mask
        if row_mask is None:
            return output
        if not isinstance(output, torch.Tensor):
            raise TypeError("LoRA target output must be a tensor")
        if args:
            inputs = args[0]
        else:
            inputs = kwargs.get("x")
        if not isinstance(inputs, torch.Tensor) or inputs.ndim < 2:
            raise TypeError("LoRA target input must be a rank-two-or-greater tensor")
        query_rows = int(inputs.shape[-2])
        if int(row_mask.shape[-1]) != query_rows:
            raise ValueError(
                f"LoRA row mask has {row_mask.shape[-1]} rows, but target input has {query_rows}"
            )
        if row_mask.ndim == 2 and inputs.ndim >= 3:
            batch = int(inputs.shape[0])
            if int(row_mask.shape[0]) not in (1, batch):
                raise ValueError(
                    f"LoRA row-mask batch {row_mask.shape[0]} does not match {batch}"
                )
        mask = self._expanded_mask(inputs, row_mask.to(device=output.device))
        adapted_output = self._adapter_delta(module, inputs, output)
        return torch.where(mask.unsqueeze(-1), adapted_output, output)


def _controller(model: Any) -> _RowSelectiveLoraController:
    attribute = "_nemotron_row_selective_lora_controller"
    controller = getattr(model, attribute, None)
    if controller is None:
        controller = _RowSelectiveLoraController(model)
        setattr(model, attribute, controller)
    return controller


@contextmanager
def row_selective_lora(model: Any, row_mask: torch.BoolTensor) -> Iterator[None]:
    """Evaluate adapter deltas, then apply them only to selected proposal rows."""
    controller = _controller(model)
    controller.activate(row_mask)
    try:
        yield
    finally:
        controller.deactivate()
