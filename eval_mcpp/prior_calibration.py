"""Subject-specific calibration of the final SongUNet output head."""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch


@dataclass(frozen=True)
class AdaptableHeadInfo:
    norm_name: str
    conv_name: str
    parameter_shapes: Dict[str, Tuple[int, ...]]
    adaptable_parameters: int
    total_parameters: int

    @property
    def adaptable_ratio(self) -> float:
        return 0.0 if self.total_parameters == 0 else self.adaptable_parameters / self.total_parameters


def _unwrap(net: torch.nn.Module) -> torch.nn.Module:
    return net.module if hasattr(net, "module") else net


def find_final_songunet_head(
    net: torch.nn.Module,
    *,
    strict_architecture: bool = True,
) -> Tuple[torch.nn.Module, torch.nn.Module, AdaptableHeadInfo]:
    """Find the final matched ``*_aux_norm`` + ``*_aux_conv`` in SongUNet decoder order."""
    root = _unwrap(net)
    model = getattr(root, "model", None)
    dec = getattr(model, "dec", None)
    if model is None or dec is None:
        raise TypeError("Expected Patch_EDMPrecond.model.dec from a SongUNet checkpoint")

    if strict_architecture:
        if root.__class__.__name__ != "Patch_EDMPrecond":
            raise TypeError(f"MCPP Stage-1 supports Patch_EDMPrecond, got {root.__class__.__name__}")
        if model.__class__.__name__ != "SongUNet":
            raise TypeError(f"MCPP Stage-1 supports SongUNet, got {model.__class__.__name__}")

    items = list(dec.items())
    conv_candidates = [
        (i, name, module)
        for i, (name, module) in enumerate(items)
        if name.endswith("_aux_conv")
    ]
    if not conv_candidates:
        raise RuntimeError("No *_aux_conv module was found in net.model.dec")

    conv_index, conv_key, conv_module = conv_candidates[-1]
    prefix = conv_key[: -len("_aux_conv")]
    norm_key = prefix + "_aux_norm"
    norm_candidates = [
        (i, module) for i, (name, module) in enumerate(items) if name == norm_key
    ]
    if not norm_candidates:
        raise RuntimeError(f"No matching {norm_key} was found for {conv_key}")
    norm_index, norm_module = norm_candidates[-1]
    if norm_index >= conv_index:
        raise RuntimeError("Final aux_norm must execute before its matching aux_conv")

    selected = {
        f"model.dec.{norm_key}.weight": getattr(norm_module, "weight", None),
        f"model.dec.{norm_key}.bias": getattr(norm_module, "bias", None),
        f"model.dec.{conv_key}.weight": getattr(conv_module, "weight", None),
        f"model.dec.{conv_key}.bias": getattr(conv_module, "bias", None),
    }
    missing = [
        name for name, parameter in selected.items()
        if not isinstance(parameter, torch.nn.Parameter)
    ]
    if missing:
        raise RuntimeError(
            "MCPP requires weight and bias for final aux_norm/aux_conv; "
            f"missing: {missing}"
        )

    shapes = {name: tuple(parameter.shape) for name, parameter in selected.items()}
    adaptable = sum(parameter.numel() for parameter in selected.values())
    total = sum(parameter.numel() for parameter in root.parameters())
    info = AdaptableHeadInfo(
        norm_name=f"model.dec.{norm_key}",
        conv_name=f"model.dec.{conv_key}",
        parameter_shapes=shapes,
        adaptable_parameters=adaptable,
        total_parameters=total,
    )
    return norm_module, conv_module, info


class MCPPHeadAdapter:
    """Freeze the population prior and maintain the subject-specific output-head state ``phi``."""

    def __init__(
        self,
        net: torch.nn.Module,
        *,
        strict_architecture: bool = True,
        print_info: bool = True,
    ) -> None:
        self.net = _unwrap(net)
        norm, conv, self.info = find_final_songunet_head(
            self.net, strict_architecture=strict_architecture
        )

        # 只让最终输出头适配；冻结参数不会阻断 measurement loss 对输入 x 的梯度。
        for parameter in self.net.parameters():
            parameter.requires_grad_(False)

        named = [
            (f"{self.info.norm_name}.weight", norm.weight),
            (f"{self.info.norm_name}.bias", norm.bias),
            (f"{self.info.conv_name}.weight", conv.weight),
            (f"{self.info.conv_name}.bias", conv.bias),
        ]
        self.names: List[str] = [name for name, _ in named]
        self.parameters: List[torch.nn.Parameter] = [parameter for _, parameter in named]
        for parameter in self.parameters:
            parameter.requires_grad_(True)

        self._phi0 = [parameter.detach().clone() for parameter in self.parameters]
        self._phi0_norm = torch.sqrt(sum(torch.sum(torch.abs(v) ** 2) for v in self._phi0)).detach()

        if print_info:
            self.print_summary()

    def print_summary(self) -> None:
        print(f"[MCPP] final aux_norm: {self.info.norm_name}")
        print(f"[MCPP] final aux_conv: {self.info.conv_name}")
        for name, shape in self.info.parameter_shapes.items():
            print(f"[MCPP] adaptable parameter: {name} shape={shape}")
        print(f"[MCPP] adaptable parameters: {self.info.adaptable_parameters}")
        print(
            "[MCPP] adaptable / total: "
            f"{self.info.adaptable_parameters}/{self.info.total_parameters} "
            f"({100.0 * self.info.adaptable_ratio:.6f}%)"
        )

    @torch.no_grad()
    def reset(self) -> None:
        """Restore the pretrained subject-independent output head before/after each subject."""
        for parameter, initial in zip(self.parameters, self._phi0):
            parameter.copy_(initial)

    @torch.no_grad()
    def step(self, normalized_gradients: Sequence[torch.Tensor], lr: float) -> None:
        """One explicit SGD calibration step; no optimizer state is introduced."""
        if len(normalized_gradients) != len(self.parameters):
            raise ValueError("Gradient count does not match the adaptable parameter count")
        for parameter, gradient in zip(self.parameters, normalized_gradients):
            parameter.add_(gradient, alpha=-float(lr))

    @torch.no_grad()
    def gradient_norm_tensor(self, gradients: Sequence[torch.Tensor]) -> torch.Tensor:
        squared = sum(torch.sum(torch.abs(gradient) ** 2) for gradient in gradients)
        return torch.sqrt(squared).detach()

    @torch.no_grad()
    def relative_delta_tensor(self, eps: float = 1e-12) -> torch.Tensor:
        delta_sq = sum(
            torch.sum(torch.abs(parameter - initial) ** 2)
            for parameter, initial in zip(self.parameters, self._phi0)
        )
        return (torch.sqrt(delta_sq) / (self._phi0_norm + eps)).detach()

    @torch.no_grad()
    def is_exactly_reset(self) -> bool:
        return all(
            torch.equal(parameter, initial)
            for parameter, initial in zip(self.parameters, self._phi0)
        )
