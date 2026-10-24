# SPDX-License-Identifier: MIT
# Copyright © 2023-2026 Apple Inc.
"""Per-expert NVFP4 global scales for the pinned mlx-lm switch layers.

Backports the switch-layer half of ml-explore/mlx-lm#1844 ("Add global scale
for nvfp4 quantized MoEs"), which landed after oMLX's mlx-lm pin. Raon NVFP4
checkpoints store one FP32 tensor scale per expert (``*.global_scale``, shape
``(num_experts,)``) next to the E4M3 block scales; ``mx.gather_qmm`` already
takes it, only the mlx-lm wrappers do not.

The pinned loader asks a switch linear for ``to_quantized(group_size, bits,
mode)`` and, for paths named in ``config["quantization"]``, also forwards
``global_scale=True``. Paths it infers (Raon's merged ``gate_up``) never get
the flag, so an NVFP4 switch linear here defaults to carrying a global scale:
every Raon NVFP4 export has one per expert, and a checkpoint without it fails
the strict weight load instead of loading unscaled.
"""

from __future__ import annotations

import inspect

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import switch_layers


def upstream_supports_global_scale() -> bool:
    """True once the pinned mlx-lm switch layers take ``global_scale``."""
    params = inspect.signature(switch_layers.QuantizedSwitchLinear.__init__).parameters
    return "global_scale" in params


def _quantize_experts(w, *, group_size, bits, mode):
    """Quantize each expert with its own nvfp4 tensor scale (as upstream)."""
    if mode != "nvfp4":
        raise ValueError(f"A global scale needs 'nvfp4' mode, got '{mode}'.")
    gs = mx.max(mx.abs(w), axis=(-2, -1)).astype(mx.float32)
    qs = [
        mx.quantize(w[e], group_size, bits, mode=mode, global_scale=gs[e])
        for e in range(w.shape[0])
    ]
    return mx.stack([q for q, _ in qs]), mx.stack([s for _, s in qs]), gs


class QuantizedSwitchLinear(switch_layers.QuantizedSwitchLinear):
    """``QuantizedSwitchLinear`` with an optional per-expert NVFP4 tensor scale."""

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
        global_scale: bool = False,
    ):
        if not global_scale:
            super().__init__(
                input_dims, output_dims, num_experts, bias, group_size, bits, mode
            )
            self.global_scale = None
            return
        if mode != "nvfp4":
            raise ValueError(f"A global scale needs 'nvfp4' mode, got '{mode}'.")
        nn.Module.__init__(self)
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        # Placeholders in the packed NVFP4 layout; load_weights replaces them.
        self.weight = mx.zeros(
            (num_experts, output_dims, input_dims * bits // 32), mx.uint32
        )
        self.scales = mx.zeros(
            (num_experts, output_dims, input_dims // group_size), mx.uint8
        )
        self.biases = None
        self.global_scale = mx.ones((num_experts,), mx.float32)
        if bias:
            self.bias = mx.zeros((num_experts, output_dims))
        self.freeze()

    def __call__(self, x, indices, sorted_indices=False):
        x = mx.gather_qmm(
            x,
            self["weight"],
            self["scales"],
            self.get("biases"),
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            global_scale=self.get("global_scale"),
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x


class SwitchLinear(switch_layers.SwitchLinear):
    """``SwitchLinear`` whose NVFP4 quantization keeps a per-expert global scale."""

    def to_quantized(
        self,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
        global_scale: bool | None = None,
    ):
        if global_scale is None:
            global_scale = mode == "nvfp4"
        num_experts, output_dims, input_dims = self.weight.shape
        ql = QuantizedSwitchLinear(
            input_dims,
            output_dims,
            num_experts,
            "bias" in self,
            group_size,
            bits,
            mode=mode,
            global_scale=global_scale,
        )
        if global_scale:
            ql.weight, ql.scales, ql.global_scale = _quantize_experts(
                self.weight, group_size=group_size, bits=bits, mode=mode
            )
        else:
            ql.weight, ql.scales, *biases = mx.quantize(
                self.weight, group_size, bits, mode=mode
            )
            ql.biases = biases[0] if biases else None
        if "bias" in self:
            ql.bias = self.bias
        return ql


def install(module) -> bool:
    """Point a vendored model module at the global-scale switch layers.

    No-op once upstream mlx-lm carries the feature (a pin bump past #1844).
    """
    if upstream_supports_global_scale():
        return False
    module.SwitchLinear = SwitchLinear
    module.QuantizedSwitchLinear = QuantizedSwitchLinear
    return True
