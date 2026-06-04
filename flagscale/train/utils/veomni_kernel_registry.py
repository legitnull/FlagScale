"""Kernel registry for the OpSlot dispatch system.

Supports lazy-loaded kernels via factory functions and direct callable registration.
"""

from __future__ import annotations

from typing import Callable, Optional


class KernelSpec:
    """Descriptor for a kernel implementation."""

    def __init__(
        self,
        name: str,
        op_name: str,
        variant: str,
        factory: Callable[[], Callable],
        description: str = "",
    ):
        self.name = name
        self.op_name = op_name
        self.variant = variant
        self.factory = factory
        self.description = description
        self._kernel: Optional[Callable] = None

    def get_kernel(self) -> Callable:
        if self._kernel is None:
            self._kernel = self.factory()
        return self._kernel


class _KernelRegistry:
    """Registry mapping (op_name, variant, impl_name) to kernel callables."""

    def __init__(self):
        self._registry: dict[tuple[str, str, str], KernelSpec] = {}

    def register(self, spec: KernelSpec):
        key = (spec.op_name, spec.variant, spec.name)
        self._registry[key] = spec

    def resolve(self, op_name: str, variant: str, impl_name: str) -> Optional[Callable]:
        if impl_name == "eager" or impl_name is None:
            return None
        spec = self._registry.get((op_name, variant, impl_name))
        if spec is None:
            return None
        return spec.get_kernel()


KERNEL_REGISTRY = _KernelRegistry()


def _fla_fused_rms_norm_gated_factory():
    from fla.modules import FusedRMSNormGated
    return FusedRMSNormGated


def _fla_causal_conv1d_factory():
    from fla.modules.convolution import causal_conv1d
    return causal_conv1d


def _fla_chunk_gated_delta_rule_factory():
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    return chunk_gated_delta_rule


KERNEL_REGISTRY.register(KernelSpec(
    name="fla",
    op_name="rms_norm_gated",
    variant="standard",
    factory=_fla_fused_rms_norm_gated_factory,
    description="flash-linear-attention FusedRMSNormGated",
))

KERNEL_REGISTRY.register(KernelSpec(
    name="fla",
    op_name="causal_conv1d",
    variant="standard",
    factory=_fla_causal_conv1d_factory,
    description="flash-linear-attention causal conv1d (Triton, varlen-aware)",
))

KERNEL_REGISTRY.register(KernelSpec(
    name="fla",
    op_name="chunk_gated_delta_rule",
    variant="standard",
    factory=_fla_chunk_gated_delta_rule_factory,
    description="flash-linear-attention chunk gated delta rule (Triton)",
))
