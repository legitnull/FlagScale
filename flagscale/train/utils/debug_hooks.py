"""
Debug hooks for logging activations and gradients during forward/backward passes.

Adapted from null_space/hamster/debug/hooks.py for use with flagscale training.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Set, Callable, Union

import logging

logger = logging.getLogger(__name__)


class Prefix:
    INIT = "INIT"
    FWD = "FWD"
    BWD = "BWD"


def get_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def tensor_stats(tensor: torch.Tensor) -> str:
    """Compute sum, mean, max, min of a tensor as float32."""
    t = tensor.detach().float()
    return f"sum={t.sum().item():.6g} mean={t.mean().item():.6g} max={t.max().item():.6g} min={t.min().item():.6g} shape={list(tensor.shape)}"


class DebugHooks:
    """Register forward and backward hooks on leaf modules to log activation/gradient stats."""

    def __init__(
        self,
        model: nn.Module,
        skip_containers: bool = True,
        skip_types: Optional[Set[type]] = None,
        print_fn: Optional[Callable[[str], None]] = None,
    ):
        self.model = model
        self.skip_containers = skip_containers
        self.skip_types = skip_types or {nn.Dropout}
        self.print_fn = print_fn or logger.info
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def _log(self, tensor: Optional[torch.Tensor], name: str, tag: str) -> None:
        if tensor is None or not isinstance(tensor, torch.Tensor):
            return
        stats = tensor_stats(tensor)
        self.print_fn(f"[Rank {get_rank()}][{tag}] {name} {stats}")

    def _log_tensors(self, data: Union[torch.Tensor, List, tuple, None], base_name: str, tag: str) -> None:
        if isinstance(data, (list, tuple)):
            for i, item in enumerate(data):
                self._log(item, f"{base_name}[{i}]", tag)
        else:
            self._log(data, base_name, tag)

    def _make_forward_hook(self, name: str):
        def hook(module, input, output):
            self._log_tensors(input, f"{name}.input", Prefix.FWD)
            self._log_tensors(output, f"{name}.output", Prefix.FWD)
        return hook

    def _make_backward_hook(self, name: str):
        def hook(module, grad_input, grad_output):
            self._log_tensors(grad_output, f"{name}.grad_output", Prefix.BWD)
            self._log_tensors(grad_input, f"{name}.grad_input", Prefix.BWD)
        return hook

    def _should_hook(self, module: nn.Module) -> bool:
        if self.skip_containers and len(list(module.children())) > 0:
            return False
        if type(module) in self.skip_types:
            return False
        return True

    def register(self) -> "DebugHooks":
        self.print_fn(f"[Rank {get_rank()}][{Prefix.INIT}] Registering debug hooks...")
        for name, module in self.model.named_modules():
            if not self._should_hook(module):
                continue
            handle_fwd = module.register_forward_hook(self._make_forward_hook(name))
            handle_bwd = module.register_full_backward_hook(self._make_backward_hook(name))
            self._handles.extend([handle_fwd, handle_bwd])
        self.print_fn(f"[Rank {get_rank()}][{Prefix.INIT}] Registered {len(self._handles)} hooks")
        return self

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "DebugHooks":
        self.register()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove()
