"""
Debug hooks for model porting between frameworks.

This module provides utilities to register forward and backward hooks on PyTorch models,
which is useful for debugging numerical differences when porting models between frameworks.
"""

import logging
import sys
import torch
import torch.nn as nn
from typing import List, Optional, Set, Callable, Union

# Module-level logger
logger = logging.getLogger(__name__)


def _default_print_fn(msg: str) -> None:
    print(msg, flush=True)

# Prefix constants for log messages
class Prefix:
    """Log message prefixes for filtering and grepping."""
    INIT = "INIT"   # Weight sums, initialization checks
    FWD = "FWD"     # Forward pass hooks
    BWD = "BWD"     # Backward pass hooks
    STATE = "STATE" # Model state diagnostics


def get_rank() -> int:
    """Get the current distributed rank, or 0 if not in distributed mode."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def tensor_sum(tensor: torch.Tensor) -> float:
    """Compute the sum of a tensor as float32 to prevent overflow."""
    return torch.sum(tensor.detach().to(torch.float32)).item()


class DebugHooks:
    """
    Context manager and utility class for registering debug hooks on PyTorch models.

    Hooks print the sum of inputs/outputs during forward pass and gradients during
    backward pass, which helps identify numerical differences when porting models.

    Example:
        >>> model = MyModel()
        >>> hooks = DebugHooks(model)
        >>> hooks.register()
        >>> output = model(input)  # Will print forward hook info
        >>> output.backward()       # Will print backward hook info
        >>> hooks.remove()

    Or as a context manager:
        >>> with DebugHooks(model):
        ...     output = model(input)
        ...     output.backward()
    """

    def __init__(
        self,
        model: nn.Module,
        skip_containers: bool = True,
        skip_types: Optional[Set[type]] = None,
        print_fn: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize debug hooks.

        Args:
            model: PyTorch model
            skip_containers: If True, only hook leaf modules (no children)
            skip_types: Set of module types to skip (default: {nn.Dropout})
            print_fn: Custom print function (default: logger.info)
        """
        self.model = model
        self.skip_containers = skip_containers
        self.skip_types = skip_types or {nn.Dropout}
        self.print_fn = print_fn or _default_print_fn
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def _log(self, tensor: Optional[torch.Tensor], name: str, tag: str) -> None:
        """Log tensor sum with rank information."""
        if tensor is None or not isinstance(tensor, torch.Tensor):
            return
        val = tensor_sum(tensor)
        self.print_fn(f"[Rank {get_rank()}][{tag}] {name} sum: {val}")

    def _log_tensors(
        self,
        data: Union[torch.Tensor, List, tuple, None],
        base_name: str,
        tag: str,
    ) -> None:
        """Log tensor or collection of tensors."""
        if isinstance(data, (list, tuple)):
            for i, item in enumerate(data):
                self._log(item, f"{base_name}[{i}]", tag)
        else:
            self._log(data, base_name, tag)

    def _make_forward_hook(self, name: str):
        """Create a forward hook for the given module name."""
        def hook(module, input, output):
            self._log_tensors(input, f"{name}.input", "FWD")
            self._log_tensors(output, f"{name}.output", "FWD")
        return hook

    def _make_backward_hook(self, name: str):
        """Create a backward hook for the given module name."""
        def hook(module, grad_input, grad_output):
            self._log_tensors(grad_output, f"{name}.grad_output", "BWD")
            self._log_tensors(grad_input, f"{name}.grad_input", "BWD")
        return hook

    def _should_hook(self, module: nn.Module) -> bool:
        """Determine if a module should have hooks registered."""
        if self.skip_containers and len(list(module.children())) > 0:
            return False
        if type(module) in self.skip_types:
            return False
        return True

    def register(self) -> "DebugHooks":
        """
        Register forward and backward hooks on the model.

        Returns:
            self for method chaining
        """
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
        """Remove all registered hooks using saved handles."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.print_fn(f"[Rank {get_rank()}][{Prefix.INIT}] Removed debug hooks")

    def __enter__(self) -> "DebugHooks":
        self.register()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove()


def register_debug_hooks(
    model: nn.Module,
    skip_containers: bool = True,
    skip_types: Optional[Set[type]] = None,
) -> DebugHooks:
    """
    Register debug hooks on a model (functional API).

    Args:
        model: PyTorch model
        skip_containers: If True, only hook leaf modules
        skip_types: Set of module types to skip

    Returns:
        DebugHooks instance with hooks registered

    Example:
        >>> hooks = register_debug_hooks(model)
        >>> # ... do forward/backward ...
        >>> hooks.remove()
    """
    return DebugHooks(model, skip_containers, skip_types).register()


def remove_debug_hooks_force(model: nn.Module) -> None:
    """
    Force remove all hooks from a model without needing handles.

    This is useful when you don't have access to the original hook handles.

    Args:
        model: PyTorch model
    """
    _default_print_fn(f"[Rank {get_rank()}][{Prefix.INIT}] Force removing all hooks...")

    for module in model.modules():
        if hasattr(module, "_forward_hooks"):
            module._forward_hooks.clear()
        if hasattr(module, "_backward_hooks"):
            module._backward_hooks.clear()
        if hasattr(module, "_forward_pre_hooks"):
            module._forward_pre_hooks.clear()
        if hasattr(module, "_backward_pre_hooks"):
            module._backward_pre_hooks.clear()

    _default_print_fn(f"[Rank {get_rank()}][{Prefix.INIT}] All hooks force removed")
