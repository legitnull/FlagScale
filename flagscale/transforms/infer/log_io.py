from typing import Any, Dict, List, Optional, Tuple

import torch

from torch import nn

from flagscale.engine.runtime_context import RuntimeContext
from flagscale.models.adapters import BaseAdapter
from flagscale.runner.utils import logger
from flagscale.transforms.hook import ModelHook, ModuleHookRegistry
from flagscale.transforms.transform import Transform, TransformSpec


class LogIOHook(ModelHook):
    """A simple hook that logs the input shapes of a module."""

    def __init__(self):
        super().__init__()

    def pre_forward(self, module: nn.Module, *args, **kwargs) -> Tuple[Tuple[Any], Dict[str, Any]]:
        def shape_of(x: torch.Tensor) -> str:
            return getattr(x, "shape", type(x).__name__)

        logger.info(
            f"[LogIOHook] {module.__class__.__name__} input shapes: "
            f"{tuple(shape_of(a) for a in args)}"
        )
        return args, kwargs


class LogIOTransform(Transform):
    def __init__(self, log_level: str = "info"):
        super().__init__()

        self._log_level = log_level
        self._spec = TransformSpec(
            name="log_io",
            phase="post_compile",
            priority=0,
            requires=set(),
            forbids=set(),
            before=set(),
            after=set(),
        )

    def spec(self) -> TransformSpec:
        return self._spec

    def supports(self, _model: BaseAdapter | nn.Module) -> bool:
        return True

    def preflight(self) -> bool:
        return True

    def apply(self, model: BaseAdapter) -> bool:
        backbone = model.backbone()

        if isinstance(backbone, nn.Module):
            reg = ModuleHookRegistry.get_or_create_registry(backbone)
            hook = LogIOHook()
            reg.register_hook(hook, "log_io")
            return True
        elif isinstance(backbone, list) and all(isinstance(m, nn.Module) for m in backbone):
            # TODO(yupu): Implement this
            raise NotImplementedError("Not implemented for multiple modules")
        else:
            raise ValueError(f"Unsupported backbone type: {type(backbone)}")
