import contextlib
import contextvars
import uuid

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Hashable, Literal, Optional, Tuple

import torch

_current_ctx: contextvars.ContextVar["RuntimeContext | None"] = contextvars.ContextVar(
    "flagscale_runtime_ctx", default=None
)


@dataclass
class RuntimeContext:
    """
    Runtime context for a model's forward pass.
    """

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    # per-step signals (managed by step tracker hook or manual step())
    total_steps: Optional[int] = None
    step_index: int = -1
    timestep: Optional[int] = None

    # small metadata only (no tensors)
    extras: Dict[str, Any] = field(default_factory=dict)

    @contextlib.contextmanager
    def session(self, total_steps: Optional[int] = None):
        token = _current_ctx.set(self)
        if total_steps is not None:
            self.total_steps = total_steps
        self.step_index, self.timestep = -1, None
        try:
            yield
        finally:
            _current_ctx.reset(token)

    # 不一定用得上
    @contextlib.contextmanager
    def step(self, step_index: Optional[int] = None, timestep: Optional[int] = None, **extras: Any):
        prev = (self.step_index, self.timestep, dict(self.extras))
        if step_index is not None:
            self.step_index = step_index
        if timestep is not None:
            self.timestep = timestep
        if extras:
            self.extras.update(extras)
        try:
            yield
        finally:
            self.step_index, self.timestep, self.extras = prev

    def partition_key(
        self, scope: Literal["session", "global"] = "session"
    ) -> Tuple[Hashable, ...]:
        if scope == "global":
            return (self.model_version, self.device, self.dtype)
        return (self.run_id, self.device, self.dtype)

    # 判断下是否存在
    def set_extra(self, key: str, value: Any) -> None:
        self.extras[key] = value

    def get_extra(self, key: str, default: Any = None) -> Any:
        return self.extras.get(key, default)

    @contextlib.contextmanager
    def push_extras(self, mapping: Dict[str, Any]):
        prev = dict(self.extras)
        self.extras.update(mapping)
        try:
            yield
        finally:
            self.extras = prev

    def to_dict(self, include_extras: bool = False) -> Dict[str, Any]:
        d = asdict(self)
        if not include_extras:
            d.pop("extras", None)
        return d

    @classmethod
    def current(cls) -> Optional["RuntimeContext"]:
        return _current_ctx.get()

    @classmethod
    def is_active(cls) -> bool:
        return cls.current() is not None


# Optional module-level alias for convenience
def current_runtime() -> Optional[RuntimeContext]:
    return RuntimeContext.current()
