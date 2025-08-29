import threading

from typing import Dict, Generic, Hashable, Optional, Type, TypeVar
from weakref import WeakKeyDictionary

import torch.nn as nn

S = TypeVar("S")


class StateStore(Generic[S]):
    def __init__(self, state_cls: Type[S]) -> None:
        self._state_cls: Type[S] = state_cls
        # State can be shared across modules or by a single module.
        self._global: Dict[Hashable, S] = {}
        self._by_module: WeakKeyDictionary[nn.Module, Dict[Hashable, S]] = WeakKeyDictionary()
        self._lock = threading.RLock()

    def get(self, name: Hashable, module: Optional[nn.Module] = None) -> S:
        with self._lock:
            if module is None:
                st = self._global.get(name)
                if st is None:
                    st = self._state_cls()
                    self._global[name] = st
                return st
            bucket = self._by_module.setdefault(module, {})
            st = bucket.get(name)
            if st is None:
                st = self._state_cls()
                bucket[name] = st
            return st

    def set(self, name: Hashable, state: S, module: Optional[nn.Module] = None) -> None:
        with self._lock:
            if module is None:
                self._global[name] = state
            else:
                self._by_module.setdefault(module, {})[name] = state

    def has(self, name: Hashable, module: Optional[nn.Module] = None) -> bool:
        with self._lock:
            if module is None:
                return name in self._global
            bucket = self._by_module.get(module)
            return bucket is not None and name in bucket

    def delete(self, name: Hashable, module: Optional[nn.Module] = None) -> None:
        with self._lock:
            if module is None:
                self._global.pop(name, None)
            else:
                bucket = self._by_module.get(module)
                if bucket:
                    bucket.pop(name, None)

    def clear_for(self, module: nn.Module) -> None:
        with self._lock:
            self._by_module.pop(module, None)

    def clear_all(self) -> None:
        with self._lock:
            self._global.clear()
            self._by_module.clear()
