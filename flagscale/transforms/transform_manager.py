from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch.nn as nn

from flagscale.transforms.transform import Transform


@dataclass
class _PhasePlan:
    pre: List[Transform]
    compile: List[Transform]
    post: List[Transform]


# TODO(yupu): support dry-run mode
# TODO(yupu): Optionally support `strict=False` mode, where invalid transforms are pruned along with dependents
class TransformManager:
    """
    Orders and executes transforms by phase:
      pre_compile → compile → post_compile
    """

    def __init__(self, transforms: Sequence[Transform], *, strict: bool = True) -> None:
        self._all: Dict[str, Transform] = {t.spec().name: t for t in transforms}
        self._strict = strict

    def _partition(
        self, names: Set[str]
    ) -> Tuple[List[Transform], List[Transform], List[Transform]]:
        """
        Partition the transforms into pre-compile, compile, and post-compile phases.
        """
        pre: List[Transform] = []
        comp: List[Transform] = []
        post: List[Transform] = []
        for n in names:
            t = self._all[n]
            ph = t.spec().phase
            if ph == "pre_compile":
                pre.append(t)
            elif ph == "compile":
                comp.append(t)
            else:
                post.append(t)
        return pre, comp, post

    def _validate_and_select(self, model: nn.Module) -> Set[str]:
        """
        1. check `supports()` and `preflight()`
        2. check requires, forbids, and cross-phase requirements
        3. return the set of valid transform names
        """
        return ""

    def _sort_phase(self, transforms: List[Transform]) -> List[Transform]:
        """
        Topological sort the transforms in the phase.
        """
        pass

    def plan(self, model: nn.Module) -> _PhasePlan:
        """
        Plan the transforms to be applied to the model.
        """
        active = self._validate_and_select(model)
        pre, comp, post = self._partition(active)
        return _PhasePlan(
            pre=self._sort_phase(pre), compile=self._sort_phase(comp), post=self._sort_phase(post)
        )

    # TODO(yupu): support List[nn.Module] or BaseAdapter?
    def apply(self, model: nn.Module) -> None:
        """
        Apply the transforms in the order specified by the plan.
        """
        plan = self.plan(model)
        for t in plan.pre:
            t.apply(model)
        for t in plan.compile:
            t.apply(model)
        for t in plan.post:
            t.apply(model)
