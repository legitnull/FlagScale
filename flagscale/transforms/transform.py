from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Set

from torch import nn

from flagscale.models.adapters import BaseAdapter

# To mark whether the transform should be applied before/after the `torch.compile`
TransformPhase = Literal["pre_compile", "compile", "post_compile"]


@dataclass
class TransformSpec:
    """
    A description of a transform.
    """

    name: str
    phase: TransformPhase
    # Used to determine the order of transforms.
    # TODO(yupu): how do we set this properly?
    priority: int = 0
    # Requirements must be met for this transform to run
    requires: Set[str] = field(default_factory=set)
    # Mutually exclusive
    forbids: Set[str] = field(default_factory=set)
    # Ensure this Transform runs before those
    before: Set[str] = field(default_factory=set)
    # Ensure this Transform runs after those
    after: Set[str] = field(default_factory=set)


class Transform(ABC):
    """
    Base transform class.

    A transform is a class that can be applied to a model.
    It can be used to modify the model in some way.

    For example, a transform can be used to apply a pre-processing step to the model,
    or to apply a post-processing step to the model.
    """

    @abstractmethod
    def spec(self) -> TransformSpec: ...

    def supports(self, model: BaseAdapter | nn.Module) -> bool:
        """
        Check if the transform supports the model.
        """
        return True

    def preflight(self) -> bool:
        """
        Check if a hardware/python package requirement is met.
        """
        return True

    # TODO(yupu): should we add a return type?
    def apply(self, model: BaseAdapter | nn.Module) -> bool:
        """
        Implement in subclasses by using `Hook`s
        """
        return False

    # TODO(yupu): Is this needed? It would be nice to have but not sure if it's worth the complexity.
    # def revert(self, model: BaseAdapter | nn.Module) -> None:
    #     """
    #     Undo what apply() did so far.
    #     """
    #     return
