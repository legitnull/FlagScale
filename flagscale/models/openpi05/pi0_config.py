import dataclasses

from flagscale.models.openpi05.model import ModelType


@dataclasses.dataclass(frozen=True)
class Pi0Config:
    dtype: str = "bfloat16"
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    pi05: bool = False
    discrete_state_input: bool = None  # type: ignore

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

    @property
    def model_type(self) -> ModelType:
        if self.pi05:
            return ModelType.PI05
        return ModelType.PI0
