from __future__ import annotations

from dataclasses import dataclass, field
from logging import getLogger
from typing import TYPE_CHECKING, Any

from flagscale.models.configs.types import NormalizationMode
from flagscale.models.utils.constants import ACTION
from flagscale.models.vla.action_model.gr00t_action_header import GR00TActionHeadConfig
from flagscale.models.vla.action_model.gr00t_action_header_dynamic import (
    GR00TDynamicActionHeadConfig,
)
from flagscale.models.vla.pretrained_config import PreTrainedConfig
from flagscale.models.vla.vlm.qwenvl_backbone import QwenVLConfig

if TYPE_CHECKING:
    from flagscale.train.train_config import TrainConfig

logger = getLogger(__name__)


@dataclass
class NFPConfig:
    """Next Frame Prediction (NFP) head configuration."""

    vl_hidden_dim: int = 2560
    expand_ratio: int = 4
    depth: int = 2
    dropout: float = 0.0
    vlm_feature_layer: int = -1
    nfp_loss_mse_weight: float = 0.1
    nfp_loss_cosine_weight: float = 1.0
    short_future_loss_weight: float = 1.0
    prev_event_loss_weight: float = 1.0
    next_event_loss_weight: float = 1.0
    half_event_loss_weight: float = 1.0
    num_query_tokens: int = 32
    learnable_query_tokens: bool = True
    allow_unsupervised_query_grad: bool = False
    # Image token ID used to locate image positions in VLM hidden states.
    # Qwen3-VL uses 151655; Qwen3.5-VL uses 248056.
    image_token_id: int = 248056


@dataclass
class Qwen35Gr00tConfig(PreTrainedConfig):
    vlm: QwenVLConfig = field(default_factory=QwenVLConfig)
    action_model: GR00TActionHeadConfig | GR00TDynamicActionHeadConfig = field(
        default_factory=GR00TActionHeadConfig
    )

    prompt_template: str | None = None

    # NFP (Next Frame Prediction) head config; None disables NFP.
    nfp: NFPConfig | None = None
    # When False, action loss is zeroed out (Stage 1: only NFP + VLM language).
    use_action_policy_loss: bool = True
    # Tokens per chunk for memory-efficient VLM cross-entropy. 0 = use HF default (no chunking).
    chunked_ce_tokens: int = 128

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    @property
    def observation_delta_indices(self) -> list[int]:
        return [0]

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.action_model.future_action_window_size + 1))

    def validate_features(self) -> None:
        if not self.output_features:
            raise ValueError("output_features must be set")
        action_ft = self.action_feature
        if action_ft is None:
            raise ValueError(f"output_features must contain '{ACTION}' with type ACTION")

    @classmethod
    def from_train_config(cls, train_config: TrainConfig) -> Qwen35Gr00tConfig:
        model_cfg = train_config.model

        vlm_section = model_cfg.vlm
        vlm = QwenVLConfig(
            type=vlm_section.get("type", "qwen3-vl"),
            base_vlm=vlm_section.get("base_vlm", ""),
            load_pretrained=vlm_section.get("load_pretrained", True),
            attn_implementation=vlm_section.get("attn_implementation", None),
        )

        # Dispatch action model config based on type field.
        action_model_section = model_cfg.action_model
        action_model_type = action_model_section.get("type", "gr00t_action_head")
        if action_model_type == "gr00t_dynamic_action_head":
            action_model = GR00TDynamicActionHeadConfig.from_omegaconf(action_model_section)
        else:
            action_model = GR00TActionHeadConfig.from_omegaconf(action_model_section)

        prompt_template = getattr(model_cfg, "prompt_template", None)

        kwargs = dict(vlm=vlm, action_model=action_model, prompt_template=prompt_template)

        # NFP config
        raw_nfp = getattr(model_cfg, "nfp", None)
        if raw_nfp is not None:
            kwargs["nfp"] = NFPConfig(
                vl_hidden_dim=raw_nfp.get("vl_hidden_dim", 2560),
                expand_ratio=raw_nfp.get("expand_ratio", 4),
                depth=raw_nfp.get("depth", 2),
                dropout=raw_nfp.get("dropout", 0.0),
                vlm_feature_layer=raw_nfp.get("vlm_feature_layer", -1),
                nfp_loss_mse_weight=raw_nfp.get("nfp_loss_mse_weight", 0.1),
                nfp_loss_cosine_weight=raw_nfp.get("nfp_loss_cosine_weight", 1.0),
                short_future_loss_weight=raw_nfp.get("short_future_loss_weight", 1.0),
                prev_event_loss_weight=raw_nfp.get("prev_event_loss_weight", 1.0),
                next_event_loss_weight=raw_nfp.get("next_event_loss_weight", 1.0),
                half_event_loss_weight=raw_nfp.get("half_event_loss_weight", 1.0),
                num_query_tokens=raw_nfp.get("num_query_tokens", 32),
                learnable_query_tokens=raw_nfp.get("learnable_query_tokens", True),
                allow_unsupervised_query_grad=raw_nfp.get("allow_unsupervised_query_grad", False),
                image_token_id=raw_nfp.get("image_token_id", 248056),
            )

        use_action_policy_loss = getattr(model_cfg, "use_action_policy_loss", True)
        kwargs["use_action_policy_loss"] = use_action_policy_loss

        chunked_ce = getattr(model_cfg, "chunked_ce_tokens", 128)
        kwargs["chunked_ce_tokens"] = chunked_ce

        raw_norm = getattr(model_cfg, "normalization_mapping", None)
        if raw_norm is not None:
            kwargs["normalization_mapping"] = {k: NormalizationMode(v) for k, v in raw_norm.items()}

        return cls(**kwargs)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> Qwen35Gr00tConfig:
        if "vlm" in data and isinstance(data["vlm"], dict):
            data["vlm"] = QwenVLConfig(**data["vlm"])
        if "action_model" in data and isinstance(data["action_model"], dict):
            am_data = data["action_model"]
            if am_data.get("type") == "gr00t_dynamic_action_head":
                data["action_model"] = GR00TDynamicActionHeadConfig(**am_data)
            else:
                data["action_model"] = GR00TActionHeadConfig(**am_data)
        if "nfp" in data and isinstance(data["nfp"], dict):
            data["nfp"] = NFPConfig(**data["nfp"])
        if "normalization_mapping" in data and isinstance(data["normalization_mapping"], dict):
            data["normalization_mapping"] = {
                k: NormalizationMode(v) for k, v in data["normalization_mapping"].items()
            }
        return cls(**data)
