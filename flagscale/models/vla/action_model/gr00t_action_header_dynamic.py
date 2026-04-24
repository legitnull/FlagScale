# Ported from starVLA GR00T_ActionHeader_dynamic_v0.py
# Dynamic FlowmatchingActionHead that accepts nfp_feature (NFP embeddings)
# and routes them to the second half of DiT cross-attention layers.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta

if TYPE_CHECKING:
    from omegaconf import DictConfig

from flagscale.logger import logger
from flagscale.models.utils.constants import ACTION
from flagscale.models.vla.action_model.flow_matching_head.cross_attention_dit_dynamic import (
    DynamicDiT,
)
from flagscale.models.vla.action_model.flow_matching_head.encoding_utils import (
    SinusoidalPositionalEncoding,
    swish,
)
from flagscale.models.vla.registry import register_action_model


# ---------------------------------------------------------------------------
# NFP helper modules (GatedMLP, GEGLU, PerLayerHeadGating)
# ---------------------------------------------------------------------------


class GEGLU(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        a, b = x.chunk(2, dim=self.dim)
        return a * F.gelu(b)


class GatedMLP(nn.Module):
    """GEGLU-based MLP with residual connection, used as NFP head."""

    def __init__(self, hidden_dim: int, expand_ratio: int = 4, depth: int = 2, dropout: float = 0.0):
        super().__init__()
        H = hidden_dim
        E = expand_ratio * H

        blocks = []
        for _ in range(depth):
            blocks += [
                nn.LayerNorm(H),
                nn.Linear(H, 2 * E),  # 2E for GEGLU
                GEGLU(),
                nn.Dropout(dropout),
                nn.Linear(E, H),
            ]
        self.blocks = nn.Sequential(*blocks)

        self.out = nn.Sequential(
            nn.LayerNorm(H),
            nn.Linear(H, H),
        )

    def forward(self, x):
        x = x + self.blocks(x)
        return self.out(x)


class PerLayerHeadGating(nn.Module):
    """Learnable per-layer gating for multi-step NFP features."""

    def __init__(self, num_layers: int, num_heads: int, init_bias: float = 0.0):
        super().__init__()
        self.logits = nn.Parameter(torch.full((num_layers, num_heads), init_bias))

    def forward(self, feats, layer_idx: int | None = None):
        if isinstance(feats, (list, tuple)):
            feats = torch.stack(feats, dim=0)  # (T, B, S, D)

        if layer_idx is not None:
            w = torch.sigmoid(self.logits[layer_idx]).view(-1, 1, 1, 1)  # (T, 1, 1, 1)
            return (w * feats).sum(dim=0)  # (B, S, D)

        # layer_idx=None -> return all layers: (L, B, S, D)
        w_all = torch.sigmoid(self.logits).view(
            self.logits.shape[0], self.logits.shape[1], 1, 1, 1
        )
        feats_ = feats.unsqueeze(0)  # (1, T, B, S, D)
        cond_all = (w_all * feats_).sum(dim=1)  # (L, B, S, D)
        return cond_all


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class GR00TDynamicActionHeadConfig:
    type: str = "gr00t_dynamic_action_head"
    action_model_type: str = "DiT-B"
    hidden_size: int = 1024
    action_dim: int = 7
    state_dim: int = 7
    future_action_window_size: int = 7
    action_horizon: int = 8
    use_state: bool = False
    repeated_diffusion_steps: int = 4
    add_pos_embed: bool = True
    max_seq_len: int = 1024
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000
    num_inference_timesteps: int = 4
    num_target_vision_tokens: int = 32
    diffusion_model_cfg: dict = field(default_factory=dict)

    @classmethod
    def from_omegaconf(cls, cfg: DictConfig) -> GR00TDynamicActionHeadConfig:
        if cfg.get("diffusion_model_cfg") is None:
            raise ValueError("diffusion_model_cfg is required in action_model config")
        diffusion_cfg = dict(cfg.diffusion_model_cfg)
        return cls(
            type=cfg.get("type", "gr00t_dynamic_action_head"),
            action_model_type=cfg.get("action_model_type", "DiT-B"),
            hidden_size=cfg.get("hidden_size", 1024),
            action_dim=cfg.get("action_dim", 7),
            state_dim=cfg.get("state_dim", 7),
            future_action_window_size=cfg.get("future_action_window_size", 7),
            action_horizon=cfg.get("action_horizon", 8),
            use_state=cfg.get("use_state", False),
            repeated_diffusion_steps=cfg.get("repeated_diffusion_steps", 4),
            add_pos_embed=cfg.get("add_pos_embed", True),
            max_seq_len=cfg.get("max_seq_len", 1024),
            noise_beta_alpha=cfg.get("noise_beta_alpha", 1.5),
            noise_beta_beta=cfg.get("noise_beta_beta", 1.0),
            noise_s=cfg.get("noise_s", 0.999),
            num_timestep_buckets=cfg.get("num_timestep_buckets", 1000),
            num_inference_timesteps=cfg.get("num_inference_timesteps", 4),
            num_target_vision_tokens=cfg.get("num_target_vision_tokens", 32),
            diffusion_model_cfg=diffusion_cfg,
        )


# ---------------------------------------------------------------------------
# Helper modules reused from the static action header
# ---------------------------------------------------------------------------

DiTConfig = {
    "DiT-B": {"input_embedding_dim": 768, "attention_head_dim": 64, "num_attention_heads": 12},
    "DiT-L": {"input_embedding_dim": 1536, "attention_head_dim": 48, "num_attention_heads": 32},
}


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        B, T, _ = actions.shape
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,) so we can replicate across T.")

        a_emb = self.layer1(actions)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.layer2(x))
        x = self.layer3(x)
        return x


# ---------------------------------------------------------------------------
# Dynamic FlowmatchingActionHead
# ---------------------------------------------------------------------------


class DynamicFlowmatchingActionHead(nn.Module):
    """Flow-matching action head with dynamic cross-attention routing.

    Accepts ``nfp_feature`` (NFP embeddings) in addition to VLM hidden states.
    Builds a *list* of encoder_hidden_states where the first half of DiT layers
    attend to VLM features and the second half attend to NFP features.
    """

    def __init__(self, config: GR00TDynamicActionHeadConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        action_model_type = config.action_model_type
        action_model_cfg = DiTConfig[action_model_type]

        self.input_embedding_dim = action_model_cfg["input_embedding_dim"]
        diffusion_model_cfg = config.diffusion_model_cfg
        diffusion_model_cfg = {**action_model_cfg, **diffusion_model_cfg}
        self.model = DynamicDiT(**diffusion_model_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.future_action_window_size + 1
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = (
            MLP(
                input_dim=config.state_dim,
                hidden_dim=self.hidden_size,
                output_dim=self.input_embedding_dim,
            )
            if config.state_dim
            else None
        )

        self.action_encoder = ActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
        )
        self.action_decoder = MLP(
            input_dim=self.model.config.output_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    @staticmethod
    def _build_encoder_hidden_states(
        vl_embs: torch.Tensor,
        nfpm_embs: torch.Tensor,
        num_layers: int,
    ) -> list[torch.Tensor]:
        """Build per-layer cross-attention source list.

        - nfpm_embs.ndim == 3  (single-step NFP): first half VLM, second half NFP
        - nfpm_embs.ndim == 4  (multi-step NFP): first layers VLM, remaining per-step NFP
        """
        if nfpm_embs.ndim == 3:
            repeat_times = num_layers // 4
            return [vl_embs] * repeat_times + [nfpm_embs] * repeat_times
        else:
            # nfpm_embs: (T, B, S, D)
            future_steps = nfpm_embs.shape[0]
            vl_repeat = num_layers // 2 - future_steps
            return [vl_embs] * vl_repeat + [nfpm_embs[i] for i in range(future_steps)]

    def forward(
        self,
        vl_embs: torch.Tensor,
        actions: torch.Tensor,
        nfpm_embs: torch.Tensor,
        state: torch.Tensor | None = None,
        encoder_attention_mask=None,
        mask: torch.Tensor | None = None,
    ):
        """Training forward.

        Args:
            vl_embs: (B, seq_len, D) VLM hidden states.
            actions: (B, T_action, action_dim) target action trajectory.
            nfpm_embs: NFP features — (B, S, D) for single-step or (T, B, S, D) for multi-step.
            state: optional (B, 1, state_dim) proprioceptive state.
            mask: optional (B, T_action) mask for variable-length actions.
        """
        device = vl_embs.device

        # Flow matching noise
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized)

        state_features = self.state_encoder(state) if state is not None else None

        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        sa_embs = (
            torch.cat((state_features, future_tokens, action_features), dim=1)
            if state_features is not None
            else torch.cat((future_tokens, action_features), dim=1)
        )

        encoder_hidden_states = self._build_encoder_hidden_states(
            vl_embs, nfpm_embs, self.model.config.num_layers
        )

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,
        )
        pred = self.action_decoder(model_output)
        pred_actions = pred[:, -actions.shape[1] :]

        raw_loss = (pred_actions - velocity) ** 2
        # Align with starVLA: always use mean, no mask
        loss = raw_loss.mean()
        return loss

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs: torch.Tensor,
        nfpm_embs: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        state_features = self.state_encoder(state) if state is not None else None

        encoder_hidden_states = self._build_encoder_hidden_states(
            vl_embs, nfpm_embs, self.model.config.num_layers
        )

        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_discretized = int(t_cont * self.num_timestep_buckets)

            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor)

            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
            sa_embs = (
                torch.cat((state_features, future_tokens, action_features), dim=1)
                if state_features is not None
                else torch.cat((future_tokens, action_features), dim=1)
            )

            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output)
            pred_velocity = pred[:, -self.action_horizon :]
            actions = actions + dt * pred_velocity

        return actions

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


# ---------------------------------------------------------------------------
# Dict-based wrapper registered in the VLA action model registry
# ---------------------------------------------------------------------------


@register_action_model("gr00t_dynamic_action_head")
class GR00TDynamicActionHead(nn.Module):
    """GR00T dynamic action head wrapper for VLA framework.

    Adapts the DynamicFlowmatchingActionHead (tensor-based interface) to the
    dict-based interface expected by the VLA framework (QwenGr00t).
    """

    def __init__(self, config: GR00TDynamicActionHeadConfig):
        super().__init__()
        self._head = DynamicFlowmatchingActionHead(config=config)

    def forward(
        self, vlm_output: dict[str, torch.Tensor], action_input: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        vl_embs = vlm_output["hidden_states"]
        nfpm_embs = vlm_output["nfp_feature"]
        actions = action_input["actions"]
        state = action_input.get("state")
        encoder_attention_mask = action_input.get("attention_mask")
        mask = action_input.get("mask")

        loss = self._head.forward(
            vl_embs=vl_embs,
            actions=actions,
            nfpm_embs=nfpm_embs,
            state=state,
            encoder_attention_mask=encoder_attention_mask,
            mask=mask,
        )
        return {"loss": loss}

    def predict_action(
        self, vlm_output: dict[str, torch.Tensor], action_input: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        vl_embs = vlm_output["hidden_states"]
        nfpm_embs = vlm_output["nfp_feature"]
        state = action_input.get("state")

        actions = self._head.predict_action(vl_embs=vl_embs, nfpm_embs=nfpm_embs, state=state)
        return {ACTION: actions}

    def fsdp_units(self) -> list[nn.Module]:
        return list(self._head.model.transformer_blocks)
