# Mainly adopted from:
# https://github.com/starVLA/starVLA/blob/3f7feefbc5fc25890ad3a7d262b8a0aea1339aa7/starVLA/model/framework/QwenGR00T.py
# Below is the original copyright:

# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Junqiu YU / Fudan University] in [2025].
# Design and Merged by [Jinhui YE / HKUST University] in [2025].

"""
Qwen-GR00T Framework
A lightweight implementation that Qwen-VL + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5,
"""

import dataclasses
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

from .configuration_qwen3_5_gr00t import Qwen35Gr00tConfig
from flagscale.logger import logger
from flagscale.models.utils.constants import (
    ACTION,
    OBS_STATE,
    SAFETENSORS_FILE,
    VLM_CONFIG_DIR,
    resolve_pretrained_dir,
)
from flagscale.models.vla.action_model.gr00t_action_header_dynamic import GatedMLP, PerLayerHeadGating
from flagscale.models.vla.base_policy import TrainablePolicy
from flagscale.models.vla.registry import build_action_model, build_vlm
from flagscale.models.vla.utils import get_vlm_config
from flagscale.platforms.platform_manager import get_platform

VISION_START_TOKEN_ID = 248053
VISION_END_TOKEN_ID = 248054


class Qwen35Gr00t(TrainablePolicy):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen VL interface for fused language/vision token embeddings
      - DiT diffusion head for future action sequence modeling

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(self, config: Qwen35Gr00tConfig):
        super().__init__(config)

        self.vlm = build_vlm(
            config.vlm.type,
            vlm_config=config.vlm,
            prompt_template=config.prompt_template,
        )

        vlm_hidden_size = get_vlm_config(self.vlm.model_config)["hidden_size"]
        config.action_model.diffusion_model_cfg["cross_attention_dim"] = vlm_hidden_size

        # NFP (Next Frame Prediction) head — init before action_model to match starVLA init order
        self.nfp_config = config.nfp
        if self.nfp_config is not None:
            nfp_head_num = getattr(self.nfp_config, "nfp_head_num", 1)
            self.nfp_head_num = nfp_head_num
            if self.nfp_head_num == 1:
                self.nfp_head = GatedMLP(
                    hidden_dim=self.nfp_config.vl_hidden_dim,
                    expand_ratio=self.nfp_config.expand_ratio,
                    depth=self.nfp_config.depth,
                    dropout=self.nfp_config.dropout,
                )
            else:
                self.nfp_head = nn.ModuleList([
                    GatedMLP(
                        hidden_dim=self.nfp_config.vl_hidden_dim,
                        expand_ratio=self.nfp_config.expand_ratio,
                        depth=self.nfp_config.depth,
                        dropout=self.nfp_config.dropout,
                    )
                    for _ in range(self.nfp_head_num)
                ])
                action_condition_mode = getattr(self.nfp_config, "action_condition_mode", "concat")
                if action_condition_mode == "gate":
                    self.nfp_head_gating = PerLayerHeadGating(
                        config.action_model.diffusion_model_cfg["num_layers"] // 4,
                        self.nfp_head_num,
                    )

            self.use_input_query_tokens = bool(
                getattr(self.nfp_config, "learnable_query_tokens", True)
            )
            self.num_input_query_tokens = int(
                getattr(self.nfp_config, "num_query_tokens", 32)
            )
            self.allow_unsupervised_query_grad = bool(
                getattr(self.nfp_config, "allow_unsupervised_query_grad", False)
            )
            self.long_event_head = GatedMLP(
                hidden_dim=self.nfp_config.vl_hidden_dim,
                expand_ratio=self.nfp_config.expand_ratio,
                depth=self.nfp_config.depth,
                dropout=self.nfp_config.dropout,
            )
            if self.use_input_query_tokens:
                self.short_query_embeddings = nn.Parameter(
                    torch.empty(self.num_input_query_tokens, self.nfp_config.vl_hidden_dim)
                )
                self.long_query_embeddings = nn.Parameter(
                    torch.empty(self.num_input_query_tokens, self.nfp_config.vl_hidden_dim)
                )
                nn.init.normal_(self.short_query_embeddings, mean=0.0, std=0.02)
                nn.init.normal_(self.long_query_embeddings, mean=0.0, std=0.02)
            else:
                self.short_query_embeddings = None
                self.long_query_embeddings = None

        self.action_model = build_action_model(
            config.action_model.type,
            config=config.action_model,
        )

        self.future_action_window_size = config.action_model.future_action_window_size
        self.use_state = config.action_model.use_state
        self.use_action_policy_loss = config.use_action_policy_loss

        if config.input_features:
            self.input_features = config.input_features
        if config.output_features:
            self.output_features = config.output_features

    def _zero_loss(self, device: torch.device) -> torch.Tensor:
        return torch.tensor(0.0, device=device, requires_grad=True)

    def _move_inputs_to_device(self, inputs: dict | None, device: torch.device) -> dict | None:
        if inputs is None:
            return None
        return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    def _reshape_image_embeddings(self, image_embeddings: torch.Tensor, batch_size: int) -> torch.Tensor:
        return image_embeddings.reshape(batch_size, -1, image_embeddings.shape[-1])

    def _build_target_image_embeddings(self, qwen_target_inputs: dict | None) -> torch.Tensor | None:
        if qwen_target_inputs is None:
            return None
        with torch.no_grad():
            image_embeddings = self.vlm.build_image_embeddings(**qwen_target_inputs)
        if image_embeddings is None:
            return None
        return image_embeddings.detach()

    @staticmethod
    def _positions_to_list(positions, batch_size: int) -> list[int]:
        if torch.is_tensor(positions):
            if positions.dim() == 0:
                return [int(positions.item())] * batch_size
            if positions.numel() != batch_size:
                raise ValueError(f"Expected {batch_size} positions, got {positions.numel()}.")
            return [int(v) for v in positions.detach().cpu().view(-1).tolist()]
        if isinstance(positions, (list, tuple)):
            if len(positions) != batch_size:
                raise ValueError(f"Expected {batch_size} positions, got {len(positions)}.")
            return [int(v) for v in positions]
        return [int(positions)] * batch_size

    def _make_partially_detached_query_embeddings(
        self,
        query_embeddings: torch.Tensor,
        supervised_token_count: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        query_embeddings = query_embeddings.to(device=device, dtype=dtype)
        if self.allow_unsupervised_query_grad:
            return query_embeddings
        supervised_token_count = max(0, min(int(supervised_token_count), query_embeddings.shape[0]))
        unsupervised_count = query_embeddings.shape[0] - supervised_token_count
        if unsupervised_count <= 0:
            return query_embeddings
        if supervised_token_count == 0:
            return query_embeddings.detach()
        return torch.cat([
            query_embeddings[:unsupervised_count].detach(),
            query_embeddings[unsupervised_count:],
        ], dim=0)

    def _insert_query_embeddings(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        fixed_positions: dict,
        query_supervision: dict | None = None,
    ) -> torch.Tensor:
        if not self.use_input_query_tokens:
            return inputs_embeds
        if self.short_query_embeddings is None or self.long_query_embeddings is None:
            return inputs_embeds

        query_supervision = query_supervision or {}
        batch_size = input_ids.shape[0]
        short_starts = self._positions_to_list(fixed_positions["short_query_start"], batch_size)
        short_ends = self._positions_to_list(fixed_positions["short_query_end"], batch_size)
        long_starts = self._positions_to_list(fixed_positions["long_query_start"], batch_size)
        long_ends = self._positions_to_list(fixed_positions["long_query_end"], batch_size)
        short_embeds = self._make_partially_detached_query_embeddings(
            self.short_query_embeddings,
            query_supervision.get("short", 0),
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        long_embeds = self._make_partially_detached_query_embeddings(
            self.long_query_embeddings,
            query_supervision.get("long", 0),
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )

        patched_rows = []
        for batch_idx in range(batch_size):
            row_embeds = inputs_embeds[batch_idx]
            replacements = []
            for name, start, end, query_embeds in (
                ("short", short_starts[batch_idx], short_ends[batch_idx] + 1, short_embeds),
                ("long", long_starts[batch_idx], long_ends[batch_idx] + 1, long_embeds),
            ):
                query_ids = input_ids[batch_idx, start:end]
                patch_mask = (query_ids != VISION_START_TOKEN_ID) & (query_ids != VISION_END_TOKEN_ID)
                if patch_mask.sum().item() != self.num_input_query_tokens:
                    raise ValueError(
                        f"Sample {batch_idx} {name} query patch-token count does not match "
                        f"num_query_tokens={self.num_input_query_tokens}."
                    )
                replacements.append((start, end, patch_mask, query_embeds))

            segments = []
            cursor = 0
            for start, end, patch_mask, query_embeds in sorted(replacements, key=lambda item: item[0]):
                segments.append(row_embeds[cursor:start])
                original_segment = row_embeds[start:end]
                patch_indices = patch_mask.cumsum(dim=0).sub(1).clamp_min(0)
                query_values = query_embeds[patch_indices]
                patched_segment = torch.where(patch_mask[:, None], query_values, original_segment)
                segments.append(patched_segment)
                cursor = end
            segments.append(row_embeds[cursor:])
            patched_rows.append(torch.cat(segments, dim=0))
        return torch.stack(patched_rows, dim=0)

    def _run_fixed_layout_hidden(
        self,
        qwen_inputs: dict,
        fixed_positions: dict,
        feature_layer_idx: int,
        query_supervision: dict | None = None,
    ) -> torch.Tensor:
        hf_model = self.vlm.model.model
        input_ids = qwen_inputs["input_ids"]
        attention_mask = qwen_inputs.get("attention_mask", None)
        mm_token_type_ids = qwen_inputs.get("mm_token_type_ids", None)
        inputs_embeds = hf_model.get_input_embeddings()(input_ids).clone()

        if qwen_inputs.get("pixel_values", None) is not None:
            image_outputs = hf_model.get_image_features(
                qwen_inputs["pixel_values"], qwen_inputs.get("image_grid_thw", None), return_dict=True
            )
            image_embeds = image_outputs.pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = hf_model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if qwen_inputs.get("pixel_values_videos", None) is not None:
            video_outputs = hf_model.get_video_features(
                qwen_inputs["pixel_values_videos"], qwen_inputs.get("video_grid_thw", None), return_dict=True
            )
            video_embeds = video_outputs.pooler_output
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = hf_model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        inputs_embeds = self._insert_query_embeddings(
            inputs_embeds, input_ids, fixed_positions, query_supervision=query_supervision
        )
        position_ids = hf_model.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=qwen_inputs.get("image_grid_thw", None),
            video_grid_thw=qwen_inputs.get("video_grid_thw", None),
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            mm_token_type_ids=mm_token_type_ids,
        )
        outputs = hf_model.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs.hidden_states[1:][feature_layer_idx]

    def _slice_query_hidden(
        self,
        hidden_states: torch.Tensor,
        qwen_inputs: dict,
        fixed_positions: dict,
        start_key: str,
        end_key: str,
    ) -> torch.Tensor:
        starts = self._positions_to_list(fixed_positions[start_key], hidden_states.shape[0])
        ends = self._positions_to_list(fixed_positions[end_key], hidden_states.shape[0])
        selected = []
        for batch_idx in range(hidden_states.shape[0]):
            query_hidden = hidden_states[batch_idx, starts[batch_idx]:ends[batch_idx] + 1, :]
            query_ids = qwen_inputs["input_ids"][batch_idx, starts[batch_idx]:ends[batch_idx] + 1]
            query_attention = qwen_inputs["attention_mask"][batch_idx, starts[batch_idx]:ends[batch_idx] + 1].to(torch.bool)
            query_patch_mask = (
                (query_ids != VISION_START_TOKEN_ID)
                & (query_ids != VISION_END_TOKEN_ID)
                & query_attention
            )
            patch_hidden = query_hidden[query_patch_mask]
            if patch_hidden.shape[0] != self.num_input_query_tokens:
                raise ValueError(
                    f"Sample {batch_idx} has {patch_hidden.shape[0]} query patch tokens, "
                    f"expected num_query_tokens={self.num_input_query_tokens}."
                )
            selected.append(patch_hidden)
        return torch.stack(selected, dim=0)

    def _slice_supervised_tail(
        self,
        predictions: torch.Tensor,
        target_token_count: int,
        name: str,
    ) -> torch.Tensor:
        if target_token_count <= 0:
            raise ValueError(f"{name} target_token_count must be positive, got {target_token_count}.")
        if predictions.shape[1] < target_token_count:
            raise ValueError(
                f"{name} has {predictions.shape[1]} query outputs, but target requires {target_token_count} patch tokens."
            )
        return predictions[:, -target_token_count:, :]

    def _masked_nfp_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        sample_mask: list[bool] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sample_mask is None:
            return self._nfp_loss(
                predictions.reshape(-1, predictions.shape[-1]),
                targets.reshape(-1, targets.shape[-1]),
            )
        if not any(bool(flag) for flag in sample_mask):
            zero = (predictions.sum() + targets.sum()) * 0.0
            return zero, zero
        mask = torch.as_tensor(sample_mask, device=predictions.device, dtype=torch.bool)
        return self._nfp_loss(
            predictions[mask].reshape(-1, predictions.shape[-1]),
            targets[mask].reshape(-1, targets.shape[-1]),
        )

    def _auxiliary_zero_connected_loss(self, device: torch.device) -> torch.Tensor:
        zero = torch.zeros((), device=device)
        for module in (getattr(self, "nfp_head", None), getattr(self, "long_event_head", None)):
            if module is None:
                continue
            for param in module.parameters():
                if param.requires_grad and param.numel() > 0:
                    zero = zero + param.reshape(-1)[0] * 0.0
        for param in (getattr(self, "short_query_embeddings", None), getattr(self, "long_query_embeddings", None)):
            if param is not None and param.requires_grad and param.numel() > 0:
                zero = zero + param.reshape(-1)[0] * 0.0
        return zero

    def _nfp_loss(self, nfp_outputs: torch.Tensor, nfp_targets: torch.Tensor):
        """Compute MSE and cosine embedding loss for NFP head."""
        if nfp_outputs.numel() == 0 or nfp_targets.numel() == 0:
            device = nfp_outputs.device
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, zero
        mse_loss = F.mse_loss(nfp_outputs, nfp_targets, reduction="none").mean(-1)
        mse_loss = mse_loss.sum() / (nfp_outputs.shape[0] + 1e-12)
        cosine_loss = F.cosine_embedding_loss(
            nfp_outputs,
            nfp_targets,
            torch.ones(nfp_outputs.size(0), device=nfp_outputs.device),
            reduction="none",
        )
        cosine_loss = cosine_loss.sum() / (nfp_outputs.shape[0] + 1e-12)
        return mse_loss, cosine_loss

    def forward(self, batch: list[dict] | dict, mode: str = "vla") -> dict[str, torch.Tensor]:
        if mode == "vlm":
            return {"vlm_loss": self.forward_vlm(batch)}
        return self.forward_vla(batch)

    def _forward_vla_fixed_layout(self, batch: dict) -> dict[str, torch.Tensor]:
        if self.nfp_config is None:
            raise ValueError("Fixed-layout WM training requires model.nfp to be configured.")
        device = next(self.parameters()).device
        qwen_inputs = self._move_inputs_to_device(batch["qwen_inputs"], device)
        qwen_future_inputs = self._move_inputs_to_device(batch.get("qwen_future_inputs"), device)
        qwen_fixed_positions = batch["qwen_fixed_positions"]
        qwen_long_event_groups = batch.get("qwen_long_event_groups", [])
        qwen_half_event_inputs = self._move_inputs_to_device(batch.get("qwen_half_event_inputs"), device)
        qwen_half_event_target_inputs = self._move_inputs_to_device(batch.get("qwen_half_event_target_inputs"), device)
        qwen_half_event_fixed_positions = batch.get("qwen_half_event_fixed_positions", None)
        has_half_event = batch.get("has_half_event", None)

        feature_layer_idx = self.nfp_config.vlm_feature_layer
        head_dtype = next(self.nfp_head.parameters()).dtype if isinstance(self.nfp_head, nn.Module) else next(self.nfp_head[0].parameters()).dtype
        future_embeddings = self._build_target_image_embeddings(qwen_future_inputs)
        if future_embeddings is None:
            raise ValueError("qwen_future_inputs are required for fixed-layout short future prediction.")
        future_targets = self._reshape_image_embeddings(
            future_embeddings.to(dtype=head_dtype), qwen_inputs["input_ids"].shape[0]
        )

        with torch.autocast(get_platform().amp_device_type(), dtype=torch.bfloat16):
            last_hidden = self._run_fixed_layout_hidden(
                qwen_inputs,
                qwen_fixed_positions,
                feature_layer_idx,
                query_supervision={"short": future_targets.shape[1], "long": 0},
            )

        short_features = self._slice_query_hidden(
            last_hidden, qwen_inputs, qwen_fixed_positions, "short_query_start", "short_query_end"
        ).to(dtype=head_dtype)
        if isinstance(self.nfp_head, nn.ModuleList):
            short_head = self.nfp_head[0]
        else:
            short_head = self.nfp_head
        short_outputs = short_head(short_features.reshape(-1, short_features.shape[-1]))
        short_outputs = short_outputs.reshape(short_features.shape[0], short_features.shape[1], -1)
        short_outputs = self._slice_supervised_tail(short_outputs, future_targets.shape[1], "short future")
        nfp_mse_loss, nfp_cosine_loss = self._nfp_loss(
            short_outputs.reshape(-1, short_outputs.shape[-1]),
            future_targets.reshape(-1, future_targets.shape[-1]),
        )
        nfp_feature = short_outputs.detach().clone()

        def compute_event_loss(event_inputs, event_positions, target_inputs, sample_mask, use_short_head=False):
            zero = self._zero_loss(last_hidden.device)
            if event_inputs is None or event_positions is None or target_inputs is None:
                return zero, zero
            event_inputs = self._move_inputs_to_device(event_inputs, device)
            target_inputs = self._move_inputs_to_device(target_inputs, device)
            target_embeddings = self._build_target_image_embeddings(target_inputs)
            if target_embeddings is None:
                return zero, zero
            event_targets = self._reshape_image_embeddings(
                target_embeddings.to(dtype=head_dtype), event_inputs["input_ids"].shape[0]
            )
            with torch.autocast(get_platform().amp_device_type(), dtype=torch.bfloat16):
                event_hidden = self._run_fixed_layout_hidden(
                    event_inputs,
                    event_positions,
                    feature_layer_idx,
                    query_supervision={
                        "short": event_targets.shape[1] if use_short_head else 0,
                        "long": 0 if use_short_head else event_targets.shape[1],
                    },
                )
            start_key = "short_query_start" if use_short_head else "long_query_start"
            end_key = "short_query_end" if use_short_head else "long_query_end"
            event_features = self._slice_query_hidden(
                event_hidden, event_inputs, event_positions, start_key, end_key
            ).to(dtype=head_dtype)
            head = short_head if use_short_head else self.long_event_head
            event_outputs = head(event_features.reshape(-1, event_features.shape[-1]))
            event_outputs = event_outputs.reshape(event_features.shape[0], event_features.shape[1], -1)
            event_outputs = self._slice_supervised_tail(
                event_outputs, event_targets.shape[1], "half event" if use_short_head else "long event"
            )
            return self._masked_nfp_loss(event_outputs, event_targets, sample_mask)

        prev_mse = self._zero_loss(last_hidden.device)
        prev_cos = self._zero_loss(last_hidden.device)
        next_mse = self._zero_loss(last_hidden.device)
        next_cos = self._zero_loss(last_hidden.device)
        half_mse = self._zero_loss(last_hidden.device)
        half_cos = self._zero_loss(last_hidden.device)
        compute_prev_events = float(self.nfp_config.prev_event_loss_weight) != 0.0
        compute_next_events = float(self.nfp_config.next_event_loss_weight) != 0.0
        compute_half_events = float(self.nfp_config.half_event_loss_weight) != 0.0
        long_event_records = []
        for group in qwen_long_event_groups:
            group_prev_mse = self._zero_loss(last_hidden.device)
            group_prev_cos = self._zero_loss(last_hidden.device)
            group_next_mse = self._zero_loss(last_hidden.device)
            group_next_cos = self._zero_loss(last_hidden.device)
            if compute_prev_events:
                group_prev_mse, group_prev_cos = compute_event_loss(
                    group.get("qwen_prev_inputs"),
                    group.get("qwen_prev_fixed_positions"),
                    group.get("qwen_prev_target_inputs"),
                    group.get("has_rnd_prev"),
                )
            if compute_next_events:
                group_next_mse, group_next_cos = compute_event_loss(
                    group.get("qwen_next_inputs"),
                    group.get("qwen_next_fixed_positions"),
                    group.get("qwen_next_target_inputs"),
                    group.get("has_rnd_next"),
                )
            prev_mse = prev_mse + group_prev_mse
            prev_cos = prev_cos + group_prev_cos
            next_mse = next_mse + group_next_mse
            next_cos = next_cos + group_next_cos
            long_event_records.append((group.get("mode", "event"), group_prev_mse, group_prev_cos, group_next_mse, group_next_cos))

        if compute_half_events:
            half_mse, half_cos = compute_event_loss(
                qwen_half_event_inputs,
                qwen_half_event_fixed_positions,
                qwen_half_event_target_inputs,
                has_half_event,
                use_short_head=False,
            )

        loss_dict = {}
        if self.use_action_policy_loss:
            actions = batch[ACTION]
            if isinstance(actions, list):
                actions = torch.tensor(np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype)
            else:
                actions = actions.to(device=last_hidden.device, dtype=last_hidden.dtype)
            actions_target = actions[:, -(self.future_action_window_size + 1):, :]
            repeated_diffusion_steps = self.config.action_model.repeated_diffusion_steps
            vlm_output = {"hidden_states": last_hidden.repeat(repeated_diffusion_steps, 1, 1)}
            vlm_output["nfp_feature"] = nfp_feature.repeat(repeated_diffusion_steps, 1, 1)
            action_input = {"actions": actions_target.repeat(repeated_diffusion_steps, 1, 1), "state": None}
            output = self.action_model.forward(vlm_output, action_input)
            action_loss = output["loss"]
            loss = action_loss
            loss_dict["raw_action_loss"] = action_loss
        else:
            loss = self._zero_loss(last_hidden.device)

        short_loss = self.nfp_config.nfp_loss_mse_weight * nfp_mse_loss + self.nfp_config.nfp_loss_cosine_weight * nfp_cosine_loss
        prev_loss = self.nfp_config.nfp_loss_mse_weight * prev_mse + self.nfp_config.nfp_loss_cosine_weight * prev_cos
        next_loss = self.nfp_config.nfp_loss_mse_weight * next_mse + self.nfp_config.nfp_loss_cosine_weight * next_cos
        half_loss = self.nfp_config.nfp_loss_mse_weight * half_mse + self.nfp_config.nfp_loss_cosine_weight * half_cos
        loss = loss + self.nfp_config.short_future_loss_weight * short_loss
        loss = loss + self.nfp_config.prev_event_loss_weight * prev_loss
        loss = loss + self.nfp_config.next_event_loss_weight * next_loss
        loss = loss + self.nfp_config.half_event_loss_weight * half_loss
        loss = loss + self._auxiliary_zero_connected_loss(last_hidden.device)

        loss_dict.update({
            "nfp_mse_loss_0": nfp_mse_loss,
            "nfp_cosine_loss_0": nfp_cosine_loss,
            "short_future_loss_0": short_loss,
            "prev_nfp_mse_loss": prev_mse,
            "prev_nfp_cosine_loss": prev_cos,
            "next_nfp_mse_loss": next_mse,
            "next_nfp_cosine_loss": next_cos,
            "prev_event_loss": prev_loss,
            "next_event_loss": next_loss,
            "half_event_mse_loss": half_mse,
            "half_event_cosine_loss": half_cos,
            "half_event_loss": half_loss,
            "action_loss": loss,
            "loss": loss,
        })
        for mode, group_prev_mse, group_prev_cos, group_next_mse, group_next_cos in long_event_records:
            loss_dict[f"{mode}_prev_nfp_mse_loss"] = group_prev_mse
            loss_dict[f"{mode}_prev_nfp_cosine_loss"] = group_prev_cos
            loss_dict[f"{mode}_next_nfp_mse_loss"] = group_next_mse
            loss_dict[f"{mode}_next_nfp_cosine_loss"] = group_next_cos
        return loss_dict

    def forward_vla(self, batch: list[dict] | dict) -> dict[str, torch.Tensor]:
        if isinstance(batch, dict) and "qwen_fixed_positions" in batch:
            return self._forward_vla_fixed_layout(batch)

        if isinstance(batch, list):  # wds: list of per-sample dicts
            images = [ex["image"] for ex in batch]
            instructions = [ex["lang"] for ex in batch]
            actions = [ex["action"] for ex in batch]
            if self.use_state and "state" in batch[0]:
                state = [ex["state"] for ex in batch]
            else:
                state = None
            qwen_inputs = self.vlm.build_qwenvl_inputs(images, instructions)
            qwen_future_inputs = None
        else:  # lerobot: single dict with batched tensors
            if "qwen_inputs" in batch:
                qwen_inputs = batch["qwen_inputs"]
            else:
                images, instructions = self.vlm.prepare_input(
                    batch, image_feature_keys=list(self.image_features.keys())
                )
                qwen_inputs = self.vlm.build_qwenvl_inputs(images, instructions)
            actions = [batch[ACTION][i] for i in range(batch[ACTION].shape[0])]
            state = batch.get(OBS_STATE) if self.use_state else None
            qwen_future_inputs = None  # will be computed below

        # NFP: build future image embeddings if NFP is enabled
        nfp_feature = None
        if self.nfp_config is not None:
            if qwen_future_inputs is None:
                if isinstance(batch, list):
                    future_images = [ex["all_future_images"] for ex in batch]
                    qwen_future_inputs = self.vlm.build_qwenvl_inputs(future_images, instructions)
                else:
                    qwen_future_inputs = batch.get("qwen_future_inputs")
                    if qwen_future_inputs is None:
                        future_keys = [f"{k}_future" for k in self.image_features]
                        future_images, _ = self.vlm.prepare_input(batch, image_feature_keys=future_keys)
                        qwen_future_inputs = self.vlm.build_qwenvl_inputs(future_images, instructions)

            image_mask = qwen_inputs["input_ids"] == self.nfp_config.image_token_id
            future_image_embeddings = self.vlm.build_image_embeddings(**qwen_future_inputs)

        # TODO: (yupu) Hard-coded autocast and dtype, matches starVLA
        with torch.autocast(get_platform().amp_device_type(), dtype=torch.bfloat16):
            vlm_output = self.vlm.forward(qwen_inputs, output_attentions=False)
            # last_hidden_state: [B, seq_len, H]
            if self.nfp_config is not None:
                last_hidden = vlm_output["hidden_states"][1:][self.nfp_config.vlm_feature_layer]
            else:
                last_hidden = vlm_output["hidden_states"][-1]  # [B, L, H]

        # NFP forward — aligned with starVLA QwenGR00TDynamic35
        nfp_mse_loss = []
        nfp_cosine_loss = []
        if self.nfp_config is not None:
            nfp_input = last_hidden[image_mask]

            if nfp_input.shape[0] == 0:
                device = last_hidden.device
                zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
                nfp_mse_loss = [zero_loss] * (self.nfp_head_num if self.nfp_head_num > 1 else 1)
                nfp_cosine_loss = [zero_loss] * (self.nfp_head_num if self.nfp_head_num > 1 else 1)
                B = last_hidden.shape[0]
                D = last_hidden.shape[-1]
                nfp_feature = torch.zeros(B, 1, D, device=device, dtype=last_hidden.dtype)

            elif self.nfp_head_num > 1:
                nfp_feature_list, nfp_mse_loss, nfp_cosine_loss = [], [], []
                B = last_hidden.shape[0]
                T = self.nfp_head_num
                V = qwen_inputs['image_grid_thw'].shape[0] // B
                P = qwen_inputs['image_grid_thw'][0].prod().item() // 4
                D = last_hidden.shape[-1]

                fut_emb = future_image_embeddings.view(B, V, T, P, D)

                for i, head in enumerate(self.nfp_head):
                    nfp_outputs = head(nfp_input)
                    mse_loss, cosine_loss = self._nfp_loss(nfp_outputs, fut_emb[:, :, i].reshape(-1, D))
                    nfp_mse_loss.append(mse_loss)
                    nfp_cosine_loss.append(cosine_loss)
                    nfp_feature_list.append(nfp_outputs.reshape(B, -1, nfp_outputs.shape[-1]).detach().clone())

                action_condition_mode = getattr(self.nfp_config, "action_condition_mode", "concat")
                if action_condition_mode == "gate":
                    nfp_feature = self.nfp_head_gating(nfp_feature_list)
                else:
                    nfp_feature = torch.cat(nfp_feature_list, dim=1)

            else:
                nfp_outputs = self.nfp_head(nfp_input)
                mse_loss, cosine_loss = self._nfp_loss(nfp_outputs, future_image_embeddings)
                nfp_mse_loss, nfp_cosine_loss = [mse_loss], [cosine_loss]
                nfp_feature = nfp_outputs.reshape(last_hidden.shape[0], -1, nfp_outputs.shape[-1]).detach().clone()

        if self.use_action_policy_loss:
            with torch.autocast(get_platform().amp_device_type(), dtype=torch.float32):
                if isinstance(actions, list):
                    if isinstance(actions[0], torch.Tensor):
                        actions = torch.stack(actions).to(device=last_hidden.device, dtype=last_hidden.dtype)
                    else:
                        actions = torch.tensor(
                            np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype,
                        )
                else:
                    actions = actions.to(device=last_hidden.device, dtype=last_hidden.dtype)
                actions_target = actions[:, -(self.future_action_window_size + 1):, :]

                repeated_diffusion_steps = self.config.action_model.repeated_diffusion_steps

                actions_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
                last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)

                state_repeated = None
                if state is not None:
                    if isinstance(state, list):
                        if isinstance(state[0], torch.Tensor):
                            state = torch.stack(state).to(device=last_hidden.device, dtype=last_hidden.dtype)
                        else:
                            state = torch.tensor(
                                np.array(state), device=last_hidden.device, dtype=last_hidden.dtype,
                            )
                    else:
                        state = state.to(device=last_hidden.device, dtype=last_hidden.dtype)
                    state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

                vlm_output_repeated = {"hidden_states": last_hidden_repeated}
                if nfp_feature is not None:
                    if nfp_feature.ndim == 4:
                        nfp_feature_repeated = nfp_feature.repeat(1, repeated_diffusion_steps, 1, 1)
                    else:
                        nfp_feature_repeated = nfp_feature.repeat(repeated_diffusion_steps, 1, 1)
                    vlm_output_repeated["nfp_feature"] = nfp_feature_repeated

                action_input = {
                    "actions": actions_repeated,
                    "state": state_repeated,
                }

                output = self.action_model.forward(vlm_output_repeated, action_input)

        # Loss composition — aligned with starVLA QwenGR00TDynamic35
        if self.nfp_config is not None:
            loss_dict = {}
            if self.use_action_policy_loss:
                action_loss = output["loss"]
                loss_dict["raw_action_loss"] = action_loss
                loss = action_loss
            else:
                loss = 0.0

            for i, (mse_loss, cosine_loss) in enumerate(zip(nfp_mse_loss, nfp_cosine_loss)):
                loss += 0.1 * mse_loss + cosine_loss
                loss_dict[f"nfp_mse_loss_{i}"] = mse_loss
                loss_dict[f"nfp_cosine_loss_{i}"] = cosine_loss
            loss = loss + self._auxiliary_zero_connected_loss(last_hidden.device)
            loss_dict["action_loss"] = loss
            result = loss_dict
            result["loss"] = loss
        else:
            result = {"loss": output["loss"]}

        return result

    def forward_vlm(self, vlm_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Separate VLM co-train forward, called after VLA backward to reduce peak memory."""
        device = next(self.parameters()).device
        vlm_batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                     for k, v in vlm_batch.items()}
        with torch.autocast(get_platform().amp_device_type(), dtype=torch.bfloat16):
            vlm_loss = self.vlm.model(**vlm_batch, return_dict=True).loss
        return vlm_loss

    @torch.inference_mode()
    def predict_action(self, batch: list[dict] | dict) -> dict:
        """
        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory
        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        if isinstance(batch, list):  # wds: list of per-sample dicts
            images = [ex["image"] for ex in batch]
            instructions = [ex["lang"] for ex in batch]
            if self.use_state and "state" in batch[0]:
                state = torch.stack([ex["state"] for ex in batch])
            else:
                state = None
        else:  # lerobot: single dict with batched tensors
            logger.info(f"[predict_action] batch keys={list(batch.keys())}")
            logger.info(f"[predict_action] image_features keys={list(self.image_features.keys())}")
            for k in self.image_features:
                if k in batch:
                    v = batch[k]
                    logger.info(f"[predict_action] image key={k} shape={v.shape} dtype={v.dtype}")
            images, instructions = self.vlm.prepare_input(
                batch, image_feature_keys=list(self.image_features.keys())
            )
            state = batch.get(OBS_STATE) if self.use_state else None

        qwen_inputs = self.vlm.build_qwenvl_inputs(images, instructions)

        with torch.autocast(get_platform().amp_device_type(), dtype=torch.bfloat16):
            vlm_output = self.vlm.forward(qwen_inputs, output_attentions=False)
            # last_hidden_state: [B, seq_len, H]
            if self.nfp_config is not None:
                last_hidden = vlm_output["hidden_states"][1:][self.nfp_config.vlm_feature_layer]
            else:
                last_hidden = vlm_output["hidden_states"][-1]  # [B, L, H]

        logger.info(
            f"[predict_action] last_hidden shape={last_hidden.shape} dtype={last_hidden.dtype}"
        )

        # NFP feature for inference — aligned with starVLA QwenGR00TDynamic35.predict_action
        nfp_feature = None
        if self.nfp_config is not None:
            image_mask = qwen_inputs["input_ids"] == self.nfp_config.image_token_id
            nfp_input = last_hidden[image_mask]
            if self.nfp_head_num > 1:
                nfp_feature = []
                for i, head in enumerate(self.nfp_head):
                    nfp_outputs = head(nfp_input)
                    nfp_feature.append(nfp_outputs.reshape(last_hidden.shape[0], -1, nfp_outputs.shape[-1]))
                action_condition_mode = getattr(self.nfp_config, "action_condition_mode", "concat")
                if action_condition_mode == "gate":
                    nfp_feature = self.nfp_head_gating(nfp_feature)
                else:
                    nfp_feature = torch.cat(nfp_feature, dim=1)
            else:
                nfp_outputs = self.nfp_head(nfp_input)
                nfp_feature = nfp_outputs.reshape(last_hidden.shape[0], -1, nfp_outputs.shape[-1])

        if state is not None:
            state = state.to(device=last_hidden.device, dtype=last_hidden.dtype)

        # Step 4: Action Expert Forward
        with torch.autocast(get_platform().amp_device_type(), dtype=torch.float32):
            vlm_output_for_action = {"hidden_states": last_hidden}
            if nfp_feature is not None:
                vlm_output_for_action["nfp_feature"] = nfp_feature
            action_input = {"state": state}
            output = self.action_model.predict_action(vlm_output_for_action, action_input)

        logger.info(f"[predict_action] output keys={list(output.keys())}")
        for k, v in output.items():
            if isinstance(v, torch.Tensor):
                logger.info(f"[predict_action] output {k} shape={v.shape} dtype={v.dtype}")

        # Assume the output of the action model is dict mapping `ACTION` to the normalized actions
        return output

    def fsdp_units(self):
        return self.vlm.fsdp_units() + self.action_model.fsdp_units()

    def _save_pretrained(self, save_directory: Path, state_dict=None) -> None:
        """Save Qwen35Gr00t checkpoint: VLM processor + config.json + weights.

        In addition to the base class artifacts, writes the VLM HF config
        and processor to a ``vlm_config/`` subdirectory so the checkpoint
        is fully self-contained (no dependency on the original VLM hub
        repo at inference time).  ``config.json`` records a *relative*
        ``base_vlm`` path pointing at this subdirectory.
        """
        save_directory = Path(save_directory)

        # 1. Save VLM config + processor
        vlm_config_dir = save_directory / VLM_CONFIG_DIR
        vlm_config_dir.mkdir(parents=True, exist_ok=True)
        self.vlm.model.config.save_pretrained(vlm_config_dir)
        self.vlm.processor.save_pretrained(vlm_config_dir)

        # 2. Save config.json with relative VLM path
        save_config = dataclasses.replace(
            self.config,
            vlm=dataclasses.replace(
                self.config.vlm,
                base_vlm=VLM_CONFIG_DIR,
                load_pretrained=False,
            ),
        )
        save_config._save_pretrained(save_directory)

        # 3. Save weights
        # Under FSDP2, model.state_dict() returns sharded DTensors that can't
        # be serialized directly. The caller must gather the full state dict
        # via get_model_state_dict() and pass it in.
        if state_dict is not None:
            state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}
        else:
            state_dict = {k: v.clone().contiguous() for k, v in self.state_dict().items()}
        save_file(state_dict, str(save_directory / SAFETENSORS_FILE))

    @classmethod
    def from_pretrained(cls, pretrained_path, device="cpu", **kwargs):
        """Load a Qwen35Gr00t checkpoint.

        Resolves the relative ``base_vlm`` path stored in ``config.json``
        against the checkpoint directory, then delegates weight loading
        to ``TrainablePolicy.from_pretrained``.
        """
        path = resolve_pretrained_dir(Path(pretrained_path), SAFETENSORS_FILE)
        config = Qwen35Gr00tConfig.from_pretrained(path)

        # Resolve relative VLM path against checkpoint directory
        if not Path(config.vlm.base_vlm).is_absolute():
            config.vlm = dataclasses.replace(
                config.vlm,
                base_vlm=str(path / config.vlm.base_vlm),
            )

        return super().from_pretrained(pretrained_path, device=device, config=config)
