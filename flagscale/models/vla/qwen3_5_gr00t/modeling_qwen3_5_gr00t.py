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

import torch
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
from flagscale.models.vla.action_model.gr00t_action_header_dynamic import GatedMLP
from flagscale.models.vla.base_policy import TrainablePolicy
from flagscale.models.vla.registry import build_action_model, build_vlm
from flagscale.models.vla.utils import get_vlm_config
from flagscale.platforms.platform_manager import get_platform


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

        self.action_model = build_action_model(
            config.action_model.type,
            config=config.action_model,
        )

        self.future_action_window_size = config.action_model.future_action_window_size
        self.use_state = config.action_model.use_state

        # NFP (Next Frame Prediction) head
        self.nfp_config = config.nfp
        if self.nfp_config is not None:
            self.nfp_head = GatedMLP(
                hidden_dim=self.nfp_config.vl_hidden_dim,
                expand_ratio=self.nfp_config.expand_ratio,
                depth=self.nfp_config.depth,
                dropout=self.nfp_config.dropout,
            )
        self.use_action_policy_loss = config.use_action_policy_loss

        if config.input_features:
            self.input_features = config.input_features
        if config.output_features:
            self.output_features = config.output_features

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

    def forward(
        self, batch: list[dict] | dict, vlm_batch: dict[str, torch.Tensor] | None = None, skip_vla: bool = False
    ) -> dict[str, torch.Tensor]:
        """ """
        if skip_vla:
            result = {"loss": torch.tensor(0.0, device=next(self.parameters()).device, requires_grad=True)}
            if vlm_batch is not None:
                with torch.autocast(get_platform().amp_device_type(), dtype=torch.bfloat16):
                    vlm_loss = self.vlm.model(**vlm_batch, return_dict=True).loss
                result["vlm_loss"] = vlm_loss
            return result

        if isinstance(batch, list):  # wds: list of per-sample dicts
            images = [ex["image"] for ex in batch]
            instructions = [ex["lang"] for ex in batch]
            actions = [ex["action"] for ex in batch]
            if self.use_state and "state" in batch[0]:
                state = [ex["state"] for ex in batch]
            else:
                state = None
            qwen_inputs = self.vlm.build_qwenvl_inputs(images, instructions)
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

        # NFP: build future image embeddings if NFP is enabled
        nfp_feature = None
        nfp_losses = {}
        if self.nfp_config is not None:
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

        # NFP forward
        if self.nfp_config is not None:
            nfp_input = last_hidden[image_mask]
            if nfp_input.shape[0] == 0:
                device = last_hidden.device
                zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
                nfp_losses["nfp_mse_loss_0"] = zero_loss
                nfp_losses["nfp_cosine_loss_0"] = zero_loss
                B, D = last_hidden.shape[0], last_hidden.shape[-1]
                nfp_feature = torch.zeros(B, 1, D, device=device, dtype=last_hidden.dtype)
            else:
                nfp_outputs = self.nfp_head(nfp_input)
                mse_loss, cosine_loss = self._nfp_loss(nfp_outputs, future_image_embeddings)
                nfp_losses["nfp_mse_loss_0"] = mse_loss
                nfp_losses["nfp_cosine_loss_0"] = cosine_loss
                nfp_feature = (
                    nfp_outputs.reshape(last_hidden.shape[0], -1, nfp_outputs.shape[-1])
                    .detach()
                    .clone()
                )

        target_horizon = self.config.action_model.action_horizon

        with torch.autocast(get_platform().amp_device_type(), dtype=torch.float32):
            # Align with starVLA: direct slice, no padding/mask
            if isinstance(actions, list):
                actions = torch.stack(actions)
            actions = actions.to(device=last_hidden.device, dtype=last_hidden.dtype)
            actions_target = actions[:, -target_horizon:, :]  # (B, action_horizon, action_dim)

            repeated_diffusion_steps = self.config.action_model.repeated_diffusion_steps

            actions_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)

            state_repeated = None
            if state is not None:
                if isinstance(state, list):
                    state = torch.stack(state)
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

        # Loss composition
        if self.nfp_config is not None:
            action_loss = output["loss"]
            result = {"raw_action_loss": action_loss}
            loss = (
                action_loss
                if self.use_action_policy_loss
                else torch.tensor(0.0, device=action_loss.device, requires_grad=True)
            )
            for k, v in nfp_losses.items():
                result[k] = v
                if "mse" in k:
                    loss = loss + self.nfp_config.nfp_loss_mse_weight * v
                else:
                    loss = loss + self.nfp_config.nfp_loss_cosine_weight * v
            result["loss"] = loss
        else:
            result = {"loss": output["loss"]}

        if vlm_batch is not None:
            with torch.autocast(get_platform().amp_device_type(), dtype=torch.bfloat16):
                vlm_loss = self.vlm.model(**vlm_batch, return_dict=True).loss
            result["vlm_loss"] = vlm_loss

        return result

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

        # NFP feature for inference
        nfp_feature = None
        if self.nfp_config is not None:
            image_mask = qwen_inputs["input_ids"] == self.nfp_config.image_token_id
            nfp_input = last_hidden[image_mask]
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
