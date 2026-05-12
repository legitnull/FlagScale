# Mainly adopted from:
# https://github.com/starVLA/starVLA/blob/3f7feefbc5fc25890ad3a7d262b8a0aea1339aa7/starVLA/model/modules/vlm/QWen3.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import (
    AutoConfig,
    AutoProcessor,
    PretrainedConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)

try:
    from transformers import Qwen3_5ForConditionalGeneration
except ImportError:
    Qwen3_5ForConditionalGeneration = None

from flagscale.logger import logger
from flagscale.models.vla.registry import register_vlm
from flagscale.platforms.platform_manager import get_platform


@dataclass
class QwenVLConfig:
    type: str = "qwen3-vl"
    base_vlm: str = ""
    load_pretrained: bool = True
    attn_implementation: str | None = None


def _to_pil(img):
    """Convert a single image (tensor, numpy, or PIL) to PIL.Image."""
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if isinstance(img, np.ndarray):
        if img.dtype == np.uint8:
            return Image.fromarray(img)
        # float [0,1] → uint8
        return Image.fromarray((img * 255).clip(0, 255).astype(np.uint8))
    return img


class QwenVLBackbone(nn.Module):
    """
    Base class for Qwen VL backends.

    Args:
        vlm_config: QwenVLConfig with base_vlm, load_pretrained, attn_implementation.
        prompt_template: Optional prompt template with {instruction} placeholder.
    """

    def __init__(self, vlm_config: QwenVLConfig, prompt_template: str | None = None, **kwargs):
        super().__init__()
        self.model_id = vlm_config.base_vlm
        self._load_pretrained = vlm_config.load_pretrained
        self._attn_implementation = vlm_config.attn_implementation

        if not self._load_pretrained and not Path(self.model_id).exists():
            raise FileNotFoundError(
                f"VLM config directory not found: {self.model_id}. "
                "Ensure the checkpoint was saved with save_pretrained."
            )

        # TODO: (yupu) The model loaded by `from_pretrained` is eval mode by default, is this expected? I removed `policy.train()` in train_qwen_gr00t.py to match starVLA, but not sure if this is the right way to do this.
        self.model = self._load_model(self.model_id)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        # FIXME: Hard-coded padding side
        self.processor.tokenizer.padding_side = "left"
        self._prompt_template = prompt_template

    def _load_model(self, model_id: str):
        raise NotImplementedError

    @property
    def model_config(self) -> PretrainedConfig:
        """HF config object (e.g., Qwen2VLConfig)."""
        return self.model.config

    def prepare_input(
        self, batch: dict, image_feature_keys: list[str]
    ) -> tuple[list[list[Image.Image]], list[str]]:
        # TODO: (yupu) hard-code task key to "task"
        instructions = batch["task"]
        if isinstance(instructions, torch.Tensor):
            instructions = instructions.detach().cpu().tolist()
        if isinstance(instructions, str):
            instructions = [instructions]

        # logger.info(f"[prepare_input] image_feature_keys={image_feature_keys}")
        batch_images: list[list[Image.Image]] | None = None
        for key in image_feature_keys:
            imgs = batch[key]
            # if isinstance(imgs, torch.Tensor):
            #     logger.info(
            #         f"[prepare_input] key={key} tensor shape={imgs.shape} dtype={imgs.dtype}"
            #     )
            if isinstance(imgs, torch.Tensor) and imgs.ndim == 3:
                imgs = [imgs]
            key_images = [_to_pil(img) for img in imgs]
            if batch_images is None:
                batch_images = [[img] for img in key_images]
            else:
                for sample_images, img in zip(batch_images, key_images):
                    sample_images.append(img)

        for idx, sample_images in enumerate(batch_images):
            batch_images[idx] = [img for img in sample_images if img is not None]

        # logger.info(
        #     f"[prepare_input] batch_size={len(batch_images)} images_per_sample={[len(s) for s in batch_images]} pil_size={batch_images[0][0].size if batch_images else None}"
        # )
        return batch_images, instructions

    def build_qwenvl_inputs(
        self, images: list[list[Image.Image]], instructions: list[str]
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def _build_messages(
        self, images: list[list[Image.Image]], instructions: list[str]
    ) -> list[list[dict]]:
        messages = []
        assert len(images) == len(instructions)
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]

            if self._prompt_template is not None:
                prompt = self._prompt_template.replace("{instruction}", instruction)
            else:
                prompt = instruction

            content.append({"type": "text", "text": prompt})
            messages.append([{"role": "user", "content": content}])
        return messages

    def build_image_embeddings(
        self,
        pixel_values: torch.FloatTensor | None = None,
        image_grid_thw: torch.FloatTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        video_grid_thw: torch.FloatTensor | None = None,
        **kwargs,
    ) -> torch.Tensor | None:
        """Extract image embeddings from the VLM visual encoder (projected to LLM dim).

        Used as NFP supervision targets — the returned embeddings represent what
        the NFP head should learn to predict for future frames.
        """
        embeds = []
        hf_model = self.model.model  # inner HF model (e.g. Qwen3VLModel)

        if hasattr(hf_model, "visual"):
            if pixel_values is not None:
                img_out = hf_model.visual(hidden_states=pixel_values, grid_thw=image_grid_thw)
                img_tensor = (
                    img_out.pooler_output
                    if hasattr(img_out, "pooler_output")
                    else (
                        img_out.last_hidden_state
                        if hasattr(img_out, "last_hidden_state")
                        else (img_out[0] if isinstance(img_out, tuple) else img_out)
                    )
                )
                embeds.append(img_tensor)

            if pixel_values_videos is not None:
                vid_out = hf_model.visual(
                    hidden_states=pixel_values_videos, grid_thw=video_grid_thw
                )
                vid_tensor = (
                    vid_out.pooler_output
                    if hasattr(vid_out, "pooler_output")
                    else (
                        vid_out.last_hidden_state
                        if hasattr(vid_out, "last_hidden_state")
                        else (vid_out[0] if isinstance(vid_out, tuple) else vid_out)
                    )
                )
                embeds.append(vid_tensor)

            return torch.cat(embeds, dim=0) if embeds else None

        # Fallback for models without .visual (e.g. older HF API)
        raw_embeds = hf_model.get_image_features(pixel_values, image_grid_thw)
        if hasattr(raw_embeds, "last_hidden_state"):
            raw_embeds = raw_embeds.last_hidden_state
        elif isinstance(raw_embeds, tuple):
            raw_embeds = raw_embeds[0]

        projector = getattr(
            hf_model, "multi_modal_projector", getattr(hf_model, "vision_language_adapter", None)
        )
        if projector is not None:
            try:
                raw_embeds = projector(raw_embeds)
            except TypeError:
                raw_embeds = projector(raw_embeds, grid_thw=image_grid_thw)

        if isinstance(raw_embeds, list):
            raw_embeds = torch.cat(raw_embeds, dim=0)

        return raw_embeds

    def forward(self, batch: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        # logger.info(
        #     f"[VLM.forward] input keys={list(batch.keys())} "
        #     + " ".join(f"{k}={v.shape}" for k, v in batch.items() if isinstance(v, torch.Tensor))
        # )
        with torch.autocast(get_platform().amp_device_type(), dtype=torch.bfloat16):
            outputs = self.model(
                **batch,
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )
        # logger.info(
        #     f"[VLM.forward] hidden_states: {len(outputs.hidden_states)} layers, last={outputs.hidden_states[-1].shape}"
        # )
        # TODO: (yupu) We should output the original outputs, not just the hidden states.
        return {"hidden_states": outputs.hidden_states}

    def fsdp_units(self) -> list[nn.Module]:
        # return list(self.model.model.visual.blocks) + list(self.model.model.language_model.layers)
        return list(self.model.model.language_model.layers)


@register_vlm("qwen2.5-vl")
class Qwen25VLBackbone(QwenVLBackbone):
    """Qwen2.5-VL backend."""

    def _load_model(self, model_id: str):
        attn_impl = self._attn_implementation or "flash_attention_2"
        if not self._load_pretrained:
            hf_config = AutoConfig.from_pretrained(
                model_id, attn_implementation=attn_impl, torch_dtype="auto"
            )
            return Qwen2_5_VLForConditionalGeneration(hf_config)
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation=attn_impl,
            torch_dtype="auto",
        )

    def build_qwenvl_inputs(
        self, images: list[list[Image.Image]], instructions: list[str]
    ) -> dict[str, torch.Tensor]:
        from qwen_vl_utils import process_vision_info

        messages = self._build_messages(images, instructions)

        # Prepare text prompts using processor
        # default process is json --> message --> texts --> input_ids
        texts = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages
        ]

        # image_inputs = list of PIL
        image_inputs, video_inputs = process_vision_info(messages)
        batch_input = self.processor(
            text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        )

        logger.info(
            "[Qwen25.build_qwenvl_inputs] "
            + " ".join(
                f"{k}={v.shape}" for k, v in batch_input.items() if isinstance(v, torch.Tensor)
            )
        )

        # Use current CUDA device instead of self.model.device, which returns
        # a DTensor device under FSDP2 and causes mixed Tensor/DTensor errors.
        return batch_input.to(get_platform().device())


@register_vlm("qwen3-vl")
class Qwen3VLBackbone(QwenVLBackbone):
    """Qwen3-VL backend."""

    def _load_model(self, model_id: str) -> Qwen3VLForConditionalGeneration:
        attn_impl = self._attn_implementation or "flash_attention_2"
        if not self._load_pretrained:
            hf_config = AutoConfig.from_pretrained(
                model_id, attn_implementation=attn_impl, torch_dtype=torch.bfloat16
            )
            model = Qwen3VLForConditionalGeneration(hf_config)
        else:
            # FIXME: hard-coded torch_dtype matches starVLA
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_id,
                attn_implementation=attn_impl,
                torch_dtype=torch.bfloat16,
            )

        return model

    def build_qwenvl_inputs(
        self, images: list[list[Image.Image]], instructions: list[str]
    ) -> dict[str, torch.Tensor]:
        messages = self._build_messages(images, instructions)

        # Preparation for inference
        # enable_thinking=False to match starVLA's no-thinking VLA prompt.
        batch_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )

        logger.info(
            "[Qwen3.build_qwenvl_inputs] "
            + " ".join(
                f"{k}={v.shape}" for k, v in batch_inputs.items() if isinstance(v, torch.Tensor)
            )
        )

        # Use current CUDA device instead of self.model.device, which returns
        # a DTensor device under FSDP2 and causes mixed Tensor/DTensor errors.
        return batch_inputs.to(get_platform().device())


@register_vlm("qwen3.5-vl")
class Qwen3_5VLBackbone(QwenVLBackbone):
    """Qwen3.5-VL backend (Qwen3_5ForConditionalGeneration)."""

    def __init__(self, vlm_config: QwenVLConfig, prompt_template: str | None = None, **kwargs):
        if Qwen3_5ForConditionalGeneration is None:
            raise ImportError(
                "Qwen3_5ForConditionalGeneration is not available. "
                "Please upgrade transformers: pip install git+https://github.com/huggingface/transformers.git"
            )
        super().__init__(vlm_config, prompt_template, **kwargs)
        # Qwen3.5 stores hidden_size in text_config; align with top-level config
        # so that get_vlm_config() can find it.
        if not hasattr(self.model.config, "hidden_size") or self.model.config.hidden_size is None:
            self.model.config.hidden_size = self.model.config.text_config.hidden_size

    def _load_model(self, model_id: str):
        attn_impl = self._attn_implementation or "flash_attention_2"
        if not self._load_pretrained:
            hf_config = AutoConfig.from_pretrained(
                model_id, attn_implementation=attn_impl, torch_dtype=torch.bfloat16
            )
            model = Qwen3_5ForConditionalGeneration(hf_config)
        else:
            model = Qwen3_5ForConditionalGeneration.from_pretrained(
                model_id,
                attn_implementation=attn_impl,
                torch_dtype=torch.bfloat16,
            )
        return model

    def train(self, mode: bool = True):
        # Qwen3.5's DynamicCache accumulates KV states via concatenation.
        # Activation checkpointing recomputes the forward pass, causing double-append
        # and shape mismatches. Disable the cache entirely during training.
        super().train(mode)
        cache_enabled = not mode
        self.model.config.use_cache = cache_enabled
        self.model.config.text_config.use_cache = cache_enabled
        self.model.model.language_model.config.use_cache = cache_enabled
        return self

    def forward(self, batch: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        with torch.autocast(get_platform().amp_device_type(), dtype=torch.bfloat16):
            outputs = self.model(
                **batch,
                output_hidden_states=True,
                return_dict=True,
                use_cache=not self.training,
                **kwargs,
            )
        return {"hidden_states": outputs.hidden_states}

    def build_qwenvl_inputs(
        self, images: list[list[Image.Image]], instructions: list[str]
    ) -> dict[str, torch.Tensor]:
        from qwen_vl_utils import process_vision_info

        messages = self._build_messages(images, instructions)

        # enable_thinking=False to keep inference distribution aligned with the
        # training-side collate in train_qwen3_5_gr00t.py.
        texts = [
            self.processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            for m in messages
        ]

        image_inputs, video_inputs = process_vision_info(messages)
        batch_input = self.processor(
            text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        )

        # logger.info(
        #     "[Qwen3_5.build_qwenvl_inputs] "
        #     + " ".join(
        #         f"{k}={v.shape}" for k, v in batch_input.items() if isinstance(v, torch.Tensor)
        #     )
        # )

        return batch_input.to(get_platform().device())
