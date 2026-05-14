# Mainly adopted from
# https://github.com/huggingface/lerobot/blob/2b304eeb841ae6c371e3dd341bbbb9dd254b07cb/src/lerobot/scripts/lerobot_train.py

import argparse
import gc
import os
import random
import time
from collections.abc import Mapping
from contextlib import nullcontext
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Any

import datasets as hf_datasets
import numpy as np
import PIL.Image as PILImage
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from qwen_vl_utils import process_vision_info
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.optim import Optimizer
from torchdata.stateful_dataloader import StatefulDataLoader

from flagscale.logger import logger
from flagscale.models.configs.types import FeatureType
from flagscale.models.utils.constants import (
    ACTION,
    OBS_PREFIX,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from flagscale.models.vla import TrainablePolicy
from flagscale.models.vla.pretrained_config import PreTrainedConfig
from flagscale.platforms import get_platform
from flagscale.train.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from flagscale.train.datasets.lerobot_mixture_dataset import LeRobotMixtureDataset
from flagscale.train.datasets.lerobot_multiimage_event_dataset import LeRobotMultiImageEventDataset
from flagscale.train.datasets.utils import (
    dataset_to_policy_features,
    get_hf_features_from_features,
    load_nested_dataset,
)
from flagscale.train.processor import PolicyProcessorPipeline
from flagscale.train.train_config import TrainConfig
from flagscale.train.utils.activation_checkpoint import (
    DEFAULT_OP_SAC_SAVE_LIST,
    apply_activation_checkpointing,
)
from flagscale.train.utils.logging_utils import (
    AverageMeter,
    MetricsTracker,
    format_big_number,
)
from flagscale.train.utils.optim_setup import setup_optimizer_and_scheduler
from flagscale.train.utils.train_utils import (
    StatefulDistributedSampler,
    get_step_checkpoint_dir,
    load_training_state_fsdp2,
    save_checkpoint,
    update_last_checkpoint,
)


_FIXED_POSITION_LOG_COUNT = 0


class LeRobotDatasetWithFutureFrames(LeRobotDataset):
    """LeRobotDataset subclass that splits future frames from visual tensors.

    When delta_timestamps includes a future offset for camera keys, LeRobotDataset
    returns tensors with an extra time dimension (e.g. [T, C, H, W] for video,
    [T, C, H, W] for image via _query_hf_dataset). This subclass splits them into
    current frame (index 0) under the original key and future frames under
    ``{cam_key}_future`` keys.
    """

    def load_hf_dataset(self) -> hf_datasets.Dataset:
        """Override base to keep images as PIL (base converts to tensor via hf_transform_to_torch).

        Uses decode=False so we can resolve relative image paths against self.root.
        See LeRobotDataset.__getitem__ (lerobot_dataset.py:1061) for the call chain:
        hf_dataset[idx] → _query_hf_dataset → _query_videos → image_transforms.
        """

        features = get_hf_features_from_features(self.features)
        # Disable auto-decoding for image columns
        for key in features:
            if isinstance(features[key], hf_datasets.Image):
                features[key] = hf_datasets.Image(decode=False)

        hf_dataset = load_nested_dataset(
            self.root / "data", features=features, episodes=self.episodes
        )

        root = self.root
        image_keys = set(self.meta.image_keys)

        def _resolve_and_transform(items_dict: dict[str, list]) -> dict:
            """set_transform callback — HF calls this on every hf_dataset access.
            Decodes image dicts to PIL; converts other columns to tensors."""

            for key in list(items_dict.keys()):
                values = items_dict[key]
                if key in image_keys:
                    decoded = []
                    for v in values:
                        if isinstance(v, dict):
                            if v.get("bytes") is not None:
                                img = PILImage.open(BytesIO(v["bytes"]))
                            elif v.get("path") is not None:
                                path = v["path"]
                                if not os.path.isabs(path):
                                    path = str(root / path)
                                img = PILImage.open(path)
                            else:
                                raise ValueError(f"Image has neither bytes nor path: {v}")
                            img.load()
                            decoded.append(img)
                        elif isinstance(v, Image.Image):
                            decoded.append(v)
                        else:
                            logger.warning(
                                f"Unexpected type {type(v)} for image key '{key}', passing through"
                            )
                            decoded.append(v)
                    items_dict[key] = decoded
                else:
                    first = values[0]
                    if first is None:
                        pass
                    elif isinstance(first, str):
                        pass
                    else:
                        items_dict[key] = [
                            x if isinstance(x, str) else torch.tensor(x) for x in values
                        ]
            return items_dict

        hf_dataset.set_transform(_resolve_and_transform)
        return hf_dataset

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        """Override base to return list[PIL.Image] for image keys instead of torch.stack().
        Video keys are skipped — they come from _query_videos (mp4 decoding)."""
        image_keys = set(self.meta.image_keys)
        non_image_indices = {k: v for k, v in query_indices.items() if k not in image_keys}
        result = super()._query_hf_dataset(non_image_indices)

        for key, q_idx in query_indices.items():
            if key not in image_keys:
                continue
            if key in self.meta.video_keys:
                continue
            relative_indices = (
                q_idx
                if self._absolute_to_relative_idx is None
                else [self._absolute_to_relative_idx[idx] for idx in q_idx]
            )
            result[key] = list(self.hf_dataset[key][relative_indices])
        return result

    @staticmethod
    def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
        """Convert a CHW float32 [0,1] tensor to a PIL Image."""
        return Image.fromarray(
            (t.permute(1, 2, 0) * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        )

    def __getitem__(self, idx) -> dict:
        """Split multi-frame camera values into current + future keys.
        After super().__getitem__: image keys are list[PIL], video keys are [T,C,H,W] tensors."""
        item = super().__getitem__(idx)
        for cam_key in self.meta.camera_keys:
            if cam_key not in item:
                continue
            val = item[cam_key]
            # PIL list (image keys with delta_timestamps)
            if isinstance(val, list) and len(val) > 1:
                item[f"{cam_key}_future"] = val[1]
                item[cam_key] = val[0]
            # Tensor (video keys with delta_timestamps)
            elif isinstance(val, torch.Tensor) and val.dim() >= 4:
                item[f"{cam_key}_future"] = self._tensor_to_pil(val[1])
                item[cam_key] = self._tensor_to_pil(val[0])
            # Single-frame video tensor
            elif isinstance(val, torch.Tensor) and val.dim() == 3:
                item[cam_key] = self._tensor_to_pil(val)
        return item


class SkipBatch(RuntimeError):
    """Raised by qwen_collate_fn when every sample in a batch is corrupted
    (has ``None`` fields). The training loop catches this and fetches the
    next batch instead of crashing the whole job."""


def _scalarize(v):
    """Best-effort convert a 0-d/1-elem tensor to a Python scalar, else return as-is."""
    if isinstance(v, torch.Tensor):
        try:
            return v.item()
        except Exception:
            return v.tolist()
    return v


def _resize_pil_images(images, target_size=(256, 256)):
    """Recursively resize PIL images, matching starVLA's resize_images pattern."""
    if isinstance(images, Image.Image):
        return images.resize(target_size)
    if isinstance(images, list):
        return [_resize_pil_images(img, target_size=target_size) for img in images]
    raise ValueError(f"Unsupported image type or structure: {type(images)}")


def qwen_collate_fn(
    batch_list: list[dict],
    *,
    processor,
    prompt_template: str,
    image_feature_keys: list[str],
    image_size: tuple[int, int] | None = (256, 256),
    fixed_layout_config: dict | None = None,
    prev_event_prompt: str | None = None,
    next_event_prompt: str | None = None,
    half_event_prompt: str | None = None,
) -> dict:
    """Collate samples into a batch with tokenized Qwen processor inputs.

    In the default path this emits the existing ``qwen_inputs`` and
    ``qwen_future_inputs`` fields. When ``fixed_layout_config.enabled`` is true,
    it instead builds the two-query layout used by the WM event objective:
    ``image -> short query -> instruction -> long query`` plus fixed token
    positions and optional event target groups.
    """
    timing_enabled = os.getenv("FS_WM_DATA_TIMING", "0") == "1"
    if not hasattr(qwen_collate_fn, "_timing_count"):
        qwen_collate_fn._timing_count = 0
    timing_limit = int(os.getenv("FS_WM_COLLATE_TIMING_LIMIT", "8"))
    timing = timing_enabled and qwen_collate_fn._timing_count < timing_limit
    t0 = time.perf_counter() if timing else None

    future_keys = [f"{k}_future" for k in image_feature_keys]
    all_image_keys = set(image_feature_keys) | set(future_keys)
    event_payload_keys = {
        "long_event_supervisions",
        "half_event_images",
        "half_event_direction",
        "has_half_event",
        "event_mode",
    }
    optional_none_keys = event_payload_keys | {
        "task_index",
        "subtask_index",
        "atomic_index",
    }

    optional_index_keys = {"task_index", "subtask_index", "atomic_index"}

    def _normalize_optional_index(value):
        if value is None:
            return torch.tensor(-1, dtype=torch.long)
        if isinstance(value, torch.Tensor):
            return value.to(dtype=torch.long)
        return torch.as_tensor(value, dtype=torch.long)

    original_size = len(batch_list)
    cleaned: list[dict] = []
    for idx, sample in enumerate(batch_list):
        none_keys = [
            k for k, v in sample.items()
            if v is None and k not in optional_none_keys
        ]
        if none_keys:
            ep = _scalarize(sample.get("episode_index"))
            fr = _scalarize(sample.get("frame_index"))
            ds_idx = _scalarize(sample.get("dataset_index"))
            print(
                f"[qwen_collate_fn] DROP bad sample batch_idx={idx} "
                f"dataset_index={ds_idx} episode_index={ep} frame_index={fr} "
                f"none_keys={none_keys}",
                flush=True,
            )
            continue
        sample = dict(sample)
        for key in optional_index_keys:
            sample[key] = _normalize_optional_index(sample.get(key))
        cleaned.append(sample)
    if not cleaned:
        raise SkipBatch(
            f"all {original_size} samples in this batch had None fields; skipping"
        )
    batch_list = cleaned

    image_data: dict[str, list[Image.Image]] = {}
    for key in all_image_keys:
        if key in batch_list[0]:
            image_data[key] = [sample[key] for sample in batch_list]

    non_image_batch = [
        {k: v for k, v in sample.items() if k not in all_image_keys and k not in event_payload_keys}
        for sample in batch_list
    ]
    batch = torch.utils.data.default_collate(non_image_batch)
    batch.update(image_data)
    t_default_collate = time.perf_counter() if timing else None

    instructions = batch["task"]
    if isinstance(instructions, torch.Tensor):
        instructions = instructions.detach().cpu().tolist()
    if isinstance(instructions, str):
        instructions = [instructions]

    def _images_for_keys(feature_keys: list[str]) -> list[list[Image.Image]]:
        batch_images: list[list[Image.Image]] | None = None
        for key in feature_keys:
            imgs = batch[key]
            if not isinstance(imgs, list):
                imgs = [imgs]
            if batch_images is None:
                batch_images = [[img] for img in imgs]
            else:
                for sample_imgs, img in zip(batch_images, imgs):
                    sample_imgs.append(img)
        if batch_images is None:
            raise ValueError(f"No images found for feature keys: {feature_keys}")
        if image_size is not None:
            batch_images = _resize_pil_images(batch_images, target_size=image_size)
        return batch_images

    def _render_prompt(instruction, prompt=None, replacements=None):
        if prompt:
            out = prompt.replace("{instruction}", str(instruction)).replace("{inst}", str(instruction))
            if replacements:
                for key, value in replacements.items():
                    out = out.replace("{" + key + "}", str(value))
            return out
        if prompt_template:
            return prompt_template.replace("{instruction}", str(instruction))
        return str(instruction)

    def _build_messages(images, texts, prompt=None, replacements_per_sample=None):
        replacements_iter = replacements_per_sample or [None] * len(images)
        messages = []
        for imgs, instruction, replacements in zip(images, texts, replacements_iter):
            content = [{"type": "image", "image": img} for img in imgs]
            content.append({"type": "text", "text": _render_prompt(instruction, prompt, replacements)})
            messages.append([{"role": "user", "content": content}])
        return messages

    def _build_inputs(images, texts, prompt=None, replacements_per_sample=None):
        messages = _build_messages(images, texts, prompt, replacements_per_sample)
        rendered = [
            processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            for m in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        result= processor(
            text=rendered,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Debug: log if any sequence exceeds model_max_length
        if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "model_max_length"):
            max_len = processor.tokenizer.model_max_length
            if "input_ids" in result:
                seq_len = (
                    result["input_ids"].shape[-1]
                    if result["input_ids"].dim() > 1
                    else result["input_ids"].shape[0]
                )
                if seq_len > max_len:
                    # Log sample info for debugging
                    ep_idx = batch.get("episode_index", None)
                    sample_idx = batch.get("index", None)
                    task_desc = batch.get("task", None)
                    img_sizes = [f"{img.size}" for imgs in (batch_images or []) for img in imgs]
                    logger.warning(
                        f"[VLA_collate] seq_len={seq_len} > max_len={max_len}. "
                        f"episode_index={ep_idx}, sample_index={sample_idx}, "
                        f"task={task_desc[:100] if isinstance(task_desc, str) else task_desc}, "
                        f"image_sizes={img_sizes}, feature_keys={feature_keys}"
                    )

        return result

    def _extract_fixed_layout_parts(single_inputs):
        vision_start_token_id = 248053
        vision_end_token_id = 248054
        ids = single_inputs["input_ids"][0]
        attention_mask = single_inputs["attention_mask"][0]
        mm_token_type_ids = single_inputs.get("mm_token_type_ids", torch.zeros_like(ids))[0]
        valid = attention_mask.to(torch.bool)
        ids = ids[valid]
        mm_token_type_ids = mm_token_type_ids[valid]
        vision_starts = torch.nonzero(ids == vision_start_token_id, as_tuple=False).flatten()
        vision_ends = torch.nonzero(ids == vision_end_token_id, as_tuple=False).flatten()
        if vision_starts.numel() == 0 or vision_ends.numel() == 0:
            raise ValueError("Fixed layout requires at least one image block in each VLA sample.")
        image_start = int(vision_starts[0].item())
        image_end = int(vision_ends[-1].item())
        return (
            ids[image_start:image_end + 1],
            mm_token_type_ids[image_start:image_end + 1],
            ids[image_end + 1:],
            mm_token_type_ids[image_end + 1:],
        )

    def _build_fixed_layout_inputs(images, texts, prompt=None, replacements_per_sample=None):
        cfg = dict(fixed_layout_config or {})
        if not cfg.get("enabled", False):
            raise ValueError("fixed_layout_config.enabled must be true for fixed-layout inputs.")
        num_query_tokens = int(cfg.get("num_query_tokens", 32))
        instruction_max_tokens = int(cfg.get("instruction_max_tokens", 192))
        pad_token_id = processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = processor.tokenizer.eos_token_id
        vision_start_token_id = 248053
        vision_end_token_id = 248054

        def build_single_inputs(imgs, text, replacements=None):
            return _build_inputs(
                [imgs],
                [text],
                prompt=prompt,
                replacements_per_sample=[replacements] if replacements is not None else None,
            )

        def build_query_wrapped_like_image(image_ids, image_mm):
            starts = torch.nonzero(image_ids == vision_start_token_id, as_tuple=False).flatten()
            ends = torch.nonzero(image_ids == vision_end_token_id, as_tuple=False).flatten()
            if starts.numel() == 0 or starts.numel() != ends.numel():
                raise ValueError("Cannot mirror query wrappers because image start/end tokens are malformed.")
            first_start = int(starts[0].item())
            first_end = int(ends[0].item())
            per_image_query_tokens = first_end - first_start - 1
            if per_image_query_tokens <= 0 or num_query_tokens % per_image_query_tokens != 0:
                raise ValueError(
                    f"num_query_tokens={num_query_tokens} must be divisible by "
                    f"per_image_query_tokens={per_image_query_tokens}."
                )
            query_image_count = num_query_tokens // per_image_query_tokens
            start_token = image_ids[first_start:first_start + 1]
            start_mm = image_mm[first_start:first_start + 1]
            end_token = image_ids[first_end:first_end + 1]
            end_mm = image_mm[first_end:first_end + 1]
            wrapped_ids, wrapped_mm = [], []
            for image_idx in range(query_image_count):
                patch_ids = torch.full((per_image_query_tokens,), pad_token_id, dtype=torch.long)
                patch_mm = torch.zeros((per_image_query_tokens,), dtype=torch.long)
                wrapped_ids.extend([start_token, patch_ids, end_token])
                wrapped_mm.extend([start_mm, patch_mm, end_mm])
            return torch.cat(wrapped_ids), torch.cat(wrapped_mm)

        replacements_iter = replacements_per_sample or [None] * len(images)
        sample_inputs = []
        for imgs, text, replacements in zip(images, texts, replacements_iter):
            single = build_single_inputs(imgs, text, replacements)
            image_ids, image_mm, instruction_ids, instruction_mm = _extract_fixed_layout_parts(single)
            if instruction_ids.numel() > instruction_max_tokens:
                instruction_ids = instruction_ids[-instruction_max_tokens:]
                instruction_mm = instruction_mm[-instruction_max_tokens:]
            short_query_ids, short_query_mm = build_query_wrapped_like_image(image_ids, image_mm)
            long_query_ids, long_query_mm = build_query_wrapped_like_image(image_ids, image_mm)
            ids = torch.cat([image_ids, short_query_ids, instruction_ids, long_query_ids])
            mm = torch.cat([image_mm, short_query_mm, instruction_mm, long_query_mm])
            sample_inputs.append((single, ids, mm, image_ids.numel(), short_query_ids.numel(), instruction_ids.numel()))

        batch_seq_len = max(int(ids.numel()) for _, ids, *_ in sample_inputs)
        padded_ids, masks, padded_mm = [], [], []
        image_ends, short_starts, short_ends = [], [], []
        instruction_starts, instruction_ends, long_starts, long_ends = [], [], [], []
        all_pixel_values, all_image_grid_thw = [], []
        for single, ids, mm, image_len, query_len, instruction_len in sample_inputs:
            left_pad_len = batch_seq_len - ids.numel()
            left_ids = torch.full((left_pad_len,), pad_token_id, dtype=torch.long)
            left_mm = torch.zeros((left_pad_len,), dtype=torch.long)
            padded_ids.append(torch.cat([left_ids, ids]))
            padded_mm.append(torch.cat([left_mm, mm]))
            masks.append(torch.cat([torch.zeros((left_pad_len,), dtype=torch.long), torch.ones_like(ids)]))
            image_end = left_pad_len + image_len - 1
            short_start = left_pad_len + image_len
            short_end = short_start + query_len - 1
            instruction_start = short_end + 1
            instruction_end = instruction_start + instruction_len - 1
            long_start = batch_seq_len - query_len
            long_end = batch_seq_len - 1
            image_ends.append(image_end)
            short_starts.append(short_start)
            short_ends.append(short_end)
            instruction_starts.append(instruction_start)
            instruction_ends.append(instruction_end)
            long_starts.append(long_start)
            long_ends.append(long_end)
            if "pixel_values" in single:
                all_pixel_values.append(single["pixel_values"])
            if "image_grid_thw" in single:
                all_image_grid_thw.append(single["image_grid_thw"])

        fixed_positions = {
            "image_end": torch.tensor(image_ends, dtype=torch.long),
            "query_start": torch.tensor(short_starts, dtype=torch.long),
            "query_end": torch.tensor(short_ends, dtype=torch.long),
            "short_query_start": torch.tensor(short_starts, dtype=torch.long),
            "short_query_end": torch.tensor(short_ends, dtype=torch.long),
            "instruction_start": torch.tensor(instruction_starts, dtype=torch.long),
            "instruction_end": torch.tensor(instruction_ends, dtype=torch.long),
            "long_query_start": torch.tensor(long_starts, dtype=torch.long),
            "long_query_end": torch.tensor(long_ends, dtype=torch.long),
            "query_patch_tokens": num_query_tokens,
            "query_block_tokens": max(x[4] for x in sample_inputs),
        }
        batch_input = {
            "input_ids": torch.stack(padded_ids),
            "attention_mask": torch.stack(masks),
            "mm_token_type_ids": torch.stack(padded_mm),
        }

        for sample_idx, ids in enumerate(batch_input["input_ids"]):
            seq_len = int(batch_input["attention_mask"][sample_idx].numel())
            image_end = int(fixed_positions["image_end"][sample_idx])
            short_start = int(fixed_positions["short_query_start"][sample_idx])
            short_end = int(fixed_positions["short_query_end"][sample_idx])
            instruction_start = int(fixed_positions["instruction_start"][sample_idx])
            instruction_end = int(fixed_positions["instruction_end"][sample_idx])
            long_start = int(fixed_positions["long_query_start"][sample_idx])
            long_end = int(fixed_positions["long_query_end"][sample_idx])
            if not (0 <= image_end < short_start <= short_end < instruction_start <= instruction_end < long_start <= long_end < seq_len):
                raise ValueError(
                    "Invalid fixed-layout positions: "
                    f"sample={sample_idx} image_end={image_end} "
                    f"short=({short_start},{short_end}) instruction=({instruction_start},{instruction_end}) "
                    f"long=({long_start},{long_end}) seq_len={seq_len}"
                )
            for name, start, end in (
                ("short", short_start, short_end),
                ("long", long_start, long_end),
            ):
                query_ids = ids[start:end + 1]
                query_mask = (query_ids != vision_start_token_id) & (query_ids != vision_end_token_id)
                patch_count = int(query_mask.sum().item())
                if patch_count != num_query_tokens:
                    raise ValueError(
                        f"Invalid {name} query patch-token count: sample={sample_idx} "
                        f"got={patch_count} expected={num_query_tokens} "
                        f"range=({start},{end})"
                    )

        global _FIXED_POSITION_LOG_COUNT
        debug_limit = int(os.environ.get("WM_DEBUG_FIXED_POSITIONS", "0") or 0)
        if debug_limit > _FIXED_POSITION_LOG_COUNT:
            _FIXED_POSITION_LOG_COUNT += 1
            logger.info(
                "[fixed_position_check] "
                f"batch={len(sample_inputs)} seq_len={batch_input["input_ids"].shape[1]} "
                f"query_patch_tokens={num_query_tokens} query_block_tokens={fixed_positions["query_block_tokens"]} "
                f"sample0_image_end={image_ends[0]} "
                f"sample0_short=({short_starts[0]},{short_ends[0]}) "
                f"sample0_instruction=({instruction_starts[0]},{instruction_ends[0]}) "
                f"sample0_long=({long_starts[0]},{long_ends[0]})"
            )
        if all_pixel_values:
            batch_input["pixel_values"] = torch.cat(all_pixel_values, dim=0)
        if all_image_grid_thw:
            batch_input["image_grid_thw"] = torch.cat(all_image_grid_thw, dim=0)
        return batch_input, fixed_positions

    batch_images = _images_for_keys(image_feature_keys)
    t_current_images = time.perf_counter() if timing else None
    use_fixed_layout = bool((fixed_layout_config or {}).get("enabled", False))
    if use_fixed_layout:
        batch["qwen_inputs"], batch["qwen_fixed_positions"] = _build_fixed_layout_inputs(
            batch_images, instructions
        )
    else:
        batch["qwen_inputs"] = _build_inputs(batch_images, instructions)
    t_qwen_current = time.perf_counter() if timing else None

    if future_keys and future_keys[0] in batch:
        batch["qwen_future_inputs"] = _build_inputs(_images_for_keys(future_keys), instructions)
    t_qwen_future = time.perf_counter() if timing else None

    if use_fixed_layout:
        long_event_supervisions = [sample.get("long_event_supervisions", []) for sample in batch_list]
        event_modes = list(LeRobotMultiImageEventDataset.EVENT_MODES)

        qwen_long_event_groups = []
        for mode in event_modes:
            per_sample_groups = [
                next((g for g in groups if g.get("mode", "event") == mode), None)
                for groups in long_event_supervisions
            ]
            prev_target_images = [
                group["prev_images"] if group and group.get("has_prev", False) else batch_images[i]
                for i, group in enumerate(per_sample_groups)
            ]
            next_target_images = [
                group["next_images"] if group and group.get("has_next", False) else batch_images[i]
                for i, group in enumerate(per_sample_groups)
            ]
            prev_event_langs = [
                group.get("prev_lang", "") if group and group.get("has_prev", False) else instructions[i]
                for i, group in enumerate(per_sample_groups)
            ]
            next_event_langs = [
                group.get("next_lang", "") if group and group.get("has_next", False) else instructions[i]
                for i, group in enumerate(per_sample_groups)
            ]
            has_prev = [bool(group and group.get("has_prev", False)) for group in per_sample_groups]
            has_next = [bool(group and group.get("has_next", False)) for group in per_sample_groups]
            event_type = mode.replace("_", " ")
            qwen_prev_inputs, qwen_prev_fixed_positions = _build_fixed_layout_inputs(
                batch_images,
                instructions,
                prompt=prev_event_prompt,
                replacements_per_sample=[
                    {"prev_instruction": text, "event_type": event_type, "event_mode": mode}
                    for text in prev_event_langs
                ],
            )
            qwen_next_inputs, qwen_next_fixed_positions = _build_fixed_layout_inputs(
                batch_images,
                instructions,
                prompt=next_event_prompt,
                replacements_per_sample=[
                    {"next_instruction": text, "event_type": event_type, "event_mode": mode}
                    for text in next_event_langs
                ],
            )
            qwen_long_event_groups.append({
                "mode": mode,
                "qwen_prev_inputs": qwen_prev_inputs,
                "qwen_prev_fixed_positions": qwen_prev_fixed_positions,
                "qwen_next_inputs": qwen_next_inputs,
                "qwen_next_fixed_positions": qwen_next_fixed_positions,
                "qwen_prev_target_inputs": _build_inputs(prev_target_images, instructions),
                "qwen_next_target_inputs": _build_inputs(next_target_images, instructions),
                "has_rnd_prev": has_prev,
                "has_rnd_next": has_next,
            })
        batch["qwen_long_event_groups"] = qwen_long_event_groups
        t_long_event = time.perf_counter() if timing else None

        has_half_event = [bool(sample.get("has_half_event", False)) for sample in batch_list]
        batch["has_half_event"] = has_half_event
        if any(has_half_event):
            half_images = [
                sample.get("half_event_images", []) if has_half_event[i] else batch_images[i]
                for i, sample in enumerate(batch_list)
            ]
            half_directions = [sample.get("half_event_direction", "opposite") for sample in batch_list]
            half_prompt = half_event_prompt or (
                "The current task is: {instruction}. Given the current observation, "
                "predict what the observation should look like in the {half_direction} half of this episode."
            )
            batch["qwen_half_event_inputs"], batch["qwen_half_event_fixed_positions"] = _build_fixed_layout_inputs(
                batch_images,
                instructions,
                prompt=half_prompt,
                replacements_per_sample=[{"half_direction": d or "opposite"} for d in half_directions],
            )
            batch["qwen_half_event_target_inputs"] = _build_inputs(half_images, instructions)
        else:
            batch["qwen_half_event_inputs"] = None
            batch["qwen_half_event_fixed_positions"] = None
            batch["qwen_half_event_target_inputs"] = None
    t_half_event = time.perf_counter() if timing else None

    for key in all_image_keys:
        batch.pop(key, None)

    if timing:
        t_end = time.perf_counter()
        qwen_collate_fn._timing_count += 1
        logger.info(
            "[collate_timing] "
            f"batch={len(batch_list)} default_collate_s={t_default_collate - t0:.4f} "
            f"resize_current_s={t_current_images - t_default_collate:.4f} "
            f"qwen_current_s={t_qwen_current - t_current_images:.4f} "
            f"qwen_future_s={(t_qwen_future - t_qwen_current) if t_qwen_future else 0.0:.4f} "
            f"long_event_s={(t_long_event - t_qwen_future) if use_fixed_layout else 0.0:.4f} "
            f"half_event_s={(t_half_event - t_long_event) if use_fixed_layout else 0.0:.4f} "
            f"total_s={t_end - t0:.4f}"
        )
    return batch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    get_platform().manual_seed_all(seed)
    if get_platform().name() == "cuda":
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True


def apply_fsdp2(policy, device_mesh):
    """Apply FSDP2 sharding to a VLA policy.

    Uses a MixedPrecisionPolicy that matches DeepSpeed bf16 behavior:
      bf16.enabled=true + ZeRO-2 → param_dtype=bf16, reduce_dtype=bf16, reshard=False
    """
    # Cast everything to fp32 first so the root param group has uniform dtype.
    policy = policy.float()

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
    )
    fsdp_config = {"mesh": device_mesh, "mp_policy": mp_policy}

    # reshard_after_forward=False keeps params unsharded during forward+backward
    reshard = True

    for unit in policy.fsdp_units():
        fully_shard(unit, **fsdp_config, reshard_after_forward=reshard)

    fully_shard(policy, **fsdp_config)

    # Enable forward/backward prefetch to overlap all-gather with compute
    lang_layers = list(policy.vlm.model.model.language_model.layers)
    for i in range(len(lang_layers) - 1):
        lang_layers[i].set_modules_to_forward_prefetch([lang_layers[i + 1]])
        lang_layers[i].set_modules_to_backward_prefetch([lang_layers[i + 1]])

    # vis_blocks = list(policy.vlm.model.model.visual.blocks)
    # for i in range(len(vis_blocks) - 1):
    #     vis_blocks[i].set_modules_to_forward_prefetch([vis_blocks[i + 1]])
    #     vis_blocks[i].set_modules_to_backward_prefetch([vis_blocks[i + 1]])


def _resolve_video_backend():
    # torchcodec depends on NVIDIA NVDEC which is not available on all platforms (e.g. MUSA);
    # fall back to pyav for non-CUDA platforms or when torchcodec is broken.
    video_backend = "pyav"
    if get_platform().name() == "cuda":
        try:
            import torchcodec  # noqa: F401

            _ = torch.ops.torchcodec_ns  # verify the C++ ops are loadable
            video_backend = "torchcodec"
        except Exception:
            logger.warning("torchcodec unavailable, falling back to pyav")
            video_backend = "pyav"
    return video_backend


def _make_image_transforms():
    def _resize_to_pil(img):
        """Resize to 256×256 PIL. Accepts PIL, tensor, or list of either."""
        if isinstance(img, list):
            return [_resize_to_pil(i) for i in img]
        if isinstance(img, Image.Image):
            return img.resize((256, 256))
        # Tensor from _query_videos: float32 CHW [0,1] or [T, C, H, W]
        if isinstance(img, torch.Tensor):
            if img.dim() == 4:
                return [_resize_to_pil(f) for f in img]
            frame_uint8 = (img.permute(1, 2, 0) * 255).round().clamp(0, 255).to(torch.uint8)
            pil = Image.fromarray(frame_uint8.cpu().numpy()).resize((256, 256))
            return pil
        return img

    return _resize_to_pil


def _make_single_dataset(
    data_path: str,
    policy_config: PreTrainedConfig,
    future_offset: float | None,
    tolerance_s: float,
    video_backend: str = "pyav",
    image_transforms=None,
) -> LeRobotDatasetWithFutureFrames:
    ds_meta = LeRobotDatasetMetadata(root=data_path, revision=None)
    delta_timestamps = _resolve_delta_timestamps(policy_config, ds_meta)

    # _resolve_delta_timestamps applies observation_delta_indices uniformly to all
    # observation keys, but the future frame offset should only apply to camera
    # keys (video or image). We inject it manually here.
    if future_offset is not None:
        if delta_timestamps is None:
            delta_timestamps = {}
        for cam_key in ds_meta.camera_keys:
            ts_list = list(delta_timestamps.get(cam_key, [0.0]))
            ts_list.append(int(future_offset) / ds_meta.fps)
            delta_timestamps[cam_key] = ts_list

    return LeRobotDatasetWithFutureFrames(
        root=data_path,
        episodes=None,
        delta_timestamps=delta_timestamps,
        image_transforms=image_transforms,
        revision=None,
        video_backend=video_backend,
        tolerance_s=tolerance_s,
    )


def make_dataset(config: TrainConfig, policy_config: PreTrainedConfig, seed: int = 42):
    future_offset = getattr(config.data, "future_offset", None)
    data_mix = getattr(config.data, "data_mix", None)
    dataset_type = getattr(config.data, "dataset_type", "lerobot")
    use_multiimage_event = dataset_type == "lerobot_multiimage_event"
    video_backend = _resolve_video_backend()
    image_transforms = None

    if data_mix is not None:
        data_root_dir = getattr(config.data, "data_root_dir", None)
        if data_root_dir is None:
            raise ValueError("data_root_dir must be set when using data_mix")

        from examples.qwen3_5_gr00t.mixtures import DATASET_MIXTURES

        if data_mix not in DATASET_MIXTURES:
            raise ValueError(
                f"Unknown data_mix: {data_mix}. Available: {list(DATASET_MIXTURES.keys())}"
            )

        image_transforms = _make_image_transforms()
        mixture_spec = DATASET_MIXTURES[data_mix]
        data_mixture = []
        for dataset_name, weight in mixture_spec:
            data_path = f"{data_root_dir}/{dataset_name}"
            if use_multiimage_event:
                ds_meta = LeRobotDatasetMetadata(root=data_path, revision=None)
                delta_timestamps = _resolve_delta_timestamps(policy_config, ds_meta)
                if future_offset is not None:
                    if delta_timestamps is None:
                        delta_timestamps = {}
                    for cam_key in ds_meta.camera_keys:
                        ts_list = list(delta_timestamps.get(cam_key, [0.0]))
                        ts_list.append(int(future_offset) / ds_meta.fps)
                        delta_timestamps[cam_key] = ts_list
                ds = LeRobotMultiImageEventDataset(
                    root=data_path,
                    episodes=None,
                    delta_timestamps=delta_timestamps,
                    image_transforms=None,
                    revision=None,
                    video_backend=video_backend,
                    tolerance_s=config.data.tolerance_s,
                )
            else:
                ds = _make_single_dataset(
                    data_path,
                    policy_config,
                    future_offset,
                    config.data.tolerance_s,
                    video_backend=video_backend,
                    image_transforms=image_transforms,
                )
            data_mixture.append((ds, weight))

        balance = getattr(config.data, "balance_dataset_weights", True)
        dataset = LeRobotMixtureDataset(
            data_mixture=data_mixture,
            mode="train",
            balance_dataset_weights=balance,
            seed=seed,
        )
    else:
        if use_multiimage_event:
            ds_meta = LeRobotDatasetMetadata(root=config.data.data_path, revision=None)
            delta_timestamps = _resolve_delta_timestamps(policy_config, ds_meta)
            if future_offset is not None:
                if delta_timestamps is None:
                    delta_timestamps = {}
                for cam_key in ds_meta.camera_keys:
                    ts_list = list(delta_timestamps.get(cam_key, [0.0]))
                    ts_list.append(int(future_offset) / ds_meta.fps)
                    delta_timestamps[cam_key] = ts_list
            dataset = LeRobotMultiImageEventDataset(
                root=config.data.data_path,
                episodes=None,
                delta_timestamps=delta_timestamps,
                image_transforms=None,
                revision=None,
                video_backend=video_backend,
                tolerance_s=config.data.tolerance_s,
            )
        else:
            dataset = _make_single_dataset(
                config.data.data_path,
                policy_config,
                future_offset,
                config.data.tolerance_s,
                video_backend=video_backend,
                image_transforms=_make_image_transforms(),
            )

    return dataset


def make_policy(cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata):
    features = dataset_to_policy_features(ds_meta.features)
    cfg.output_features = {k: f for k, f in features.items() if f.type is FeatureType.ACTION}
    cfg.input_features = {k: f for k, f in features.items() if k not in cfg.output_features}
    policy = TrainablePolicy.from_config(cfg)
    policy.to(get_platform().name())
    policy.train()
    return policy


def _resolve_delta_timestamps(
    cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg (PreTrainedConfig): The policy config to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith(OBS_PREFIX) and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def format_train_tracker_step(train_tracker: MetricsTracker) -> str:
    def _format_meter_val(meter: AverageMeter) -> str:
        fmt = meter.fmt[1:] if meter.fmt.startswith(":") else meter.fmt
        return f"{meter.name}:{format(meter.val, fmt)}"

    display_list = [
        f"step:{format_big_number(train_tracker.steps)}",
        f"smpl:{format_big_number(train_tracker.samples)}",
        f"ep:{format_big_number(train_tracker.episodes)}",
        f"epch:{train_tracker.epochs:.2f}",
        *[_format_meter_val(m) for m in train_tracker.metrics.values()],
    ]
    return " ".join(display_list)


def make_pre_post_processors(
    policy,
    data_config,
    dataset_stats: dict[str, Any],
    device: str,
) -> tuple[PolicyProcessorPipeline | None, PolicyProcessorPipeline | None]:
    """Build pre- and post-processor pipelines from YAML config + policy config.

    The policy config is the single source of truth for features and norm_map.
    YAML (``data_config.preprocessor`` / ``data_config.postprocessor``) defines
    the step list; runtime values (stats, features, norm_map, device) are
    injected as overrides.

    Args:
        policy: The policy model — provides input_features, output_features,
            and config.normalization_mapping.
        data_config: The ``data`` section of the training config (OmegaConf).
            Must have ``preprocessor`` and/or ``postprocessor`` fields, each
            with ``name`` and ``steps``.
        dataset_stats: Per-feature statistics from the dataset metadata.
        device: Target device string (e.g. ``"cuda"``).

    Returns:
        (preprocessor, postprocessor) — either may be None if not configured.
    """
    features = {**policy.input_features, **policy.output_features}
    norm_map = policy.config.normalization_mapping

    preprocessor = None
    if getattr(data_config, "preprocessor", None) is not None:
        preprocessor = _build_pipeline_from_config(
            data_config.preprocessor,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
            overrides={
                "device_processor": {"device": device},
                "normalizer_processor": {
                    "stats": dataset_stats,
                    "features": features,
                    "norm_map": norm_map,
                },
            },
        )

    postprocessor = None
    if getattr(data_config, "postprocessor", None) is not None:
        postprocessor = _build_pipeline_from_config(
            data_config.postprocessor,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            overrides={
                "unnormalizer_processor": {
                    "stats": dataset_stats,
                    "features": features,
                    "norm_map": norm_map,
                },
            },
        )

    return preprocessor, postprocessor


def _build_pipeline_from_config(
    config: dict[str, Any] | list[str | dict[str, Any]],
    name: str,
    overrides: dict[str, Any] | None = None,
) -> PolicyProcessorPipeline[dict[str, Any], dict[str, Any]]:
    """
    Create a processor pipeline from step configurations with optional overrides.

    This function creates a PolicyProcessorPipeline directly from step configurations,
    without requiring a pretrained path. It supports overriding step configurations
    similar to PolicyProcessorPipeline.from_pretrained().

    Args:
        config: Can be either:
            - A dict with "name" and "steps" fields (JSON format):
              {"name": "policy_preprocessor", "steps": [...]}
            - A list of step configurations (concise format):
              ["step_name", {"step_name": {...}}]
        name: Pipeline name (e.g. "policy_preprocessor", "policy_postprocessor").
        overrides: Optional dictionary to override step configurations. Keys should
            match the step's registry_name.

    Returns:
        A PolicyProcessorPipeline instance with the configured steps.


    Raises:
        ValueError: If a step configuration is invalid or step cannot be instantiated.
        KeyError: If a registry name is not found.
    """
    from flagscale.train.processor.pipeline import ProcessorStepRegistry

    overrides = overrides or {}

    # Determine format and extract step configs
    if isinstance(config, (dict, DictConfig)) and "steps" in config:
        # JSON format: {"name": "...", "steps": [...]}
        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)
        step_configs = config["steps"]
    elif isinstance(config, list):
        # Concise list format
        step_configs = config
    else:
        raise ValueError(f"Config must be a dict with 'steps' key or a list, got {type(config)}")

    steps = []
    for step_entry in step_configs:
        # Determine step format and normalize to standard dict
        if isinstance(step_entry, str):
            # Concise format: "step_name"
            step_dict = {"registry_name": step_entry, "config": {}}
        elif isinstance(step_entry, (dict, DictConfig)):
            if "registry_name" in step_entry:
                # JSON format: {"registry_name": "...", "config": {...}}
                if isinstance(step_entry, DictConfig):
                    step_entry = OmegaConf.to_container(step_entry, resolve=True)
                step_dict = step_entry
            elif len(step_entry) == 1:
                # Concise format: {"step_name": {...}}
                step_name = next(iter(step_entry.keys()))
                step_config = step_entry[step_name]
                if isinstance(step_config, DictConfig):
                    step_config = OmegaConf.to_container(step_config, resolve=True)
                step_dict = {"registry_name": step_name, "config": step_config}
            else:
                raise ValueError(
                    f"Step config dict must have either 'registry_name' or exactly one key, "
                    f"got {list(step_entry.keys())}"
                )
        else:
            raise ValueError(
                f"Step config must be str or dict, got {type(step_entry)}: {step_entry}"
            )

        # Get step class
        registry_name = step_dict["registry_name"]
        step_class = ProcessorStepRegistry.get(registry_name)

        # Merge config with overrides (overrides take precedence)
        try:
            base_config = step_dict.get("config", {})
            step_overrides = overrides.get(registry_name, {})
            merged_config = {**base_config, **step_overrides}

            step_instance = step_class(**merged_config)
            steps.append(step_instance)
        except Exception as e:
            raise ValueError(
                f"Failed to instantiate processor step '{registry_name}' "
                f"with config {merged_config}. Error: {e!s}"
            ) from e

    return PolicyProcessorPipeline(
        steps=steps,
        name=name,
    )


def has_method(cls: object, method_name: str) -> bool:
    return hasattr(cls, method_name) and callable(getattr(cls, method_name))


def update_policy(
    train_metrics: MetricsTracker,
    policy,
    batch: Any,
    optimizer: Optimizer,
    use_amp: bool,
    grad_clip_norm: float,
    lr_scheduler=None,
    lock=None,
    vlm_batch: Any = None,
    vlm_loss_scale: float = 0.0,
) -> MetricsTracker:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained (FSDP2-sharded).
        batch: A batch of VLA training data (robot observations + actions).
        optimizer: The optimizer used to update the policy's parameters.
        use_amp: Whether to use automatic mixed precision.
        grad_clip_norm: The maximum norm for gradient clipping.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.
        vlm_batch: Optional batch of VLM co-training data. When provided, the policy
            computes an additional language modelling loss on this batch (via the VLM
            backbone's causal LM head) and adds it to the action loss. Expected keys
            match the HF Qwen model inputs: input_ids, attention_mask, labels, and
            optionally pixel_values / image_grid_thw for multimodal samples.
        vlm_loss_scale: Weight applied to the VLM loss before adding to action loss.

    Returns:
        The updated MetricsTracker with new statistics for this step.
    """
    with torch.profiler.record_function("optimizer_zero_grad"):
        optimizer.zero_grad()

    autocast_context = (
        torch.amp.autocast(get_platform().amp_device_type(), dtype=torch.bfloat16)
        if use_amp
        else nullcontext()
    )

    vla_batches = batch if isinstance(batch, list) else [batch]
    vla_outputs = []
    for micro_idx, vla_batch in enumerate(vla_batches):
        is_last_vla = micro_idx == len(vla_batches) - 1
        with torch.profiler.record_function("forward_vla"):
            with autocast_context:
                micro_output = policy(vla_batch)
                vla_outputs.append(micro_output)
                vla_loss = micro_output["loss"] / len(vla_batches)

        with torch.profiler.record_function("backward_vla"):
            if hasattr(policy, "set_is_last_backward"):
                # WARNING: We need to check if this is ok
                policy.set_is_last_backward(is_last_vla and vlm_batch is None)
            vla_loss.backward()

    if not vla_outputs:
        raise RuntimeError("update_policy received no VLA batches")

    output = dict(vla_outputs[-1])
    for key in vla_outputs[-1].keys():
        values = [out[key] for out in vla_outputs if key in out and torch.is_tensor(out[key]) and out[key].ndim == 0]
        if len(values) == len(vla_outputs):
            output[key] = torch.stack([value.detach() for value in values]).mean()

    if vlm_batch is not None:
        with torch.profiler.record_function("forward_vlm"), autocast_context:
            policy.set_is_last_backward(True)
            vlm_output = policy(vlm_batch, mode="vlm")
            vlm_loss_scaled = vlm_loss_scale * vlm_output["vlm_loss"]

        with torch.profiler.record_function("backward_vlm"):
            vlm_loss_scaled.backward()
        output["vlm_loss"] = vlm_output["vlm_loss"]

    with torch.profiler.record_function("grad_clip"):
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), grad_clip_norm if grad_clip_norm > 0 else float("inf")
        )

    with torch.profiler.record_function("optimizer_step"):
        with lock if lock is not None else nullcontext():
            optimizer.step()

    with torch.profiler.record_function("lr_scheduler_step"):
        if lr_scheduler is not None:
            lr_scheduler.step()

    # Update internal buffers if policy has update method
    if has_method(policy, "update"):
        policy.update()

    train_metrics.grad_norm = (
        grad_norm.full_tensor().item() if hasattr(grad_norm, "full_tensor") else grad_norm.item()
    )
    train_metrics.lr = next(
        (g["lr"] for g in optimizer.param_groups if g.get("name") == "vlm"),
        optimizer.param_groups[0]["lr"],
    )
    if "raw_action_loss" in output and "action_loss" in train_metrics.metrics:
        train_metrics.action_loss = output["raw_action_loss"].item()
    if "nfp_mse_loss_0" in output and "nfp_mse_loss" in train_metrics.metrics:
        train_metrics.nfp_mse_loss = output["nfp_mse_loss_0"].item()
    if "nfp_cosine_loss_0" in output and "nfp_cosine_loss" in train_metrics.metrics:
        train_metrics.nfp_cosine_loss = output["nfp_cosine_loss_0"].item()
    if "short_future_loss_0" in output and "short_future_loss" in train_metrics.metrics:
        train_metrics.short_future_loss = output["short_future_loss_0"].item()
    if "prev_event_loss" in output and "prev_event_loss" in train_metrics.metrics:
        train_metrics.prev_event_loss = output["prev_event_loss"].item()
    if "next_event_loss" in output and "next_event_loss" in train_metrics.metrics:
        train_metrics.next_event_loss = output["next_event_loss"].item()
    if "long_event_loss" in train_metrics.metrics:
        prev_loss = output.get("prev_event_loss")
        next_loss = output.get("next_event_loss")
        if prev_loss is not None and next_loss is not None:
            train_metrics.long_event_loss = (prev_loss + next_loss).item()
    if "vlm_loss" in output and "vlm_loss" in train_metrics.metrics:
        train_metrics.vlm_loss = output["vlm_loss"].item() * vlm_loss_scale

    if vlm_batch is not None and "vlm_bsz" in train_metrics.metrics:
        vlm_bsz = vlm_batch["input_ids"].shape[0]
        train_metrics.vlm_bsz = vlm_bsz

    vla_batches = batch if isinstance(batch, list) else [batch]
    vla_total_bsz = sum(mb["action"].shape[0] for mb in vla_batches)
    train_metrics.vla_bsz = vla_total_bsz

    return train_metrics


def main(config: TrainConfig, seed: int):
    set_seed(seed)

    policy_config = PreTrainedConfig.from_train_config(config)

    local_rank = int(os.environ["LOCAL_RANK"])
    get_platform().set_device(local_rank)
    device = get_platform().device(local_rank)
    dist.init_process_group(backend=get_platform().dist_backend())
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    is_main_process = rank == 0

    if config.data.dataset_type == "wds":
        from megatron.energon import WorkerConfig, get_loader, get_train_dataset

        from flagscale.models.vla.qwen3_5_gr00t import Qwen35Gr00tConfig
        from flagscale.models.vla.qwen_gr00t.task_encoder_qwen_gr00t import TaskEncoder

        if not isinstance(policy_config, Qwen35Gr00tConfig):
            raise ValueError(
                f"wds dataset_type only supports Qwen35Gr00t, got {type(policy_config).__name__}"
            )

        policy = TrainablePolicy.from_config(policy_config)
        policy.to(get_platform().name())

        ds = get_train_dataset(
            config.data.data_path,
            batch_size=config.system.batch_size,
            task_encoder=TaskEncoder(config.data.wds),
            shuffle_buffer_size=1000,
            max_samples_per_sequence=100,
            worker_config=WorkerConfig.default_worker_config(
                num_workers=config.system.num_workers,
                data_parallel_group=None,
            ),
            repeat=True,
        )
        dataloader = get_loader(ds)
        dl_iter = iter(dataloader)

        vlm_dl = None
        vlm_dl_iter = None
        vlm_sampler = None
        if getattr(config.data, "vlm_data_path", None):
            vlm_ds = get_train_dataset(
                config.data.vlm_data_path,
                batch_size=config.system.batch_size,
                task_encoder=TaskEncoder(config.data.wds),
                shuffle_buffer_size=1000,
                max_samples_per_sequence=100,
                worker_config=WorkerConfig.default_worker_config(
                    num_workers=config.system.num_workers,
                    data_parallel_group=None,
                ),
                repeat=True,
            )
            vlm_dl_iter = iter(get_loader(vlm_ds))
        preprocessor = None
        postprocessor = None
        sampler = None
        # Only to make the `MetricsTracker` work for now
        num_frames = 1
        num_episodes = 1
    else:
        dataset = make_dataset(config, policy_config, seed=seed)
        dist.barrier(device_ids=[local_rank])

        policy = make_policy(policy_config, dataset.meta)
        dist.barrier(device_ids=[local_rank])

        dataset_stats = (
            dataset.merged_stats
            if isinstance(dataset, LeRobotMixtureDataset)
            else dataset.meta.stats
        )
        preprocessor, postprocessor = make_pre_post_processors(
            policy,
            config.data,
            dataset_stats=dataset_stats,
            device=device.type,
        )

        num_workers = config.system.num_workers
        shuffle = config.system.shuffle

        sampler = StatefulDistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=False,
            seed=seed,
        )

        collate_fn = partial(
            qwen_collate_fn,
            processor=policy.vlm.processor,
            prompt_template=policy.vlm._prompt_template,
            image_feature_keys=list(policy.image_features.keys()),
            fixed_layout_config=getattr(config.data, "fixed_layout", None),
            prev_event_prompt=getattr(config.data, "prev_event_prompt", None),
            next_event_prompt=getattr(config.data, "next_event_prompt", None),
            half_event_prompt=getattr(config.data, "half_event_prompt", None),
        )

        dataloader = StatefulDataLoader(
            dataset,
            num_workers=num_workers,
            batch_size=config.system.batch_size,
            shuffle=False,  # Must be False when using sampler
            sampler=sampler,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=False,
            prefetch_factor=2 if num_workers > 0 else None,
        )

        dl_iter = iter(dataloader)
        num_frames = dataset.num_frames
        num_episodes = dataset.num_episodes
        vlm_dl = None
        vlm_dl_iter = None
        vlm_sampler = None
        if getattr(config.data, "vlm_data", None) is not None:
            # Set data root paths from YAML config before importing qwen_data_config
            vlm_data_cfg = config.data.vlm_data
            for env_key, cfg_key in [
                ("VLM_DATA_ROOT", "vlm_data_root"),
                ("VLM_VIDEO_ROOT", "video_data_root"),
                ("VLM_IMAGE_ROOT", "image_root"),
            ]:
                val = getattr(vlm_data_cfg, cfg_key, None)
                if val is not None:
                    os.environ[env_key] = str(val)

            from types import SimpleNamespace

            from flagscale.train.datasets.vlm_datasets_qwen35 import make_vlm_dataloader

            vlm_cfg = SimpleNamespace(
                datasets=SimpleNamespace(vlm_data=config.data.vlm_data),
                framework=SimpleNamespace(
                    qwenvl=SimpleNamespace(base_vlm=config.model.vlm.base_vlm)
                ),
            )
            vlm_data_module = make_vlm_dataloader(
                vlm_cfg, rank=rank, world_size=world_size, seed=seed
            )
            vlm_dl = vlm_data_module["train_dataloader"]
            vlm_sampler = vlm_data_module["sampler"]
            vlm_dl_iter = iter(vlm_dl)

    # --- Apply Activation Checkpointing (before FSDP) ---
    ac_config = config.system.activation_checkpoint
    apply_activation_checkpointing(
        policy,
        ac_config,
        units=policy.fsdp_units(),
        op_sac_save_list=DEFAULT_OP_SAC_SAVE_LIST,
    )

    # --- Apply FSDP2 ---
    device_mesh = init_device_mesh(get_platform().name(), (world_size,))
    apply_fsdp2(policy, device_mesh)

    # Setup optimizer and scheduler (applies freeze config internally)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(policy, config)

    dist.barrier(device_ids=[local_rank])

    step = 0
    resume_from = config.system.checkpoint.resume_from
    if resume_from:
        step, dl_state = load_training_state_fsdp2(
            Path(resume_from),
            policy,
            optimizer,
            lr_scheduler,
        )
        if dl_state is not None:
            if "vla" in dl_state:
                dataloader.load_state_dict(dl_state["vla"])
            if "vlm" in dl_state and vlm_dl is not None:
                vlm_dl.load_state_dict(dl_state["vlm"])
        epoch = sampler.epoch if sampler is not None else 0
        if isinstance(dataset, LeRobotMixtureDataset):
            dataset.set_epoch(epoch)
        dl_iter = iter(dataloader)
        if vlm_dl is not None:
            vlm_dl_iter = iter(vlm_dl)
        logger.info(f"Resumed from checkpoint at step {step}")

    train_metrics = {
        "action_loss": AverageMeter("act_loss", ":.3f"),
        "nfp_mse_loss": AverageMeter("nfp_mse", ":.4f"),
        "nfp_cosine_loss": AverageMeter("nfp_cos", ":.4f"),
        "short_future_loss": AverageMeter("short_q", ":.3f"),
        "prev_event_loss": AverageMeter("prev_q", ":.3f"),
        "next_event_loss": AverageMeter("next_q", ":.3f"),
        "long_event_loss": AverageMeter("long_q", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
        "vla_bsz": AverageMeter("vla_bsz", ":.1f"),
    }
    if vlm_dl_iter is not None:
        train_metrics["vlm_loss"] = AverageMeter("vlm_loss", ":.3f")
        train_metrics["vlm_bsz"] = AverageMeter("vlm_bsz", ":.1f")

    train_tracker = MetricsTracker(
        config.system.batch_size,
        num_frames,
        num_episodes,
        train_metrics,
        initial_step=step,
    )

    epoch = sampler.epoch if sampler is not None else 0

    def _to_device(v, dev):
        if isinstance(v, torch.Tensor):
            return v.to(dev, non_blocking=True)
        # BatchEncoding (from HF processor) is a UserDict, not a dict
        if isinstance(v, Mapping):
            return {k: _to_device(val, dev) for k, val in v.items()}
        return v

    def _next_vla_batch():
        """Fetch the next VLA batch, skipping batches that qwen_collate_fn
        has flagged as fully corrupted (all-None). Returns the new dl_iter
        too, since StopIteration bumps the epoch and rebuilds it.
        """
        nonlocal dl_iter, epoch
        max_skip_in_a_row = 64
        skipped = 0
        while True:
            try:
                return next(dl_iter)
            except StopIteration:
                epoch += 1
                if sampler is not None:
                    sampler.set_epoch(epoch)
                if isinstance(dataset, LeRobotMixtureDataset):
                    dataset.set_epoch(epoch)
                dl_iter = iter(dataloader)
                continue
            except SkipBatch as e:
                skipped += 1
                logger.warning(f"[train] skipping bad batch ({skipped}): {e}")
                if skipped >= max_skip_in_a_row:
                    raise RuntimeError(
                        f"giving up after {skipped} consecutive bad batches"
                    ) from e
                continue

    # --- Torch profiler ---
    # Produces per-rank Chrome trace JSON at <output_dir>/profiler_traces/rank_<N>/
    profiler_enabled = False #True
    profiler = None
    profiler_stop_step = 12  # wait(5) + warmup(2) + active(5)

    if profiler_enabled:
        profiler_output_dir = (
            Path(config.system.checkpoint.output_directory) / "profiler_traces" / f"rank_{rank}"
        )
        profiler_output_dir.mkdir(parents=True, exist_ok=True)

        def _chrome_trace_handler(prof):
            trace_file = profiler_output_dir / f"trace_rank{rank}_step{step}.json"
            prof.export_chrome_trace(str(trace_file))
            logger.info(f"[Profiler] Rank {rank}: trace saved to {trace_file}")

        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=5,
                warmup=2,
                active=5,
                repeat=1,
            ),
            on_trace_ready=_chrome_trace_handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        profiler.start()
        logger.info(f"[Profiler] Rank {rank}: enabled. Traces -> {profiler_output_dir}")

    # Freeze all objects created during setup (model, optimizer, buffers) so GC
    # never scans them. Without this, GC triggers random 20s+ stalls when Python
    # destructor → cudaFree coincides with NCCL collectives. Short-lived training
    # objects are still collected normally via refcount + gen-0 GC.
    gc.collect()
    gc.freeze()

    # Log any automatic GC that still triggers during training. Since we only
    # froze setup objects, gen-0 collections can still fire on new allocations.
    # This lets us detect if GC is causing stalls we can't otherwise see.
    # def _gc_callback(phase, info):
    #     if phase == "start":
    #         _gc_callback._start_time = time.perf_counter()
    #     elif phase == "stop":
    #         elapsed = time.perf_counter() - _gc_callback._start_time
    #         logger.info(f"[GC] auto collection gen={info['generation']} "
    #                     f"collected={info.get('collected', '?')} elapsed={elapsed:.4f}s")
    # _gc_callback._start_time = 0
    # gc.callbacks.append(_gc_callback)

    vla_gradient_accumulation_steps = max(1, int(getattr(config.system, "vla_gradient_accumulation_steps", 1)))

    for _ in range(step, config.system.train_steps):
        # --- Dataloader phase ---
        with torch.profiler.record_function("dataloader_vla"):
            data_start = time.perf_counter()
            vla_micro_batches = []
            for _micro_idx in range(vla_gradient_accumulation_steps):
                micro_batch = _next_vla_batch()
                if isinstance(micro_batch, dict):
                    micro_batch = {k: _to_device(v, device) for k, v in micro_batch.items()}
                vla_micro_batches.append(micro_batch)
            batch = vla_micro_batches[0] if vla_gradient_accumulation_steps == 1 else vla_micro_batches

        with torch.profiler.record_function("dataloader_vlm"):
            if vlm_dl_iter is not None:
                try:
                    vlm_batch = next(vlm_dl_iter)
                except StopIteration:
                    if vlm_sampler is not None:
                        vlm_sampler.set_epoch(epoch)
                    vlm_dl_iter = iter(vlm_dl)
                    vlm_batch = next(vlm_dl_iter)
            else:
                vlm_batch = None
            if vlm_batch is not None:
                vlm_batch = {
                    k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in vlm_batch.items()
                }

        with torch.profiler.record_function("preprocessor"):
            if preprocessor is not None:
                if isinstance(batch, list):
                    batch = [preprocessor(micro_batch) for micro_batch in batch]
                else:
                    batch = preprocessor(batch)
            train_tracker.dataloading_s = time.perf_counter() - data_start

        if step % config.system.log_freq == 0:

            def _log_vla_batch(b):
                if b is None:
                    return
                parts = []
                for k, v in b.items():
                    if isinstance(v, torch.Tensor):
                        parts.append(f"{k}: {list(v.shape)}")
                    elif isinstance(v, Mapping):
                        # Log nested Mapping (e.g. qwen_inputs BatchEncoding)
                        nested = []
                        for nk, nv in v.items():
                            if isinstance(nv, torch.Tensor):
                                nested.append(f"{nk}: {list(nv.shape)}")
                        if nested:
                            parts.append(f"{k}: {{{', '.join(nested)}}}")
                if parts:
                    logger.info(f"[VLA_batch] rank={rank} " + ", ".join(parts))
                # Per-sample dataset info
                ds_names = b.get("_dataset_name", None)
                ep_idx = b.get("episode_index", None)
                sample_idx = b.get("index", None)
                if ds_names is not None:
                    ep_list = ep_idx.tolist() if isinstance(ep_idx, torch.Tensor) else ep_idx
                    idx_list = sample_idx.tolist() if isinstance(sample_idx, torch.Tensor) else sample_idx
                    for i, name in enumerate(ds_names):
                        logger.info(
                            f"[VLA_sample {i}] dataset={name} episode={ep_list[i]} index={idx_list[i]}"
                        )

            def _log_vlm_batch(b):
                if b is None:
                    return
                parts = []
                for k, v in b.items():
                    if isinstance(v, torch.Tensor):
                        parts.append(f"{k}: {list(v.shape)}")
                    elif isinstance(v, Mapping):
                        # Log nested Mapping (e.g. qwen_inputs BatchEncoding)
                        nested = []
                        for nk, nv in v.items():
                            if isinstance(nv, torch.Tensor):
                                nested.append(f"{nk}: {list(nv.shape)}")
                        if nested:
                            parts.append(f"{k}: {{{', '.join(nested)}}}")
                if parts:
                    logger.info(f"[VLM_batch] rank={rank} " + ", ".join(parts))
                # Per-sample dataset info
                vlm_data_paths = b.get("_vlm_data_path", None)
                vlm_files = b.get("_vlm_file", None)
                if vlm_data_paths is not None and vlm_files is not None:
                    input_ids = b.get("input_ids", None)
                    for i, (data_path, file) in enumerate(zip(vlm_data_paths, vlm_files)):
                        seqlen = input_ids[i].shape[0] if input_ids is not None else "?"
                        logger.info(
                            f"[VLM_sample {i}] data_path={data_path} file={file} seqlen={seqlen}"
                        )

            # _log_vla_batch( batch[0] if isinstance(batch, list) else batch)
            # _log_vlm_batch( vlm_batch[0] if isinstance(vlm_batch, list) else vlm_batch)

        with torch.profiler.record_function("update_policy"):
            update_start = time.perf_counter()
            train_tracker = update_policy(
                train_tracker,
                policy,
                batch,
                optimizer,
                use_amp=config.system.use_amp,
                grad_clip_norm=config.system.grad_clip_norm,
                lr_scheduler=lr_scheduler,
                vlm_batch=vlm_batch,
                vlm_loss_scale=getattr(config.system, "vlm_loss_scale", 0.1),
            )
            train_tracker.update_s = time.perf_counter() - update_start

        step += 1
        train_tracker.step()

        # Advance torch profiler schedule
        if profiler is not None:
            profiler.step()
            if step >= profiler_stop_step:
                profiler.stop()
                logger.info(f"[Profiler] Rank {rank}: stopped. Traces saved.")
                profiler = None

        if step % config.system.log_freq == 0:
            avg_step_s = train_tracker.metrics["update_s"].avg + train_tracker.metrics["dataloading_s"].avg
            vla_tput = train_tracker.metrics["vla_bsz"].avg / avg_step_s
            tput_msg = f" vla_tput:{vla_tput:.1f}samples/s/gpu"
            if "vlm_bsz" in train_tracker.metrics and train_tracker.metrics["vlm_bsz"].count > 0:
                vlm_tput = train_tracker.metrics["vlm_bsz"].avg / avg_step_s
                tput_msg += f" vlm_tput:{vlm_tput:.1f}samples/s/gpu"
            logger.info(f"step: {step} {format_train_tracker_step(train_tracker)}{tput_msg}")
            train_tracker.reset_averages()

        if (
            config.system.checkpoint.save_checkpoint
            and step % config.system.checkpoint.save_freq == 0
        ):
            dist.barrier(device_ids=[local_rank])

            # get_model_state_dict and get_optimizer_state_dict are collectives — all ranks must call
            options = StateDictOptions(full_state_dict=True, cpu_offload=True)
            state_dict = get_model_state_dict(policy, options=options)
            optimizer_state_dict = get_optimizer_state_dict(policy, optimizer, options=options)

            if is_main_process:
                logger.info(f"Saving checkpoint at step {step}")
                output_dir = Path(config.system.checkpoint.output_directory)
                checkpoint_dir = get_step_checkpoint_dir(
                    output_dir, config.system.train_steps, step
                )
                dl_state = {"vla": dataloader.state_dict()}
                if vlm_dl is not None:
                    dl_state["vlm"] = vlm_dl.state_dict()
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    config=config,
                    policy=policy,
                    optimizer_state_dict=optimizer_state_dict,
                    lr_scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    state_dict=state_dict,
                    dataloader_state=dl_state,
                )
                update_last_checkpoint(checkpoint_dir)

            dist.barrier(device_ids=[local_rank])

    if is_main_process:
        logger.info("Training completed")

    dist.barrier(device_ids=[local_rank])
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Qwen35Gr00t model. This script is typically called by the flagscale runner, not directly."
    )
    parser.add_argument(
        "--config-file", type=str, required=True, help="Path to the configuration YAML file"
    )
    args = parser.parse_args()

    config_file_path = args.config_file

    # Load config from YAML file (Hydra-generated config.yaml contains both train and experiment)
    config = OmegaConf.load(config_file_path)

    logger.info(f"full config: {config}")

    # Extract train config and convert to Pydantic TrainConfig (preserves raw configs)
    train_config = TrainConfig.from_hydra_config(config)

    # Extract experiment config (seed, exp_dir, etc.)
    experiment_config = OmegaConf.to_container(config.experiment, resolve=True)
    seed = experiment_config.get("seed", 42)

    logger.info("=" * 100)
    logger.info(f"Experiment: {experiment_config}")
    logger.info(f"Train config: {train_config}")

    # torch.cuda.memory._set_allocator_settings("expandable_segments:True")

    main(train_config, seed)
