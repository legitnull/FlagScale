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


def qwen_collate_fn(
    batch_list: list[dict],
    *,
    processor,
    prompt_template: str,
    image_feature_keys: list[str],
) -> dict:
    """Collate samples into a batch with tokenized Qwen processor inputs.

    PIL images can't go through default_collate, so we pull them out first,
    collate the rest (action, state, etc.), then build Qwen inputs from the
    PIL images + task instructions.

    Adds ``qwen_inputs`` and (if future frames exist) ``qwen_future_inputs``
    to the batch dict.
    """
    future_keys = [f"{k}_future" for k in image_feature_keys]
    all_image_keys = set(image_feature_keys) | set(future_keys)

    # Separate PIL images from tensor-compatible fields
    image_data: dict[str, list[Image.Image]] = {}
    for key in all_image_keys:
        if key in batch_list[0]:
            image_data[key] = [sample[key] for sample in batch_list]

    non_image_batch = [
        {k: v for k, v in sample.items() if k not in all_image_keys} for sample in batch_list
    ]
    batch = torch.utils.data.default_collate(non_image_batch)
    batch.update(image_data)

    instructions = batch["task"]
    if isinstance(instructions, torch.Tensor):
        instructions = instructions.detach().cpu().tolist()
    if isinstance(instructions, str):
        instructions = [instructions]

    def _build_inputs(feature_keys: list[str]):
        """Build Qwen processor inputs from PIL images across camera keys.

        Groups images per sample: [[cam1, cam2, ...], [cam1, cam2, ...], ...]
        then formats as Qwen chat messages for tokenization.
        """
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

        messages = []
        for imgs, instruction in zip(batch_images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]
            prompt = (
                prompt_template.replace("{instruction}", instruction)
                if prompt_template
                else instruction
            )
            content.append({"type": "text", "text": prompt})
            messages.append([{"role": "user", "content": content}])

        # NOTE: apply_chat_template(tokenize=True) is simpler but uses the processor's
        # min/max_pixels config for resizing, which differs from process_vision_info's
        # qwen_vl_utils defaults (e.g. 256x256 → 252x252 vs different size). Sticking
        # with the 3-step API for now.
        texts = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        result = processor(
            text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
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

    batch["qwen_inputs"] = _build_inputs(image_feature_keys)

    if future_keys and future_keys[0] in batch:
        batch["qwen_future_inputs"] = _build_inputs(future_keys)

    # Remove raw PIL images — they've been consumed by the processor above.
    # Leaving them would break downstream steps (e.g. normalize_processor)
    # that expect tensors.
    for key in all_image_keys:
        batch.pop(key, None)

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
    video_backend = _resolve_video_backend()

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


class CudaTimer:
    """GPU-synchronized timer using CUDA events for accurate measurement."""

    def __init__(self, benchmark: bool = True):
        self._benchmark = benchmark
        if self._benchmark:
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)

    def start(self):
        if self._benchmark:
            self._start.record()

    def stop(self) -> float:
        """Stop timer, synchronize, and return elapsed time in seconds."""
        if not self._benchmark:
            return 0.0
        self._end.record()
        torch.cuda.synchronize()
        return self._start.elapsed_time(self._end) / 1000.0  # ms -> s


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
    timer = CudaTimer()
    timer.start()

    # Per-phase timers for diagnosing spikes
    t_fwd_vla = CudaTimer()
    t_bwd_vla = CudaTimer()
    t_fwd_vlm = CudaTimer()
    t_bwd_vlm = CudaTimer()
    t_opt = CudaTimer()

    with torch.profiler.record_function("optimizer_zero_grad"):
        optimizer.zero_grad()

    autocast_context = (
        torch.amp.autocast(get_platform().amp_device_type(), dtype=torch.bfloat16)
        if use_amp
        else nullcontext()
    )

    t_fwd_vla.start()
    with torch.profiler.record_function("forward_vla"), autocast_context:
        output = policy(batch)
        vla_loss = output["loss"]
    fwd_vla_s = t_fwd_vla.stop()

    t_bwd_vla.start()
    with torch.profiler.record_function("backward_vla"):
        # WARNING: We need to check if this is ok
        if vlm_batch is not None:
            policy.set_is_last_backward(False)
        vla_loss.backward()
    bwd_vla_s = t_bwd_vla.stop()

    fwd_vlm_s = 0.0
    bwd_vlm_s = 0.0
    if vlm_batch is not None:
        t_fwd_vlm.start()
        with torch.profiler.record_function("forward_vlm"), autocast_context:
            policy.set_is_last_backward(True)
            vlm_output = policy(vlm_batch, mode="vlm")
            vlm_loss_scaled = vlm_loss_scale * vlm_output["vlm_loss"]
        fwd_vlm_s = t_fwd_vlm.stop()

        t_bwd_vlm.start()
        with torch.profiler.record_function("backward_vlm"):
            vlm_loss_scaled.backward()
        bwd_vlm_s = t_bwd_vlm.stop()
        output["vlm_loss"] = vlm_output["vlm_loss"]

    t_opt.start()
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
    opt_s = t_opt.stop()

    train_metrics.grad_norm = (
        grad_norm.full_tensor().item() if hasattr(grad_norm, "full_tensor") else grad_norm.item()
    )
    train_metrics.lr = next(
        (g["lr"] for g in optimizer.param_groups if g.get("name") == "vlm"),
        optimizer.param_groups[0]["lr"],
    )
    train_metrics.update_s = timer.stop()
    train_metrics.fwd_vla_s = fwd_vla_s
    train_metrics.bwd_vla_s = bwd_vla_s
    train_metrics.fwd_vlm_s = fwd_vlm_s
    train_metrics.bwd_vlm_s = bwd_vlm_s
    train_metrics.opt_s = opt_s
    if "raw_action_loss" in output and "action_loss" in train_metrics.metrics:
        train_metrics.action_loss = output["raw_action_loss"].item()
    if "nfp_mse_loss_0" in output and "nfp_mse_loss" in train_metrics.metrics:
        train_metrics.nfp_mse_loss = output["nfp_mse_loss_0"].item()
    if "nfp_cosine_loss_0" in output and "nfp_cosine_loss" in train_metrics.metrics:
        train_metrics.nfp_cosine_loss = output["nfp_cosine_loss_0"].item()
    if "vlm_loss" in output and "vlm_loss" in train_metrics.metrics:
        train_metrics.vlm_loss = output["vlm_loss"].item() * vlm_loss_scale

    return train_metrics


def main(config: TrainConfig, seed: int):
    set_seed(seed)

    policy_config = PreTrainedConfig.from_train_config(config)

    dist.init_process_group(backend=get_platform().dist_backend())
    local_rank = int(os.environ["LOCAL_RANK"])
    get_platform().set_device(local_rank)
    device = get_platform().device(local_rank)
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
        dist.barrier()

        policy = make_policy(policy_config, dataset.meta)
        dist.barrier()

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

    dist.barrier()

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
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
        "fwd_vla_s": AverageMeter("fwd_vla", ":.3f"),
        "bwd_vla_s": AverageMeter("bwd_vla", ":.3f"),
        "fwd_vlm_s": AverageMeter("fwd_vlm", ":.3f"),
        "bwd_vlm_s": AverageMeter("bwd_vlm", ":.3f"),
        "opt_s": AverageMeter("opt_s", ":.3f"),
    }
    if vlm_dl_iter is not None:
        train_metrics["vlm_loss"] = AverageMeter("vlm_loss", ":.3f")

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

    # --- Torch profiler ---
    # Produces per-rank Chrome trace JSON at <output_dir>/profiler_traces/rank_<N>/
    profiler_enabled = False
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

    for _ in range(step, config.system.train_steps):
        # --- Dataloader phase ---
        with torch.profiler.record_function("dataloader_vla"):
            data_timer = CudaTimer()
            data_timer.start()
            try:
                batch = next(dl_iter)
            except StopIteration:
                epoch += 1
                if sampler is not None:
                    sampler.set_epoch(epoch)
                if isinstance(dataset, LeRobotMixtureDataset):
                    dataset.set_epoch(epoch)
                dl_iter = iter(dataloader)
                batch = next(dl_iter)
            if isinstance(batch, dict):
                batch = {k: _to_device(v, device) for k, v in batch.items()}

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

        # Pad VLM batch to model_max_length unconditionally for memory profiling.
        # Remove this block after profiling.
        # if vlm_batch is not None:
        #     max_seq = 6000
        #     cur_seq = vlm_batch["input_ids"].shape[-1]
        #     if cur_seq < max_seq:
        #         pad_len = max_seq - cur_seq
        #         bs = vlm_batch["input_ids"].shape[0]
        #         dev = vlm_batch["input_ids"].device
        #         for key in ("input_ids", "labels", "mm_token_type_ids"):
        #             if key in vlm_batch:
        #                 pad = torch.zeros(bs, pad_len, dtype=vlm_batch[key].dtype, device=dev)
        #                 vlm_batch[key] = torch.cat([vlm_batch[key], pad], dim=1)
        #         if "attention_mask" in vlm_batch:
        #             pad = torch.zeros(bs, pad_len, dtype=vlm_batch["attention_mask"].dtype, device=dev)
        #             vlm_batch["attention_mask"] = torch.cat([vlm_batch["attention_mask"], pad], dim=1)

        with torch.profiler.record_function("preprocessor"):
            if preprocessor is not None:
                batch = preprocessor(batch)
            train_tracker.dataloading_s = data_timer.stop()

        # Debug: log tensor shapes and GPU memory for memory profiling
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

            _log_vla_batch(batch)
            _log_vlm_batch(vlm_batch)

        with torch.profiler.record_function("update_policy"):
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
            logger.info(f"step: {step} {format_train_tracker_step(train_tracker)}")
            train_tracker.reset_averages()

        if (
            config.system.checkpoint.save_checkpoint
            and step % config.system.checkpoint.save_freq == 0
        ):
            dist.barrier()

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

            dist.barrier()

    if is_main_process:
        logger.info("Training completed")

    dist.barrier()
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
