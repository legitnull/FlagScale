# Mainly adopted from
# https://github.com/huggingface/lerobot/blob/2b304eeb841ae6c371e3dd341bbbb9dd254b07cb/src/lerobot/scripts/lerobot_train.py

import argparse
import os
import random
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf, DictConfig
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.checkpoint.state_dict import get_model_state_dict, get_optimizer_state_dict, StateDictOptions
from torch.optim import Optimizer

from flagscale.logger import logger
from flagscale.train.train_config import TrainConfig, DataConfig
from flagscale.train.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from flagscale.train.datasets.lerobot_mixture_dataset import LeRobotMixtureDataset
from flagscale.train.datasets.utils import dataset_to_policy_features
from flagscale.models.configs.types import FeatureType
from flagscale.train.processor import PolicyProcessorPipeline
from flagscale.models.utils.constants import (
    ACTION,
    OBS_PREFIX,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from flagscale.train.utils.logging_utils import (
    AverageMeter,
    MetricsTracker,
    format_big_number,
)
from flagscale.train.utils.train_utils import (
    StatefulDistributedSampler,
    get_step_checkpoint_dir,
    load_training_state_fsdp2,
    save_checkpoint,
    update_last_checkpoint,
)
from flagscale.train.utils.optim_setup import setup_optimizer_and_scheduler
from flagscale.models.vla import TrainablePolicy
from flagscale.models.vla.pretrained_config import PreTrainedConfig
from flagscale.models.vla.vlm.qwenvl_backbone import _to_pil
from flagscale.platforms import get_platform
from qwen_vl_utils import process_vision_info
from torchdata.stateful_dataloader import StatefulDataLoader


class LeRobotDatasetWithFutureFrames(LeRobotDataset):
    """LeRobotDataset subclass that splits future frames from visual tensors.

    When delta_timestamps includes a future offset for camera keys, LeRobotDataset
    returns tensors with an extra time dimension (e.g. [T, C, H, W] for video,
    [T, C, H, W] for image via _query_hf_dataset). This subclass splits them into
    current frame (index 0) under the original key and future frames under
    ``{cam_key}_future`` keys, matching starVLA's naming convention.
    """

    def load_hf_dataset(self):
        """Load HF dataset with decode=False for image features, so we can
        resolve relative paths against self.root before decoding."""
        from flagscale.train.datasets.utils import get_hf_features_from_features, load_nested_dataset
        import datasets as hf_datasets

        features = get_hf_features_from_features(self.features)
        # Disable auto-decoding for image columns
        for key in features:
            if isinstance(features[key], hf_datasets.Image):
                features[key] = hf_datasets.Image(decode=False)

        hf_dataset = load_nested_dataset(
            self.root / "data", features=features, episodes=self.episodes
        )

        root = self.root
        image_keys = [k for k, ft in self.features.items() if ft["dtype"] == "image"]

        def _resolve_and_transform(items_dict):
            from io import BytesIO
            import PIL.Image as PILImage
            from torchvision import transforms
            to_tensor = transforms.ToTensor()

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
                            decoded.append(to_tensor(img))
                        else:
                            decoded.append(to_tensor(v) if not isinstance(v, torch.Tensor) else v)
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

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        for cam_key in self.meta.camera_keys:
            if cam_key in item and item[cam_key].dim() >= 4:
                item[f"{cam_key}_future"] = item[cam_key][1]
                item[cam_key] = item[cam_key][0]
        return item


def qwen_collate_fn(batch_list, *, processor, prompt_template, image_feature_keys):
    batch = torch.utils.data.default_collate(batch_list)

    instructions = batch["task"]
    if isinstance(instructions, torch.Tensor):
        instructions = instructions.detach().cpu().tolist()
    if isinstance(instructions, str):
        instructions = [instructions]

    def _build_inputs(feature_keys):
        batch_images = None
        for key in feature_keys:
            imgs = batch[key]
            if isinstance(imgs, torch.Tensor) and imgs.ndim == 3:
                imgs = [imgs]
            key_images = [_to_pil(img) for img in imgs]
            if batch_images is None:
                batch_images = [[img] for img in key_images]
            else:
                for sample_imgs, img in zip(batch_images, key_images):
                    sample_imgs.append(img)

        messages = []
        for imgs, instruction in zip(batch_images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]
            prompt = prompt_template.replace("{instruction}", instruction) if prompt_template else instruction
            content.append({"type": "text", "text": prompt})
            messages.append([{"role": "user", "content": content}])

        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
        image_inputs, video_inputs = process_vision_info(messages)
        return processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

    batch["qwen_inputs"] = _build_inputs(image_feature_keys)

    future_keys = [f"{k}_future" for k in image_feature_keys]
    if future_keys and future_keys[0] in batch:
        batch["qwen_future_inputs"] = _build_inputs(future_keys)

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

    # `reduce_dtype=torch.float32` would make evaluation on libero_goal drop to 94.8% (from 97.0%)
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
    )
    fsdp_config = {"mesh": device_mesh, "mp_policy": mp_policy}

    # reshard_after_forward=False keeps params unsharded during forward+backward
    reshard = False

    for unit in policy.fsdp_units():
        fully_shard(unit, **fsdp_config, reshard_after_forward=reshard)

    fully_shard(policy, **fsdp_config)


def _resolve_video_backend():
    # torchcodec depends on NVIDIA NVDEC which is not available on all platforms (e.g. MUSA);
    # fall back to pyav for non-CUDA platforms or when torchcodec is broken.
    video_backend = "pyav"
    if get_platform().name() == "cuda":
        try:
            import torchcodec  # noqa: F401
            torch.ops.torchcodec_ns  # verify the C++ ops are loadable
            video_backend = "torchcodec"
        except Exception:
            logger.info("torchcodec unavailable, falling back to pyav")
            video_backend = "pyav"
    return video_backend


def _make_image_transforms():
    def _resize_to_uint8_hwc(frame: torch.Tensor) -> torch.Tensor:
        """float32 CHW [0,1] from torchcodec → uint8 HWC 224x224 via PIL resize."""
        if frame.dim() == 4:
            # delta_timestamps adds a leading T dim; squeeze single-frame case
            if frame.shape[0] == 1:
                frame = frame.squeeze(0)
            else:
                return torch.stack([_resize_to_uint8_hwc(f) for f in frame])

        frame_uint8 = (frame.permute(1, 2, 0) * 255).round().clamp(0, 255).to(torch.uint8)
        # PIL default is BICUBIC, matching starVLA's Image.fromarray(image).resize((224, 224))
        pil = Image.fromarray(frame_uint8.cpu().numpy()).resize((224, 224))
        return torch.from_numpy(np.array(pil))

    return _resize_to_uint8_hwc


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
    image_transforms = _make_image_transforms()

    if data_mix is not None:
        data_root_dir = getattr(config.data, "data_root_dir", None)
        if data_root_dir is None:
            raise ValueError("data_root_dir must be set when using data_mix")

        from examples.qwen3_5_gr00t.mixtures import DATASET_MIXTURES
        if data_mix not in DATASET_MIXTURES:
            raise ValueError(f"Unknown data_mix: {data_mix}. Available: {list(DATASET_MIXTURES.keys())}")

        mixture_spec = DATASET_MIXTURES[data_mix]
        data_mixture = []
        for dataset_name, weight in mixture_spec:
            data_path = f"{data_root_dir}/{dataset_name}"
            ds = _make_single_dataset(
                data_path, policy_config, future_offset, config.data.tolerance_s,
                video_backend=video_backend, image_transforms=image_transforms,
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
            config.data.data_path, policy_config, future_offset, config.data.tolerance_s,
            video_backend=video_backend, image_transforms=image_transforms,
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
    start_time = time.perf_counter()

    skip_vla = False  # Set to False to enable VLA forward+backward
    optimizer.zero_grad()

    autocast_context = (
        torch.amp.autocast(get_platform().amp_device_type(), dtype=torch.bfloat16) if use_amp else nullcontext()
    )

    with autocast_context:
        output = policy(batch, vlm_batch=None, skip_vla=skip_vla)
        vla_loss = output["loss"]

    if not skip_vla:
        if vlm_batch is not None:
            policy.set_is_last_backward(False)
        vla_loss.backward()

    if vlm_batch is not None:
        policy.set_is_last_backward(True)
        vlm_output = policy(batch, vlm_batch=vlm_batch, skip_vla=True)
        vlm_loss_scaled = vlm_loss_scale * vlm_output["vlm_loss"]
        vlm_loss_scaled.backward()
        output["vlm_loss"] = vlm_output["vlm_loss"]

    loss = vla_loss.detach() + (vlm_loss_scale * output["vlm_loss"].detach() if "vlm_loss" in output else 0.0)

    # Clip gradients (torch.nn.utils.clip_grad_norm_ works with DTensors in PyTorch ≥2.6)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(), grad_clip_norm if grad_clip_norm > 0 else float("inf")
    )

    with lock if lock is not None else nullcontext():
        optimizer.step()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Update internal buffers if policy has update method
    if has_method(policy, "update"):
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.full_tensor().item() if hasattr(grad_norm, 'full_tensor') else grad_norm.item()
    train_metrics.lr = next((g["lr"] for g in optimizer.param_groups if g.get("name") == "vlm"), optimizer.param_groups[0]["lr"])
    train_metrics.update_s = time.perf_counter() - start_time
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
        from megatron.energon import get_train_dataset, get_loader, WorkerConfig
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

        dataset_stats = dataset.merged_stats if isinstance(dataset, LeRobotMixtureDataset) else dataset.meta.stats
        preprocessor, postprocessor = make_pre_post_processors(
            policy, config.data, dataset_stats=dataset_stats, device=device.type,
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

            from flagscale.train.datasets.vlm_datasets_qwen35 import make_vlm_dataloader
            from types import SimpleNamespace

            vlm_cfg = SimpleNamespace(
                datasets=SimpleNamespace(vlm_data=config.data.vlm_data),
                framework=SimpleNamespace(
                    qwenvl=SimpleNamespace(base_vlm=config.model.vlm.base_vlm)
                ),
            )
            vlm_data_module = make_vlm_dataloader(vlm_cfg, rank=rank, world_size=world_size, seed=seed)
            vlm_dl = vlm_data_module["train_dataloader"]
            vlm_sampler = vlm_data_module["sampler"]
            vlm_dl_iter = iter(vlm_dl)

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
            Path(resume_from), policy, optimizer, lr_scheduler,
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
        "loss": AverageMeter("loss", ":.3f"),
        "action_loss": AverageMeter("act_loss", ":.3f"),
        "nfp_mse_loss": AverageMeter("nfp_mse", ":.4f"),
        "nfp_cosine_loss": AverageMeter("nfp_cos", ":.4f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    if vlm_dl_iter is not None:
        train_metrics["vlm_loss"] = AverageMeter("vlm_loss", ":.3f")

    effective_batch_size = config.system.batch_size * world_size

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
        if isinstance(v, dict):
            return {k: _to_device(val, dev) for k, val in v.items()}
        return v

    for _ in range(step, config.system.train_steps):
        start_time = time.perf_counter()
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

        if preprocessor is not None:
            batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

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

    main(train_config, seed)
