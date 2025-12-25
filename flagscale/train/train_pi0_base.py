import argparse
from typing import Any, Iterator, TypedDict
import os
import pathlib
import platform
import random
from dataclasses import dataclass
from typing_extensions import Unpack
import math
import time
from contextlib import nullcontext


import etils.epath as epath
import numpy as np
import torch
import torch.distributed as dist
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

import wandb

# from megatron.energon import WorkerConfig, get_loader, get_train_dataset
# from tools.datasets.vla.data.dataset_helpers import TaskEncoder

from flagscale.runner.utils import logger

# TODO(yupu): prune
from lerobot.datasets.transforms import ImageTransforms
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from lerobot.datasets.utils import dataset_to_policy_features

from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from flagscale.models.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from flagscale.models.pi0_base.configuration_pi0 import PI0Config
from flagscale.models.utils.constants import ACTION, OBS_PREFIX, REWARD
from flagscale.models.pi0_base.modeling_pi0 import PI0Policy
from flagscale.models.configs.types import FeatureType
from flagscale.train.utils.logging_utils import AverageMeter, MetricsTracker

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}

from pprint import pformat
import dataclasses, json

def dump_runtime(config, pi0_config, processor_kwargs, postprocessor_kwargs, optimizer=None, lr_scheduler=None, device=None):
    print("=== train args (Namespace) ===")
    print(config)

    print("\n=== PI0Config ===")
    if dataclasses.is_dataclass(pi0_config):
        print(json.dumps(dataclasses.asdict(pi0_config), indent=2, default=str))
    else:
        print(pi0_config)

    print("\n=== Processor kwargs ===")
    print(pformat(processor_kwargs))
    print("\n=== Postprocessor kwargs ===")
    print(pformat(postprocessor_kwargs))

    if optimizer:
        print("\n=== Optimizer ===")
        print(type(optimizer).__name__)
        for i, g in enumerate(optimizer.param_groups):
            print(f"group {i}: lr={g['lr']}, betas={g.get('betas')}, eps={g.get('eps')}, wd={g.get('weight_decay')}")

    if lr_scheduler:
        print("\n=== LR Scheduler state ===")
        print(lr_scheduler.state_dict())

    print("\n=== Device / AMP ===")
    print(f"device: {device}")
    print(f"cudnn.benchmark: {torch.backends.cudnn.benchmark}, deterministic: {torch.backends.cudnn.deterministic}")


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True

def init_ddp(config):
    # TODO(yupu): to a function
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # To match lerobot_train.py
    torch.backends.cuda.matmul.allow_tf32 = True

    return local_rank


def init_wandb(config, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = pathlib.Path(config.checkpoint_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name, config=vars(config), project=config.project_name
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def make_dataset(cfg, pi0_config: PI0Config):
    # TODO(yupu): to config
    cfg.enable_image_transform = False
    cfg.tolerance_s = 0.0001
    cfg.video_backend = "pyav"
    cfg.use_imagenet_stats = True

    image_transforms = (
        ImageTransforms(cfg.image_transforms) if cfg.enable_image_transform else None
    )
    # Leave the revision to None
    # TODO(yupu): Remove repo_id and use local data without downloading from hub
    ds_meta = LeRobotDatasetMetadata(cfg.repo_id, root=cfg.data_path, revision=None)

    delta_timestamps = resolve_delta_timestamps(pi0_config, ds_meta)
    # # TODO(yupu): Remove repo_id
    dataset = LeRobotDataset(
        cfg.repo_id,
        root=cfg.data_path,
        episodes=None,
        delta_timestamps=delta_timestamps,
        image_transforms=image_transforms,
        revision=None,
        video_backend=cfg.video_backend,
        tolerance_s=cfg.tolerance_s,
    )

    if cfg.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(
                    stats, dtype=torch.float32
                )

    return dataset


def resolve_delta_timestamps(
    cfg: PI0Config, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg (PI0Config): The PI0Config to read delta_indices from.
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
        if key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith(OBS_PREFIX) and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [
                i / ds_meta.fps for i in cfg.observation_delta_indices
            ]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


# TODO(yupu): check if this is correct
def is_main_process():
    return dist.get_rank() == 0 and int(os.environ["LOCAL_RANK"]) == 0


# datasets/utils.py
def cycle(iterable: Any) -> Iterator[Any]:
    """Create a dataloader-safe cyclical iterator.

    This is an equivalent of `itertools.cycle` but is safe for use with
    PyTorch DataLoaders with multiple workers.
    See https://github.com/pytorch/pytorch/issues/23900 for details.

    Args:
        iterable: The iterable to cycle over.

    Yields:
        Items from the iterable, restarting from the beginning when exhausted.
    """
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


def make_policy(
    cfg: PI0Config,
    ds_meta: LeRobotDatasetMetadata | None = None,
    rename_map: dict[str, str] | None = None,
) -> PI0Policy:
    """
    Instantiate a policy model.

    This factory function handles the logic of creating a policy, which requires
    determining the input and output feature shapes. These shapes can be derived
    either from a `LeRobotDatasetMetadata` object or an `EnvConfig` object. The function
    can either initialize a new policy from scratch or load a pretrained one.

    Args:
        cfg: The configuration for the policy to be created. If `cfg.pretrained_path` is
             set, the policy will be loaded with weights from that path.
        ds_meta: Dataset metadata used to infer feature shapes and types. Also provides
                 statistics for normalization layers.
        rename_map: Optional mapping of dataset or environment feature keys to match
                 expected policy feature names (e.g., `"left"` → `"camera1"`).

    Returns:
        An instantiated and device-placed policy model.
    """

    policy_cls = PI0Policy

    kwargs = {}
    features = dataset_to_policy_features(ds_meta.features)

    cfg.output_features = {
        # Changed from ft.type is FeatureType.ACTION to ft.type == FeatureType.ACTION for different enum classes: flagscale.FeatureType vs lerobot.FeatureType
        key: ft for key, ft in features.items() if ft.type == FeatureType.ACTION
    }
    if not cfg.input_features:
        cfg.input_features = {
            key: ft for key, ft in features.items() if key not in cfg.output_features
        }
    kwargs["config"] = cfg

    # PI0 finetuning, so always load a pretrained policy.
    # Load a pretrained policy and override the config if needed (for example, if there are inference-time
    # hyperparameters that we want to vary).
    kwargs["pretrained_name_or_path"] = cfg.pretrained_path
    policy = policy_cls.from_pretrained(cfg.pretrained_path, config=cfg)

    policy.to(cfg.device)
    assert isinstance(policy, torch.nn.Module)

    # policy = torch.compile(policy, mode="reduce-overhead")

    # TODO(yupu): Risky
    # if not rename_map:
    #     validate_visual_features_consistency(cfg, features)
    # TODO: (jadechoghari) - add a check_state(cfg, features) and check_action(cfg, features)

    return policy


class ProcessorConfigKwargs(TypedDict, total=False):
    """
    A TypedDict defining the keyword arguments for processor configuration.

    This provides type hints for the optional arguments passed to `make_pre_post_processors`,
    improving code clarity and enabling static analysis.

    Attributes:
        preprocessor_config_filename: The filename for the preprocessor configuration.
        postprocessor_config_filename: The filename for the postprocessor configuration.
        preprocessor_overrides: A dictionary of overrides for the preprocessor configuration.
        postprocessor_overrides: A dictionary of overrides for the postprocessor configuration.
        dataset_stats: Dataset statistics for normalization.
    """

    preprocessor_config_filename: str | None
    postprocessor_config_filename: str | None
    preprocessor_overrides: dict[str, Any] | None
    postprocessor_overrides: dict[str, Any] | None
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None


def make_pre_post_processors(
    policy_cfg: PI0Config,
    pretrained_path: str | None = None,
    **kwargs: Unpack[ProcessorConfigKwargs],
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Create or load pre- and post-processor pipelines for a given policy.

    This function acts as a factory. It can either load existing processor pipelines
    from a pretrained path or create new ones from scratch based on the policy
    configuration. Each policy type has a dedicated factory function for its
    processors (e.g., `make_tdmpc_pre_post_processors`).

    Args:
        policy_cfg: The configuration of the policy for which to create processors.
        pretrained_path: An optional path to load pretrained processor pipelines from.
            If provided, pipelines are loaded from this path.
        **kwargs: Keyword arguments for processor configuration, as defined in
            `ProcessorConfigKwargs`.

    Returns:
        A tuple containing the input (pre-processor) and output (post-processor) pipelines.

    Raises:
        NotImplementedError: If a processor factory is not implemented for the given
            policy configuration type.
    """
    return (
        PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_path,
            config_filename=kwargs.get(
                "preprocessor_config_filename",
                f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
            ),
            overrides=kwargs.get("preprocessor_overrides", {}),
            to_transition=batch_to_transition,
            to_output=transition_to_batch,
        ),
        PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_path,
            config_filename=kwargs.get(
                "postprocessor_config_filename",
                f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
            ),
            overrides=kwargs.get("postprocessor_overrides", {}),
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


@dataclass
class CosineDecayWithWarmupSchedulerConfig:
    """Used by Physical Intelligence to train Pi0.

    Automatically scales warmup and decay steps if num_training_steps < num_decay_steps.
    This ensures the learning rate schedule completes properly even with shorter training runs.
    """

    num_warmup_steps: int
    num_decay_steps: int
    peak_lr: float
    decay_lr: float

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        # Auto-scale scheduler parameters if training steps are shorter than configured decay steps
        actual_warmup_steps = self.num_warmup_steps
        actual_decay_steps = self.num_decay_steps

        if num_training_steps < self.num_decay_steps:
            # Calculate scaling factor to fit the schedule into the available training steps
            scale_factor = num_training_steps / self.num_decay_steps
            actual_warmup_steps = int(self.num_warmup_steps * scale_factor)
            actual_decay_steps = num_training_steps

            logger.info(
                f"Auto-scaling LR scheduler: "
                f"num_training_steps ({num_training_steps}) < num_decay_steps ({self.num_decay_steps}). "
                f"Scaling warmup: {self.num_warmup_steps} → {actual_warmup_steps}, "
                f"decay: {self.num_decay_steps} → {actual_decay_steps} "
                f"(scale factor: {scale_factor:.3f})"
            )

        def lr_lambda(current_step):
            def linear_warmup_schedule(current_step):
                if current_step <= 0:
                    return 1 / (actual_warmup_steps + 1)
                frac = 1 - current_step / actual_warmup_steps
                return (1 / (actual_warmup_steps + 1) - 1) * frac + 1

            def cosine_decay_schedule(current_step):
                step = min(current_step, actual_decay_steps)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * step / actual_decay_steps))
                alpha = self.decay_lr / self.peak_lr
                decayed = (1 - alpha) * cosine_decay + alpha
                return decayed

            if current_step < actual_warmup_steps:
                return linear_warmup_schedule(current_step)

            return cosine_decay_schedule(current_step)

        return LambdaLR(optimizer, lr_lambda, -1)


def has_method(cls: object, method_name: str) -> bool:
    return hasattr(cls, method_name) and callable(getattr(cls, method_name))


def update_policy(
    train_metrics: MetricsTracker,
    policy: PI0Policy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        accelerator: The Accelerator instance for distributed training and mixed precision.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    policy.train()

    # Let accelerator handle mixed precision
    with accelerator.autocast():
        loss, output_dict = policy.forward(batch)
    logger.info(f"loss: {loss.item()}")
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    # Use accelerator's backward method
    accelerator.backward(loss)

    # Clip gradients if specified
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    # Optimizer step
    with lock if lock is not None else nullcontext():
        optimizer.step()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Update internal buffers if policy has update method
    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


"""
1. optimizer/scheduler setup without presets

TODO:
1. organize configs into different parts like lerobot does
others:
1. resume
"""


def main(config):
    import debugpy
    import os

    # Choose between Accelerator (like lerobot) or manual DDP
    use_accelerator = True  # config.use_accelerator
    accelerator = None

    set_seed(config.seed)

    if use_accelerator:
        # Use Accelerator like lerobot does
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False, kwargs_handlers=[ddp_kwargs]
        )
        device = accelerator.device
        rank = accelerator.process_index
        is_main_process = accelerator.is_main_process
    else:
        # Manual DDP setup
        local_rank = init_ddp(config)
        device = torch.device("cuda", local_rank)
        rank = dist.get_rank()
        is_main_process = rank == 0

    # if is_main_process:
    #     (
    #         debugpy.listen(("0.0.0.0", 9096)),
    #         debugpy.wait_for_client(),
    #         debugpy.breakpoint(),
    #     ) if not debugpy.is_client_connected() else None

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    if is_main_process:
        logger.info(f"Running on: {platform.node()}")
        logger.info(f"config: {config}")
        # if config.batch_size % torch.cuda.device_count() != 0:
        #     raise ValueError(
        #         f"Batch size {config.batch_size} must be divisible by the number of devices {torch.cuda.device_count()}."
        #     )
        logger.info(f"use_accelerator: {use_accelerator}")
        resuming = config.resume
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    pi0_config = PI0Config.from_pretrained(config.checkpoint_dir)
    # TODO(yupu): Ugly
    pi0_config.pretrained_path = config.checkpoint_dir
    pi0_config.device = device
    print(f"pi0_config: {pi0_config}")

    if (use_accelerator and is_main_process) or not use_accelerator:
        # Each process needs its own dataset for DDP training
        dataset = make_dataset(config, pi0_config)

    if use_accelerator and not is_main_process:
        dataset = make_dataset(config, pi0_config)

    if use_accelerator:
        accelerator.wait_for_everyone()

    policy = make_policy(
        cfg=pi0_config,
        ds_meta=dataset.meta,
        rename_map=None,
    )



    # policy.eval()
    # batch = torch.load(f"batch_after_preprocessor_0_{accelerator.process_index}.pt")
    # with torch.no_grad():
    #     loss, output_dict = policy.forward(batch)
    # logger.info(f"loss: {loss.item()}")
    # import sys
    # sys.exit()

    if use_accelerator:
        accelerator.wait_for_everyone()

    # Create processors - only provide dataset_stats if not resuming from saved processors
    processor_kwargs = {}
    postprocessor_kwargs = {}
    # Only provide dataset_stats when not resuming from saved processor state
    processor_kwargs["dataset_stats"] = dataset.meta.stats

    processor_kwargs["preprocessor_overrides"] = {
        "device_processor": {"device": device.type},
        "normalizer_processor": {
            "stats": dataset.meta.stats,
            "features": {
                **policy.config.input_features,
                **policy.config.output_features,
            },
            "norm_map": policy.config.normalization_mapping,
        },
    }

    # TODO(yupu): Hard-coded rename map
    rename_map = {
        "observation.images.cam_high": "observation.images.base_0_rgb",
        "observation.images.cam_left_wrist": "observation.images.left_wrist_0_rgb",
        "observation.images.cam_right_wrist": "observation.images.right_wrist_0_rgb",
    }

    processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
        "rename_map": rename_map
    }
    postprocessor_kwargs["postprocessor_overrides"] = {
        "unnormalizer_processor": {
            "stats": dataset.meta.stats,
            "features": policy.config.output_features,
            "norm_map": policy.config.normalization_mapping,
        },
    }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=pi0_config,
        pretrained_path=pi0_config.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    # TODO(yupu): to config
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=pi0_config.optimizer_lr,
        betas=pi0_config.optimizer_betas,
        eps=pi0_config.optimizer_eps,
        weight_decay=pi0_config.optimizer_weight_decay,
    )
    scheduler_config = CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=pi0_config.scheduler_warmup_steps,
        num_decay_steps=pi0_config.scheduler_decay_steps,
        peak_lr=pi0_config.optimizer_lr,
        decay_lr=pi0_config.scheduler_decay_lr,
    )
    lr_scheduler = scheduler_config.build(optimizer, config.train_steps)

    # Note for actual training later: You'll want shuffle=True and call sampler.set_epoch(epoch) at the start of each epoch to ensure different shuffling per epoch.
    # TODO(yupu): to config
    config.num_workers = 4
    shuffle = False  # Set to False for reproducible comparison

    # TODO(yupu): drop last?
    if not use_accelerator:
        # DistributedSampler ensures each rank gets different data
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=shuffle,
            drop_last=False,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=config.num_workers,
            batch_size=config.batch_size,
            shuffle=False,  # Must be False when using sampler
            sampler=sampler,
            pin_memory=True,  # Assume all data is on GPU
            drop_last=False,
            prefetch_factor=2 if config.num_workers > 0 else None,
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=config.num_workers,
            batch_size=config.batch_size,
            shuffle=False,  # Must be False when using sampler
            sampler=None,
            pin_memory=True,  # Assume all data is on GPU
            drop_last=False,
            prefetch_factor=2 if config.num_workers > 0 else None,
        )

    if use_accelerator:
        accelerator.wait_for_everyone()
        policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            policy, optimizer, dataloader, lr_scheduler
        )
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    # Use effective batch size for proper epoch calculation in distributed training
    if use_accelerator:
        effective_batch_size = config.batch_size * accelerator.num_processes
    else:
        effective_batch_size = config.batch_size * dist.get_world_size()

    step = 0  # number of policy updates (forward + backward + optim)

    train_tracker = MetricsTracker(
        effective_batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    # TODO(yupu): to config
    grad_clip_norm = 1.0

    if is_main_process:
        dump_runtime(
            config=config,
            pi0_config=pi0_config,
            processor_kwargs=processor_kwargs,
            postprocessor_kwargs=postprocessor_kwargs,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=device,
        )

    for _ in range(step, config.train_steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)

        if not use_accelerator:
            batch = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

        batch = preprocessor(batch)

        # if use_accelerator:
        #     torch.save(
        #         batch, f"batch_after_preprocessor_{step}_{accelerator.process_index}.pt"
        #     )
        # else:
        #     torch.save(batch, f"batch_after_preprocessor_{step}_{dist.get_rank()}.pt")

        train_tracker.dataloading_s = time.perf_counter() - start_time

        # TODO(yupu): Remove accelerator from here
        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
        )

        print(f"train_tracker at step {step}: {train_tracker}")

        import sys

        # if step == 5:
        #     sys.exit()


        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()

        # if use_accelerator:
        #     torch.save(
        #         batch, f"batch_after_preprocessor_{step}_{accelerator.process_index}.pt"
        #     )
        # else:
        #     torch.save(batch, f"batch_after_preprocessor_{step}_{dist.get_rank()}.pt")

    # if use_accelerator:
    #     torch.save(loss, f"loss_{accelerator.process_index}.pt")
    # else:
    #     torch.save(loss, f"loss_{dist.get_rank()}.pt")

    print("done for now")
    import sys

    sys.exit()

    # ds = get_train_dataset(
    #     config.data_path,
    #     batch_size=config.batch_size,
    #     shuffle_buffer_size=10000,
    #     max_samples_per_sequence=100,
    #     worker_config=WorkerConfig.default_worker_config(
    #         num_workers=1, data_parallel_group=None
    #     ),
    #     task_encoder=TaskEncoder(config),
    #     repeat=True,
    # )
    # loader = get_loader(ds)
    data_iter = None  # iter(loader)

    model_config = PI0PolicyConfig.from_pretrained(config.checkpoint_dir)
    model_config.n_action_steps = config.action_steps
    model_config.tokenizer_max_length = config.tokenizer_max_length
    policy = PI0Policy.from_pretrained(
        model_path=config.checkpoint_dir,
        tokenizer_path=config.tokenizer_path,
        stat_path=config.stat_path,
        config=model_config,
    )
    policy = policy.cuda()
    policy = DDP(
        policy, device_ids=[int(os.environ["LOCAL_RANK"])], find_unused_parameters=True
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    step = 0
    done = False
    while not done:
        # sampler.set_epoch(epoch)  # Uncomment when using DistributedSampler
        batch = next(data_iter)
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: {v.shape}")

        loss, _ = policy.forward(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step % config.log_freq == 0:
            logger.info(f"step: {step} loss: {loss.item():.3f}")
        step += 1
        if step >= config.train_steps:
            done = True
            break
    if dist.get_rank() == 0 and local_rank == 0:
        policy.module.save_pretrained(config.output_directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoint_path_not_set")
    parser.add_argument("--project-name", type=str, default="default_project")
    parser.add_argument("--exp-name", type=str, default="default_exp")
    parser.add_argument("--data-path", type=str, default="energon data_path not set")
    parser.add_argument("--tokenizer-path", type=str, default="tokenizer_path not set")
    parser.add_argument("--state-key", type=str, default="state_key not set")
    parser.add_argument("--action-key", type=str, default="action_key not set")
    parser.add_argument(
        "--action-token-key", type=str, default="action_token_key not set"
    )
    parser.add_argument("--stat-path", type=str, default="stat_path not set")
    parser.add_argument(
        "--output-directory", type=str, default="output_directory not set"
    )
    parser.add_argument("--vision-root", type=str, default="")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--tensor-model-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-model-parallel-size", type=int, default=1)
    parser.add_argument("--context-parallel-size", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=10000)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--action-horizon", type=int, default=30)
    parser.add_argument("--action-steps", type=int, default=50)
    parser.add_argument("--tokenizer-max-length", type=int, default=256)

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ckpt-overwrite", action="store_true")
    parser.add_argument("--wandb-enabled", action="store_true")
    parser.add_argument(
        "--use-accelerator",
        action="store_true",
        help="Use HuggingFace Accelerator (like lerobot) instead of manual DDP",
    )
    parser.add_argument("--cli-overrides", type=str, default="")
    parser.add_argument("--repo-id", type=str, default="")

    config = parser.parse_args()

    logger.info("=" * 100)
    logger.info(f"train_pi0_base.py config: {config}")
    main(config)
