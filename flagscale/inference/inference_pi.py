import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision import transforms

from flagscale.models.configs.types import FeatureType, NormalizationMode, PolicyFeature
from flagscale.models.pi0.configuration_pi0 import PI0Config
from flagscale.models.pi0.modeling_pi0 import PI0Policy
from flagscale.models.pi05.configuration_pi05 import PI05Config
from flagscale.models.pi05.modeling_pi05 import PI05Policy
from flagscale.models.utils.constants import ACTION, OBS_STATE
from flagscale.platforms import get_platform  # noqa: F401 must be before model imports
from flagscale.runner.utils import logger
from flagscale.train.train_pi import make_pre_post_processors


def load_image(image_path: str) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    # Convert RGB to BGR to match source inference behavior (model was deployed with cv2.imread which returns BGR)
    img_array = img_array[:, :, ::-1].copy()
    # Keep as uint8 to avoid lossy float->uint8 round-trip during resize
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    return img_tensor


def load_state_from_file(state_path: str) -> torch.Tensor:
    # (1, state_dim)
    state = torch.load(state_path, map_location="cpu")
    return state


def run_inference(config_path: str):
    """
    Run inference with pi0/pi05 model.

    Args:
        config_path: Path to config YAML file

    Returns:
        Predicted action tensor with shape (batch_size, action_dim)
    """

    logger.info(f"Loading config from {config_path}...")
    cfg = OmegaConf.load(config_path)
    assert isinstance(cfg, DictConfig)

    engine_cfg = cfg.get("engine", {})
    generate_cfg = cfg.get("generate", {})

    # Get model variant from config (defaults to "pi0")
    model_variant = engine_cfg.get("model_variant", "pi0").lower()
    if model_variant not in ["pi0", "pi0.5"]:
        raise ValueError(f"Invalid model_variant: {model_variant}. Must be 'pi0' or 'pi0.5'")

    pretrained_path = engine_cfg.model
    logger.info(f"Loading {model_variant} model from {pretrained_path}...")

    # Select config and policy classes based on model variant
    if model_variant == "pi0.5":
        policy_config = PI05Config.from_pretrained(pretrained_path)
        policy_cls = PI05Policy
    else:
        policy_config = PI0Config.from_pretrained(pretrained_path)
        policy_cls = PI0Policy

    policy_config.pretrained_path = pretrained_path
    policy_config.device = engine_cfg.device

    # Override dtype if specified in engine config
    if engine_cfg.get("dtype"):
        policy_config.dtype = engine_cfg.dtype
        logger.info(f"Overriding model dtype to: {engine_cfg.dtype}")

    images = generate_cfg.images  # dict mapping image_key -> path
    state_path = generate_cfg.get("state_path")
    task_path = generate_cfg.get("task_path")

    image_keys = list(images.keys())
    logger.info(f"Loading {len(image_keys)} images...")
    loaded_images = {}
    for img_key, img_path in images.items():
        img = load_image(img_path)
        loaded_images[img_key] = img
        logger.info(f"Loaded image: {img_key} from {img_path} with shape {img.shape}")

    # Load state
    state_key = OBS_STATE
    logger.info(f"Loading state from {state_path}...")
    state = load_state_from_file(state_path)
    logger.info(f"Loaded state with shape: {state.shape}")

    logger.info(f"Loading task from {task_path}...")
    with open(task_path, "r", encoding="utf-8") as f:
        prompt = f.read().strip()
    logger.info(f"Loaded task prompt: '{prompt}'")

    rename_map = generate_cfg.get("rename_map")

    policy = policy_cls.from_pretrained(pretrained_path, config=policy_config)
    policy = policy.to(device=engine_cfg.device)
    policy.eval()
    logger.info(f"{model_variant} model loaded successfully")

    use_compile = engine_cfg.get("compile", False)
    if use_compile:
        compile_mode = engine_cfg.get("compile_mode", "default")
        logger.info(f"Applying torch.compile (mode={compile_mode}) to denoise_step...")
        policy.model.denoise_step = torch.compile(policy.model.denoise_step, mode=compile_mode)
        logger.info("torch.compile applied to denoise_step")

    # Set normalization mapping for pi0.5 (when not using quantiles)
    use_quantiles = engine_cfg.get("use_quantiles", False)
    if not use_quantiles and model_variant == "pi0.5":
        policy.config.normalization_mapping = {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
        logger.info("Set normalization_mapping for pi0.5 model")

    # TODO: (yupu) Load the stats from pretrained model
    logger.info(f"Loading dataset stats from {engine_cfg.stat_path}...")
    with open(engine_cfg.stat_path, "r", encoding="utf-8") as f:
        stats_dict = json.load(f)
    dataset_stats = {}
    for key, sub_dict in stats_dict.items():
        dataset_stats[key] = {k: torch.tensor(v).to(engine_cfg.device) for k, v in sub_dict.items()}

    # Set output_features from stats to get the actual action dimension
    if ACTION in dataset_stats:
        actual_action_dim = dataset_stats[ACTION]["mean"].shape[-1]
        policy_config.output_features[ACTION] = PolicyFeature(
            type=FeatureType.ACTION, shape=(actual_action_dim,)
        )
        logger.info(f"Set output_features[ACTION] to actual dimension: {actual_action_dim}")

    processor_kwargs = {}
    processor_kwargs["preprocessor_overrides"] = {
        "device_processor": {"device": engine_cfg.device},
        "normalizer_processor": {
            "stats": dataset_stats,
            "features": {**policy_config.input_features},
            "norm_map": policy.config.normalization_mapping,
        },
        "tokenizer_processor": {"tokenizer_name": engine_cfg.tokenizer},
    }

    if rename_map:
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": rename_map
        }

    postprocessor_kwargs = {}
    postprocessor_kwargs["postprocessor_overrides"] = {
        "unnormalizer_processor": {
            "stats": dataset_stats,
            "features": policy.config.output_features,
            "norm_map": policy.config.normalization_mapping,
        }
    }

    preprocessor, postprocessor = make_pre_post_processors(
        pretrained_path=pretrained_path, **processor_kwargs, **postprocessor_kwargs
    )

    batch = {}

    for img_key, img in loaded_images.items():
        batch[img_key] = img

    # Add state
    batch[state_key] = state
    batch["task"] = [prompt]

    batch = {
        k: v.to(policy_config.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

    logger.info("Preprocessing batch...")
    batch = preprocessor(batch)

    logger.info("Running inference...")
    with torch.no_grad():
        # warmup_iters = 5 if use_compile else 3
        # logger.info(f"Warming up with {warmup_iters} iterations (compile={use_compile})...")
        # for i in range(warmup_iters):
        #     _ = policy.predict_action_chunk(batch)
        #     logger.info(f"  Warmup iteration {i+1}/{warmup_iters} done")

        logger.info("Running timed inference...")
        action = policy.predict_action_chunk(batch)
        logger.info(f"action before postprocessor: {action.shape}")

    logger.info("Applying postprocessor...")
    action = postprocessor(action)
    logger.info(f"action after postprocessor: {action.shape}")

    # Apply AbsoluteActions: convert delta actions to absolute by adding state
    # mask=(True,True,True,True,True,True,False) means first 6 dims are deltas, 7th (gripper) is absolute
    original_state = load_state_from_file(state_path).to(action.device)  # (1, state_dim)
    action[..., :6] += original_state[:, :6].unsqueeze(-2)

    logger.info(f"Applied AbsoluteActions (state + delta for first 6 dims)")

    logger.info(f"Final action: {action}")

    return action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, required=True, help="Path to config YAML file")

    args = parser.parse_args()
    run_inference(config_path=args.config_path)


if __name__ == "__main__":
    main()
