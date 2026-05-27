import argparse
import json
import time

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request
from omegaconf import DictConfig, OmegaConf
from PIL import Image as PIL_Image

from flagscale.models.configs.types import FeatureType, PolicyFeature
from flagscale.models.pi05.configuration_pi05 import PI05Config
from flagscale.models.pi05.modeling_pi05 import PI05Policy
from flagscale.models.utils.constants import ACTION, OBS_STATE
from flagscale.runner.utils import logger
from flagscale.train.train_pi import make_pre_post_processors

app = Flask(__name__)

IMAGE_KEYS = [
    "observation.images.base_0_rgb",
    "observation.images.left_wrist_0_rgb",
    "observation.images.right_wrist_0_rgb",
]

policy = None
preprocessor = None
postprocessor = None
engine_cfg = None
policy_config = None


def load_model(config_path: str):
    global policy, preprocessor, postprocessor, engine_cfg, policy_config

    cfg = OmegaConf.load(config_path)
    assert isinstance(cfg, DictConfig)

    engine_cfg = cfg.get("engine", {})
    generate_cfg = cfg.get("generate", {})

    pretrained_path = engine_cfg.model
    logger.info(f"Loading pi0.5 model from {pretrained_path}...")

    policy_config = PI05Config.from_pretrained(pretrained_path)
    policy_config.pretrained_path = pretrained_path
    policy_config.device = engine_cfg.device

    if engine_cfg.get("dtype"):
        policy_config.dtype = engine_cfg.dtype

    policy = PI05Policy.from_pretrained(pretrained_path, config=policy_config)
    policy = policy.to(device=engine_cfg.device)
    policy.eval()
    logger.info("pi0.5 model loaded successfully")

    use_compile = engine_cfg.get("compile", False)
    if use_compile:
        compile_mode = engine_cfg.get("compile_mode", "max-autotune")
        policy.model.denoise_step = torch.compile(policy.model.denoise_step, mode=compile_mode)
        logger.info(f"torch.compile applied (mode={compile_mode})")

    stat_path = f"{pretrained_path}/stats.json"
    logger.info(f"Loading dataset stats from {stat_path}...")
    with open(stat_path, "r", encoding="utf-8") as f:
        stats_dict = json.load(f)
    dataset_stats = {}
    for key, sub_dict in stats_dict.items():
        dataset_stats[key] = {k: torch.tensor(v).to(engine_cfg.device) for k, v in sub_dict.items()}

    if ACTION in dataset_stats:
        actual_action_dim = dataset_stats[ACTION]["mean"].shape[-1]
        policy_config.output_features[ACTION] = PolicyFeature(
            type=FeatureType.ACTION, shape=(actual_action_dim,)
        )

    processor_kwargs = {
        "preprocessor_overrides": {
            "device_processor": {"device": engine_cfg.device},
            "normalizer_processor": {
                "stats": dataset_stats,
                "features": {**policy_config.input_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "tokenizer_processor": {"tokenizer_name": engine_cfg.tokenizer},
        }
    }

    rename_map = generate_cfg.get("rename_map")
    if rename_map:
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": rename_map
        }

    postprocessor_kwargs = {
        "postprocessor_overrides": {
            "unnormalizer_processor": {
                "stats": dataset_stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            }
        }
    }

    preprocessor, postprocessor = make_pre_post_processors(
        pretrained_path=pretrained_path, **processor_kwargs, **postprocessor_kwargs
    )

    # Warmup
    # warmup_iters = 5 if use_compile else 3
    # logger.info(f"Warming up with {warmup_iters} iterations...")
    # dummy_batch = _build_batch(
    #     state=np.zeros(7, dtype=np.float32),
    #     input_img=np.zeros((224, 224, 3), dtype=np.uint8),
    #     prompt="warmup",
    # )
    # with torch.no_grad():
    #     for i in range(warmup_iters):
    #         dummy_processed = preprocessor(dummy_batch)
    #         _ = policy.predict_action_chunk(dummy_processed)
    #         logger.info(f"  Warmup {i+1}/{warmup_iters} done")

    logger.info("Server ready for inference requests")


def _build_batch(state: np.ndarray, input_img: np.ndarray, prompt: str) -> dict:
    img_tensor = torch.from_numpy(input_img).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W) uint8

    batch = {}
    for key in IMAGE_KEYS:
        if key == "observation.images.left_wrist_0_rgb":
            batch[key] = img_tensor
        else:
            batch[key] = torch.zeros(1, 3, 224, 224, dtype=torch.uint8)

    batch[OBS_STATE] = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # (1, state_dim)
    batch["task"] = [prompt]

    batch = {
        k: v.to(policy_config.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
    return batch


def sample_actions(state: list, input_img: np.ndarray, prompt: str = "pick the apple and put into the basket") -> np.ndarray:
    import time as _time
    state_array = np.array(state, dtype=np.float32)

    _t0 = _time.perf_counter()
    batch = _build_batch(state=state_array, input_img=input_img, prompt=prompt)
    _t1 = _time.perf_counter()
    print(f"[TIMER] _build_batch: {(_t1-_t0)*1000:.2f}ms", flush=True)

    batch = preprocessor(batch)
    _t2 = _time.perf_counter()
    print(f"[TIMER] preprocessor: {(_t2-_t1)*1000:.2f}ms", flush=True)

    with torch.no_grad():
        action = policy.predict_action_chunk(batch)
    _t3 = _time.perf_counter()
    print(f"[TIMER] predict_action_chunk: {(_t3-_t2)*1000:.2f}ms", flush=True)

    action = postprocessor(action)
    _t4 = _time.perf_counter()
    print(f"[TIMER] postprocessor: {(_t4-_t3)*1000:.2f}ms", flush=True)

    # AbsoluteActions: first 6 dims are deltas, add current state
    state_tensor = torch.tensor(state_array, dtype=torch.float32, device=action.device).unsqueeze(0)
    action[..., :6] += state_tensor[:, :6].unsqueeze(-2)

    result = action.cpu().numpy()
    _t5 = _time.perf_counter()
    print(f"[TIMER] postprocess+cpu: {(_t5-_t3)*1000:.2f}ms", flush=True)

    return result


@app.route("/hello", methods=["GET"])
def hello():
    return jsonify({
        "message": "Hello from Pi0.5 server (flagscale)!",
        "status": "success"
    })


@app.route("/inference_pi05", methods=["POST"])
def inference_pi05():
    import time as _time
    start_time = _time.perf_counter()

    _t0 = _time.perf_counter()
    image_file = request.files["image"]
    json_data = request.form["json"]
    data = json.loads(json_data)
    _t1 = _time.perf_counter()
    print(f"[TIMER] HTTP parse: {(_t1-_t0)*1000:.2f}ms", flush=True)

    image = PIL_Image.open(image_file.stream).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    _t2 = _time.perf_counter()
    print(f"[TIMER] Image decode: {(_t2-_t1)*1000:.2f}ms", flush=True)

    prompt = data.get("prompt", "pick the apple and put into the basket")
    action_chunk = sample_actions(state=data["state"], input_img=img_bgr, prompt=prompt)
    _t3 = _time.perf_counter()
    print(f"[TIMER] sample_actions total: {(_t3-_t2)*1000:.2f}ms", flush=True)

    inference_time = _time.perf_counter() - start_time
    logger.info(f"Inference time: {inference_time:.3f}s, action shape: {action_chunk.shape}")

    return jsonify({
        "inference_pi05_time": inference_time,
        "action_chunk": action_chunk.flatten().tolist(),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, required=True, help="Path to inference config YAML")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    load_model(args.config_path)
    app.run(host=args.host, port=args.port)
