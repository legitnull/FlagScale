import argparse
import json
import os
import time

import cv2
import numpy as np
import safetensors.torch
import torch
import yaml
from flask import Flask, jsonify, request
from PIL import Image as PIL_Image

from flagscale.models.openpi05 import image_tools, normalize
from flagscale.models.openpi05.model import ModelType, Observation
from flagscale.models.openpi05.pi0_config import Pi0Config
from flagscale.models.openpi05.pi0_pytorch import PI0Pytorch
from flagscale.models.openpi05.tokenizer import PaligemmaTokenizer
from flagscale.models.openpi05.transforms import AbsoluteActions, MyArxInputs, MyArxOutputs


def normalize_quantile(x, stats):
    assert stats.q01 is not None
    assert stats.q99 is not None
    q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
    return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def unnormalize_quantile(x, stats):
    assert stats.q01 is not None
    assert stats.q99 is not None
    q01, q99 = stats.q01, stats.q99
    if (dim := q01.shape[-1]) < x.shape[-1]:
        return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
    return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def create_observation(state, input_img, prompt="pick the apple and put into the basket"):
    return {
        "state": np.array(state),
        "image": input_img,
        "wrist_image": input_img,
        "prompt": prompt,
    }


def input_transform(observation, norm_stats):
    inputs = {
        "observation/state": observation["state"],
        "observation/image": observation["image"],
        "observation/wrist_image": observation["wrist_image"],
        "prompt": observation["prompt"],
    }

    myarx_inputs_transform = MyArxInputs(ModelType.PI05)
    inputs = myarx_inputs_transform(inputs)

    state = inputs["state"]
    state = normalize_quantile(state, norm_stats["state"])
    inputs["state"] = state

    for k, v in inputs["image"].items():
        input_img = v
        input_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(input_img, 224, 224))
        inputs["image"][k] = input_img

    tokens, token_masks = PaligemmaTokenizer(max_len=200).tokenize(inputs["prompt"], state=inputs["state"])
    inputs = {**inputs, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}
    inputs.pop("prompt")

    inputs["state"] = pad_to_dim(inputs["state"], 32, axis=-1)

    return inputs


def output_transform(inputs, actions, norm_stats):
    state = unnormalize_quantile(inputs.state.cpu().numpy(), norm_stats["state"])
    actions = unnormalize_quantile(actions.cpu().numpy(), norm_stats["actions"])

    outputs = {
        "state": state,
        "actions": actions,
    }

    absolute_actions_transform = AbsoluteActions(mask=(True, True, True, True, True, True, False))
    outputs = absolute_actions_transform(outputs)

    outputs["actions"] = outputs["actions"][0]
    myarx_outputs_transform = MyArxOutputs()
    outputs = myarx_outputs_transform(outputs)

    return outputs["actions"]


def create_model(cfg):
    checkpoint_dir = cfg["engine"]["checkpoint_dir"]
    asset_id = cfg["engine"]["asset_id"]
    device = cfg["engine"].get("device", "cuda:0")

    norm_stats = normalize.load(os.path.join(checkpoint_dir, "assets", asset_id))

    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    paligemma_variant = cfg["engine"].get("paligemma_variant", "gemma_2b_lora")
    action_expert_variant = cfg["engine"].get("action_expert_variant", "gemma_300m_lora")
    model_config = Pi0Config(paligemma_variant=paligemma_variant, action_expert_variant=action_expert_variant, pi05=True)
    model = PI0Pytorch(config=model_config)
    safetensors.torch.load_model(model, weight_path)

    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    model = model.to(device)
    model.eval()

    if cfg["engine"].get("compile", False):
        model.denoise_step = torch.compile(model.denoise_step, mode="max-autotune")

    return model, norm_stats, device


def sample_actions(model, norm_stats, device, state, input_img, prompt="pick the apple and put into the basket"):
    observation = create_observation(state, input_img, prompt=prompt)
    inputs = input_transform(observation, norm_stats)

    def to_torch(x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(device)[None, ...]
        elif isinstance(x, dict):
            return {k: to_torch(v) for k, v in x.items()}
        elif isinstance(x, (bool, np.bool_)):
            return torch.tensor([x], dtype=torch.bool, device=device)
        else:
            return x

    inputs = to_torch(inputs)
    inputs = Observation.from_dict(inputs)

    with torch.no_grad():
        generator = torch.Generator(device=device).manual_seed(42)
        noise_shape = (1, 50, 32)
        noise = torch.normal(mean=0.0, std=1.0, size=noise_shape, dtype=torch.float32, device=device, generator=generator)
        actions = model.sample_actions(device, inputs, noise=noise)

    actions = output_transform(inputs, actions, norm_stats)
    return actions


def run_server(cfg):
    model, norm_stats, device = create_model(cfg)
    host = cfg["server"].get("host", "0.0.0.0")
    port = cfg["server"].get("port", 5000)
    prompt = cfg["server"].get("prompt", "pick the apple and put into the basket")

    app = Flask(__name__)

    @app.route("/hello", methods=["GET"])
    def hello():
        return jsonify({"message": "Hello from Pi05 server!", "status": "success"})

    @app.route("/inference_pi05", methods=["POST"])
    def inference_pi05():
        start_time = time.time()

        image_file = request.files["image"]
        json_data = request.form["json"]
        data = json.loads(json_data)

        image = PIL_Image.open(image_file.stream)
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        action_chunk = sample_actions(model, norm_stats, device, state=data["state"], input_img=img_bgr, prompt=prompt)

        end_time = time.time()
        inference_pi05_time = end_time - start_time

        json_output = {}
        json_output["inference_pi05_time"] = inference_pi05_time
        json_output["action_chunk"] = action_chunk.flatten().tolist()

        return jsonify(json_output)

    app.run(host=host, port=port)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        cfg = yaml.safe_load(f)

    run_server(cfg)


if __name__ == "__main__":
    main()
