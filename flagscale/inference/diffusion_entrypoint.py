import argparse
import os
import sys

from typing import Union

from omegaconf import DictConfig, ListConfig, OmegaConf

from flagscale.engine.diffusion_engine import DiffusionEngine


def parse_config() -> Union[DictConfig, ListConfig]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path", type=str, required=True, help="Path to the configuration YAML file"
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    # TODO: Any checks?

    return config


def inference(cfg: DictConfig):
    print(f"cfg: {cfg}")
    model_cfg = cfg.get("diffusion", {})
    engine = DiffusionEngine(model_cfg)

    generate_cfg = cfg.get("generate", {})
    outputs = engine.generate(**generate_cfg)

    engine.save(outputs)


if __name__ == "__main__":
    cfg = parse_config()
    assert isinstance(cfg, DictConfig)  # To make pyright happy
    inference(cfg)
    print("done")
