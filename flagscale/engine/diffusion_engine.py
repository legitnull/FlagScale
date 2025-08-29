import os

from typing import Union

from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from omegaconf import DictConfig, OmegaConf
from pydantic import ConfigDict, dataclasses
from typing_extensions import Optional, Self

from vllm.config import config

from flagscale.models.adapters import BaseAdapter, create_adapter
from flagscale.runner.utils import logger
from flagscale.transforms.infer.log_io import LogIOTransform


# @config
# @dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class DiffusionEngineConfig:
    model: str = ""
    # adapter hint
    # tranforms
    #


# TODO: supports all kinds of outputs
# e.g. `StableDiffusionPipelineOutput`
# read more from https://github.com/huggingface/diffusers/blob/main/src/diffusers/utils/outputs.py#L40
class RequestOutput:
    result: str = ""


class DiffusionEngine:
    """ """

    def __init__(self, config_dict: DictConfig) -> None:
        # 1) Build pipeline from engine_config
        # 2) Build adapter over the core module (e.g., pipeline.unet)
        # 3) Register optional caps on adapter (simple callables)
        # 4) Select transforms from engine_args/config (pre/compile/post)
        # 5) Plan and apply transforms on adapter.transformers()

        self.validate_config(config_dict)

        self.pipeline = self.load(self.model_name, **self.model_config)
        print(f"DiffusionEngine: pipeline: {self.pipeline}")

        self.adapter: BaseAdapter = create_adapter(
            self.engine_config.get("adapter", None), self.pipeline
        )
        print(f"self.adapter type: {type(self.adapter)}")

    def validate_config(self, config_dict: DictConfig) -> None:
        """Validate the config dict

        The config dict consists of 3 parts:
        - model_config: model related config
        - engine_config: engine related config
        - transforms: transforms to apply to the pipeline

        The minimum required config is:
        - model_config.model:
        - engine_config.results_path
        """
        self.model_config = config_dict.get("model_config", {})
        if self.model_config is None or self.model_config.get("model", None) is None:
            raise ValueError("model_config.model is required")

        self.model_name = self.model_config.model
        self.model_config.pop("model")

        self.engine_config = config_dict.get("engine_config", {})
        if self.engine_config is None or self.engine_config.get("results_path", None) is None:
            raise ValueError("engine_config.results_path is required")
        self.results_path = self.engine_config.results_path
        self.engine_config.pop("results_path")

        self.transforms = config_dict.get("transforms", [])

    @classmethod
    def from_engine_config(cls, engine_config: DiffusionEngineConfig) -> "DiffusionEngine":
        return cls(engine_config)

    def add_request(self, request_id: str, prompt: str, **kwargs) -> None:
        pass

    def generate(self, **kwargs) -> RequestOutput:
        backbone = self.adapter.backbone()
        print(f"backbone type: {type(backbone)}")
        print(f"backbone: {backbone}")

        log_io_transform = LogIOTransform()
        log_io_transform.apply(self.adapter)

        # try:
        outputs = self.pipeline(**kwargs)
        # except ValueError:
        #     logger.error("Unsupported arguments received: {**kwargs}")
        return outputs

    # TODO: load custom models
    def load(
        self, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs
    ) -> DiffusionPipeline:
        pipeline = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path, **kwargs)
        # TODO: Read from config
        pipeline.to("cuda")
        print(f"unet: {pipeline.unet}")

        return pipeline

    def save(self, outputs) -> bool:
        os.makedirs(self.results_path, exist_ok=True)
        image = outputs.images[0]
        image.save(self.results_path + "/result.png")

        return True
