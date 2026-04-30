"""
VLM dataset for Qwen3.5-VL.

Forked from vlm_datasets.py with the following changes for Qwen3.5 compatibility:
  - Removed pre-computed position_ids (Qwen3.5 computes 3D M-RoPE internally)
  - Added mm_token_type_ids generation (required by Qwen3.5's get_rope_index)
  - Token IDs: image_pad=248056, video_pad=248057
"""

import copy
import itertools
import json
import logging
import os
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch
import transformers
from decord import VideoReader
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import Dataset

from flagscale.train.datasets.qwen_data_config import data_list

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 248056  # <|image_pad|> in Qwen3.5
VIDEO_TOKEN_INDEX = 248057  # <|video_pad|> in Qwen3.5

DEFAULT_IMAGE_TOKEN = "<image>\n"
DEFAULT_VIDEO_TOKEN = "<video>\n"

local_rank = None
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)
logging.getLogger("decord").setLevel(logging.CRITICAL)
logging.getLogger("decord.video_reader").setLevel(logging.CRITICAL)
from contextlib import contextmanager


@contextmanager
def suppress_ffmpeg_stderr():
    null_fd = os.open(os.devnull, os.O_RDWR)
    save_fd = os.dup(2)
    try:
        os.dup2(null_fd, 2)
        yield
    finally:
        os.dup2(save_fd, 2)
        os.close(null_fd)
        os.close(save_fd)


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def build_mm_token_type_ids(input_ids: torch.Tensor) -> torch.Tensor:
    """
    Build mm_token_type_ids from input_ids for Qwen3.5.
    text -> 0, image_pad (248056) -> 1, video_pad (248057) -> 2.
    """
    mm_type = torch.zeros_like(input_ids, dtype=torch.int32)
    mm_type[input_ids == IMAGE_TOKEN_INDEX] = 1
    mm_type[input_ids == VIDEO_TOKEN_INDEX] = 2
    return mm_type


def preprocess_qwen_2_visual(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw: list = [],
    visual_type: str = "image",
) -> dict:
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."
    if visual_type not in ["image", "video"]:
        raise ValueError("visual_type must be either 'image' or 'video'")

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    visual_replicate_index = 0
    input_ids, targets = [], []

    for i, source in enumerate(sources):
        try:
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
        except:
            print(sources)

        input_id, target = [], []
        sys_out = tokenizer.apply_chat_template([{"role": "system", "content": system_message}])

        if isinstance(sys_out, dict) or hasattr(sys_out, "keys"):
            sys_id = sys_out.get("input_ids", [])
        else:
            sys_id = sys_out

        if hasattr(sys_id, "tolist"):
            sys_id = sys_id.tolist()
        if len(sys_id) > 0 and isinstance(sys_id[0], list):
            sys_id = sys_id[0]

        input_id += sys_id
        target += [IGNORE_INDEX] * len(sys_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)
            if role == "user":
                visual_tag = f"<{visual_type}>"
                if visual_tag not in content and visual_type == "video" and "<image>" in content:
                    visual_tag = "<image>"
                if visual_tag in content:
                    parts = content.split(visual_tag)
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        blocks = grid_thw[visual_replicate_index]
                        replacement = "".join(
                            [
                                "<|vision_start|>"
                                + f"<|{visual_type}_pad|>" * block_tokens
                                + "<|vision_end|>"
                                for block_tokens in blocks
                            ]
                        )
                        new_parts.append(replacement)
                        visual_replicate_index += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

            conv = [{"role": role, "content": content}]
            encode_out = tokenizer.apply_chat_template(conv)

            if isinstance(encode_out, dict) or hasattr(encode_out, "keys"):
                encode_id = encode_out.get("input_ids", [])
            else:
                encode_id = encode_out

            if hasattr(encode_id, "tolist"):
                encode_id = encode_id.tolist()
            if len(encode_id) > 0 and isinstance(encode_id[0], list):
                encode_id = encode_id[0]

            try:
                encode_id = [int(x) for x in encode_id]
            except (ValueError, TypeError):
                print(f"⚠️ [警告] 发现无法转为 token 的脏数据内容: {content[:100]}...")
                encode_id = [0]

            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target_mask = encode_id.copy()
                target_mask[:3] = [IGNORE_INDEX] * 3
                target += target_mask

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets.append(target)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning (Qwen3.5 version)."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super().__init__()

        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        rank0_print(f"[Qwen3.5 VLM] Loading datasets: {dataset_list}")
        self.video_max_total_pixels = getattr(data_args, "video_max_total_pixels", 1664 * 28 * 28)
        self.video_min_total_pixels = getattr(data_args, "video_min_total_pixels", 256 * 28 * 28)

        list_data_dict = []

        for data in dataset_list:
            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data["annotation_path"])
            else:
                annotations = json.load(open(data["annotation_path"], "r"))
            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(annotations, int(len(annotations) * sampling_rate))
                print(f"sampling {len(annotations)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")
            for ann in annotations:
                if data["data_path"] != "":
                    ann["data_path"] = data["data_path"]
                elif "raw_data" in ann.keys():
                    ann["data_path"] = ann["raw_data"]["data_root"]
            list_data_dict += annotations

        list_data_dict = self.pre_filter_long_case(
            list_data_dict, max_words=tokenizer.max_len_single_sentence
        )
        random.shuffle(list_data_dict)

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

        rank0_print(f"[Qwen3.5 VLM] Total training samples: {len(self.list_data_dict)}")

    def __len__(self):
        return len(self.list_data_dict)

    def pre_filter_long_case(self, list_data_dict, max_words=1024):
        def count_total_words(convs):
            total = 0
            for entry in convs:
                value = entry.get("value", "")
                total += len(value.strip().split())
            return total

        return [
            item
            for item in list_data_dict
            if count_total_words(item.get("conversations", [])) <= max_words
        ]

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if ("images" in sample) or ("videos" in sample) else -cur_len
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def process_image_unified(self, image_file):
        processor = copy.deepcopy(self.data_args.image_processor)
        image = Image.open(image_file).convert("RGB")
        if getattr(self.data_args, "fix_image_size", None) is not None:
            image = image.resize(
                self.data_args.fix_image_size,
                resample=Image.BICUBIC,
            )
        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, list):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw

    def process_video(self, video_file):
        if not os.path.exists(video_file):
            print(f"File not exist: {video_file}")
        with suppress_ffmpeg_stderr():
            vr = VideoReader(video_file, num_threads=4)
            total_frames = len(vr)
            avg_fps = vr.get_avg_fps()
            video_length = total_frames / avg_fps
            interval = getattr(self.data_args, "base_interval", 4)

            num_frames_to_sample = round(video_length / interval)
            video_min_frames = getattr(self.data_args, "video_min_frames", 4)
            video_max_frames = getattr(self.data_args, "video_max_frames", 8)

            target_frames = min(max(num_frames_to_sample, video_min_frames), video_max_frames)
            frame_idx = np.linspace(0, total_frames - 2, target_frames, dtype=int)
            frame_idx = np.unique(frame_idx)
            video = vr.get_batch(frame_idx).asnumpy()
        fps = len(frame_idx) / video_length
        processor = copy.deepcopy(self.data_args.video_processor)
        # 把“单帧像素上限”乘以“实际抽取的帧数”，得到整个视频的额度
        actual_num_frames = len(frame_idx)
        processor.max_pixels = self.data_args.video_max_frame_pixels * actual_num_frames
        processor.min_pixels = self.data_args.video_min_frame_pixels * actual_num_frames
        processor.size["longest_edge"] = processor.max_pixels
        processor.size["shortest_edge"] = processor.min_pixels
        video_processed = processor.preprocess(
            videos=video, do_sample_frames=False, return_tensors="pt"
        )
        video_tensor = video_processed["pixel_values_videos"]
        grid_thw = video_processed["video_grid_thw"][0]
        second_per_grid_ts = [self.data_args.image_processor.temporal_patch_size / fps] * len(
            grid_thw
        )
        return video_tensor, grid_thw, second_per_grid_ts

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        num_base_retries = 5
        num_final_retries = 40

        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(0.5)

        for attempt_idx in range(num_final_retries):
            try:
                next_index = random.randint(0, len(self.list_data_dict) - 1)
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                src = self.list_data_dict[next_index]
                video_path = src.get("video", src.get("image", "unknown"))
                data_path = src.get("data_path", "")
                print(
                    f"[Try other #{attempt_idx}] Failed sample {next_index}, "
                    f"file: {data_path}/{video_path}. Exception: {e}"
                )

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, i) -> dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"

        if "image" in sources[0] and len(sources[0]["image"]):
            image_folder = self.list_data_dict[i]["data_path"]
            image_file = self.list_data_dict[i]["image"]
            if isinstance(image_file, list):
                if len(image_file) > 1:
                    image_file = [os.path.join(image_folder, file) for file in image_file]
                    results = [self.process_image_unified(file) for file in image_file]
                    image, grid_thw = zip(*results)
                else:
                    image_file = image_file[0]
                    image_file = os.path.join(image_folder, image_file)
                    image, grid_thw = self.process_image_unified(image_file)
                    image = [image]
            else:
                image_file = os.path.join(image_folder, image_file)
                image, grid_thw = self.process_image_unified(image_file)
                image = [image]
            grid_thw_merged = copy.deepcopy(grid_thw)
            if not isinstance(grid_thw, Sequence):
                grid_thw_merged = [grid_thw_merged]
                grid_thw = [grid_thw]
            image_blocks = []
            for merged_thw in grid_thw_merged:
                tokens = (
                    merged_thw.prod().item()
                    if hasattr(merged_thw.prod(), "item")
                    else merged_thw.prod()
                )
                tokens = tokens // (self.data_args.image_processor.merge_size**2)
                image_blocks.append([int(tokens)])
            grid_thw_merged = image_blocks
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(
                sources, self.tokenizer, grid_thw=grid_thw_merged, visual_type="image"
            )

        elif "video" in sources[0] and len(sources[0]["video"]):
            video_file = self.list_data_dict[i]["video"]
            video_folder = self.list_data_dict[i]["data_path"]
            if isinstance(video_file, list):
                if len(video_file) > 1:
                    video_file = [os.path.join(video_folder, file) for file in video_file]
                    results = [self.process_video(file) for file in video_file]
                    video, grid_thw, second_per_grid_ts = zip(*results)
                else:
                    video_file = video_file[0]
                    video_file = os.path.join(video_folder, video_file)
                    video, grid_thw, second_per_grid_ts = self.process_video(video_file)
                    video = [video]
            else:
                video_file = os.path.join(video_folder, video_file)
                video, grid_thw, second_per_grid_ts = self.process_video(video_file)
                video = [video]
            grid_thw_merged = copy.deepcopy(grid_thw)
            if not isinstance(grid_thw, Sequence):
                grid_thw_merged = [grid_thw_merged]
                grid_thw = [grid_thw]
            video_blocks = []
            for merged_thw in grid_thw_merged:
                # 视频的 merged_thw 维度是 [T, H, W]
                t = merged_thw[0].item() if hasattr(merged_thw[0], "item") else merged_thw[0]
                h = merged_thw[1].item() if hasattr(merged_thw[1], "item") else merged_thw[1]
                w = merged_thw[2].item() if hasattr(merged_thw[2], "item") else merged_thw[2]
                tokens_per_frame = (h * w) // (self.data_args.image_processor.merge_size**2)
                # 为该视频生成 T 个 frame 的 token 数列表
                video_blocks.append([int(tokens_per_frame)] * int(t))
            grid_thw_merged = video_blocks
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(
                sources, self.tokenizer, grid_thw=grid_thw_merged, visual_type="video"
            )

        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(sources, self.tokenizer, grid_thw=None)

        # Build mm_token_type_ids from input_ids (Qwen3.5 requirement)
        mm_token_type_ids = build_mm_token_type_ids(data_dict["input_ids"])  # [1, seq_len]

        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0],
                labels=data_dict["labels"][0],
                mm_token_type_ids=mm_token_type_ids[0],  # [seq_len]
            )

        if "image" in self.list_data_dict[i]:
            data_dict["pixel_values"] = image
            data_dict["image_grid_thw"] = grid_thw
        elif "video" in self.list_data_dict[i]:
            data_dict["pixel_values_videos"] = video
            data_dict["video_grid_thw"] = grid_thw

        max_len = self.tokenizer.max_len_single_sentence
        if data_dict["input_ids"].shape[0] > max_len:
            src = self.list_data_dict[i]
            file_info = src.get("video", src.get("image", "unknown"))
            data_path = src.get("data_path", "")
            print(
                f"Sample too long (len={data_dict['input_ids'].shape[0]}, max_len={max_len}); "
                f"clip sample {i}, file: {data_path}/{file_info}"
            )
            data_dict["input_ids"] = data_dict["input_ids"][:max_len]
            data_dict["labels"] = data_dict["labels"][:max_len]
            data_dict["mm_token_type_ids"] = data_dict["mm_token_type_ids"][:max_len]

            if "image" in self.list_data_dict[i]:
                num_image_tokens = (data_dict["input_ids"] == IMAGE_TOKEN_INDEX).sum().item()
                expected_tokens = sum(
                    t.prod().item() // self.data_args.image_processor.merge_size**2
                    for t in grid_thw
                )
                if num_image_tokens != expected_tokens:
                    raise ValueError(
                        f"Image tokens truncated: {num_image_tokens} vs {expected_tokens}, skipping sample {i}"
                    )

        return data_dict


@dataclass
class DataCollatorForQwen35:
    """Collate examples for Qwen3.5 VLM training. No position_ids — model computes them."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[dict]) -> dict[str, torch.Tensor]:
        input_ids, labels, mm_token_type_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "mm_token_type_ids")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
            padding_side=self.tokenizer.padding_side,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX,
            padding_side=self.tokenizer.padding_side,
        )
        # mm_token_type_ids: pad with 0 (text type)
        mm_token_type_ids = torch.nn.utils.rnn.pad_sequence(
            mm_token_type_ids,
            batch_first=True,
            padding_value=0,
            padding_side=self.tokenizer.padding_side,
        )

        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        mm_token_type_ids = mm_token_type_ids[:, : self.tokenizer.model_max_length]

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            mm_token_type_ids=mm_token_type_ids,
        )

        images = list(
            itertools.chain(
                *(instance["pixel_values"] for instance in instances if "pixel_values" in instance)
            )
        )
        videos = list(
            itertools.chain(
                *(
                    instance["pixel_values_videos"]
                    for instance in instances
                    if "pixel_values_videos" in instance
                )
            )
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = list(
                itertools.chain(
                    *(
                        instance["image_grid_thw"]
                        for instance in instances
                        if "image_grid_thw" in instance
                    )
                )
            )
            grid_thw = torch.stack(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = list(
                itertools.chain(
                    *(
                        instance["video_grid_thw"]
                        for instance in instances
                        if "video_grid_thw" in instance
                    )
                )
            )
            video_grid_thw = torch.stack(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> dict:
    """Make dataset and collator for supervised fine-tuning (Qwen3.5)."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)

    eval_dataset = None
    if hasattr(data_args, "eval_dataset") and data_args.eval_dataset:
        eval_data_args = copy.deepcopy(data_args)
        eval_data_args.dataset_use = data_args.eval_dataset
        eval_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=eval_data_args)

    data_collator = DataCollatorForQwen35(tokenizer=tokenizer)

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )


def make_vlm_dataloader(cfg, rank: int, world_size: int, seed: int = 42):
    from transformers import AutoProcessor

    data_args = cfg.datasets.vlm_data
    processor = AutoProcessor.from_pretrained(cfg.framework.qwenvl.base_vlm)
    image_processor = processor.image_processor
    video_processor = getattr(processor, "video_processor", None)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.framework.qwenvl.base_vlm,
        model_max_length=data_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )

    image_processor.max_pixels = int(data_args.max_pixels)
    image_processor.min_pixels = int(data_args.min_pixels)
    image_processor.size["longest_edge"] = int(data_args.max_pixels)
    image_processor.size["shortest_edge"] = int(data_args.min_pixels)
    data_args_ns = SimpleNamespace(**OmegaConf.to_container(data_args, resolve=True))
    data_args_ns.image_processor = image_processor
    data_args_ns.video_processor = video_processor
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args_ns)

    train_dataset = data_module["train_dataset"]
    data_collator = data_module["data_collator"]
    from torchdata.stateful_dataloader import StatefulDataLoader

    from flagscale.train.utils.train_utils import StatefulDistributedSampler

    sampler = StatefulDistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=False,
        seed=seed,
    )

    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=cfg.datasets.vlm_data.per_device_batch_size,
        collate_fn=data_collator,
        num_workers=8,
        shuffle=False,  # Must be False when using sampler
        sampler=sampler,
        pin_memory=True,
        prefetch_factor=4,
    )

    return {
        "train_dataloader": train_dataloader,
        "sampler": sampler,
    }
