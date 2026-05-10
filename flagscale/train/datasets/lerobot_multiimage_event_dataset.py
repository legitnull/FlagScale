from __future__ import annotations

import os
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import datasets as hf_datasets
import numpy as np
import pandas as pd
import torch
from PIL import Image

from flagscale.logger import logger
from flagscale.train.datasets.lerobot_dataset import LeRobotDataset
from flagscale.train.datasets.utils import get_hf_features_from_features, load_nested_dataset


class LeRobotMultiImageEventDataset(LeRobotDataset):
    """LeRobot v3.0 multi-image dataset with event-causal supervision.

    The public event data stores JPEG paths in parquet image fields and event text
    in the index of meta/{subtasks,atomics,tasks}.parquet. This loader keeps the
    base FlagScale LeRobot v3.0 indexing/statistics path, but adds:
      - current/future image split for camera keys
      - subtask/atomic prev-next event targets when columns exist
      - half-episode fallback for datasets with only task-level labels
    """

    EVENT_MODES = ("subtask", "atomic")
    EVENT_META = {
        "subtask": ("subtasks.parquet", "subtask_index"),
        "atomic": ("atomics.parquet", "atomic_index"),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._event_text_lookup = {
            mode: self._load_event_lookup(filename, idx_col)
            for mode, (filename, idx_col) in self.EVENT_META.items()
        }
        self._timing_enabled = os.getenv("FS_WM_DATA_TIMING", "0") == "1"
        self._timing_limit = int(os.getenv("FS_WM_DATA_TIMING_LIMIT", "4"))
        self._timing_count = 0

    def load_hf_dataset(self) -> hf_datasets.Dataset:
        features = get_hf_features_from_features(self.features)
        for key in features:
            if isinstance(features[key], hf_datasets.Image):
                features[key] = hf_datasets.Image(decode=False)

        # Some converted LeRobot v3.0 parquet files order columns differently
        # from meta/info.json. PyArrow requires identical field ordering when
        # casting filtered episode tables, so mirror the physical parquet order.
        first_parquet = sorted((self.root / "data").glob("*/*.parquet"))[0]
        physical_columns = list(pd.read_parquet(first_parquet, engine="pyarrow").columns)
        features = hf_datasets.Features({
            col: features[col] for col in physical_columns if col in features
        })

        hf_dataset = load_nested_dataset(
            self.root / "data", features=features, episodes=self.episodes
        )
        root = self.root
        image_keys = set(self.meta.image_keys)

        def _resolve_and_transform(items_dict: dict[str, list]) -> dict:
            for key in list(items_dict.keys()):
                values = items_dict[key]
                if key in image_keys:
                    decoded = []
                    for value in values:
                        decoded.append(self._decode_image_value(root, value))
                    items_dict[key] = decoded
                    continue

                first = values[0] if values else None
                if first is None or isinstance(first, str):
                    continue
                items_dict[key] = [
                    x if isinstance(x, str) else torch.tensor(x) for x in values
                ]
            return items_dict

        hf_dataset.set_transform(_resolve_and_transform)
        return hf_dataset

    @staticmethod
    def _decode_image_value(root: Path, value: Any) -> Image.Image:
        if isinstance(value, Image.Image):
            return value
        if isinstance(value, dict):
            if value.get("bytes") is not None:
                img = Image.open(BytesIO(value["bytes"]))
            elif value.get("path") is not None:
                path = value["path"]
                if not os.path.isabs(path):
                    path = str(root / path)
                img = Image.open(path)
            else:
                raise ValueError(f"Image has neither bytes nor path: {value}")
            img.load()
            return img.convert("RGB")
        if isinstance(value, str):
            path = value if os.path.isabs(value) else str(root / value)
            img = Image.open(path)
            img.load()
            return img.convert("RGB")
        raise ValueError(f"Unsupported image value type: {type(value)}")

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        image_keys = set(self.meta.image_keys)
        non_image_indices = {k: v for k, v in query_indices.items() if k not in image_keys}
        result = super()._query_hf_dataset(non_image_indices)
        for key, q_idx in query_indices.items():
            if key not in image_keys or key in self.meta.video_keys:
                continue
            relative_indices = [self._to_relative_index(idx) for idx in q_idx]
            result[key] = list(self.hf_dataset[relative_indices][key])
        return result

    def _to_relative_index(self, absolute_idx: int) -> int:
        if self._absolute_to_relative_idx is None:
            return int(absolute_idx)
        return int(self._absolute_to_relative_idx[int(absolute_idx)])

    @staticmethod
    def _scalar(value):
        if isinstance(value, torch.Tensor):
            return value.item() if value.numel() == 1 else value.detach().cpu().tolist()
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return value
        return value

    def _load_event_lookup(self, filename: str, idx_col: str) -> dict[int, str] | None:
        path = self.root / "meta" / filename
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            logger.warning(f"Failed to read event table {path}: {exc}")
            return None
        if idx_col not in df.columns:
            return None
        return {int(row[idx_col]): str(index) for index, row in df.iterrows()}

    def _episode_bounds(self, ep_idx: int) -> tuple[int, int]:
        episode = self.meta.episodes[int(ep_idx)]
        return int(episode["dataset_from_index"]), int(episode["dataset_to_index"])

    def _episode_values(self, ep_idx: int, col: str) -> list[Any]:
        start, end = self._episode_bounds(ep_idx)
        rel = [self._to_relative_index(i) for i in range(start, end)]
        values = self.hf_dataset[rel][col]
        return [self._scalar(v) for v in values]

    @staticmethod
    def _is_null(value) -> bool:
        try:
            return bool(pd.isna(value))
        except (TypeError, ValueError):
            return False

    def _lookup_text(self, value, lookup: dict[int, str] | None) -> str | None:
        if lookup is None or self._is_null(value):
            return None
        try:
            return lookup.get(int(value))
        except (TypeError, ValueError):
            return None

    def _image_at(self, absolute_idx: int, key: str) -> Image.Image:
        rel = self._to_relative_index(absolute_idx)
        return self.hf_dataset[rel][key]

    def _build_neighbor_sample(
        self,
        absolute_idx: int,
        lang: str | None,
        ep_start: int,
    ) -> dict:
        images_by_key = {
            key: self._image_at(absolute_idx, key).resize((256, 256))
            for key in self.meta.camera_keys
        }
        return {
            "images_by_key": images_by_key,
            "lang": str(lang) if lang is not None else "",
            "step_index": int(absolute_idx - ep_start),
        }

    def _sample_event_neighbors(
        self,
        ep_idx: int,
        base_abs_idx: int,
        idx_col: str,
        lookup: dict[int, str] | None,
        rng: np.random.Generator,
    ) -> dict:
        ep_start, ep_end = self._episode_bounds(ep_idx)
        values = self._episode_values(ep_idx, idx_col)
        base_local = int(base_abs_idx - ep_start)
        if base_local < 0 or base_local >= len(values) or self._is_null(values[base_local]):
            return {"rnd_prev": None, "rnd_next": None, "current_lang_override": None}

        segments = []
        i = 0
        while i < len(values):
            value = values[i]
            j = i + 1
            while j < len(values) and (
                values[j] == value or (self._is_null(values[j]) and self._is_null(value))
            ):
                j += 1
            segments.append((value, i, j - 1))
            i = j

        current_seg = None
        for seg_idx, (_, start, end) in enumerate(segments):
            if start <= base_local <= end:
                current_seg = seg_idx
                break
        if current_seg is None:
            return {"rnd_prev": None, "rnd_next": None, "current_lang_override": None}

        def sample_segment(seg_idx: int) -> dict | None:
            if seg_idx < 0 or seg_idx >= len(segments):
                return None
            value, start, end = segments[seg_idx]
            if self._is_null(value):
                return None
            sampled_local = int(rng.integers(start, end + 1))
            sampled_abs = ep_start + sampled_local
            return self._build_neighbor_sample(sampled_abs, self._lookup_text(value, lookup), ep_start)

        current_value = segments[current_seg][0]
        return {
            "rnd_prev": sample_segment(current_seg - 1),
            "rnd_next": sample_segment(current_seg + 1),
            "current_lang_override": self._lookup_text(current_value, lookup),
        }

    def _sample_half_event(
        self,
        ep_idx: int,
        base_abs_idx: int,
        current_task: str,
        rng: np.random.Generator,
    ) -> dict | None:
        ep_start, ep_end = self._episode_bounds(ep_idx)
        n = ep_end - ep_start
        if n < 2:
            return None
        base_local = int(base_abs_idx - ep_start)
        half = n // 2
        if base_local < half:
            candidates = np.arange(half, n)
            direction = "later"
        else:
            candidates = np.arange(0, half)
            direction = "earlier"
        if len(candidates) == 0:
            return None
        sampled_abs = ep_start + int(rng.choice(candidates))
        sample = self._build_neighbor_sample(sampled_abs, current_task, ep_start)
        sample["direction"] = direction
        return sample

    def _get_event_targets(
        self,
        ep_idx: int,
        base_abs_idx: int,
        current_task: str,
        rng: np.random.Generator,
    ) -> dict:
        groups = []
        for mode in self.EVENT_MODES:
            _, idx_col = self.EVENT_META[mode]
            lookup = self._event_text_lookup.get(mode)
            if lookup is None or idx_col not in self.features:
                continue
            neighbors = self._sample_event_neighbors(ep_idx, base_abs_idx, idx_col, lookup, rng)
            if neighbors["rnd_prev"] is None and neighbors["rnd_next"] is None:
                continue
            neighbors["mode"] = mode
            groups.append(neighbors)

        if groups:
            return {"long_event_supervisions": groups, "half_event": None}
        return {
            "long_event_supervisions": [],
            "half_event": self._sample_half_event(ep_idx, base_abs_idx, current_task, rng),
        }

    @staticmethod
    def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
        return Image.fromarray(
            (t.permute(1, 2, 0) * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        )

    @staticmethod
    def _neighbor_images(neighbor: dict | None, keys: list[str]) -> list[Image.Image]:
        if neighbor is None:
            return []
        images_by_key = neighbor.get("images_by_key", {})
        return [images_by_key[k] for k in keys if k in images_by_key]

    def __getitem__(self, idx) -> dict:
        timing = self._timing_enabled and self._timing_count < self._timing_limit
        t0 = time.perf_counter() if timing else None
        item = super().__getitem__(idx)
        t_base = time.perf_counter() if timing else None
        ep_idx = int(self._scalar(item["episode_index"]))
        base_abs_idx = int(self._scalar(item.get("index", idx)))

        for cam_key in self.meta.camera_keys:
            if cam_key not in item:
                continue
            value = item[cam_key]
            if isinstance(value, list) and len(value) > 1:
                item[f"{cam_key}_future"] = value[1]
                item[cam_key] = value[0]
            elif isinstance(value, torch.Tensor) and value.dim() >= 4:
                item[f"{cam_key}_future"] = self._tensor_to_pil(value[1])
                item[cam_key] = self._tensor_to_pil(value[0])
            elif isinstance(value, torch.Tensor) and value.dim() == 3:
                item[cam_key] = self._tensor_to_pil(value)
            if isinstance(item.get(cam_key), Image.Image):
                item[cam_key] = item[cam_key].resize((256, 256))
            future_key = f"{cam_key}_future"
            if isinstance(item.get(future_key), Image.Image):
                item[future_key] = item[future_key].resize((256, 256))

        t_resize = time.perf_counter() if timing else None
        current_task = str(item.get("task", ""))
        rng = np.random.default_rng(base_abs_idx)
        event_targets = self._get_event_targets(ep_idx, base_abs_idx, current_task, rng)
        t_event = time.perf_counter() if timing else None

        camera_keys = list(self.meta.camera_keys)
        long_event_supervisions = []
        for group in event_targets["long_event_supervisions"]:
            prev_images = self._neighbor_images(group.get("rnd_prev"), camera_keys)
            next_images = self._neighbor_images(group.get("rnd_next"), camera_keys)
            prev_lang = group["rnd_prev"].get("lang", "") if group.get("rnd_prev") else ""
            next_lang = group["rnd_next"].get("lang", "") if group.get("rnd_next") else ""
            long_event_supervisions.append({
                "mode": group.get("mode", "event"),
                "prev_images": prev_images,
                "next_images": next_images,
                "prev_lang": prev_lang,
                "next_lang": next_lang,
                "has_prev": bool(prev_images and prev_lang),
                "has_next": bool(next_images and next_lang),
            })

        half_event = event_targets.get("half_event")
        half_event_images = self._neighbor_images(half_event, camera_keys)
        item["long_event_supervisions"] = long_event_supervisions
        item["half_event_images"] = half_event_images
        item["half_event_direction"] = half_event.get("direction", "") if half_event else ""
        item["has_half_event"] = bool(half_event_images)
        item["event_mode"] = "joint" if long_event_supervisions else "half_episode"
        if timing:
            t_end = time.perf_counter()
            self._timing_count += 1
            logger.info(
                "[data_timing] "
                f"dataset={self.root.name} idx={idx} abs_idx={base_abs_idx} "
                f"base_hf_decode_s={t_base - t0:.4f} "
                f"current_resize_s={t_resize - t_base:.4f} "
                f"event_neighbor_sample_s={t_event - t_resize:.4f} "
                f"event_pack_s={t_end - t_event:.4f} "
                f"total_s={t_end - t0:.4f} mode={item['event_mode']}"
            )
        return item
