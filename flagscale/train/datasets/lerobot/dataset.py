# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch

from torch.utils.data import Dataset

from .metadata import LeRobotMetadata, load_metadata
from .reader import LeRobotReader
from flagscale.train.datasets.utils import check_delta_timestamps, get_delta_indices


class LeRobotDataset(Dataset):
    """
    PyTorch Dataset for LeRobot format.

    Simplified version without hub upload/download or backward compatibility.
    """

    def __init__(
        self,
        root: str | Path,
        episodes: Optional[list[int]] = None,
        delta_timestamps: Optional[Dict[str, list[float]]] = None,
        image_transforms: Optional[Callable] = None,
        video_backend: str = "pyav",
        tolerance_s: float = 1e-4,
        revision: str | None = None,
    ):
        """Initialize LeRobot dataset

        Args:
            root: Path to LeRobot dataset directory
            episodes: Optional list of episode indices to load
            delta_timestamps: Temporal offsets for each key
            image_transforms: Optional image transformations
            video_backend: Video decoding backend ("pyav" or "torchvision")
            tolerance_s: Tolerance for timestamp matching
            revision: Unused, kept for backward compatibility
        """
        super().__init__()
        self.root = Path(root)
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.delta_indices = None

        # Load metadata
        self.meta = load_metadata(self.root)

        # Create reader
        self.reader = LeRobotReader(
            root=self.root,
            metadata=self.meta,
            episodes=episodes,
            video_backend=video_backend,
            tolerance_s=tolerance_s,
        )

        # Setup delta timestamps/indices
        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        return self.meta.fps

    @property
    def num_frames(self) -> int:
        """Number of frames in selected episodes."""
        if self.episodes is not None:
            return len(self.reader.hf_dataset)
        return self.meta.total_frames

    @property
    def num_episodes(self) -> int:
        """Number of episodes selected."""
        return len(self.episodes) if self.episodes is not None else self.meta.total_episodes

    @property
    def features(self) -> dict[str, dict]:
        """All features contained in the dataset."""
        return self.meta.features

    def _get_query_indices(
        self, idx: int, ep_idx: int
    ) -> tuple[dict[str, list[int]], dict[str, torch.Tensor]]:
        """Get query indices for delta timestamps

        Args:
            idx: Current frame index
            ep_idx: Current episode index

        Returns:
            Tuple of (query_indices, padding_mask)
        """
        ep = self.meta.episodes[ep_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]

        query_indices = {
            key: [max(ep_start, min(ep_end - 1, idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }

        padding = {  # Pad values outside of current episode range
            f"{key}_is_pad": torch.BoolTensor(
                [(idx + delta < ep_start) | (idx + delta >= ep_end) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }

        return query_indices, padding

    def _get_query_timestamps(
        self, current_ts: float, query_indices: dict[str, list[int]] | None = None
    ) -> dict[str, list[float]]:
        """Get query timestamps for videos

        Args:
            current_ts: Current frame timestamp
            query_indices: Optional query indices for temporal context

        Returns:
            Dictionary mapping video keys to query timestamps
        """
        query_timestamps = {}
        for key in self.meta.video_keys:
            if query_indices is not None and key in query_indices:
                if self.reader._absolute_to_relative_idx is not None:
                    relative_indices = [
                        self.reader._absolute_to_relative_idx[idx] for idx in query_indices[key]
                    ]
                    timestamps = self.reader.hf_dataset[relative_indices]["timestamp"]
                else:
                    timestamps = self.reader.hf_dataset[query_indices[key]]["timestamp"]
                query_timestamps[key] = torch.stack(timestamps).tolist()
            else:
                query_timestamps[key] = [current_ts]

        return query_timestamps

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        """Query dataset for indices across keys, skipping video keys

        Args:
            query_indices: Dict mapping keys to index lists to retrieve

        Returns:
            Dict with stacked tensors of queried data (video keys excluded)
        """
        result: dict = {}
        for key, q_idx in query_indices.items():
            if key in self.meta.video_keys:
                continue
            # Map absolute indices to relative indices if needed
            relative_indices = (
                q_idx
                if self.reader._absolute_to_relative_idx is None
                else [self.reader._absolute_to_relative_idx[idx] for idx in q_idx]
            )
            try:
                result[key] = torch.stack(self.reader.hf_dataset[key][relative_indices])
            except (KeyError, TypeError, IndexError):
                result[key] = torch.stack(self.reader.hf_dataset[relative_indices][key])
        return result

    def _query_videos(
        self, query_timestamps: dict[str, list[float]], ep_idx: int
    ) -> dict[str, torch.Tensor]:
        """Query videos for frames at specified timestamps

        Args:
            query_timestamps: Dict mapping video keys to timestamp lists
            ep_idx: Episode index

        Returns:
            Dict mapping video keys to frame tensors
        """
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            frames = self.reader.read_video_frames(ep_idx, vid_key, query_ts)
            item[vid_key] = frames
        return item

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a processed sample

        Args:
            idx: Frame index

        Returns:
            Dictionary containing processed frame data
        """
        # 1. Read base frame
        item = self.reader.read_frame(idx)
        ep_idx = item["episode_index"].item()

        query_indices = None
        # 2. Apply temporal queries (delta timestamps)
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        # 3. Load video frames
        if len(self.meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

        # 4. Apply image transforms
        if self.image_transforms is not None:
            image_keys = self.meta.camera_keys
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])

        # 5. Add task as a string
        task_idx = item["task_index"].item()
        item["task"] = self.meta.tasks.iloc[task_idx].name

        return item

    def __len__(self) -> int:
        return self.num_frames
