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
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

from .metadata import LeRobotMetadata
from flagscale.train.datasets.utils import (
    get_hf_features_from_features,
    hf_transform_to_torch,
    load_nested_dataset,
)
from flagscale.train.datasets.video_utils import decode_video_frames, get_safe_default_codec

# Import datasets types only for type checking
if TYPE_CHECKING:
    from datasets import Dataset


class LeRobotReader:
    """Handles I/O operations for LeRobot datasets"""

    def __init__(
        self,
        root: Path,
        metadata: LeRobotMetadata,
        episodes: Optional[list[int]] = None,
        video_backend: str | None = None,
        tolerance_s: float = 1e-4,
    ):
        """Initialize LeRobot reader

        Args:
            root: Path to dataset root directory
            metadata: LeRobotMetadata instance
            episodes: Optional list of episode indices to load
            video_backend: Video decoding backend ("pyav" or "torchvision")
            tolerance_s: Tolerance for timestamp matching
        """
        self.root = root
        self.metadata = metadata
        self.episodes = episodes
        self.video_backend = video_backend if video_backend else get_safe_default_codec()
        self.tolerance_s = tolerance_s

        # Load HuggingFace dataset
        self.hf_dataset = self._load_hf_dataset()

        # Create mapping from absolute indices to relative indices when only a subset of episodes are loaded
        self._absolute_to_relative_idx = None
        if self.episodes is not None:
            self._absolute_to_relative_idx = {
                abs_idx.item() if isinstance(abs_idx, torch.Tensor) else abs_idx: rel_idx
                for rel_idx, abs_idx in enumerate(self.hf_dataset["index"])
            }

    def _load_hf_dataset(self) -> "Dataset":
        """Load HuggingFace dataset from disk

        Returns:
            HuggingFace Dataset instance
        """
        features = get_hf_features_from_features(self.metadata.features)
        hf_dataset = load_nested_dataset(
            self.root / "data", features=features, episodes=self.episodes
        )
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def read_frame(self, idx: int) -> Dict[str, Any]:
        """Read a single frame by index

        Args:
            idx: Frame index to read

        Returns:
            Dictionary containing frame data
        """
        return dict(self.hf_dataset[idx])

    def read_video_frames(
        self, episode_index: int, video_key: str, query_timestamps: list[float]
    ) -> torch.Tensor:
        """Read specific frames from a video file

        Args:
            episode_index: Episode index
            video_key: Key identifying the video/camera
            query_timestamps: List of timestamps to query

        Returns:
            Tensor of video frames
        """
        ep = self.metadata.episodes[episode_index]

        # Episodes are stored sequentially on a single mp4 to reduce the number of files.
        # Thus we load the start timestamp of the episode on this mp4 and,
        # shift the query timestamp accordingly.
        from_timestamp = ep[f"videos/{video_key}/from_timestamp"]
        shifted_query_ts = [from_timestamp + ts for ts in query_timestamps]

        video_path = self.root / self.metadata.get_video_file_path(episode_index, video_key)
        frames = decode_video_frames(
            video_path, shifted_query_ts, self.tolerance_s, self.video_backend
        )
        return frames.squeeze(0)

    def __len__(self) -> int:
        """Return number of frames in the dataset"""
        return len(self.hf_dataset)
