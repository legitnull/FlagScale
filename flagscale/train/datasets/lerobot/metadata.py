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

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from flagscale.train.datasets.utils import load_episodes, load_info, load_stats, load_tasks


@dataclass
class LeRobotMetadata:
    """Metadata for LeRobot v3.0 datasets"""

    root: Path
    info: Dict[str, Any]
    episodes: pd.DataFrame
    tasks: pd.DataFrame
    stats: Dict[str, Dict[str, Any]]

    @property
    def features(self) -> Dict[str, Any]:
        """All features contained in the dataset."""
        return self.info["features"]

    @property
    def total_frames(self) -> int:
        """Total number of frames saved in this dataset."""
        return self.info.get("total_frames", len(self.episodes))

    @property
    def total_episodes(self) -> int:
        """Total number of episodes available."""
        return self.info.get("total_episodes", len(self.episodes))

    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        return self.info.get("fps", 30)

    @property
    def video_keys(self) -> list[str]:
        """Keys to access visual modalities stored as videos."""
        return [k for k, v in self.features.items() if v.get("dtype") == "video"]

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access visual modalities (regardless of their storage method)."""
        return [k for k, v in self.features.items() if v.get("dtype") in ["video", "image"]]

    @property
    def image_keys(self) -> list[str]:
        """Keys to access visual modalities stored as images."""
        return [key for key, ft in self.features.items() if ft.get("dtype") == "image"]

    @property
    def data_path(self) -> str:
        """Formattable string for the parquet files."""
        return self.info["data_path"]

    @property
    def video_path(self) -> str | None:
        """Formattable string for the video files."""
        return self.info.get("video_path")

    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path:
        """Get path to video file for a specific episode and camera key."""
        if self.episodes is None:
            raise ValueError("Episodes metadata not loaded")
        if ep_index >= len(self.episodes):
            raise IndexError(
                f"Episode index {ep_index} out of range. Episodes: {len(self.episodes)}"
            )
        ep = self.episodes[ep_index]
        chunk_idx = ep[f"videos/{vid_key}/chunk_index"]
        file_idx = ep[f"videos/{vid_key}/file_index"]
        fpath = self.video_path.format(
            video_key=vid_key, chunk_index=chunk_idx, file_index=file_idx
        )
        return Path(fpath)


def load_metadata(root: Path) -> LeRobotMetadata:
    """Load all metadata for a LeRobot dataset

    Args:
        root: Path to LeRobot dataset directory

    Returns:
        LeRobotMetadata instance containing all dataset metadata
    """
    info = load_info(root)
    episodes = load_episodes(root)
    tasks = load_tasks(root)
    stats = load_stats(root)

    return LeRobotMetadata(root=root, info=info, episodes=episodes, tasks=tasks, stats=stats)
