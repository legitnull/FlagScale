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

import pytest
import torch


def test_reader_initialization(test_dataset_path, our_metadata):
    """Test that reader initializes correctly"""
    from flagscale.train.datasets.lerobot import LeRobotReader

    reader = LeRobotReader(root=test_dataset_path, metadata=our_metadata)
    assert reader is not None
    assert reader.hf_dataset is not None
    assert len(reader) > 0


def test_reader_read_frame(test_dataset_path, our_metadata):
    """Test reading a single frame"""
    from flagscale.train.datasets.lerobot import LeRobotReader

    reader = LeRobotReader(root=test_dataset_path, metadata=our_metadata)
    frame = reader.read_frame(0)
    assert isinstance(frame, dict)
    assert "episode_index" in frame
    assert "timestamp" in frame


def test_reader_length(test_dataset_path, our_metadata):
    """Test reader length"""
    from flagscale.train.datasets.lerobot import LeRobotReader

    reader = LeRobotReader(root=test_dataset_path, metadata=our_metadata)
    assert len(reader) == our_metadata.total_frames


def test_reader_with_episodes_filter(test_dataset_path, our_metadata):
    """Test reader with episode filtering"""
    from flagscale.train.datasets.lerobot import LeRobotReader

    if our_metadata.total_episodes < 2:
        pytest.skip("Need at least 2 episodes for this test")

    reader = LeRobotReader(root=test_dataset_path, metadata=our_metadata, episodes=[0])
    assert reader is not None
    assert len(reader) < our_metadata.total_frames
