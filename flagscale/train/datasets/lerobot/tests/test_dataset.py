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


def test_dataset_initialization(test_dataset_path):
    """Test that dataset initializes correctly"""
    from flagscale.train.datasets.lerobot import LeRobotDataset

    dataset = LeRobotDataset(root=test_dataset_path)
    assert dataset is not None
    assert len(dataset) > 0


def test_dataset_properties(our_dataset):
    """Test dataset properties"""
    assert isinstance(our_dataset.fps, int)
    assert isinstance(our_dataset.num_frames, int)
    assert isinstance(our_dataset.num_episodes, int)
    assert our_dataset.fps > 0
    assert our_dataset.num_frames > 0
    assert our_dataset.num_episodes > 0


def test_dataset_getitem(our_dataset):
    """Test __getitem__ returns valid data"""
    item = our_dataset[0]
    assert isinstance(item, dict)
    assert "episode_index" in item
    assert "timestamp" in item
    assert "task" in item
    assert isinstance(item["task"], str)


def test_dataset_with_delta_timestamps(test_dataset_path):
    """Test dataset with delta timestamps"""
    from flagscale.train.datasets.lerobot import LeRobotDataset

    delta_timestamps = {"action": [-0.1, 0.0, 0.1], "observation.state": [-0.1, 0.0]}

    dataset = LeRobotDataset(root=test_dataset_path, delta_timestamps=delta_timestamps)
    assert dataset.delta_indices is not None
    assert "action" in dataset.delta_indices
    assert "observation.state" in dataset.delta_indices

    # Test that we can get an item
    item = dataset[10]  # Use index 10 to avoid edge cases
    assert isinstance(item, dict)


def test_dataset_length(our_dataset, our_metadata):
    """Test dataset length matches metadata"""
    assert len(our_dataset) == our_metadata.total_frames
