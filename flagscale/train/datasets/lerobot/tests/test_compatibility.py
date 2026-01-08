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


def test_dataset_length(original_lerobot_dataset, our_dataset):
    """Test that dataset lengths match"""
    assert len(our_dataset) == len(original_lerobot_dataset)


def test_metadata_compatibility(test_dataset_path, original_lerobot_dataset, our_metadata):
    """Test that metadata matches"""
    assert our_metadata.total_frames == original_lerobot_dataset.num_frames
    assert our_metadata.total_episodes == original_lerobot_dataset.num_episodes
    assert our_metadata.fps == original_lerobot_dataset.fps


def test_sample_compatibility(original_lerobot_dataset, our_dataset):
    """Test that samples match exactly"""
    # Test first, middle, and last samples
    test_indices = [0, len(our_dataset) // 2, len(our_dataset) - 1]

    for idx in test_indices:
        orig_sample = original_lerobot_dataset[idx]
        our_sample = our_dataset[idx]

        # Compare keys
        assert set(orig_sample.keys()) == set(our_sample.keys()), f"Key mismatch at idx {idx}"

        # Compare values
        for key in orig_sample.keys():
            orig_val = orig_sample[key]
            our_val = our_sample[key]

            if isinstance(orig_val, torch.Tensor):
                assert torch.allclose(
                    orig_val, our_val, rtol=1e-5, atol=1e-5
                ), f"Tensor mismatch in {key} at idx {idx}"
            elif isinstance(orig_val, str):
                assert orig_val == our_val, f"String mismatch in {key} at idx {idx}"
            else:
                assert orig_val == our_val, f"Value mismatch in {key} at idx {idx}"


def test_delta_timestamps_compatibility(test_dataset_path):
    """Test that delta timestamps work correctly"""
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset as OrigLeRobotDataset
    except ImportError:
        pytest.skip("lerobot package not installed")

    from flagscale.train.datasets.lerobot import LeRobotDataset

    delta_timestamps = {"action": [-0.1, 0.0, 0.1], "observation.state": [-0.1, 0.0]}

    orig = OrigLeRobotDataset(root=test_dataset_path, delta_timestamps=delta_timestamps)
    ours = LeRobotDataset(root=test_dataset_path, delta_timestamps=delta_timestamps)

    # Test multiple samples (skip first few and last few to avoid edge cases)
    start_idx = 10
    num_samples = min(10, len(orig) - 20)

    for i in range(num_samples):
        idx = start_idx + i
        orig_sample = orig[idx]
        our_sample = ours[idx]

        for key in orig_sample.keys():
            if isinstance(orig_sample[key], torch.Tensor):
                assert torch.allclose(
                    orig_sample[key], our_sample[key], rtol=1e-5, atol=1e-5
                ), f"Delta timestamp mismatch: {key} at idx {idx}"


def test_episode_filtering_compatibility(test_dataset_path, original_lerobot_dataset):
    """Test that episode filtering works correctly"""
    from flagscale.train.datasets.lerobot import LeRobotDataset

    if original_lerobot_dataset.num_episodes < 2:
        pytest.skip("Need at least 2 episodes for this test")

    episodes = [0]

    orig = original_lerobot_dataset.__class__(root=test_dataset_path, episodes=episodes)
    ours = LeRobotDataset(root=test_dataset_path, episodes=episodes)

    assert len(orig) == len(ours), "Length mismatch with episode filtering"

    # Test a few samples
    for idx in [0, len(ours) // 2]:
        orig_sample = orig[idx]
        our_sample = ours[idx]

        # Check episode_index matches
        assert (
            orig_sample["episode_index"] == our_sample["episode_index"]
        ), f"Episode index mismatch at idx {idx}"
