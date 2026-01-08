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

import pytest


@pytest.fixture
def test_dataset_path():
    """Path to test LeRobot dataset"""
    return Path("/share/project/fengyupu/hf_hub/aloha_mobile_cabinet")


@pytest.fixture
def original_lerobot_dataset(test_dataset_path):
    """Original LeRobotDataset from lerobot package"""
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset as OrigLeRobotDataset

        return OrigLeRobotDataset(root=test_dataset_path)
    except ImportError:
        pytest.skip("lerobot package not installed")


@pytest.fixture
def our_dataset(test_dataset_path):
    """Our new implementation"""
    from flagscale.train.datasets.lerobot import LeRobotDataset

    return LeRobotDataset(root=test_dataset_path)


@pytest.fixture
def our_metadata(test_dataset_path):
    """Our metadata implementation"""
    from flagscale.train.datasets.lerobot import load_metadata

    return load_metadata(test_dataset_path)
