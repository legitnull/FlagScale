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


def test_metadata_loading(our_metadata):
    """Test that metadata loads successfully"""
    assert our_metadata is not None
    assert our_metadata.info is not None
    assert our_metadata.episodes is not None
    assert our_metadata.tasks is not None
    assert our_metadata.stats is not None


def test_metadata_properties(our_metadata):
    """Test metadata properties"""
    assert isinstance(our_metadata.features, dict)
    assert isinstance(our_metadata.total_frames, int)
    assert isinstance(our_metadata.total_episodes, int)
    assert isinstance(our_metadata.fps, int)
    assert our_metadata.total_frames > 0
    assert our_metadata.total_episodes > 0
    assert our_metadata.fps > 0


def test_metadata_camera_keys(our_metadata):
    """Test camera-related properties"""
    assert isinstance(our_metadata.camera_keys, list)
    assert isinstance(our_metadata.video_keys, list)
    assert isinstance(our_metadata.image_keys, list)


def test_metadata_paths(our_metadata):
    """Test path properties"""
    assert our_metadata.data_path is not None
    assert isinstance(our_metadata.data_path, str)
