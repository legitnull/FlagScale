# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Helper functions for Ulysses sequence parallelism in training."""

import torch
import torch.distributed as dist
from typing import Any, Dict, Optional

from .parallel_state import get_parallel_state
from .sequence_parallel import slice_input_tensor, sp_pad_and_slice


def apply_ulysses_sp_to_model(model, ulysses_group: Optional[dist.ProcessGroup] = None):
    """
    Apply Ulysses sequence parallelism to a model.

    For Ulysses SP, the attention layers handle sequence sharding internally via
    all-to-all collectives. This function is a placeholder that can be used to
    set any model-level SP state if needed.

    Args:
        model: The model to enable Ulysses SP on
        ulysses_group: The process group for Ulysses SP communication
    """
    # Ulysses SP is primarily handled inside attention layers via gather_seq_scatter_heads
    # and gather_heads_scatter_seq. No explicit model-level patching needed for most cases.
    pass


def patch_qwen3_5_model_for_vlm_ulysses_sp(model, ulysses_group: Optional[dist.ProcessGroup] = None):
    """
    Patch a Qwen3.5 VLM model for Ulysses sequence parallelism.

    For Ulysses SP in linear attention (GatedDeltaNet), the patched_modeling_qwen3_5_gpu
    module already includes SP support via gather_seq_scatter_heads/gather_heads_scatter_seq.
    This function is a placeholder for any additional VLM-specific setup.

    Args:
        model: The Qwen3.5 VLM model
        ulysses_group: The process group for Ulysses SP communication
    """
    # The patched GatedDeltaNet layers already check get_parallel_state().ulysses_enabled
    # and apply all-to-all internally. No additional patching needed.
    pass


def shard_vlm_batch_for_ulysses_sp(
    batch: Dict[str, Any],
    ulysses_group: Optional[dist.ProcessGroup] = None
) -> Dict[str, Any]:
    """
    Shard a VLM batch for Ulysses sequence parallelism.

    In Ulysses SP, each rank receives a contiguous chunk of the sequence dimension.
    This function slices input_ids, attention_mask, labels, etc. along the sequence
    dimension according to the rank's position in the ulysses_group.

    Args:
        batch: Input batch dict with keys like input_ids, attention_mask, labels, etc.
        ulysses_group: The process group for Ulysses SP (if None, no sharding)

    Returns:
        Sharded batch dict where sequence tensors are sliced to local chunks
    """
    if ulysses_group is None:
        return batch

    ulysses_size = dist.get_world_size(ulysses_group)
    ulysses_rank = dist.get_rank(ulysses_group)

    if ulysses_size == 1:
        return batch

    sharded_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.ndim >= 2:
            # Assume sequence dimension is dim=1 for [batch, seq, ...] tensors
            # Use sp_pad_and_slice for proper padding and slicing
            sharded_value, _ = sp_pad_and_slice(
                value,
                group=ulysses_group,
                dim=1,
            )
            sharded_batch[key] = sharded_value
        else:
            # Non-tensor or 1D tensors pass through unchanged
            sharded_batch[key] = value

    return sharded_batch


def compute_ulysses_sp_corrected_loss(
    loss_tensor: torch.Tensor,
    labels: torch.Tensor,
    ulysses_group: Optional[dist.ProcessGroup] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute corrected loss for Ulysses sequence parallelism.

    In Ulysses SP, each rank computes loss on its local sequence chunk. To get the
    correct global loss, we need to:
    1. Count valid (non-ignored) tokens on each rank
    2. All-reduce to get global valid token count
    3. Scale local loss by (local_valid_tokens / global_valid_tokens)

    Args:
        loss_tensor: Per-token loss tensor of shape [batch, local_seq_len]
        labels: Labels tensor of shape [batch, local_seq_len] (for counting valid tokens)
        ulysses_group: The process group for Ulysses SP

    Returns:
        (loss_for_backward, loss_for_log): Both are scalars
            - loss_for_backward: Scaled loss for backprop (maintains correct gradients)
            - loss_for_log: Globally averaged loss for logging
    """
    if ulysses_group is None or dist.get_world_size(ulysses_group) == 1:
        # No SP, return mean loss
        loss_scalar = loss_tensor.mean()
        return loss_scalar, loss_scalar

    # Count valid tokens (non-ignored, assuming ignore_index=-100)
    valid_mask = (labels != -100)
    local_valid_tokens = valid_mask.sum().float()

    # All-reduce to get global valid token count
    global_valid_tokens = local_valid_tokens.clone()
    dist.all_reduce(global_valid_tokens, op=dist.ReduceOp.SUM, group=ulysses_group)

    # Compute local mean loss
    if local_valid_tokens > 0:
        local_loss = loss_tensor[valid_mask].mean()
    else:
        local_loss = torch.tensor(0.0, device=loss_tensor.device, dtype=loss_tensor.dtype)

    # Scale for correct backprop: each rank's gradient contribution should be
    # proportional to its fraction of valid tokens
    loss_for_backward = local_loss * (local_valid_tokens / global_valid_tokens.clamp(min=1.0))

    # For logging, all-reduce to get global average
    loss_for_log = local_loss.clone()
    dist.all_reduce(loss_for_log, op=dist.ReduceOp.SUM, group=ulysses_group)
    loss_for_log = loss_for_log / dist.get_world_size(ulysses_group)

    return loss_for_backward, loss_for_log
