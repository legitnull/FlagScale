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
) -> tuple[Dict[str, Any], int]:
    """
    Shard a VLM batch for Ulysses sequence parallelism.

    In Ulysses SP, each rank receives a contiguous chunk of the sequence dimension.
    This function slices input_ids, attention_mask, labels, etc. along the sequence
    dimension according to the rank's position in the ulysses_group.

    Vision tensors (pixel_values, pixel_values_videos) are sharded along dim=0
    (patch sequence), while language tensors are sharded along dim=1 (token sequence).
    Metadata tensors (grid_thw) are passed through unchanged.

    Args:
        batch: Input batch dict with keys like input_ids, attention_mask, labels, etc.
        ulysses_group: The process group for Ulysses SP (if None, no sharding)

    Returns:
        (sharded_batch, global_valid_tokens): Sharded batch dict where sequence tensors
        are sliced to local chunks, and the total number of valid (non -100) label
        tokens across all ulysses ranks. global_valid_tokens is computed here via a
        CPU all_reduce to avoid NCCL communicator conflicts during backward.
    """
    if ulysses_group is None:
        labels = batch.get("labels")
        global_valid_tokens = int((labels != -100).sum().item()) if labels is not None else 0
        return batch, global_valid_tokens

    ulysses_size = dist.get_world_size(ulysses_group)
    ulysses_rank = dist.get_rank(ulysses_group)

    if ulysses_size == 1:
        labels = batch.get("labels")
        global_valid_tokens = int((labels != -100).sum().item()) if labels is not None else 0
        return batch, global_valid_tokens

    SEQUENCE_KEYS = {"input_ids", "attention_mask", "labels", "mm_token_type_ids"}
    PAD_VALUES = {"labels": -100, "input_ids": 0, "attention_mask": 0, "mm_token_type_ids": 0}

    sharded_batch = {}
    for key, value in batch.items():
        if not isinstance(value, torch.Tensor):
            sharded_batch[key] = value
        elif key in SEQUENCE_KEYS and value.ndim >= 2:
            sharded_batch[key] = sp_pad_and_slice(value, dim=1, pad_value=PAD_VALUES.get(key, 0))
        elif key == "position_ids" and value.ndim == 3:
            # M-RoPE position_ids: (batch, 3, seq_len) — shard along seq dim
            sharded_batch[key] = sp_pad_and_slice(value, dim=2, pad_value=0)
        else:
            # Vision keys (pixel_values, image_grid_thw, etc.) are replicated —
            # the ViT must see all patches together for correct self-attention.
            sharded_batch[key] = value

    # Signal to the model that vision tokens are NOT sharded across SP ranks.
    sharded_batch["sp_vision_replicated"] = True

    # Count valid tokens across all ulysses ranks using a CPU tensor all_reduce.
    # Done here (before forward) to avoid adding NCCL collectives between forward and
    # backward, which can deadlock with FSDP's reduce-scatter when using a flattened
    # dp_shard_sp mesh.
    local_labels = sharded_batch.get("labels")
    local_valid = int((local_labels != -100).sum().item()) if local_labels is not None else 0
    count_tensor = torch.tensor([local_valid], dtype=torch.long, device="cuda")
    dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM, group=ulysses_group)
    global_valid_tokens = int(count_tensor.item())

    return sharded_batch, global_valid_tokens


def compute_ulysses_sp_corrected_loss(
    loss_tensor: torch.Tensor,
    labels: torch.Tensor,
    global_valid_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute corrected loss for Ulysses sequence parallelism.

    The model computes cross_entropy(reduction='mean') over local valid tokens,
    giving a scalar local mean loss. To get correct global gradients we scale by
    (local_valid_tokens / global_valid_tokens) so each rank's gradient contribution
    is proportional to its fraction of valid tokens.

    No NCCL collectives are performed here — global_valid_tokens is pre-computed
    at shard time via a CPU all_reduce in shard_vlm_batch_for_ulysses_sp, avoiding
    communicator conflicts with FSDP's backward reduce-scatter.

    Args:
        loss_tensor: Scalar mean loss from model forward (CE with reduction='mean')
        labels: Labels tensor of shape [batch, local_seq_len] (for counting valid tokens)
        global_valid_tokens: Total valid tokens across all ulysses ranks (pre-computed)

    Returns:
        (loss_for_backward, loss_for_log): Both are scalars
            - loss_for_backward: Scaled loss for backprop (maintains correct gradients)
            - loss_for_log: Local weighted loss (caller may all_reduce after backward for logging)
    """
    if global_valid_tokens == 0:
        zero = torch.zeros_like(loss_tensor)
        return zero, zero

    local_valid_tokens = (labels != -100).sum().float().to(loss_tensor.device)

    loss_tensor = torch.where(
        torch.isnan(loss_tensor), torch.zeros_like(loss_tensor), loss_tensor
    )

    scale = local_valid_tokens / float(global_valid_tokens)
    loss_for_backward = loss_tensor * scale

    loss_for_log = loss_for_backward.detach().clone()

    return loss_for_backward, loss_for_log
