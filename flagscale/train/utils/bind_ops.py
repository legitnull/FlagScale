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

"""Bind OpSlots to FLA kernel implementations."""

from __future__ import annotations

from .veomni_ops_dispatch import OpSlot
from . import patched_modeling_qwen3_5_gpu as _modeling_module


def bind_qwen3_5_fla_ops() -> None:
    """Bind Qwen3.5 OpSlots to FLA kernel implementations.

    Must be called before creating any Qwen3_5GatedDeltaNet layers.
    Binds:
    - veomni_rms_norm_gated -> fla.modules.FusedRMSNormGated
    - veomni_causal_conv1d -> fla.modules.convolution.causal_conv1d
    - veomni_chunk_gated_delta_rule -> fla.ops.gated_delta_rule.chunk_gated_delta_rule
    """
    bound = []
    for name in dir(_modeling_module):
        obj = getattr(_modeling_module, name, None)
        if not isinstance(obj, OpSlot):
            continue
        if obj.op_name in ("rms_norm_gated", "causal_conv1d", "chunk_gated_delta_rule"):
            obj.bind("fla")
            bound.append(f"{obj.op_name} (fla)")

    if bound:
        print(f"[OpSlot] Bound: {', '.join(bound)}")
