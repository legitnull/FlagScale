"""Smoke test for Qwen3.5 GatedDeltaNet with Ulysses SP integration.

Validates:
1. OpSlot binding to FLA kernels
2. Single-GPU forward pass correctness
3. SP batch sharding logic
"""

import sys
sys.path.insert(0, '/share/project/fengyupu/github/fs5/flagscale/train')

import torch
from utils.bind_ops import bind_qwen3_5_fla_ops

bind_qwen3_5_fla_ops()

from transformers import Qwen3_5Config
from utils.patched_modeling_qwen3_5_gpu import (
    Qwen3_5GatedDeltaNet,
    veomni_causal_conv1d,
    veomni_chunk_gated_delta_rule,
    veomni_rms_norm_gated,
)


def test_opslot_binding():
    assert veomni_causal_conv1d.use_non_eager_impl, "causal_conv1d not bound"
    assert veomni_chunk_gated_delta_rule.use_non_eager_impl, "chunk_gated_delta_rule not bound"
    assert veomni_rms_norm_gated.use_non_eager_impl, "rms_norm_gated not bound"
    print("[PASS] OpSlot binding")


def test_forward_pass():
    config = Qwen3_5Config(
        hidden_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        head_dim=32,
        intermediate_size=256,
        num_hidden_layers=1,
        rms_norm_eps=1e-6,
        hidden_act='silu',
    )

    layer = Qwen3_5GatedDeltaNet(config, layer_idx=0).to('cuda').to(torch.bfloat16)

    batch_size = 2
    seq_len = 64
    hidden_states = torch.randn(batch_size, seq_len, 128, device='cuda', dtype=torch.bfloat16)

    output = layer(hidden_states, attention_mask=None, cu_seq_lens_q=None)
    out_tokens = output[0].shape[0]
    assert out_tokens == batch_size * seq_len or out_tokens == seq_len, f"Unexpected shape: {output[0].shape}"
    assert output[0].shape[-1] == 128, f"Wrong hidden dim: {output[0].shape}"
    assert not torch.isnan(output[0]).any(), "NaN in output"
    print(f"[PASS] Forward pass: output shape {output[0].shape}")


def test_forward_deterministic():
    config = Qwen3_5Config(
        hidden_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        head_dim=32,
        intermediate_size=256,
        num_hidden_layers=1,
        rms_norm_eps=1e-6,
        hidden_act='silu',
    )

    layer = Qwen3_5GatedDeltaNet(config, layer_idx=0).to('cuda').to(torch.bfloat16)
    hidden_states = torch.randn(2, 32, 128, device='cuda', dtype=torch.bfloat16)

    with torch.no_grad():
        ref = layer(hidden_states, attention_mask=None, cu_seq_lens_q=None)[0].clone()
        for _ in range(5):
            out = layer(hidden_states, attention_mask=None, cu_seq_lens_q=None)[0]
            assert torch.equal(ref, out), "Forward not deterministic"
    print("[PASS] Forward determinism")


def test_sp_batch_sharding():
    from utils.ulysses_sp_helpers import shard_vlm_batch_for_ulysses_sp

    batch = {
        'input_ids': torch.randint(0, 1000, (2, 64)),
        'attention_mask': torch.ones(2, 64, dtype=torch.long),
        'labels': torch.randint(0, 1000, (2, 64)),
        'scalar_value': 42,
    }

    # With no group, should pass through unchanged
    result = shard_vlm_batch_for_ulysses_sp(batch, ulysses_group=None)
    assert result is batch, "Should pass through with no group"
    print("[PASS] SP batch sharding (no-op path)")


if __name__ == '__main__':
    test_opslot_binding()
    test_forward_pass()
    test_forward_deterministic()
    test_sp_batch_sharding()
    print("\n=== All tests passed! ===")
