"""Context Parallelism utilities for hybrid attention models.

Supports:
- FLA native CP for GatedDeltaNet (linear attention) layers
- Gather/scatter CP for full attention layers
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from fla.ops.cp.context import build_cp_context


class GatherAlongSeq(torch.autograd.Function):
    """All-gather along sequence dimension with autograd support.

    Forward: all_gather [B, S_local, D] -> [B, S_total, D]
    Backward: reduce_scatter (sum gradients across ranks, return local shard)
    """

    @staticmethod
    def forward(ctx, input_tensor, group):
        ctx.group = group
        world_size = dist.get_world_size(group)
        ctx.world_size = world_size

        gathered = [torch.empty_like(input_tensor) for _ in range(world_size)]
        dist.all_gather(gathered, input_tensor.contiguous(), group=group)
        return torch.cat(gathered, dim=1)

    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        world_size = ctx.world_size

        # reduce_scatter: sum across ranks and scatter
        # grad_output shape: [B, S_total, D] -> split into world_size chunks
        # Each rank gets the sum of its chunk's gradients from all ranks
        grad_input = torch.zeros_like(grad_output.chunk(world_size, dim=1)[0])
        grad_chunks = list(grad_output.chunk(world_size, dim=1))
        dist.reduce_scatter(grad_input, [c.contiguous() for c in grad_chunks],
                           op=dist.ReduceOp.SUM, group=group)
        return grad_input, None


def gather_along_seq(tensor, group):
    """All-gather tensor along sequence dimension with autograd support."""
    return GatherAlongSeq.apply(tensor, group)


def patch_gated_delta_net_for_cp(model, cp_group, conv1d_kernel_size=4):
    """Patch GatedDeltaNet layers to use FLA's native CP.

    FLA's CP splits a single sequence across ranks. For batched inputs [B, S_local, D],
    we process each batch element independently: each is treated as one sequence of
    S_full = S_local * world_size tokens, with this rank holding its local shard.

    build_cp_context expects GLOBAL (pre-shard) cu_seqlens=[0, S_full] and partitions
    internally, giving each rank its local portion.

    Args:
        model: Qwen3_5Gr00tForCausalLM model
        cp_group: ProcessGroup for context parallelism
        conv1d_kernel_size: Kernel size for conv1d (Qwen3.5 uses 4)
    """
    if cp_group is None:
        return

    layer_types = model.config.layer_types
    language_model = model.language_model if hasattr(model, "language_model") else model

    patched_count = 0
    for i, layer in enumerate(language_model.layers):
        if layer_types[i] != "linear_attention":
            continue

        linear_attn = layer.linear_attn
        original_forward = linear_attn.forward

        def make_patched_forward(attn_module, group, kernel_size):
            def patched_forward(hidden_states, cache_params=None, attention_mask=None):
                B, S_local, D = hidden_states.shape
                world_size = dist.get_world_size(group)
                S_full = S_local * world_size

                from transformers.models.qwen3_5.modeling_qwen3_5 import apply_mask_to_padding_states
                from fla.modules.conv.causal_conv1d import causal_conv1d

                hidden_states_masked = apply_mask_to_padding_states(hidden_states, attention_mask)

                # Single-sequence global cu_seqlens — FLA partitions internally
                cu_seqlens = hidden_states.new_tensor([0, S_full], dtype=torch.long)
                cp_ctx = build_cp_context(
                    cu_seqlens=cu_seqlens,
                    group=group,
                    conv1d_kernel_size=kernel_size,
                )

                outputs = []
                for b_idx in range(B):
                    h = hidden_states_masked[b_idx:b_idx+1]  # [1, S_local, D]

                    mixed_qkv = attn_module.in_proj_qkv(h)  # [1, S_local, D_qkv]
                    z = attn_module.in_proj_z(h)
                    z = z.reshape(1, S_local, -1, attn_module.head_v_dim)
                    b_proj = attn_module.in_proj_b(h)
                    a = attn_module.in_proj_a(h)

                    mixed_qkv = causal_conv1d(
                        x=mixed_qkv,
                        weight=attn_module.conv1d.weight.squeeze(1),
                        bias=attn_module.conv1d.bias,
                        activation=attn_module.activation,
                        cp_context=cp_ctx,
                    )[0]

                    query, key, value = torch.split(
                        mixed_qkv,
                        [attn_module.key_dim, attn_module.key_dim, attn_module.value_dim],
                        dim=-1,
                    )

                    query = query.reshape(1, S_local, -1, attn_module.head_k_dim)
                    key = key.reshape(1, S_local, -1, attn_module.head_k_dim)
                    value = value.reshape(1, S_local, -1, attn_module.head_v_dim)

                    beta = b_proj.sigmoid()
                    g = -attn_module.A_log.float().exp() * F.softplus(a.float() + attn_module.dt_bias)

                    if attn_module.num_v_heads // attn_module.num_k_heads > 1:
                        query = query.repeat_interleave(
                            attn_module.num_v_heads // attn_module.num_k_heads, dim=2
                        )
                        key = key.repeat_interleave(
                            attn_module.num_v_heads // attn_module.num_k_heads, dim=2
                        )

                    core_attn_out, _ = attn_module.chunk_gated_delta_rule(
                        query, key, value,
                        g=g, beta=beta,
                        initial_state=None,
                        output_final_state=False,
                        use_qk_l2norm_in_kernel=True,
                        cp_context=cp_ctx,
                    )

                    core_attn_out = attn_module.norm(core_attn_out, z)
                    out = attn_module.out_proj(core_attn_out.reshape(1, S_local, -1))
                    outputs.append(out)

                return torch.cat(outputs, dim=0)  # [B, S_local, D]

            return patched_forward

        linear_attn.forward = make_patched_forward(linear_attn, cp_group, conv1d_kernel_size)
        patched_count += 1

    return patched_count


def _warmup_triton_kernels(language_model, cp_group, conv1d_kernel_size):
    """Run a dummy forward pass to trigger Triton autotuning.

    Triton's @autotune decorator benchmarks multiple kernel configs on the first
    call with new shapes. During benchmarking, the output buffer gets overwritten
    by non-winning configs. This warmup ensures autotuning completes before any
    real computation.
    """
    import torch.distributed as dist
    from fla.modules.conv.causal_conv1d import causal_conv1d

    layer_types = language_model.config.layer_types
    world_size = dist.get_world_size(cp_group)

    for i, layer in enumerate(language_model.layers):
        if layer_types[i] != "linear_attention":
            continue
        attn_module = layer.linear_attn
        # Use a small dummy input that matches expected shapes
        device = next(attn_module.parameters()).device
        dtype = next(attn_module.parameters()).dtype
        S_local = 64  # minimal length for warmup
        S_full = S_local * world_size
        dummy_h = torch.zeros(1, S_local, language_model.config.hidden_size, device=device, dtype=dtype)

        cu_seqlens = dummy_h.new_tensor([0, S_full], dtype=torch.long)
        cp_ctx = build_cp_context(
            cu_seqlens=cu_seqlens,
            group=cp_group,
            conv1d_kernel_size=conv1d_kernel_size,
        )

        with torch.no_grad():
            mixed_qkv = attn_module.in_proj_qkv(dummy_h)
            _ = causal_conv1d(
                x=mixed_qkv,
                weight=attn_module.conv1d.weight.squeeze(1),
                bias=attn_module.conv1d.bias,
                activation=attn_module.activation,
                cp_context=cp_ctx,
            )
        break  # all layers share the same kernel shapes


def _patch_gated_delta_net_no_cp(model, conv1d_kernel_size=4):
    """Patch GatedDeltaNet layers to use FLA's kernels WITHOUT CP.

    Same code path as the CP version but without any distributed communication.
    Used as a baseline for gradient comparison tests.
    """
    layer_types = model.config.layer_types
    language_model = model.language_model if hasattr(model, "language_model") else model

    for i, layer in enumerate(language_model.layers):
        if layer_types[i] != "linear_attention":
            continue

        linear_attn = layer.linear_attn

        def make_patched_forward(attn_module):
            def patched_forward(hidden_states, cache_params=None, attention_mask=None):
                B, S, D = hidden_states.shape

                from transformers.models.qwen3_5.modeling_qwen3_5 import apply_mask_to_padding_states
                from fla.modules.conv.causal_conv1d import causal_conv1d

                hidden_states_masked = apply_mask_to_padding_states(hidden_states, attention_mask)

                outputs = []
                for b_idx in range(B):
                    h = hidden_states_masked[b_idx:b_idx+1]  # [1, S, D]

                    mixed_qkv = attn_module.in_proj_qkv(h)
                    z = attn_module.in_proj_z(h)
                    z = z.reshape(1, S, -1, attn_module.head_v_dim)
                    b_proj = attn_module.in_proj_b(h)
                    a = attn_module.in_proj_a(h)

                    # Use FLA's causal_conv1d WITHOUT cp_context
                    mixed_qkv = causal_conv1d(
                        x=mixed_qkv,
                        weight=attn_module.conv1d.weight.squeeze(1),
                        bias=attn_module.conv1d.bias,
                        activation=attn_module.activation,
                    )[0]

                    query, key, value = torch.split(
                        mixed_qkv,
                        [attn_module.key_dim, attn_module.key_dim, attn_module.value_dim],
                        dim=-1,
                    )

                    query = query.reshape(1, S, -1, attn_module.head_k_dim)
                    key = key.reshape(1, S, -1, attn_module.head_k_dim)
                    value = value.reshape(1, S, -1, attn_module.head_v_dim)

                    beta = b_proj.sigmoid()
                    g = -attn_module.A_log.float().exp() * F.softplus(a.float() + attn_module.dt_bias)

                    if attn_module.num_v_heads // attn_module.num_k_heads > 1:
                        query = query.repeat_interleave(
                            attn_module.num_v_heads // attn_module.num_k_heads, dim=2
                        )
                        key = key.repeat_interleave(
                            attn_module.num_v_heads // attn_module.num_k_heads, dim=2
                        )

                    core_attn_out, _ = attn_module.chunk_gated_delta_rule(
                        query, key, value,
                        g=g, beta=beta,
                        initial_state=None,
                        output_final_state=False,
                        use_qk_l2norm_in_kernel=True,
                    )

                    core_attn_out = attn_module.norm(core_attn_out, z)
                    out = attn_module.out_proj(core_attn_out.reshape(1, S, -1))
                    outputs.append(out)

                return torch.cat(outputs, dim=0)

            return patched_forward

        linear_attn.forward = make_patched_forward(linear_attn)

def patch_full_attention_gather_scatter(model, cp_group):
    """Apply gather/scatter to full attention layers (simpler than ring attention).

    Gathers hidden_states and position_embeddings across CP ranks before the
    attention computation, then scatters the output back to local shards.

    Args:
        model: Qwen3_5Gr00tForCausalLM model
        cp_group: ProcessGroup for context parallelism
    """
    if cp_group is None:
        return

    layer_types = model.config.layer_types
    language_model = model.language_model if hasattr(model, "language_model") else model

    patched_count = 0
    for i, layer in enumerate(language_model.layers):
        if layer_types[i] != "full_attention":
            continue

        attn = layer.self_attn
        original_forward = attn.forward

        def make_wrapped(orig_fwd, group):
            def wrapped(hidden_states, position_embeddings=None, attention_mask=None, **kwargs):
                world_size = dist.get_world_size(group)
                rank = dist.get_rank(group)

                # All-gather hidden_states: [B, S/N, H] -> [B, S, H] (differentiable)
                full_hidden = gather_along_seq(hidden_states, group)

                # All-gather position_embeddings (cos, sin): [B, S/N, D] -> [B, S, D]
                # Position embeddings don't need gradients
                full_pos_emb = None
                if position_embeddings is not None:
                    cos, sin = position_embeddings
                    full_cos = gather_along_seq(cos.detach(), group)
                    full_sin = gather_along_seq(sin.detach(), group)
                    full_pos_emb = (full_cos, full_sin)

                output = orig_fwd(full_hidden, full_pos_emb, attention_mask=None, **kwargs)

                if isinstance(output, tuple):
                    full_output = output[0]
                    local_output = full_output.chunk(world_size, dim=1)[rank].contiguous()
                    return (local_output,) + output[1:]
                else:
                    local_output = output.chunk(world_size, dim=1)[rank].contiguous()
                    return local_output

            return wrapped

        attn.forward = make_wrapped(original_forward, cp_group)
        patched_count += 1

    return patched_count


def apply_context_parallel_to_model(model, cp_group):
    """Apply CP patches to both linear and full attention layers.

    Args:
        model: Qwen35Gr00t policy model (wraps the VLM)
        cp_group: ProcessGroup for context parallelism

    Returns:
        Tuple of (linear_count, full_count) — number of patched layers
    """
    # Navigate to the inner language model: policy.vlm.model.model.language_model
    # The outer `model` is Qwen35Gr00t, which has .vlm (QwenVLBackbone wrapper)
    # .vlm.model is Qwen3_5ForConditionalGeneration
    # .vlm.model.model.language_model is the Qwen3_5Model with layers
    if hasattr(model, "vlm") and hasattr(model.vlm, "model"):
        if hasattr(model.vlm.model, "model") and hasattr(model.vlm.model.model, "language_model"):
            inner_model = model.vlm.model.model.language_model
        else:
            inner_model = model.vlm.model
    else:
        # Fallback: assume model is already the language model
        inner_model = model

    linear_count = patch_gated_delta_net_for_cp(inner_model, cp_group, conv1d_kernel_size=4)
    full_count = patch_full_attention_gather_scatter(inner_model, cp_group)
    return linear_count, full_count


def shard_vlm_batch_for_cp(vlm_batch, cp_group):
    """Shard VLM batch inputs along the sequence dimension for CP.

    Handles both 2D tensors [B, S] and 3D position_ids [3 or 4, B, S] for MRoPE.

    Args:
        vlm_batch: Dict with input_ids, attention_mask, labels, position_ids, etc.
        cp_group: ProcessGroup for context parallelism

    Returns:
        Sharded vlm_batch (mutated in-place and returned)
    """
    if cp_group is None or vlm_batch is None:
        return vlm_batch

    rank = dist.get_rank(cp_group)
    world_size = dist.get_world_size(cp_group)

    seq_keys = ["input_ids", "attention_mask", "labels", "position_ids", "mm_token_type_ids"]
    for key in seq_keys:
        if key in vlm_batch and vlm_batch[key] is not None:
            tensor = vlm_batch[key]

            # Handle 3D position_ids [3 or 4, B, S] for MRoPE
            if key == "position_ids" and tensor.ndim == 3:
                seq_dim = 2
                seq_len = tensor.shape[seq_dim]
            else:
                seq_dim = 1
                seq_len = tensor.shape[seq_dim]

            if seq_len % world_size != 0:
                pad_len = world_size - (seq_len % world_size)
                if key == "labels":
                    pad_value = -100
                elif key == "attention_mask":
                    pad_value = 0
                else:
                    pad_value = 0

                if seq_dim == 1:
                    tensor = F.pad(tensor, (0, pad_len), value=pad_value)
                else:  # seq_dim == 2 for 3D position_ids
                    tensor = F.pad(tensor, (0, pad_len), value=pad_value)

            vlm_batch[key] = tensor.chunk(world_size, dim=seq_dim)[rank].contiguous()

    return vlm_batch


def compute_cp_corrected_loss(vlm_loss, shift_labels, cp_group, ignore_index=-100):
    """Compute gradient-correct loss for context parallelism.

    Returns loss_for_backward (with correct gradient scaling) and loss_for_log
    (true global mean for logging).

    Args:
        vlm_loss: Local mean loss from chunked_cross_entropy_loss
        shift_labels: Shifted labels tensor (local shard)
        cp_group: ProcessGroup for context parallelism
        ignore_index: Label value to ignore

    Returns:
        (loss_for_backward, loss_for_log) tuple
    """
    if cp_group is None:
        return vlm_loss, vlm_loss

    local_valid = (shift_labels != ignore_index).sum().float()
    global_valid = local_valid.clone()
    dist.all_reduce(global_valid, op=dist.ReduceOp.SUM, group=cp_group)

    # vlm_loss = local_sum / local_valid
    # We need: local_sum / global_valid = vlm_loss * (local_valid / global_valid)
    # This ensures FSDP's gradient sum gives: Σ(local_sum_i) / global_valid = total_sum / global_valid
    if global_valid > 0:
        loss_for_backward = vlm_loss * (local_valid / global_valid)
    else:
        loss_for_backward = vlm_loss * 0.0

    with torch.no_grad():
        local_loss_sum = vlm_loss.detach() * local_valid
        dist.all_reduce(local_loss_sum, op=dist.ReduceOp.SUM, group=cp_group)
        loss_for_log = local_loss_sum / global_valid if global_valid > 0 else local_loss_sum

    return loss_for_backward, loss_for_log
