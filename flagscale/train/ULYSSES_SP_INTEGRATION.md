# Ulysses Sequence Parallelism Integration - Complete

## Summary

Successfully integrated Ulysses SP for Qwen3.5 GatedDeltaNet (linear attention) in the flagscale training framework. All VeOmni SP-related files copied, imports fixed, FLA kernels bound, and forward pass verified.

## Files Created/Modified

### Created Files
1. **`flagscale/train/utils/bind_ops.py`**
   - Binds OpSlots to FLA kernel implementations
   - Must be called before model instantiation

2. **`flagscale/train/utils/ulysses_sp_helpers.py`**
   - Replacement functions for the old context_parallel module
   - `apply_ulysses_sp_to_model()` - placeholder (SP handled in attention layers)
   - `patch_qwen3_5_model_for_vlm_ulysses_sp()` - placeholder
   - `shard_vlm_batch_for_ulysses_sp()` - slices batch tensors along seq dim
   - `compute_ulysses_sp_corrected_loss()` - all-reduces for correct gradient scaling

3. **`flagscale/train/utils/veomni_kernel_registry.py`**
   - Upgraded from stub to full registry with KernelSpec support
   - Registers FLA kernels for: rms_norm_gated, causal_conv1d, chunk_gated_delta_rule

4. **`flagscale/train/tests/test_qwen3_5_ulysses_sp_smoke.py`**
   - Smoke test validating OpSlot binding, forward pass, determinism
   - **All tests pass**

### Modified Files
1. **`flagscale/train/train_qwen3_5_gr00t.py`**
   - Added `from flagscale.train.utils.bind_ops import bind_qwen3_5_fla_ops`
   - Added `bind_qwen3_5_fla_ops()` call after dist.init_process_group
   - Updated to use new Ulysses SP helper functions

2. **`flagscale/train/utils/patched_modeling_qwen3_5_gpu.py`**
   - All VeOmni imports fixed to local relative imports
   - `accepts_precomputed_kwargs` made optional with no-op fallback

3. **`flagscale/train/utils/parallel_state.py`**
   - Fixed `from ..distributed.sequence_parallel` → `from .sequence_parallel`

4. **`flagscale/train/utils/sequence_parallel/ulysses.py`**
   - Fixed device import path

5. **`flagscale/train/utils/sequence_parallel/data.py`**
   - Fixed parallel_state import path

6. **`flagscale/train/utils/sequence_parallel/async_ulysses.py`**
   - Fixed device imports

7. **`flagscale/train/utils/veomni_device.py`**
   - Fixed logging imports with try/except

8. **`flagscale/train/utils/veomni_model_outputs.py`**
   - Wrapped all transformers model-specific imports in try/except

9. **`flagscale/train/utils/veomni_ops_dispatch.py`**
   - Fixed logging and kernel_registry imports

## Key Technical Details

### OpSlot Binding
- OpSlots are module-level globals in `patched_modeling_qwen3_5_gpu.py`
- Must call `bind_qwen3_5_fla_ops()` BEFORE model instantiation
- Binding happens at import time via direct module import (not importlib)

### FLA Kernels
All installed in conda env `flagscale-train-yupu-qwen35`:
- `fla-core 0.4.2`
- `flash-linear-attention 0.4.2`
- `causal_conv1d 1.6.1`

### Ulysses SP vs Context Parallelism
- **Old**: Context Parallelism (ring attention, deprecated)
- **New**: Ulysses SP (all-to-all based, head sharding)
- Ulysses handles SP inside attention layers via `gather_seq_scatter_heads` / `gather_heads_scatter_seq`

### Loss Correction
`compute_ulysses_sp_corrected_loss()` properly scales gradients:
1. Count valid (non-ignored) tokens per rank
2. All-reduce to get global count
3. Scale local loss by `local_count / global_count`

## Testing Status

✅ **Single-GPU Tests Pass**
- OpSlot binding: ✅
- Forward pass: ✅ (output shape [64, 128])
- Forward determinism: ✅
- SP batch sharding: ✅

⏳ **Multi-GPU Tests Pending**
- Requires 2+ GPUs to test actual Ulysses all-to-all collectives
- Pattern available in VeOmni: `tests/parallel/ulysses/test_qwen3_5_gated_deltanet_ulysses.py`

## Usage

```python
from flagscale.train.utils.bind_ops import bind_qwen3_5_fla_ops
bind_qwen3_5_fla_ops()  # Call BEFORE model creation

# Then create model as usual
policy = TrainablePolicy.from_config(config)
```

## Next Steps

1. **Multi-GPU validation**: Run VeOmni-style SP correctness test on 2+ GPUs
2. **Integration test**: Run actual training with `cp_degree > 1`
3. **Performance**: Profile all-to-all overhead vs sequence length
