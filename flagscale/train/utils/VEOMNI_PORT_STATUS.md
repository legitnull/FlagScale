# VeOmni Ulysses SP Port Status

## Files Copied (2025-06-04)

### From VeOmni → flagscale/train/utils/

1. **sequence_parallel/** (directory)
   - `__init__.py` - Main exports
   - `comm.py` - Process group management for Ulysses/CP/Unified SP
   - `ulysses.py` - Core Ulysses all-to-all primitives (_SeqAllToAll, gather_heads_scatter_seq, etc.)
   - `utils.py` - Padding/unpadding utilities for SP
   - `data.py` - Input slicing and output gathering for SP
   - `loss.py` - ReduceLoss autograd function for SP-aware loss computation
   - `async_ulysses.py` - Async Ulysses primitives for overlapping comm/compute
   - `async_ulysses_dit.py` - Async Ulysses for DiT models

2. **parallel_state.py**
   - Global parallel state manager (DataParallelState dataclass)
   - Manages DP/FSDP/SP/Ulysses/CP groups and ranks

3. **patched_modeling_qwen3_5_gpu.py** (2973 lines)
   - Full VeOmni patched Qwen3.5 model with Ulysses SP support
   - Contains `Qwen3_5GatedDeltaNet` with SP-aware forward pass (lines ~660-830)
   - VLM integration with SP slicing/gathering

## Import Dependencies That Need Fixing

### parallel_state.py imports:
- `from ..utils import logging` → needs flagscale logging
- `from ..utils.device import get_device_type` → needs device util

### sequence_parallel/data.py imports:
- `from ...distributed.parallel_state import get_parallel_state` → fix relative import

### sequence_parallel/ulysses.py imports:
- `from ...utils.device import get_device_id` → needs device util

### sequence_parallel/async_ulysses.py imports:
- `from veomni.utils.device import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE` → needs device flags

### patched_modeling_qwen3_5_gpu.py imports:
- `from veomni.distributed.parallel_state import get_parallel_state`
- `from veomni.distributed.sequence_parallel import gather_outputs, slice_input_tensor, sp_pad_and_slice`
- `from veomni.distributed.sequence_parallel.ulysses import gather_heads_scatter_seq, gather_seq_scatter_heads`
- `from veomni.utils.constants import IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX`
- `from veomni.utils.device import get_device_id`
- `from veomni.utils.model_outputs import CausalLMOutputWithLogProbs, FusedLinearAuxOutputMixin`
- `from veomni.ops.dispatch import OpSlot`

## Missing VeOmni Modules Needed

1. **veomni/utils/device.py**
   - `get_device_type()`, `get_device_id()`, `IS_CUDA_AVAILABLE`, `IS_NPU_AVAILABLE`

2. **veomni/utils/constants.py**
   - `IMAGE_INPUT_INDEX`, `VIDEO_INPUT_INDEX`

3. **veomni/utils/model_outputs.py**
   - `CausalLMOutputWithLogProbs`, `FusedLinearAuxOutputMixin`

4. **veomni/ops/dispatch.py**
   - `OpSlot` - OpSlot dispatch system for kernel selection

5. **veomni/utils/logging.py** (or use flagscale logger)

## Next Steps (to be done by user)

1. **Option A: Copy missing VeOmni utilities**
   - Copy `veomni/utils/device.py`, `constants.py`, `model_outputs.py`
   - Copy `veomni/ops/dispatch.py` and its dependencies
   - Fix all imports to point to copied files

2. **Option B: Create minimal adapters**
   - Create minimal `device_utils.py` in flagscale that implements just the needed functions
   - Create minimal `model_outputs.py` with the required dataclasses
   - Stub out or remove OpSlot dependencies if not needed

3. **Fix import paths**
   - Replace all `from veomni.*` imports with `from flagscale.train.utils.*`
   - Fix relative imports (e.g., `from ...distributed.parallel_state` → `from flagscale.train.utils.parallel_state`)

4. **Integrate into train_qwen3_5_gr00t.py**
   - Remove old `context_parallel` imports (already deleted)
   - Import from new `sequence_parallel` module
   - Initialize Ulysses SP groups using `init_sequence_parallel()`
   - Modify forward pass to use Ulysses all-to-all instead of gather/scatter

## Test File Reference

The VeOmni test file shows expected usage:
- `/share/project/fengyupu/github/infra_workspace/third_party/VeOmni/tests/parallel/ulysses/test_qwen3_5_gated_deltanet_ulysses.py`
