# Update: FS_WM Multi-Image Event WM Training

## Summary

This update migrates the StarVLA-style multi-image event world-model training path into FS_WM with LeRobot 3.0 event data support. The current focus is the dataset + collate + launch configuration closure, so the model forward path can consume fixed Qwen image-query positions, long event groups, and future image inputs before larger distributed training changes are made.

## Code Changes

- Added tolerance for optional event hierarchy indices in `qwen_collate_fn`.
  - `task_index`, `subtask_index`, and `atomic_index` can now be `None` without dropping the whole sample.
  - Event payload fields remain optional, while unrelated `None` fields still trigger bad-sample filtering.
- Preserved the multi-image event collate outputs required by the new WM path:
  - `qwen_fixed_positions`
  - `qwen_long_event_groups`
  - `qwen_future_inputs`

## Training Config

- Added `examples/qwen3_5_gr00t/conf/train_multiimage_event.yaml` as the top-level launch config.
- Added `examples/qwen3_5_gr00t/conf/train/qwen3_5_gr00t_multiimage_event.yaml` as the train config.
- The new config uses:
  - `dataset_type: lerobot_multiimage_event`
  - `data_mix: ego1_multiimage`
  - `data_root_dir: /share/project/hotel/lerobot30_multiimage_data_1fps/event`
  - fixed Qwen layout with `num_query_tokens: 256`
  - previous, next, and half-event prompts for event prediction
  - two-node hostfile path `/share/project/mycao/wm_exp/hostfile.2`
  - experiment name `Fs_wm_multiimage_event_two_query`

## External Launch Script

A new helper script was created outside this repository:

```bash
/share/project/mycao/run_multiimage_event.sh
```

Useful commands:

```bash
/share/project/mycao/run_multiimage_event.sh run
/share/project/mycao/run_multiimage_event.sh run train.system.train_steps=2 train.system.num_workers=0
/share/project/mycao/run_multiimage_event.sh stop
```

Because this script is outside `/share/project/mycao/FS_WM`, it is documented here but not included in this git commit.

## Validation

Single-process smoke test completed earlier with a real batch and forward pass:

- produced `qwen_fixed_positions`
- produced long event groups for `subtask` and `atomic`
- produced `qwen_future_inputs`
- forward completed with losses including short future, previous event, and next event losses

Two-node launch smoke test was also run with:

```bash
/share/project/mycao/run_multiimage_event.sh run train.system.train_steps=2 train.system.num_workers=0
```

Observed results:

- `WORLD_SIZE=2`
- hostfile contained master and worker nodes, 8 GPUs each
- torchrun rendezvous completed for 16 ranks
- NCCL initialized successfully on both nodes
- Qwen weights loaded successfully on both nodes
- DynamicDiT initialized successfully on both nodes
- no `Traceback`, `ERROR`, `Exception`, or `DROP bad sample` was observed in the short test window

The two-node test was stopped after initialization verification to release the 16 GPUs. It did not wait long enough to record the first training step/loss line.

## Distributed Event Training Fixes and Full VLM Test

This round fixes the two-node distributed hang observed after enabling previous, next, and half-event prediction losses. The root cause was that different ranks could execute different event-loss graph paths when a batch had no valid samples for one event branch. That caused mismatched FSDP/NCCL collectives during backward.

### Model Forward Fixes

- Reworked fixed-layout query embedding insertion in `flagscale/models/vla/qwen3_5_gr00t/modeling_qwen3_5_gr00t.py` to avoid in-place writes into embedding views.
  - `_insert_query_embeddings()` now rebuilds each row with patched query segments and returns a stacked tensor.
  - `_run_fixed_layout_hidden()` clones token embeddings before replacing image/query regions.
- Kept event forward execution consistent across ranks when event losses are enabled.
  - `compute_event_loss()` no longer skips the whole event branch just because the local rank has an all-false sample mask.
  - `_masked_nfp_loss()` now returns a graph-connected zero when the sample mask is all false:
    - this preserves the event branch autograd graph and lets FSDP hooks run consistently.
- Added event-branch gating by loss weight.
  - Previous, next, and half-event branches are only skipped when their configured loss weight is `0.0`.
  - This keeps short-only tests cheap while preserving full-event distributed consistency when event weights are enabled.

### Dataset and Collate Fixes

- Updated `qwen_collate_fn()` in `flagscale/train/train_qwen3_5_gr00t.py` so every rank builds the same fixed event mode structure.
  - Event groups now use the fixed `LeRobotMultiImageEventDataset.EVENT_MODES` list instead of deriving modes from the current local batch.
- Always builds `qwen_half_event_inputs`, `qwen_half_event_fixed_positions`, and `qwen_half_event_target_inputs` in fixed-layout mode.
  - Invalid half-event samples use the current batch image as a dummy target.
  - `has_half_event` remains the loss mask, so invalid samples do not contribute to the half-event loss.

### Distributed Runtime Fixes

- Sets the local CUDA device before `dist.init_process_group()`.
- Passes `device_ids=[local_rank]` to distributed barriers in the training script.
- Removed temporary `[WM_TRACE]` debug prints after confirming the hang location.

### Validation

Two-node, 16-rank tests were run through `/share/project/mycao/run_multiimage_event.sh`.

1. Short/event-disabled smoke test:
   - Used event loss weights set to `0.0`.
   - Completed one training step successfully.

2. Full-event frozen-VLM/action test:
   - Used default previous, next, and half-event loss weights.
   - Completed one training step successfully.
   - Confirmed the earlier all-reduce/all-gather hang was fixed.

3. Full-event VLM-open test with visual encoder and action model frozen:
   - Override used:
     - `action_model\..*`
     - `vlm\.model\.model\.visual\..*`
   - Parameter summary confirmed:
     - `action_model: 0 trainable, 161,472,775 frozen`
     - `vlm: 4,205,751,296 trainable, 333,514,240 frozen`
   - Completed one full training step on 16 ranks:
     - `smpl:16`
     - `nfp_mse` around `0.35-0.36`
     - `nfp_cos` around `0.97-0.98`
     - `grdn:8.521`
     - `updt_s` around `26.6-29.1s`
   - Training ended with `Training completed`, and both nodes released all GPU memory afterward.
