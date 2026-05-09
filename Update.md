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
