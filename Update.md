# FS_WM Update

## Summary

This version contains the current StarVLA-style multi-image event world-model training path in FS_WM, adapted to LeRobot 3.0 event data and validated for mixed VLA + VLM training on the 24-node job.

The active path supports:

- VLA multi-image event data from `/share/project/hotel/lerobot30_multiimage_data_1fps/event`
- VLM data mixed into the same training loop
- fixed Qwen image/query positions
- short future image prediction
- previous/next long-event prediction
- safe handling for missing event hierarchy indices
- distributed-consistent zero-loss paths for event branches with no valid local samples

## Main Changes

### Dataset and Collate

- Added LeRobot 3.0 multi-image event data support.
- Added optional handling for `task_index`, `subtask_index`, and `atomic_index`.
- Samples without `subtask_index` and `atomic_index` now train short future prediction only.
- Added collate outputs required by the event WM path:
  - `qwen_fixed_positions`
  - `qwen_long_event_groups`
  - `qwen_future_inputs`
- Added fixed-position checks for image, short query, instruction, and long query blocks.

### Event Prediction

- Removed half-event training as a semantic target.
- Set `half_event_loss_weight: 0.0` in the active config.
- Kept graph-connected zero paths so FSDP/NCCL ranks take consistent autograd paths when a branch has no valid local event samples.
- Logged short and long losses separately:
  - `short_q`
  - `prev_q`
  - `next_q`
  - `long_q`

### VLM Data

- Added token-length filtered HoneyData file:
  - `honey_data_rel_1071666_qwen35_tokenlen1024.json`
- Updated VLM config to use Qwen-token-length filtering with `model_max_length: 1024`.
- Observed that some VLM samples are still skipped at runtime when they exceed the token limit.

### Dataset Mix

The `ego1_multiimage` mix includes:

- `agibot_world`
- `EgoExoLearn`
- `GenEgoData`
- `something_something_v2`
- `hoi4d`
- `em_interaction`
- `ego4d`
- `epic_kitchens`
- `egocentric10k`
- `egodex`
- `kinetics`
- `llava-178k`
- `pe_video_extended`
- `pe_video_traintest`
- `psi_ego`

Some datasets have only one event and may not provide the newer event hierarchy keys; those samples are handled by the short-future path.

## Active Training Config

Top-level config:

```text
examples/qwen3_5_gr00t/conf/train_multiimage_event.yaml
```

Train config:

```text
examples/qwen3_5_gr00t/conf/train/qwen3_5_gr00t_multiimage_event.yaml
```

Current launch settings:

```yaml
experiment:
  exp_name: Fs_wm_multiimage_event_two_query_24node
  runner:
    hostfile: /share/project/mycao/wm_exp/hostfile.24

system:
  batch_size: 2
  vla_gradient_accumulation_steps: 2
  train_steps: 46500
  log_freq: 10
  num_workers: 4
  vlm_loss_scale: 0.4

checkpoint:
  save_freq: 5000

model:
  action_model:
    max_seq_len: 1024
  nfp:
    short_future_loss_weight: 0.1
    prev_event_loss_weight: 0.25
    next_event_loss_weight: 0.25
    half_event_loss_weight: 0.0
  optimizer:
    lr: 5.0e-05
    param_groups:
      vlm:
        lr: 1.5e-05
      action_model:
        lr: 1.0e-04

data:
  dataset_type: lerobot_multiimage_event
  data_mix: ego1_multiimage
  data_root_dir: /share/project/hotel/lerobot30_multiimage_data_1fps/event
  vlm_data:
    per_device_batch_size: 1
    model_max_length: 1024
```

Current intended loss balance:

```text
VLM : short : long = 0.4 : 0.1 : 0.5
```

The long-event weight is split across previous and next event prediction:

```yaml
prev_event_loss_weight: 0.25
next_event_loss_weight: 0.25
```

## 24-Node Run Status

Current 24-node launch uses:

```text
/share/project/mycao/wm_exp/hostfile.24
```

This gives:

```text
24 nodes * 8 GPUs = global_world_size 192
```

Gradient accumulation does not change `global_world_size`; it changes how many VLA micro-batches are accumulated per optimizer step.

The active training run has produced normal loss logs through at least step 80 and later continued past step 300. Typical steady-state timing is around 4.8-5.1 seconds per logged step, with occasional long-tail steps caused by data decode retries or distributed synchronization.

Observed runtime data issues:

- Some VLA video samples fail ffmpeg decode and are retried before being skipped.
- Some VLM samples exceed the 1024 token limit and are skipped.
- Decode retry failures can create long-tail steps because distributed training waits for the slowest rank.

## Notes

- `vit` and `action_model` remain frozen in the active training setup.
- VLM is trainable in the current mixed run.
- The current config targets one approximate epoch using VLA as the step-count baseline under the 24-node setting.
