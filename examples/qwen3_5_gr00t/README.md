# Qwen3.5-GR00T: Training with Qwen3.5-VL Backbone

This guide covers how to train Qwen3.5-GR00T models using FlagScale. Qwen3.5-GR00T uses a Qwen3.5-VL backbone as the vision-language model with a dynamic DiT-based flow matching action head.

## Key Differences from Qwen-GR00T

- Uses **Qwen3.5-VL** (instead of Qwen3-VL / Qwen2.5-VL) as the VLM backbone
- Requires `transformers>=5.5.0` for `Qwen3_5ForConditionalGeneration`
- Supports **NFP** (Next Frame Prediction) head for world model training
- Uses dynamic action head (`GR00TDynamicActionHead` with DiT-B architecture)
- Supports **VLM co-training** with separate VLM datasets

## Installation

### Clone Repository

```sh
git clone https://github.com/FlagOpen/FlagScale.git
cd FlagScale/
```

### Setup Conda Environment

Create a new conda environment for robotics training:

```sh
conda create -n flagos-robo python=3.12
conda activate flagos-robo
```

Install FlagScale and training dependencies:

```sh
cd FlagScale/
# "[cuda-train]" is for NVIDIA GPUs; replace with "[ascend-train]" on Huawei Ascend, or "[musa-train]" on Moore Threads MUSA
pip install ".[cuda-train]" --verbose
```

## Training

### Prepare Dataset

FlagScale uses the **LeRobotDataset v3.0** format. For detailed information about the format structure, see the [LeRobotDataset v3.0 documentation](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3).


### Configure Training

The default training config is at `examples/qwen3_5_gr00t/conf/train/qwen3_5_gr00t.yaml`.

Key config fields:

- `model.vlm.base_vlm`: Path to the Qwen3.5-VL model
- `model.vlm.type`: `qwen3.5-vl`
- `model.action_model.type`: `gr00t_dynamic_action_head`
- `model.nfp`: NFP head configuration (set to `null` to disable)
- `model.use_action_policy_loss`: Set `false` for Stage 1 (NFP pre-training), `true` for Stage 2 (action training)
- `model.freeze`: Module freezing patterns for staged training
- `data.data_path`: Path to the LeRobot VLA dataset
- `data.vlm_data`: VLM co-training data configuration

#### Training Stages

Qwen3.5-GR00T supports two-stage training:

**Stage 1 — NFP Pre-training:** Train NFP head + VLM language model, freeze action model and visual encoder.

```yaml
model:
  use_action_policy_loss: false
  freeze:
    freeze_patterns:
      - "action_model\\..*"
      - "vlm\\.model\\.model\\.visual\\..*"
```

**Stage 2 — Action Training:** Enable action policy loss, adjust freeze patterns as needed.

```yaml
model:
  use_action_policy_loss: true
  freeze:
    freeze_patterns:
      - "vlm\\.model\\.model\\.visual\\..*"
```

#### VLM Co-training

To enable VLM co-training, configure the `data.vlm_data` section:

```yaml
data:
  vlm_data:
    vlm_data_root: /path/to/vlm_data
    video_data_root: /path/to/video_data
    image_root: /path/to/image_data
    dataset_use: activitynetqa_0_30_s
    per_device_batch_size: 1
    model_max_length: 4608
    max_pixels: 307200
    min_pixels: 784
    video_max_frame_pixels: 65536
    video_min_frame_pixels: 784
    base_interval: 2
```

The VLM loss is scaled by `system.vlm_loss_scale` (default `0.1`).

### Quick Start

```sh
cd FlagScale/
flagscale train qwen3_5_gr00t -c ./examples/qwen3_5_gr00t/conf/train_multiimage_event.yaml
```

### Stop Training
```sh
cd FlagScale/
flagscale train qwen3_5_gr00t --stop
```
