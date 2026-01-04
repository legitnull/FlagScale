#!/bin/bash
# Script to run the pi0 client using paths from examples/pi0_lerobot/conf/inference/pi0.yaml

set -e

# Values from examples/pi0_lerobot/conf/inference/pi0.yaml
BASE_IMG="/share/project/fengyupu/github/FlagScale/inference_inputs/frame_10_observation_images_cam_high.jpg"
LEFT_WRIST_IMG="/share/project/fengyupu/github/FlagScale/inference_inputs/frame_10_observation_images_cam_left_wrist.jpg"
RIGHT_WRIST_IMG="/share/project/fengyupu/github/FlagScale/inference_inputs/frame_10_observation_images_cam_right_wrist.jpg"
STATE_PATH="/share/project/fengyupu/github/FlagScale/inference_inputs/frame_10_state.pt"
TASK_PATH="/share/project/fengyupu/github/FlagScale/inference_inputs/frame_10_task.txt"

# Server settings
HOST="${1:-127.0.0.1}"
PORT="${2:-5000}"

# Read instruction from task file
INSTRUCTION=$(cat "$TASK_PATH")

# Run the client
python examples/pi0_lerobot/client_pi0.py \
    --host "$HOST" \
    --port "$PORT" \
    --base-img "$BASE_IMG" \
    --left-wrist-img "$LEFT_WRIST_IMG" \
    --right-wrist-img "$RIGHT_WRIST_IMG" \
    --state-path "$STATE_PATH" \
    --instruction "$INSTRUCTION"

