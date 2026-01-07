#!/usr/bin/env python3
"""
Download models or datasets from HuggingFace Hub or ModelScope to a user-defined folder.

Usage:
    # Download model from HuggingFace
    python download.py \
        --repo_id lerobot/pi0_base \
        --output_dir ~/models \
        --source huggingface
    # Downloads to: ~/models/lerobot/pi0_base

    # Download dataset from HuggingFace
    python download.py \
        --repo_id lerobot/aloha_mobile_cabinet \
        --output_dir ~/datasets \
        --repo_type dataset \
        --source huggingface
    # Downloads to: ~/datasets/lerobot/aloha_mobile_cabinet
"""

import argparse
import sys

from pathlib import Path
from typing import Optional


def download_from_huggingface(
    repo_id: str,
    output_dir: Path,
    repo_type: str = "model",
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> Path:
    """Download model or dataset from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repository ID (e.g., "lerobot/pi0_base")
        output_dir: Base directory to save the repository
            (will be saved to output_dir/repo_id)
        repo_type: Type of repository - "model" or "dataset" (default: "model")
        revision: Git revision (branch, tag, or commit hash). Defaults to "main"
        token: HuggingFace token for private repos. If None, uses cached token

    Returns:
        Path to downloaded repository directory
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface_hub is not installed.")
        print("Install it with: pip install huggingface_hub")
        sys.exit(1)

    final_output_dir = output_dir / repo_id

    print(f"Downloading {repo_type} {repo_id} from HuggingFace Hub...")
    print(f"Output directory: {final_output_dir}")

    final_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            local_dir=str(final_output_dir),
            local_dir_use_symlinks=False,
            token=token,
        )
        print(f"✓ Successfully downloaded to: {downloaded_path}")
        return Path(downloaded_path)
    except Exception as e:
        print(f"✗ Error downloading from HuggingFace: {e}")
        if "401" in str(e) or "authentication" in str(e).lower():
            print("\nTip: You may need to set a HuggingFace token:")
            print("  export HF_TOKEN=your_token_here")
            print("  or run: huggingface-cli login")
        elif "404" in str(e):
            print(f"\nTip: Repository '{repo_id}' not found. Check the repo ID.")
        sys.exit(1)


def download_from_modelscope(
    repo_id: str, output_dir: Path, repo_type: str = "model", revision: Optional[str] = None
) -> Path:
    """Download model or dataset from ModelScope.

    Args:
        repo_id: ModelScope repository ID (e.g., "lerobot/pi0_base")
        output_dir: Base directory to save the repository
            (will be saved to output_dir/repo_id)
        repo_type: Type of repository - "model" or "dataset" (default: "model")
        revision: Git revision (branch, tag, or commit hash). Defaults to "master"

    Returns:
        Path to downloaded repository directory
    """
    try:
        from modelscope import snapshot_download as ms_snapshot_download
    except ImportError:
        print("Error: modelscope is not installed.")
        print("Install it with: pip install modelscope")
        sys.exit(1)

    final_output_dir = output_dir / repo_id

    print(f"Downloading {repo_type} {repo_id} from ModelScope...")
    print(f"Output directory: {final_output_dir}")

    final_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        cache_dir = output_dir.parent if output_dir.parent != Path.cwd() else None
        downloaded_path = ms_snapshot_download(
            model_id=repo_id, cache_dir=str(cache_dir) if cache_dir else None, revision=revision
        )

        downloaded_path = Path(downloaded_path)

        if downloaded_path != final_output_dir:
            import shutil

            if final_output_dir.exists():
                print(f"Output directory {final_output_dir} already exists. Removing...")
                shutil.rmtree(final_output_dir)
            print(f"Copying from cache to {final_output_dir}...")
            shutil.copytree(downloaded_path, final_output_dir)
            print(f"✓ Successfully downloaded to: {final_output_dir}")
            return final_output_dir
        else:
            print(f"✓ Successfully downloaded to: {downloaded_path}")
            return downloaded_path

    except Exception as e:
        print(f"✗ Error downloading from ModelScope: {e}")
        if "401" in str(e) or "authentication" in str(e).lower():
            print("\nTip: You may need to set ModelScope credentials:")
            print("  export MODELSCOPE_API_TOKEN=your_token_here")
        elif "404" in str(e) or "not found" in str(e).lower():
            print(f"\nTip: Repository '{repo_id}' not found on ModelScope.")
            print("  Note: ModelScope repo IDs may differ from HuggingFace.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download models or datasets from HuggingFace Hub or ModelScope",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download model from HuggingFace (saves to ~/models/lerobot/pi0_base)
  python download.py --repo_id lerobot/pi0_base \\
      --output_dir ~/models --source huggingface

  # Download dataset from HuggingFace (saves to ~/datasets/lerobot/aloha_mobile_cabinet)
  python download.py --repo_id lerobot/aloha_mobile_cabinet \\
      --output_dir ~/datasets --repo_type dataset --source huggingface

  # Download from ModelScope (China users, saves to ~/models/lerobot/pi0_base)
  python download.py --repo_id lerobot/pi0_base \\
      --output_dir ~/models --source modelscope

  # Download tokenizer (saves to ~/models/google/paligemma-3b-pt-224)
  python download.py --repo_id google/paligemma-3b-pt-224 \\
      --output_dir ~/models --source huggingface

Note: For private repositories, set HF_TOKEN environment variable:
  export HF_TOKEN=your_token_here
        """,
    )

    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Repository ID (e.g., 'lerobot/pi0_base' or 'lerobot/aloha_mobile_cabinet')",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help=(
            "Base output directory (repository will be saved to output_dir/repo_id, "
            "e.g., '~/models' -> '~/models/lerobot/pi0_base')"
        ),
    )

    parser.add_argument(
        "--repo_type",
        type=str,
        choices=["model", "dataset"],
        default="model",
        help="Type of repository: 'model' or 'dataset' (default: model)",
    )

    parser.add_argument(
        "--source",
        type=str,
        choices=["huggingface", "modelscope"],
        default="huggingface",
        help="Source to download from: 'huggingface' or 'modelscope' (default: huggingface)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()

    if args.source == "huggingface":
        downloaded_path = download_from_huggingface(
            repo_id=args.repo_id,
            output_dir=output_dir,
            repo_type=args.repo_type,
            revision=None,
            token=None,
        )
    elif args.source == "modelscope":
        downloaded_path = download_from_modelscope(
            repo_id=args.repo_id, output_dir=output_dir, repo_type=args.repo_type, revision=None
        )
    else:
        raise ValueError(f"Unknown source: {args.source}")

    repo_type_name = "Dataset" if args.repo_type == "dataset" else "Model"
    print(f"\n{repo_type_name} downloaded successfully to: {downloaded_path}")
    print("You can now use this path in your config file:")
    if args.repo_type == "dataset":
        print(f"  data_path: {downloaded_path}")
    else:
        print(f"  checkpoint_dir: {downloaded_path}")


if __name__ == "__main__":
    main()
