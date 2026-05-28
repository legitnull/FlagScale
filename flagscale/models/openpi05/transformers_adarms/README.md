# Transformers ADARMS Patch

This directory contains a simplified import shim for the adarms (adaptive RMS norm) support in PI0.5 models.

## Setup

The actual patching is done by replacing transformers source files in the conda environment. Run:

```bash
bash scripts/transformers_adarms_patch.sh apply
```

This backs up the original files and copies the patched versions from the source repo.

## What Gets Patched

The script replaces 4 files in `transformers/models/`:
- `gemma/modeling_gemma.py` - Adds `GemmaRMSNorm` with `cond_dim` support, modifies decoder layer and model forward to handle `adarms_cond`
- `gemma/configuration_gemma.py` - Adds `use_adarms` and `adarms_cond_dim` config params
- `paligemma/modeling_paligemma.py` - Compatible with patched Gemma
- `siglip/modeling_siglip.py` - Compatible with patched vision tower

## Reverting

To restore the original transformers files:

```bash
bash scripts/transformers_adarms_patch.sh revert
```

## Status Check

To see if the patch is currently applied:

```bash
bash scripts/transformers_adarms_patch.sh status
```
