"""
Adaptive RMS norm (adarms) support for transformers Gemma models.

This module requires that the transformers package has been patched with the
adarms-modified source files. Run the patch script before using:

    bash scripts/transformers_adarms_patch.sh apply

The patched files replace:
  - transformers/models/gemma/modeling_gemma.py
  - transformers/models/gemma/configuration_gemma.py
  - transformers/models/paligemma/modeling_paligemma.py
  - transformers/models/siglip/modeling_siglip.py
"""

from transformers.models.gemma.modeling_gemma import GemmaRMSNorm


def _gated_residual(x, y, gate):
    if x is None and y is None:
        return None
    if x is None or y is None:
        return x if x is not None else y
    if gate is None:
        return x + y
    return x + y * gate


__all__ = ["GemmaRMSNorm", "_gated_residual"]
