"""Speculative-decoding support (Phase 12).

Loads the EAGLE-3 draft head (``thoughtworks/Llama-3.2-3B-Instruct-Eagle3``)
and drives it from an external target model's fused hidden states. The draft
layer itself is the vendored EAGLE-3 inference ``Model`` class
(``vendor/eagle/eagle/model/cnets.py``); we reuse it rather than reimplement.
"""
from .eagle_draft import EAGLE3_FUSION_LAYERS, EagleDraft, load_eagle3_draft

__all__ = ["EagleDraft", "load_eagle3_draft", "EAGLE3_FUSION_LAYERS"]
