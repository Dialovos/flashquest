"""DuoAttention head split utilities. Phase 4 + Phase 5."""
from .dispatch import quest_duo_eager_sdpa
from .fused_dispatch import quest_duo_fused_sdpa
from .pattern import load_duo_pattern

__all__ = ["load_duo_pattern", "quest_duo_eager_sdpa", "quest_duo_fused_sdpa"]
