"""DuoAttention head split utilities. Phase 4."""
from .dispatch import quest_duo_eager_sdpa
from .pattern import load_duo_pattern

__all__ = ["load_duo_pattern", "quest_duo_eager_sdpa"]
