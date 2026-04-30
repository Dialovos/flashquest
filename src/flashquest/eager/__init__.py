"""Eager (pure-PyTorch) Quest reference. Phase 1 milestone."""
from .attention import quest_eager_sdpa
from .criticality import page_scores
from .page_summary import compute_page_summary
from .selection import select_pages

__all__ = [
    "quest_eager_sdpa",
    "page_scores",
    "compute_page_summary",
    "select_pages",
]
