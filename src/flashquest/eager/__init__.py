"""Eager (pure-PyTorch) Quest reference. Phase 1 milestone."""
from .attention import quest_eager_sdpa
from .criticality import page_scores
from .page_summary import compute_page_summary
from .selection import select_pages
from .sparse_int8 import quest_eager_sparse_int8
from .streaming import streaming_eager_sdpa

__all__ = [
    "quest_eager_sdpa",
    "page_scores",
    "compute_page_summary",
    "select_pages",
    "quest_eager_sparse_int8",
    "streaming_eager_sdpa",
]
