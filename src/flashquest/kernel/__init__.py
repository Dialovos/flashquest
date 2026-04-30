"""flashquest Triton kernels. Phase 2: dense FA-2 forward. Phase 3: sparse INT8."""
from .flash_fwd import flash_attn_fwd
from .sparse_fwd import flash_attn_sparse_fwd

__all__ = ["flash_attn_fwd", "flash_attn_sparse_fwd"]
