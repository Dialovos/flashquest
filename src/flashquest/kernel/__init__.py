"""flashquest Triton kernels. Phase 2: dense FA-2 forward."""
from .flash_fwd import flash_attn_fwd

__all__ = ["flash_attn_fwd"]
