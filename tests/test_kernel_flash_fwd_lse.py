import math

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _ref_lse(Q, K, sm_scale, is_causal):
    """Reference logsumexp from raw QK; matches what FA-2 exposes.

    Causal convention matches the kernel: q_pos = (S_kv - S_q) + i.
    """
    qk = (Q.float() @ K.float().transpose(-2, -1)) * sm_scale
    if is_causal:
        S_q = Q.shape[-2]
        S_kv = K.shape[-2]
        m = torch.zeros(S_q, S_kv, device=qk.device, dtype=torch.bool)
        for i in range(S_q):
            qp = (S_kv - S_q) + i
            m[i, : qp + 1] = True
        qk = qk.masked_fill(~m, float("-inf"))
    return torch.logsumexp(qk, dim=-1)


@cuda
def test_lse_shape_and_value_non_causal():
    from flashquest.kernel import flash_attn_fwd

    torch.manual_seed(0)
    B, H, S, D = 1, 2, 64, 64
    sm_scale = 1.0 / math.sqrt(D)
    Q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")

    _, lse = flash_attn_fwd(Q, K, V, causal=False)

    assert lse.shape == (B, H, S)
    assert lse.dtype == torch.float32

    ref = _ref_lse(Q, K, sm_scale, is_causal=False)
    torch.testing.assert_close(lse, ref, rtol=1e-2, atol=1e-2)


@cuda
def test_lse_causal():
    from flashquest.kernel import flash_attn_fwd

    torch.manual_seed(1)
    B, H, S, D = 1, 2, 128, 64
    sm_scale = 1.0 / math.sqrt(D)
    Q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")

    _, lse = flash_attn_fwd(Q, K, V, causal=True)

    ref = _ref_lse(Q, K, sm_scale, is_causal=True)
    torch.testing.assert_close(lse, ref, rtol=1e-2, atol=1e-2)


@cuda
def test_lse_skipped_when_disabled():
    from flashquest.kernel import flash_attn_fwd

    Q = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16, device="cuda")
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)
    O, lse = flash_attn_fwd(Q, K, V, causal=False, return_lse=False)
    assert lse is None
