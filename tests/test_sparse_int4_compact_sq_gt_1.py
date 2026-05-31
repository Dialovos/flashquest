import math
import torch
import pytest
from flashquest.kernel.kv_quant import quantize_k_int4, quantize_v_int4, dequantize_k_int4, dequantize_v_int4
from flashquest.kernel.sparse_int4_fwd_compact import flash_attn_sparse_int4_fwd_compact_sq

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

def _ref_sq(Q, K_packed, K_scale, K_mn, V_packed, V_scale, V_mn, sel, page_size):
    # Q: (B,Hq,Sq,D). sel: (B,Hq,BUCKET) int32, -1 sentinel. Non-causal over selected pages.
    B, Hq, Sq, D = Q.shape
    Kd = dequantize_k_int4(K_packed, K_scale, K_mn, page_size=page_size)  # (B,Hkv,Scompleted,D)
    Vd = dequantize_v_int4(V_packed, V_scale, V_mn)
    Hkv = Kd.shape[1]; n_rep = Hq // Hkv
    Kd = Kd.repeat_interleave(n_rep, dim=1); Vd = Vd.repeat_interleave(n_rep, dim=1)
    sm = 1.0 / math.sqrt(D)
    O = torch.zeros_like(Q.float())
    for b in range(B):
        for h in range(Hq):
            pages = [int(p) for p in sel[b, h].tolist() if p >= 0]
            idx = torch.cat([torch.arange(p*page_size, (p+1)*page_size) for p in pages]) if pages else torch.empty(0, dtype=torch.long)
            idx = idx[idx < Kd.shape[2]]
            if idx.numel() == 0:
                continue
            k = Kd[b, h, idx].float(); v = Vd[b, h, idx].float()       # (n,D)
            qk = (Q[b, h].float() @ k.T) * sm                           # (Sq,n)
            p = torch.softmax(qk, dim=-1)
            O[b, h] = p @ v
    return O.to(Q.dtype)

@pytest.mark.parametrize("Sq", [2, 4, 8])
def test_compact_sq_gt_1_matches_reference(Sq):
    torch.manual_seed(0)
    B, Hq, Hkv, D, page_size, n_pages = 1, 8, 2, 128, 64, 6
    Scompleted = n_pages * page_size
    Q = torch.randn(B, Hq, Sq, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, Hkv, Scompleted, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, Hkv, Scompleted, D, dtype=torch.bfloat16, device="cuda")
    K_packed, K_scale, K_mn = quantize_k_int4(K, page_size=page_size)
    V_packed, V_scale, V_mn = quantize_v_int4(V)
    sel = torch.full((B, Hq, 4), -1, dtype=torch.int32, device="cuda")
    sel[..., 0] = 0; sel[..., 1] = 2; sel[..., 2] = 4
    O, lse = flash_attn_sparse_int4_fwd_compact_sq(
        Q, K_packed, K_scale, K_mn, V_packed, V_scale, V_mn,
        selected_page_ids=sel, page_size=page_size, return_lse=True,
    )
    ref = _ref_sq(Q, K_packed, K_scale, K_mn, V_packed, V_scale, V_mn, sel, page_size)
    assert O.shape == (B, Hq, Sq, D)
    assert torch.allclose(O.float(), ref.float(), atol=2e-2, rtol=2e-2), (O.float()-ref.float()).abs().max()
    assert lse.shape == (B, Hq, Sq)
    assert not torch.any(torch.isnan(lse))
