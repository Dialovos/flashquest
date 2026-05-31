import math, torch, pytest
from flashquest.eager.llama_persistent_patch import _bf16_dense_attn_offset_causal_with_lse
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")

def _ref(Q, K, V, q_offset):
    B,H_q,S_q,D = Q.shape; H_kv = K.shape[1]; n_rep = H_q//H_kv
    Kf = K.repeat_interleave(n_rep,1).float(); Vf = V.repeat_interleave(n_rep,1).float()
    sm = 1/math.sqrt(D); O = torch.zeros(B,H_q,S_q,D)
    for b in range(B):
        for h in range(H_q):
            for i in range(S_q):
                lim = q_offset + i
                qk = (Q[b,h,i].float() @ Kf[b,h,:lim+1].T) * sm
                p = torch.softmax(qk, -1)
                O[b,h,i] = p @ Vf[b,h,:lim+1]
    return O
def test_offset_causal_matches_reference():
    torch.manual_seed(0)
    B,H_q,H_kv,D,S_q,q_offset = 1,8,2,128,4,5
    S_kv = q_offset + S_q
    Q = torch.randn(B,H_q,S_q,D,dtype=torch.bfloat16,device="cuda")
    K = torch.randn(B,H_kv,S_kv,D,dtype=torch.bfloat16,device="cuda")
    V = torch.randn(B,H_kv,S_kv,D,dtype=torch.bfloat16,device="cuda")
    O,lse = _bf16_dense_attn_offset_causal_with_lse(Q,K,V,q_offset)
    ref = _ref(Q,K,V,q_offset)
    assert O.shape==(B,H_q,S_q,D) and lse.shape==(B,H_q,S_q)
    assert torch.allclose(O.float(), ref.cuda(), atol=2e-2, rtol=2e-2)
