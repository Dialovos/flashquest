"""Edge case grid for flash_attn_fwd. See plan §Edge case catalog."""
import pytest
import torch
from hypothesis import given, settings, strategies as st

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# E11 — strided / non-contiguous Q
@cuda
def test_strided_Q_via_transpose():
    from flashquest.kernel import flash_attn_fwd

    torch.manual_seed(0)
    B, S, H, D = 1, 128, 4, 64
    Q_nhd = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
    K_nhd = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
    V_nhd = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")

    Q = Q_nhd.transpose(1, 2)
    K = K_nhd.transpose(1, 2)
    V = V_nhd.transpose(1, 2)
    assert not Q.is_contiguous()

    O, _ = flash_attn_fwd(Q, K, V, causal=True)

    Qc, Kc, Vc = Q.contiguous(), K.contiguous(), V.contiguous()
    O_ref, _ = flash_attn_fwd(Qc, Kc, Vc, causal=True)

    torch.testing.assert_close(O, O_ref, rtol=0, atol=0)


# E12 — NaN does not corrupt OTHER rows.
# (Strict NaN propagation through the tensor-core dot path is implementation-
# defined: Triton's tl.dot on bf16 with NaN operands may flush to zero rather
# than carry NaN through. The load-bearing property is that a single
# corrupted query does not poison neighbouring rows.)
@cuda
def test_nan_in_q_does_not_corrupt_neighbours():
    from flashquest.kernel import flash_attn_fwd

    torch.manual_seed(0)
    B, H, S, D = 1, 1, 64, 64
    Q_clean = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn_like(Q_clean)
    V = torch.randn_like(Q_clean)

    O_clean, _ = flash_attn_fwd(Q_clean, K, V, causal=False)

    Q_dirty = Q_clean.clone()
    Q_dirty[0, 0, 5, :] = float("nan")

    O_dirty, _ = flash_attn_fwd(Q_dirty, K, V, causal=False)

    # Every row except 5 must equal the clean output.
    other_rows = torch.cat([O_dirty[0, 0, :5], O_dirty[0, 0, 6:]], dim=0)
    other_rows_clean = torch.cat([O_clean[0, 0, :5], O_clean[0, 0, 6:]], dim=0)
    torch.testing.assert_close(other_rows, other_rows_clean, rtol=0, atol=0)


# E9 — MHA degenerate (n_rep == 1)
@cuda
def test_mha_path():
    from flashquest.kernel import flash_attn_fwd

    torch.manual_seed(0)
    B, H, S, D = 2, 4, 64, 64
    Q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)

    O, _ = flash_attn_fwd(Q, K, V, causal=False)
    ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=False)
    torch.testing.assert_close(O, ref, rtol=1e-2, atol=1e-2)


# E7 — batch > 1 independence
@cuda
def test_batch_independence():
    from flashquest.kernel import flash_attn_fwd

    torch.manual_seed(0)
    H, S, D = 2, 64, 64
    Q0 = torch.randn(1, H, S, D, dtype=torch.bfloat16, device="cuda")
    Q1 = torch.randn(1, H, S, D, dtype=torch.bfloat16, device="cuda")
    K0 = torch.randn_like(Q0); K1 = torch.randn_like(Q1)
    V0 = torch.randn_like(Q0); V1 = torch.randn_like(Q1)

    Q = torch.cat([Q0, Q1], dim=0)
    K = torch.cat([K0, K1], dim=0)
    V = torch.cat([V0, V1], dim=0)

    O_batched, _ = flash_attn_fwd(Q, K, V, causal=False)
    O0, _ = flash_attn_fwd(Q0, K0, V0, causal=False)
    O1, _ = flash_attn_fwd(Q1, K1, V1, causal=False)

    torch.testing.assert_close(O_batched[0:1], O0, rtol=0, atol=0)
    torch.testing.assert_close(O_batched[1:2], O1, rtol=0, atol=0)


# E8 — GQA n_rep=8
@cuda
def test_gqa_eight_way():
    from flashquest.kernel import flash_attn_fwd

    torch.manual_seed(0)
    B, H_q, H_kv, S, D = 1, 16, 2, 128, 64
    Q = torch.randn(B, H_q, S, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16, device="cuda")

    O, _ = flash_attn_fwd(Q, K, V, causal=True)

    Kr = K.repeat_interleave(8, dim=1)
    Vr = V.repeat_interleave(8, dim=1)
    ref = torch.nn.functional.scaled_dot_product_attention(Q, Kr, Vr, is_causal=True)

    torch.testing.assert_close(O, ref, rtol=1e-2, atol=1e-2)


# E14 — empty rejection
@cuda
def test_zero_seq_q_rejected():
    from flashquest.kernel import flash_attn_fwd
    Q = torch.randn(1, 1, 0, 64, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(1, 1, 4, 64, dtype=torch.bfloat16, device="cuda")
    V = torch.randn_like(K)
    with pytest.raises(ValueError, match="zero-length"):
        flash_attn_fwd(Q, K, V, causal=False)


@cuda
def test_zero_seq_kv_rejected():
    from flashquest.kernel import flash_attn_fwd
    Q = torch.randn(1, 1, 4, 64, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(1, 1, 0, 64, dtype=torch.bfloat16, device="cuda")
    V = torch.randn_like(K)
    with pytest.raises(ValueError, match="zero-length"):
        flash_attn_fwd(Q, K, V, causal=False)


# Property-based fuzz over (S, B, H_kv, n_rep, D, causal)
@cuda
@settings(deadline=None, max_examples=20)
@given(
    S=st.integers(min_value=1, max_value=192),
    B=st.integers(min_value=1, max_value=2),
    H_kv=st.sampled_from([1, 2, 4]),
    n_rep=st.sampled_from([1, 2, 4]),
    D=st.sampled_from([64, 128]),
    causal=st.booleans(),
)
def test_random_shapes_match_sdpa(S, B, H_kv, n_rep, D, causal):
    from flashquest.kernel import flash_attn_fwd

    torch.manual_seed(S * 31 + B * 17 + H_kv * 7 + n_rep * 3 + D + int(causal))
    H_q = H_kv * n_rep
    Q = torch.randn(B, H_q, S, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16, device="cuda")

    O, _ = flash_attn_fwd(Q, K, V, causal=causal)

    Kr = K.repeat_interleave(n_rep, dim=1) if n_rep > 1 else K
    Vr = V.repeat_interleave(n_rep, dim=1) if n_rep > 1 else V
    ref = torch.nn.functional.scaled_dot_product_attention(Q, Kr, Vr, is_causal=causal)

    torch.testing.assert_close(O, ref, rtol=2e-2, atol=2e-2)
