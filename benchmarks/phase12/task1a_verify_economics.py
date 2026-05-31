"""Phase 12 gate 1a — verifier economics. Times the S_q>1 verify kernel vs the
S_q=1 decode kernel on a synthetic 32k INT4 cache at the Llama-3.2-3B shape.
Per-launch ratio is layer-count-invariant (both scale x28), so one layer suffices.
PASS iff ratio_q4 <= 1.6 OR ratio_q8 <= 2.4."""
import math, json, statistics
from pathlib import Path
import torch
from flashquest.kernel.kv_quant import quantize_k_int4, quantize_v_int4
from flashquest.kernel.sparse_int4_fwd_compact import (
    flash_attn_sparse_int4_fwd_compact, flash_attn_sparse_int4_fwd_compact_sq)

def _time(fn, iters=50, warmup=15):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)

def main():
    torch.manual_seed(0); dev = "cuda"
    B, Hq, Hkv, D, page_size = 1, 24, 8, 128, 64          # Llama-3.2-3B
    ctx = 32768; n_pages = ctx // page_size
    retention = 0.20
    k_pages = math.ceil(retention * n_pages) + 4 + 2       # +sinks +window  (=109)
    K = torch.randn(B, Hkv, ctx, D, dtype=torch.bfloat16, device=dev)
    V = torch.randn(B, Hkv, ctx, D, dtype=torch.bfloat16, device=dev)
    Kp, Ks, Kmn = quantize_k_int4(K, page_size=page_size); Vp, Vs, Vmn = quantize_v_int4(V)
    del K, V; torch.cuda.empty_cache()
    sel1 = torch.arange(k_pages, device=dev).int().view(1, 1, k_pages).expand(B, Hq, k_pages).contiguous()
    torch.cuda.reset_peak_memory_stats()
    Q1 = torch.randn(B, Hq, 1, D, dtype=torch.bfloat16, device=dev)
    t1 = _time(lambda: flash_attn_sparse_int4_fwd_compact(
        Q1, Kp, Ks, Kmn, Vp, Vs, Vmn, selected_page_ids=sel1, page_size=page_size))
    out = {"shape": "Llama-3.2-3B (Hq=24,Hkv=8,D=128)", "ctx": ctx, "retention": retention,
           "k_pages": int(k_pages), "T_decode_q1_ms": t1}
    for q in (2, 4, 8):
        Qn = torch.randn(B, Hq, q, D, dtype=torch.bfloat16, device=dev)
        tn = _time(lambda: flash_attn_sparse_int4_fwd_compact_sq(
            Qn, Kp, Ks, Kmn, Vp, Vs, Vmn, selected_page_ids=sel1, page_size=page_size))
        out[f"T_verify_q{q}_ms"] = tn; out[f"ratio_q{q}"] = tn / t1
    out["peak_mib"] = torch.cuda.max_memory_allocated() / 2**20
    out["PASS_1a"] = bool(out["ratio_q4"] <= 1.6 or out["ratio_q8"] <= 2.4)
    Path("benchmarks/phase12").mkdir(parents=True, exist_ok=True)
    Path("benchmarks/phase12/task1a.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out

if __name__ == "__main__":
    main()
