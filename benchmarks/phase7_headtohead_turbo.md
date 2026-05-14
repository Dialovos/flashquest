| Backend | Quant | 8 k decode tok/s | 32 k decode tok/s | 128 k fits? | Peak VRAM @ max fit |
|---|---|---|---|---|---|
| flashquest | AWQ-INT4 + TurboQuant K3-V3 paged KV + Quest top-k retention=0.25 | 2.05 | 1.93 | ✗ | 6105 MiB |
| llama.cpp | Q4_K_M, FP16 KV | 38.45 | ✗ (timeout > 30 min) | ✗ | n/a |
| vLLM 0.7.3 | AWQ-INT4, FP16 KV | OOM | ✗ (timeout) | OOM | n/a |
