| Backend | Quant | 8 k decode tok/s | 32 k decode tok/s | 128 k fits? | Peak VRAM @ max fit |
|---|---|---|---|---|---|
| flashquest | AWQ-INT4 + INT4 paged KV + Quest top-k retention=0.25 | 1.81 | OOM | ✗ | 4475 MiB |
| llama.cpp | Q4_K_M, FP16 KV | 40.43 | ✗ | ✗ | n/a |
| vLLM 0.7.3 | AWQ-INT4, FP16 KV | OOM | OOM | ✗ | n/a |