"""Weight-only INT8 quantization for the EAGLE-3 draft head (Phase 12 task 12).

The bf16 draft head is ~486 MiB resident on-GPU; its ``nn.Linear`` weights
(q/k/v/o + gate/up/down + the ``fc`` 9216->3072 fusion + the big ``lm_head``
3072->32000) account for ~463 MiB of that. Weight-only INT8 halves those weights
to ~232 MiB — verified to free ~232 MiB resident (measured: spec peak 4759 ->
4530 MiB at 8k) while preserving acceptance exactly (1.684 == bf16, draft-token
argmax agreement 98.7% / pos0 100% for the per-channel backend).

Two backends are provided:

* ``"perchannel"`` (default) — a hand-rolled per-output-channel symmetric INT8
  ``nn.Linear`` (int8 weight + bf16 scale, dequant + a single bf16 GEMM). No
  extra dependency; higher fidelity than bnb (98.7% vs 95.4% argmax agreement);
  only one layer's bf16 weight is materialised at a time so allocator peak stays
  below the always-resident bf16 head.
* ``"bnb"`` — ``bitsandbytes.nn.Linear8bitLt`` (``has_fp16_weights=False``):
  a true int8 GEMM with an fp16 outlier sub-path (LLM.int8). bnb's path is
  fp16-typed, so we wrap each layer to cast input fp16 / output back to bf16,
  preserving the head's bf16 contract (norm/residuals/KV ``cat``). Note the
  Int8Params only quantize on the CPU->CUDA ``.to(cuda)`` transition — see the
  CPU-stage in ``_Bf16Int8Linear`` (a silent fp16-stays footgun otherwise).

MEASURED OUTCOME (Phase 12 task 12, RTX 3050 Ti 4 GB, this model):
the VRAM saving is real (~232 MiB), but **neither backend speeds up the draft**.
At batch=1/seq_len=1 the draft is a GEMV with no batch reuse, so per-channel
eager-dequant rereads/rewrites the full bf16 ``lm_head`` every token (~5x
slower: 89 ms -> 456 ms ``T_draft4`` at 4k) and bnb's throughput-tuned int8
kernels are launch-overhead-bound at one token (even slower). End-to-end the
spec-decode speedup *drops* (8k: 0.992x bf16 -> 0.606x int8; modeled 4k 1.22x ->
0.63x). Worse, the ~232 MiB head saving does **not** make 16k/32k fit — the
spec path OOMs in prefill on its own state (the full-sequence ``fused_seq`` +
the draft head's growing KV + the verify sandbox), which dwarfs the head-weight
saving. A fused low-bit GEMV kernel (Marlin / torchao tinygemm) is the only
escape hatch for the speed axis and is out of scope here. Kept opt-in for the
documented VRAM-vs-speed tradeoff and future kernel work; the bf16 default path
is unchanged.

Quantization is opt-in: ``load_eagle3_draft(..., quantize="int8")`` (or
``"int8:perchannel"`` / ``"int8:bnb"``) or the ``quantize_eagle_head(draft)``
helper.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Linear submodule names on the vendored EAGLE-3 ``Model`` to quantize. The
# frozen ``embed_tokens`` (an nn.Embedding shared with the target) is left bf16 —
# it is not a Linear and quantizing it would perturb the input embeddings.
_QUANT_TARGETS = (
    "midlayer.self_attn.q_proj",
    "midlayer.self_attn.k_proj",
    "midlayer.self_attn.v_proj",
    "midlayer.self_attn.o_proj",
    "midlayer.mlp.gate_proj",
    "midlayer.mlp.up_proj",
    "midlayer.mlp.down_proj",
    "fc",
    "lm_head",
)


class _PerChannelInt8Linear(nn.Module):
    """Hand-rolled per-output-channel symmetric weight-only INT8 ``nn.Linear``.

    Stores the weight as ``int8 [out, in]`` (1 byte) + a per-output-channel bf16
    ``scale [out]``; ``forward`` dequantizes to bf16 (``w_int8 * scale``) and runs
    a normal ``F.linear``. No bnb dependency, and — unlike a fp32 full-precision
    fallback — only *one* layer's bf16 weight is materialised at a time, so the
    allocator peak is ``int8_resident(all layers) + max single-layer bf16
    transient`` (~232 + 188 = ~420 MiB) which is still **below** the bf16 head's
    ~463 MiB always-resident Linears. The dequant+bf16-GEMM is also a single
    plain matmul (no LLM.int8 outlier decomposition), so at seq_len=1 it is far
    cheaper than bnb's int8 path. The int8 weight is the only persistent copy
    (the bf16 transient is freed each call).
    """

    def __init__(self, linear: nn.Linear, out_dtype: torch.dtype):
        super().__init__()
        self.out_dtype = out_dtype
        self.out_features = linear.out_features
        self.in_features = linear.in_features
        w = linear.weight.data.detach().to(torch.float32)  # [out, in]
        # per-output-channel symmetric scale: max|w| over the input dim / 127
        amax = w.abs().amax(dim=1).clamp_(min=1e-8)         # [out]
        scale = (amax / 127.0)                              # [out]
        q = torch.round(w / scale[:, None]).clamp_(-127, 127).to(torch.int8)
        self.register_buffer("weight_int8", q.contiguous())
        self.register_buffer("scale", scale.to(out_dtype).contiguous())
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.data.detach().to(out_dtype))
        else:
            self.bias = None

    @property
    def weight(self):  # introspection helper (matches an nn.Linear surface)
        return self.weight_int8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequant one layer's weight to bf16 (transient, freed after the call),
        # then a single plain GEMM. Keeps the bf16 I/O contract exactly.
        w = self.weight_int8.to(self.out_dtype) * self.scale[:, None]
        return torch.nn.functional.linear(x.to(self.out_dtype), w, self.bias)


class _Bf16Int8Linear(nn.Module):
    """``bitsandbytes`` ``Linear8bitLt`` wrapped to keep a bf16 I/O contract.

    bnb's int8 GEMM consumes/produces fp16; the EAGLE-3 head runs in bf16
    (norm, residual adds, KV ``cat``). We cast the input to fp16 going in and the
    output back to ``out_dtype`` (bf16) coming out, so the wrapped layer is a
    drop-in bf16 ``nn.Linear`` from the head's point of view. The weight is the
    int8 ``Int8Params`` resident on-GPU (half the bf16 footprint); no full bf16
    weight copy is ever materialised.
    """

    def __init__(self, linear: nn.Linear, out_dtype: torch.dtype):
        super().__init__()
        from bitsandbytes.nn import Linear8bitLt

        self.out_dtype = out_dtype
        # has_fp16_weights=False => store int8 weight + scale, true int8 GEMM.
        # threshold=6.0 is the LLM.int8 default outlier cutoff (keeps the few
        # large-magnitude feature columns in fp16 for accuracy).
        int8 = Linear8bitLt(
            linear.in_features, linear.out_features, bias=linear.bias is not None,
            has_fp16_weights=False, threshold=6.0,
        )
        with torch.no_grad():
            # Stage the trained weight as a *CPU* fp16 ``Int8Params``. bnb only
            # quantizes to int8 on the CPU->CUDA ``.to(cuda)`` transition; if the
            # source weight is already on CUDA the move is a no-op and the weight
            # stays fp16 (a silent footgun — the layer would run fp16, NOT int8).
            # Forcing a CPU stage guarantees the subsequent ``.to(dev)`` quantizes.
            int8.weight = type(int8.weight)(
                linear.weight.data.detach().to("cpu", torch.float16).contiguous(),
                requires_grad=False,
            )
            if linear.bias is not None:
                int8.bias = nn.Parameter(
                    linear.bias.data.detach().to("cpu", torch.float16),
                    requires_grad=False,
                )
        self.int8 = int8

    @property
    def weight(self):  # some callers introspect ``.weight`` (e.g. dtype checks)
        return self.int8.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.int8(x.to(torch.float16))
        return out.to(self.out_dtype)


def _get_submodule(root: nn.Module, dotted: str) -> tuple[nn.Module, str]:
    """Return ``(parent_module, attr_name)`` for a dotted path under ``root``."""
    parts = dotted.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def quantize_eagle_head(draft, *, backend: str = "perchannel",
                        verbose: bool = False) -> "object":
    """Replace the draft head's ``nn.Linear`` weights with weight-only INT8.

    Mutates ``draft.model`` in place (the q/k/v/o, gate/up/down, ``fc`` and
    ``lm_head`` Linears become weight-only INT8) and returns ``draft`` for
    chaining. The ``embed_tokens`` embedding and all ``LlamaRMSNorm`` layers stay
    bf16.

    Args:
        backend: ``"perchannel"`` (default) uses a hand-rolled per-output-channel
            symmetric INT8 ``nn.Linear`` (int8 weight + bf16 scale, dequant +
            single bf16 GEMM). No extra dependency, peak VRAM
            ``~int8_resident + one bf16 transient`` (~420 MiB < bf16's 463 MiB),
            and far cheaper than bnb at seq_len=1. ``"bnb"`` uses bitsandbytes
            ``Linear8bitLt`` (true int8 GEMM, lowest persistent footprint, but
            ~3x slower per draft step at seq_len=1 on this hardware — measured).
        verbose: print per-layer / summary footprint lines.
    """
    if backend not in ("perchannel", "bnb"):
        raise ValueError(f"quantize_eagle_head: unknown backend={backend!r} "
                         f"(supported: 'perchannel', 'bnb')")
    if backend == "bnb":
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as e:  # pragma: no cover - environment guard
            raise ImportError(
                "backend='bnb' needs bitsandbytes; install it with "
                "`pip install bitsandbytes --no-deps` (compatible with the "
                "pinned torch 2.5.1+cu121 stack), or use backend='perchannel'."
            ) from e

    model = draft.model
    dev = next(model.parameters()).device
    out_dtype = draft.dtype

    bf16_mib = 0.0
    int8_mib = 0.0
    for dotted in _QUANT_TARGETS:
        parent, attr = _get_submodule(model, dotted)
        lin = getattr(parent, attr)
        if not isinstance(lin, nn.Linear):
            # already quantized / unexpected — skip defensively
            continue
        bf16_mib += lin.weight.numel() * lin.weight.element_size() / 2**20
        if backend == "bnb":
            wrapped = _Bf16Int8Linear(lin, out_dtype=out_dtype).to(dev)
            w = wrapped.int8.weight
            if w.dtype != torch.int8:
                # The CPU-stage in _Bf16Int8Linear should guarantee the .to(dev)
                # above quantized to int8; bail loudly rather than silently
                # shipping an fp16 head that gives no VRAM saving.
                raise RuntimeError(
                    f"quantize_eagle_head: {dotted} did not quantize to int8 "
                    f"(weight dtype {w.dtype}); bnb int8 conversion failed."
                )
        else:
            wrapped = _PerChannelInt8Linear(lin, out_dtype=out_dtype).to(dev)
            w = wrapped.weight_int8
        # Free the original bf16 weight promptly so the saving is realised.
        setattr(parent, attr, wrapped)
        del lin
        int8_mib += w.numel() * w.element_size() / 2**20
        if verbose:
            print(f"  quantized {dotted}: -> int8 {w.numel()/2**20:.1f}M params")

    torch.cuda.empty_cache()
    if verbose:
        print(f"draft head Linears: {bf16_mib:.1f} MiB bf16 -> {int8_mib:.1f} MiB "
              f"int8 ({backend}, freed ~{bf16_mib - int8_mib:.1f} MiB)")
    # Annotate for downstream reporting / introspection.
    draft.quantized = f"int8:{backend}"
    return draft
