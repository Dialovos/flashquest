"""EAGLE-3 draft head loader + chain proposer.

The draft layer is the vendored EAGLE-3 inference ``Model``
(``vendor/eagle/eagle/model/cnets.py``). We do *not* reimplement it; we load the
``thoughtworks/Llama-3.2-3B-Instruct-Eagle3`` checkpoint into it and drive a
greedy *chain* (depth N) from a target model's fused hidden states.

Wiring facts (verified against the checkpoint + vendored base model):

* The head's ``fc`` is ``Linear(3*hidden -> hidden)``. EAGLE-3 fuses the hidden
  states *entering* target layers ``2``, ``L//2`` and ``L-3`` (see
  ``modeling_llama_kv.py`` line ~1138, where the input ``hidden_states`` is
  appended *before* the layer runs). For the 28-layer Llama-3.2-3B that is
  layers ``2, 14, 25``. With HF ``output_hidden_states=True`` the tuple is
  ``[embed_out, layer0_out, ... layer27_out]`` (length 29) and the *input* to
  layer ``idx`` equals ``hidden_states[idx]`` — so we tap indices ``[2, 14, 25]``
  and concatenate in that order to form the ``9216``-wide fused vector.
* The head predicts in a *draft vocab* of 32000. ``d2t`` maps a draft id back to
  the full vocab via ``full_id = draft_id + d2t[draft_id]`` (matches both the
  training ``t2d`` column-select and the inference ``+ self.d2t[...]``).
* ``Model.forward(hidden_states, input_ids, ...)`` applies ``fc`` iff the hidden
  width (9216) differs from the embedding width (3072), so we always pass the
  raw fused 9216-wide tensor.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# Layers (HF hidden_states indices) EAGLE-3 fuses, low->mid->high, for the
# 28-layer Llama-3.2-3B target.
EAGLE3_FUSION_LAYERS = (2, 14, 25)

_DEFAULT_HEAD = "thoughtworks/Llama-3.2-3B-Instruct-Eagle3"

# Vendored EAGLE-3 code lives here; add to sys.path so its bare
# ``from configs import EConfig`` / ``from utils_c import *`` imports resolve.
_VENDOR_MODEL_DIR = (
    Path(__file__).resolve().parents[3] / "vendor" / "eagle" / "eagle" / "model"
)


def _import_vendored_model():
    """Return (Model, EConfig) from the vendored EAGLE-3 inference module."""
    d = str(_VENDOR_MODEL_DIR)
    if d not in sys.path:
        sys.path.insert(0, d)
    # Imported by path (bare module names) to hit cnets.py's ``except`` branch,
    # which does ``from configs import EConfig`` / ``from utils_c import *``.
    import cnets  # type: ignore  # noqa: E402
    from configs import EConfig  # type: ignore  # noqa: E402

    return cnets.Model, EConfig


def fuse_target_hidden(hidden_states, layers=EAGLE3_FUSION_LAYERS):
    """Concatenate the three fused target layers along the feature dim.

    ``hidden_states`` is the HF tuple/list from ``output_hidden_states=True``
    (length ``num_layers + 1``). Returns ``[B, T, 3*hidden]``.
    """
    return torch.cat([hidden_states[i] for i in layers], dim=-1)


@dataclass
class EagleDraft:
    """Thin driver around the vendored EAGLE-3 draft ``Model``."""

    model: torch.nn.Module  # vendored cnets.Model with checkpoint loaded
    device: str
    dtype: torch.dtype
    fusion_layers: tuple = EAGLE3_FUSION_LAYERS

    @property
    def d2t(self) -> torch.Tensor:
        return self.model.d2t

    def _draft_to_full(self, draft_ids: torch.Tensor) -> torch.Tensor:
        """draft-vocab id -> full-vocab id via ``id + d2t[id]`` (long)."""
        d2t = self.model.d2t.to(draft_ids.device)
        return (draft_ids + d2t[draft_ids]).long()

    @torch.no_grad()
    def _head_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """norm + lm_head -> draft-vocab logits. ``hidden`` is ``[..., H]``."""
        m = self.model
        return m.lm_head(m.norm(hidden))

    @torch.no_grad()
    def prefill(
        self,
        target_hidden_fused: torch.Tensor,
        last_token_id,
        context_ids: torch.Tensor,
        chunk: int = 512,
    ):
        """Seed the head's KV over the verified context and return its last
        hidden + a ``past_key_values`` state for chaining.

        Mirrors the prefill half of the vendored ``Model.topK_genrate``:
        ``input_ids = [context, sample_token]``, then drop the first id so each
        per-position token aligns with the hidden state that *precedes* it (the
        EAGLE one-step shift). After the shift, ``input_ids`` has the same length
        ``T`` as ``target_hidden_fused`` (``[1, T, 3*hidden]``).

        The vendored draft attention materialises a full ``[1, H, q_len, kv_len]``
        fp32 weight matrix (no flash/SDPA), so a single ``q_len = T`` prefill is
        O(T^2) in memory — at T=4k that alone is ~1.6 GB and OOMs a 4 GB GPU. We
        therefore feed the context in **append-only chunks** of ``chunk`` tokens,
        each with ``position_ids=None`` so the model derives true absolute
        positions from the cached length (``past_key_values[0][0].shape[2]``).
        Peak attention weights drop to ``[1, H, chunk, kv]``. With regular RoPE
        (no ``rope_scaling``) this is bit-for-bit identical to a single full
        prefill — the rotary cache is grown from ``kv_seq_len`` each call, so
        positions never index out of bounds.

        Returns:
            last_hidden: ``[1, 1, hidden]`` the head hidden at the tip (its
                ``norm`` + ``lm_head`` give the first draft token).
            past: the head's ``past_key_values`` after the prefill.
        """
        m = self.model
        dev = target_hidden_fused.device
        hidden = target_hidden_fused.to(self.dtype)

        if not torch.is_tensor(last_token_id):
            last_token_id = torch.tensor([last_token_id], device=dev)
        last_token_id = last_token_id.reshape(1, 1).to(dev).long()
        context_ids = context_ids.to(dev).long()

        m.reset_kv()
        m.reset()  # clear any tree_mask
        full_ids = torch.cat([context_ids, last_token_id], dim=1)  # [1, T+1]
        shifted_ids = full_ids[:, 1:]                               # [1, T]
        T = shifted_ids.shape[1]

        past = None
        out_hidden = None
        for s in range(0, T, chunk):
            e = min(s + chunk, T)
            out_hidden, past = m(
                hidden[:, s:e, :], input_ids=shifted_ids[:, s:e],
                past_key_values=past, use_cache=True,  # position_ids=None
            )
        return out_hidden[:, -1:, :], past

    @torch.no_grad()
    def step(self, last_hidden: torch.Tensor, past):
        """One draft step: emit the next full-vocab token from ``last_hidden``,
        then advance the head by feeding (that token, its own hidden) with the
        head's cached KV (append-only, ``position_ids=None``). Returns
        ``(full_token_id[1], new_last_hidden, new_past)``.

        This is the unit the gate times as the "draft step" — a single cached-KV
        forward through the 1-layer head.
        """
        m = self.model
        logits = self._head_logits(last_hidden[:, -1])  # [1, draft_vocab]
        draft_id = logits.argmax(dim=-1)                # [1] draft vocab
        full_id = self._draft_to_full(draft_id)         # [1] full vocab

        out_hidden, past = m(
            last_hidden, input_ids=full_id.view(1, 1),
            past_key_values=past, use_cache=True,  # position_ids=None
        )
        return full_id, out_hidden[:, -1:, :], past

    @torch.no_grad()
    def propose_chain(
        self,
        target_hidden_fused: torch.Tensor,
        last_token_id,
        n_draft: int = 4,
        context_ids: torch.Tensor | None = None,
        chunk: int = 512,
    ) -> torch.Tensor:
        """Greedy depth-``n_draft`` draft chain in full-vocab ids.

        Mirrors the vendored ``Model.topK_genrate`` exactly, but greedy (top-1)
        and linear (a chain, not a tree): a chunked prefill over the context
        followed by ``n_draft`` single-token steps.

        Args:
            target_hidden_fused: ``[1, T, 3*hidden]`` fused target hidden states
                over the verified context (the ``torch.cat`` of the three fused
                layers EAGLE-3 feeds its head). ``T`` must match ``context_ids``.
            last_token_id: the target's freshly-decoded greedy token (the
                free/bonus token); ``topK_genrate`` takes it as
                ``sample_token = input_ids[:, -1]``.
            n_draft: chain depth.
            context_ids: ``[1, T]`` target token ids aligned with
                ``target_hidden_fused``. Required — the head embeds them and the
                rotary positions are derived over them during prefill.
            chunk: prefill chunk size (bounds the O(T^2) draft attention).

        Returns:
            ``LongTensor[n_draft]`` of full-vocab token ids.
        """
        if context_ids is None:
            raise ValueError(
                "propose_chain needs context_ids ([1,T] target token ids "
                "aligned with target_hidden_fused) — the head embeds them and "
                "rotary positions are derived over them during prefill."
            )
        last_hidden, past = self.prefill(
            target_hidden_fused, last_token_id, context_ids, chunk=chunk
        )
        out_tokens = []
        for _ in range(n_draft):
            full_id, last_hidden, past = self.step(last_hidden, past)
            out_tokens.append(full_id)
        return torch.cat(out_tokens, dim=0)  # [n_draft]


def load_eagle3_draft(
    model_id: str = _DEFAULT_HEAD,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    embed_weight: torch.Tensor | None = None,
) -> EagleDraft:
    """Load the EAGLE-3 draft head into the vendored inference ``Model``.

    Everything trainable (``fc``, ``midlayer``, ``norm``, ``lm_head``,
    ``d2t``/``t2d``) comes from the EAGLE-3 checkpoint. The frozen input
    ``embed_tokens`` is *not* in the checkpoint and must match the target
    tokenizer's embedding:

    * Pass ``embed_weight`` (e.g. ``target_model.model.embed_tokens.weight``) to
      reuse the target's embedding directly — preferred, avoids any extra
      download and guarantees the embedding matches the running target.
    * Otherwise the embedding is pulled from the target checkpoint named in the
      head config (gated for meta-llama; override via ``FLASHQUEST_EAGLE_TARGET``).
    """
    Model, EConfig = _import_vendored_model()

    cfg_path = hf_hub_download(model_id, "config.json")
    with open(cfg_path) as f:
        raw = json.loads(f.read())
    config = EConfig.from_pretrained(cfg_path)
    # EConfig defaults that the vendored Model.__init__ touches but the slim
    # EAGLE-3 config.json omits.
    for k, v in (
        ("pad_token_id", 0),
        ("max_position_embeddings", 4096),
        ("rope_scaling", None),
        ("pretraining_tp", 1),
    ):
        if not hasattr(config, k) or getattr(config, k) is None:
            setattr(config, k, v)

    if embed_weight is not None:
        # Skip the target download; we'll copy the embedding in after build.
        load_emb = False
        target_path = None
    else:
        load_emb = True
        target_path = raw.get("base_model_name_or_path") or os.environ.get(
            "FLASHQUEST_EAGLE_TARGET", "meta-llama/Llama-3.2-3B-Instruct"
        )

    # Build the vendored Model. total_tokens/depth/top_k are only used by the
    # tree path (topK_genrate); our propose_chain ignores them.
    model = Model(
        config,
        load_emb=load_emb,
        path=target_path,
        bias=raw.get("bias", False),
        total_tokens=63,
        depth=5,
        top_k=8,
        threshold=1.0,
    )

    if embed_weight is not None:
        with torch.no_grad():
            model.embed_tokens.weight.data = embed_weight.detach().to(
                model.embed_tokens.weight.dtype
            ).cpu().clone()

    # Load the EAGLE-3 head weights (fc/midlayer/norm/lm_head/d2t/t2d).
    ckpt_path = hf_hub_download(model_id, "model.safetensors")
    state = load_file(ckpt_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # embed_tokens is expected-missing (loaded from target); flag anything else.
    leftover = [k for k in missing if not k.startswith("embed_tokens")]
    if leftover:
        raise RuntimeError(f"EAGLE-3 head: unexpected missing keys: {leftover}")
    if unexpected:
        raise RuntimeError(f"EAGLE-3 head: unexpected keys in checkpoint: {unexpected}")

    model = model.to(dtype).to(device)
    model.eval()
    model.init_tree()  # sets tree_mask_init / position_ids buffers on device
    return EagleDraft(model=model, device=device, dtype=dtype)
