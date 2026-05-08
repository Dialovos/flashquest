"""Integration: full-forward parity between bool-mask path (use_compact_kernel=False)
and compact-kernel path (use_compact_kernel=True) on a small synthetic Llama-like model.

Phase 8a only supports kv_bits=4 compact path (INT8/Turbo deferred to Phase 8b).
"""
import pytest
import torch


def test_compact_kernel_full_forward_matches_bool_mask_kv4():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    from transformers import AutoConfig
    from transformers.models.llama.modeling_llama import LlamaModel

    from flashquest.cache.persistent_int4 import PersistentInt4KVCache
    from flashquest.eager.llama_persistent_patch import patch_llama_for_quest_persistent

    cfg = AutoConfig.for_model(
        "llama",
        hidden_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        intermediate_size=256,
        max_position_embeddings=512,
        vocab_size=128,
    )
    cfg.head_dim = 32  # 128/4
    torch.manual_seed(42)
    model_a = LlamaModel(cfg).cuda().to(torch.bfloat16).eval()

    cache_a = PersistentInt4KVCache(
        num_layers=2, batch_size=1, num_kv_heads=2, head_dim=32,
        page_size=8, max_seq_len=128,
    )
    cache_b = PersistentInt4KVCache(
        num_layers=2, batch_size=1, num_kv_heads=2, head_dim=32,
        page_size=8, max_seq_len=128,
    )

    head_pattern = torch.ones(2, 2, dtype=torch.bool)  # all retrieval

    patch_llama_for_quest_persistent(
        model_a, cache=cache_a, head_pattern=head_pattern,
        retention=0.5, num_sinks=2, window_pages=1, page_size=8,
        use_compact_kernel=False,
    )
    torch.manual_seed(0)
    inp = torch.randint(0, 128, (1, 64), device="cuda")
    with torch.no_grad():
        out_a = model_a(inp).last_hidden_state

    # Path B: compact kernel — same model weights
    model_b = LlamaModel(cfg).cuda().to(torch.bfloat16).eval()
    model_b.load_state_dict(model_a.state_dict())
    patch_llama_for_quest_persistent(
        model_b, cache=cache_b, head_pattern=head_pattern,
        retention=0.5, num_sinks=2, window_pages=1, page_size=8,
        use_compact_kernel=True,
    )
    with torch.no_grad():
        out_b = model_b(inp).last_hidden_state

    torch.testing.assert_close(out_b, out_a, atol=5e-2, rtol=5e-2)


def test_compact_kernel_rejects_int8():
    """use_compact_kernel=True with kv_bits=8 should raise NotImplementedError
    until Phase 8b lands the INT8 compact kernel."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from flashquest.cache.persistent_int8 import PersistentInt8KVCache
    from flashquest.eager.llama_persistent_patch import make_quest_persistent_forward

    cache = PersistentInt8KVCache(
        num_layers=1, batch_size=1, num_kv_heads=2, head_dim=32,
        page_size=8, max_seq_len=128,
    )
    head_pattern_layer = torch.ones(2, dtype=torch.bool)

    with pytest.raises(NotImplementedError, match="kv_bits=4"):
        make_quest_persistent_forward(
            cache=cache,
            head_pattern_layer=head_pattern_layer,
            retention=0.5, num_sinks=2, window_pages=1, page_size=8,
            use_compact_kernel=True,
        )
