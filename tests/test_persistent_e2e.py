"""End-to-end: a Quest-persistent-patched HF model produces logits within
INT8 quant tolerance of Phase 4's BF16-eager Duo patch."""
import pytest
import torch

pytestmark = pytest.mark.slow


def _have_model() -> bool:
    try:
        from transformers import AutoConfig
        AutoConfig.from_pretrained("unsloth/Llama-3.2-1B-Instruct")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _have_model(), reason="model checkpoint not available offline")
def test_persistent_cache_logits_within_int8_tolerance():
    """Persistent INT8 cache + fused dispatch logits should match Phase 4
    BF16-eager Duo logits to roughly INT8 quant noise (rtol=5e-2 atol=5e-2)."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from flashquest.cache import PersistentInt8KVCache
    from flashquest.eager.llama_duo_patch import patch_llama_for_quest_duo
    from flashquest.eager.llama_persistent_patch import patch_llama_for_quest_persistent

    name = "unsloth/Llama-3.2-1B-Instruct"
    tok = AutoTokenizer.from_pretrained(name)

    cfg = AutoConfig.from_pretrained(name)
    pattern = torch.ones(cfg.num_hidden_layers, cfg.num_key_value_heads, dtype=torch.bool)

    m4 = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).cuda().eval()
    patch_llama_for_quest_duo(
        m4, head_pattern=pattern,
        retention=0.25, num_sinks=4, window_pages=2, page_size=64,
    )
    inp = tok("The capital of France is Paris. The capital of Spain is", return_tensors="pt").to("cuda")
    with torch.no_grad():
        ref_logits = m4(**inp).logits
    del m4
    torch.cuda.empty_cache()

    m5 = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).cuda().eval()
    cache = PersistentInt8KVCache(
        batch_size=1, num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim,
        max_seq_len=512, page_size=64, device="cuda",
    )
    patch_llama_for_quest_persistent(
        m5, cache=cache, head_pattern=pattern,
        retention=0.25, num_sinks=4, window_pages=2, page_size=64,
    )
    with torch.no_grad():
        out_logits = m5(**inp).logits

    torch.testing.assert_close(out_logits, ref_logits, rtol=5e-2, atol=5e-2)
