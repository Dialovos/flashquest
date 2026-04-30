"""End-to-end: a Quest-Duo-eager-patched HF model produces the same logits as
the Phase 1 retrieval-only patch when *all* heads are classified as retrieval."""
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
def test_all_retrieval_matches_phase1_patch():
    """When the DuoAttention pattern says every head is retrieval, the Duo
    patch must produce identical logits to Phase 1's quest_eager patch."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from flashquest.eager.llama_duo_patch import patch_llama_for_quest_duo
    from flashquest.eager.llama_patch import patch_llama_for_quest_eager

    name = "unsloth/Llama-3.2-1B-Instruct"
    tok = AutoTokenizer.from_pretrained(name)

    cfg = AutoConfig.from_pretrained(name)
    num_layers = cfg.num_hidden_layers
    num_kv = cfg.num_key_value_heads
    pattern_all_retrieval = torch.ones(num_layers, num_kv, dtype=torch.bool)

    model_phase1 = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda().eval()
    patch_llama_for_quest_eager(
        model_phase1, retention=0.25, num_sinks=4, window_pages=2, page_size=64
    )

    inp = tok("The capital of France is", return_tensors="pt").to("cuda")
    with torch.no_grad():
        ref_logits = model_phase1(**inp).logits

    del model_phase1
    torch.cuda.empty_cache()

    model_phase4 = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda().eval()
    patch_llama_for_quest_duo(
        model_phase4,
        head_pattern=pattern_all_retrieval,
        retention=0.25, num_sinks=4, window_pages=2, page_size=64,
    )

    with torch.no_grad():
        out_logits = model_phase4(**inp).logits

    torch.testing.assert_close(out_logits, ref_logits, rtol=5e-3, atol=5e-3)
