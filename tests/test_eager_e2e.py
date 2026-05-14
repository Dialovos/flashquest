"""End-to-end: a Quest-eager-patched HF model produces the same logits as the
unpatched model when retention=1.0, num_sinks=0, window_pages=0."""
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
def test_full_retention_matches_unpatched_logits():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from flashquest.eager.llama_patch import patch_llama_for_quest_eager

    name = "unsloth/Llama-3.2-1B-Instruct"
    tok = AutoTokenizer.from_pretrained(name)
    # Use the SDPA backend as the reference: our patch also routes through
    # F.scaled_dot_product_attention, so this is a like-for-like comparison.
    # (HF "eager" backend orders ops differently, leaking BF16 drift > 1e-2.)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda().eval()

    inp = tok("The capital of France is", return_tensors="pt").to("cuda")
    with torch.no_grad():
        ref_logits = model(**inp).logits

    patch_llama_for_quest_eager(
        model, retention=1.0, num_sinks=0, window_pages=0, page_size=64
    )

    with torch.no_grad():
        out_logits = model(**inp).logits

    # SDPA -> SDPA: tight tolerance.
    torch.testing.assert_close(out_logits, ref_logits, rtol=5e-3, atol=5e-3)
