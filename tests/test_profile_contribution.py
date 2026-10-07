"""CPU-safe oracles plus CUDA tests of the real packed kernel and runtime tail."""
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import profile_contribution as P


def test_script_import_does_not_import_gpu_runtime():
    script = "import sys; sys.path.insert(0, 'scripts'); import profile_contribution; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", script], check=True)


def test_exact_dequant_preserves_fp32_affine_values():
    packed = torch.tensor([[[[0xF0, 0xA5], [0x21, 0x43]]]], dtype=torch.uint8)
    scale = torch.tensor([[[[.1015625] * 4]]], dtype=torch.bfloat16)
    minimum = torch.tensor([[[[.19921875] * 4]]], dtype=torch.bfloat16)
    result = P.exact_dequant(packed, scale, minimum, page_size=2)
    codes = torch.tensor([[[[0, 15, 5, 10], [1, 2, 3, 4]]]])
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, codes.float() * scale.float() + minimum.float(), rtol=0, atol=0)
    assert not torch.equal(result, result.bfloat16().float())


def test_summary_scores_match_explicit_gqa_channelwise_bound():
    torch.manual_seed(9)
    q = torch.randn(1, 6, 1, 8)
    low = torch.randn(1, 2, 4, 8)
    high = low + torch.rand_like(low)
    repeated_low = low.repeat_interleave(3, dim=1).unsqueeze(2)
    repeated_high = high.repeat_interleave(3, dim=1).unsqueeze(2)
    expected = torch.maximum(q.unsqueeze(3) * repeated_low, q.unsqueeze(3) * repeated_high).sum(-1)
    torch.testing.assert_close(P.summary_scores(q, low, high), expected)


def test_exact_attention_gqa_mask_and_tail_match_independent_loop():
    torch.manual_seed(5)
    q, k, v = torch.randn(1, 4, 1, 8), torch.randn(1, 2, 7, 8), torch.randn(1, 2, 7, 8)
    selection = torch.tensor([[[[True, False]], [[False, True]], [[True, True]], [[False, False]]]])
    mask = P.token_mask(selection, 6, 1, 3)
    actual, lse = P.exact_attention(q, k, v, mask)
    for head in range(4):
        valid = mask[0, head, 0]
        logits = q[0, head, 0] @ k[0, head // 2, valid].T / math.sqrt(8)
        expected = logits.softmax(-1) @ v[0, head // 2, valid]
        torch.testing.assert_close(actual[0, head, 0], expected)
        torch.testing.assert_close(lse[0, head, 0], logits.logsumexp(-1))
    torch.testing.assert_close(P.sdpa_attention(q, k, v, mask), actual)


def test_empty_mask_returns_zero_and_negative_infinity_lse():
    q, k, v = torch.ones(1, 2, 1, 4), torch.ones(1, 1, 2, 4), torch.ones(1, 1, 2, 4)
    result, lse = P.exact_attention(q, k, v, torch.zeros(1, 2, 1, 2, dtype=torch.bool))
    assert torch.equal(result, torch.zeros_like(result))
    assert torch.isneginf(lse).all()


def test_selection_ties_and_empty_union_diagnostics():
    a = torch.tensor([[[[True, False, True]], [[False, False, False]]]])
    assert P.overlap(a, a).flatten().tolist() == [1.0, 1.0]
    assert P.boundary_margin(torch.ones(1, 2, 1, 3), 1) == [0.0, 0.0]
    assert P.boundary_margin(torch.ones(1, 2, 1, 3), 3) is None


def test_constant_keys_expose_clamped_scale_and_byte_accounting_on_cpu():
    keys = torch.ones(1, 1, 129, 4, dtype=torch.bfloat16)
    views = {"completed_len": 128, "partial_len": 1,
             "K_packed": torch.zeros(1, 1, 128, 2, dtype=torch.uint8),
             "K_scale": torch.full((1, 1, 2, 4), 1e-6, dtype=torch.bfloat16),
             "K_mn": torch.ones(1, 1, 2, 4, dtype=torch.bfloat16)}
    def summarize(tensor, page_size):
        pages = tensor.reshape(1, 1, 2, page_size, 4)
        return pages.amin(dim=3), pages.amax(dim=3)

    def select(scores, retention, sinks, window, k_max_static):
        mask = torch.zeros_like(scores, dtype=torch.bool)
        return mask.scatter(-1, scores.topk(k_max_static, dim=-1).indices, True)

    diagnostics, _, _, _ = P.score_diagnostics(torch.ones(1, 2, 1, 4), keys, views,
        page_size=64, retention=.5, sinks=0, window=0,
        score_fn=lambda q, scale, low: P.summary_scores(q, low, low.float() + 15 * scale.float()),
        summary_fn=summarize, select_fn=select)
    assert diagnostics["raw_scale_at_or_below_epsilon_channels"] == 8
    assert diagnostics["metadata_near_tie_query_heads"] == 2
    assert diagnostics["metadata_reconstructed_max_error"]["max_abs_error"] > 0
    assert diagnostics["bytes"]["separate_bf16_summaries"] == 32
    assert diagnostics["bytes"]["analytic_separate_bf16_summaries"] == 32
    assert diagnostics["bytes"]["separate_summaries_fraction_of_packed_k"] == .125
    assert all(0 <= mass <= 1 for mass in
               diagnostics["metadata_retained_full_context_mass_per_query_head"])


@pytest.mark.parametrize("arguments", [
    ["--mode", "model"], ["--contexts", "129"], ["--query-heads", "3", "--kv-heads", "2"],
    ["--tail-len", "64"], ["--retention", "0"], ["--reps", "0"],
])
def test_cli_rejects_invalid_measurements(arguments):
    with pytest.raises(SystemExit):
        P.parse_args(arguments)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires scheduled CUDA slot")
@pytest.mark.parametrize("tail", [0, 1, 63])
def test_real_fused_kernel_exact_oracle_and_components(tail):
    args = SimpleNamespace(kv_heads=2, query_heads=4, head_dim=64, tail_len=tail,
                           page_size=64, seed=12, retention=.5, num_sinks=0,
                           window_pages=0, warmup=1, reps=2)
    q, keys, views = P.synthetic_fixture(128, args)
    row = P.profile_fixture(q, keys, views, args)
    assert row["attention"]["fused_output_vs_fp32_oracle"]["max_abs_error"] < .02
    assert row["attention"]["fused_lse_vs_fp32_oracle"]["max_abs_error"] < .01
    assert row["attention"]["sdpa_output_vs_fp32_oracle"]["max_abs_error"] < 1e-5
    assert row["scoring"]["bytes"]["separate_summaries_fraction_of_packed_k"] == .125
    assert row["scoring"]["bytes"]["incremental_metadata_for_scoring"] == 0
    assert row["actual_cache_tokens"] == 128 + tail
    assert ("tail_attention" in row["components"]) == bool(tail)
    assert ("merge" in row["components"]) == bool(tail)
    for measured in row["components"].values():
        assert len(measured["cuda_ms"]) == len(measured["incremental_peak_allocated_bytes"]) == 2
        assert min(measured["cuda_ms"]) > 0
        assert min(measured["incremental_peak_allocated_bytes"]) >= 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires scheduled CUDA slot")
def test_tiny_model_captures_restore_forwards_and_real_decode_positions():
    from transformers import LlamaConfig, LlamaForCausalLM

    class Tokenizer:
        def encode(self, text, **kwargs):
            return [1] * max(1, len(text.split()) // 4)

        def __call__(self, text, **kwargs):
            return SimpleNamespace(input_ids=torch.tensor([self.encode(text)]))

    config = LlamaConfig(vocab_size=64, hidden_size=128, intermediate_size=256,
                         num_hidden_layers=1, num_attention_heads=2,
                         num_key_value_heads=1, max_position_embeddings=1024)
    model = LlamaForCausalLM(config).to(device="cuda", dtype=torch.float16).eval()
    module = model.model.layers[0].self_attn
    original = module.forward
    args = SimpleNamespace(layers=[0], task="single", seed=0, decode_steps=[0, 1],
                           page_size=64, retention=.5, num_sinks=0, window_pages=0,
                           warmup=1, reps=1)
    rows = list(P.model_rows(model, Tokenizer(), 768, args))
    assert len(rows) == 2
    assert module.forward == original
    assert not module._forward_pre_hooks
    assert all(row["layer"] == 0 and 0 < row["actual_cache_tokens"] <= 1024 for row in rows)
    assert [row["actual_cache_tokens"] for row in rows] == [
        rows[0]["input_tokens"] + 1, rows[0]["input_tokens"] + 2]
    assert rows[0]["input_sha256"] == rows[1]["input_sha256"]
    assert all(row["attention"]["fused_lse_vs_fp32_oracle"]["max_abs_error"] < .01 for row in rows)
