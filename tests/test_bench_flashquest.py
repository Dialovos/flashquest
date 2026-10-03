"""Exercise repeated benchmark requests through the real persistent dispatcher."""
import json
import sys
import time
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import bench_common as C
import bench_flashquest
import gpu_memory as M
import phase6_run_ruler_4k_int4 as Q
from bench_common import content_hash


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_repeated_requests_reset_cache_and_advance_positions(tmp_path, monkeypatch):
    from transformers import LlamaConfig, LlamaForCausalLM

    from flashquest.runtime import awq_load

    # Random local weights exercise the dispatcher without model downloads.
    config = LlamaConfig(vocab_size=64, hidden_size=128, intermediate_size=256,
                         num_hidden_layers=1, num_attention_heads=2,
                         num_key_value_heads=1, max_position_embeddings=512)
    model = LlamaForCausalLM(config).to(device="cuda", dtype=torch.float16).eval()
    monkeypatch.setattr(awq_load, "load_awq_model", lambda *args, **kwargs: (model, range(64)))
    model_identity = {"model": "random-local-test", "revision": None,
                      "files": {}, "content_sha256": content_hash({})}
    monkeypatch.setattr(bench_flashquest, "model_identity", lambda *args: model_identity)
    observed = []

    def observe_positions(module, inputs, kwargs):
        observed.append(kwargs["cache_position"].cpu().tolist())

    hook = model.register_forward_pre_hook(observe_positions, with_kwargs=True)
    output = tmp_path / "bench.json"
    markers = tmp_path / "markers.jsonl"
    monkeypatch.setenv("FLASHQUEST_MARKERS", str(markers))
    monkeypatch.setattr(sys, "argv", ["bench_flashquest.py", "--ctx-len", "128",
                                    "--n-decode", "4", "--reps", "2", "--out", str(output)])
    try:
        assert bench_flashquest.main() == 0
    finally:
        hook.remove()
    record = json.loads(output.read_text())
    assert record["error"] is None and len(record["samples"]) == 2
    assert record["decode_steps"] == 3 and record["config"]["retention"] == 0.20
    assert all(s["input_tokens"] == 128 and s["output_tokens"] == 4 for s in record["samples"])
    assert record["decode_tok_s"] > 0 and record["peak_allocated_mib"] > 0
    assert content_hash(C.canonical_identity(record["identity"])) == record["run_identity"]
    events = [json.loads(line) for line in markers.read_text().splitlines()]
    windows = M.phase_windows(events, 0, time.monotonic_ns())
    assert len(windows["prefill"]) == len(windows["decode"]) == 2
    assert len(windows["warmup"]) == len(windows["load"]) == 1
    # Each prefill begins a new request. Every subsequent token follows it;
    # neither a warmup nor a previous repetition leaks into the next positions.
    requests = []
    for positions in observed:
        if len(positions) == 128:
            assert positions == list(range(128))
            requests.append([])
        else:
            requests[-1].extend(positions)
    assert requests == [list(range(128, 193)), [128, 129, 130], [128, 129, 130]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_quality_generate_resets_positions_and_restores_patch(tmp_path, monkeypatch):
    import weakref

    from transformers import LlamaConfig, LlamaForCausalLM

    monkeypatch.setattr(Q, "REPO_ROOT", tmp_path)
    runtime = Q.production_runtime()
    config = LlamaConfig(vocab_size=64, hidden_size=128, intermediate_size=256,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                         max_position_embeddings=512, eos_token_id=None, pad_token_id=0)
    model = LlamaForCausalLM(config).to(device="cuda", dtype=torch.float16).eval()
    original = model.model.layers[0].self_attn.forward

    class Tokenizer:
        def decode(self, ids, **kwargs):
            return " ".join(str(int(value)) for value in ids)

    runtime.loader = lambda *args, **kwargs: (model, Tokenizer())
    examples = [{"example_id": f"single:0:{i}", "task": "single", "seed": 0, "index": i,
                 "input_ids": [3] * length, "input_sha256": content_hash([3] * length),
                 "expected": ["1234567"]} for i, length in enumerate([129, 72])]
    runtime.manifest_builder = lambda *args: examples
    factory = runtime.cache_factory
    refs = []

    def observe_factory(**kwargs):
        cache = factory(**kwargs)
        refs.append(weakref.ref(cache))
        return cache

    runtime.cache_factory = observe_factory
    observed = []

    def observe_positions(module, inputs, kwargs):
        observed.append(kwargs["cache_position"].cpu().tolist())

    hook = model.register_forward_pre_hook(observe_positions, with_kwargs=True)
    try:
        args = Q.parse_args(["--ctx-len", "512", "--tasks", "single", "--n-samples", "2",
                             "--max-new-tokens", "8"])
        result, _ = Q.run_quality(args, runtime,
                                  {"model": "random-local-test", "revision": None,
                                   "files": {}, "content_sha256": content_hash({})}, {})
    finally:
        hook.remove()
    assert result["status"] == "complete" and len(result["cells"]) == 3
    assert model.model.layers[0].self_attn.forward == original
    assert len(refs) == 1 and refs[0]() is None
    requests = []
    for positions in observed:
        if len(positions) > 1:
            assert positions == list(range(len(positions)))
            requests.append((len(positions), []))
        else:
            requests[-1][1].extend(positions)
    assert requests == [(length, list(range(length, length + 7)))
                        for _ in range(3) for length in [129, 72]]
