"""Phase 6 task 4 — head-to-head orchestrator tests."""
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import phase6_run_headtohead as H


def test_planned_matrix_default():
    """3 backends × 3 contexts = 9 cells in canonical order."""
    matrix = H.planned_matrix()
    assert len(matrix) == 9
    backends = {b for b, _ in matrix}
    ctxs = {c for _, c in matrix}
    assert backends == {"flashquest", "llamacpp", "vllm"}
    assert ctxs == {8192, 32768, 131072}
    backend_order = [b for b, _ in matrix]
    assert backend_order == (
        ["flashquest"] * 3 + ["llamacpp"] * 3 + ["vllm"] * 3
    )


def test_dry_run_prints_matrix(capsys):
    """--dry-run prints all 9 cells and returns 0 without launching anything."""
    rc = H.main(["--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "flashquest @ 8192" in out
    assert "vllm @ 131072" in out
    assert "9 cells" in out


def test_render_markdown_table_has_three_backend_rows():
    """render_markdown produces a table with one row per backend."""
    cells = [
        {"backend": "flashquest", "quant": "AWQ-INT4 + INT8", "ctx_len": 8192,
         "decode_tok_s": 7.2, "peak_vram_mib": 4500, "oom": False, "error": None},
        {"backend": "flashquest", "quant": "AWQ-INT4 + INT8", "ctx_len": 32768,
         "decode_tok_s": 5.14, "peak_vram_mib": 6379, "oom": False, "error": None},
        {"backend": "flashquest", "quant": "AWQ-INT4 + INT8", "ctx_len": 131072,
         "decode_tok_s": None, "peak_vram_mib": None, "oom": True, "error": "OOM"},
        {"backend": "llama.cpp", "quant": "Q4_K_M", "ctx_len": 8192,
         "decode_tok_s": 39.6, "peak_vram_mib": 3543, "oom": False, "error": None},
        {"backend": "llama.cpp", "quant": "Q4_K_M", "ctx_len": 32768,
         "decode_tok_s": 12.0, "peak_vram_mib": 3700, "oom": False, "error": None},
        {"backend": "llama.cpp", "quant": "Q4_K_M", "ctx_len": 131072,
         "decode_tok_s": None, "peak_vram_mib": None, "oom": True, "error": "OOM"},
        {"backend": "vLLM 0.7.3", "quant": "AWQ-INT4", "ctx_len": 8192,
         "decode_tok_s": None, "oom": True, "error": "OOM"},
        {"backend": "vLLM 0.7.3", "quant": "AWQ-INT4", "ctx_len": 32768,
         "decode_tok_s": None, "oom": True, "error": "OOM"},
        {"backend": "vLLM 0.7.3", "quant": "AWQ-INT4", "ctx_len": 131072,
         "decode_tok_s": None, "oom": True, "error": "OOM"},
    ]
    md = H.render_markdown(cells)
    body_lines = [l for l in md.splitlines() if l.startswith("|")]
    assert len(body_lines) >= 5
    assert "flashquest" in md
    assert "llama.cpp" in md
    assert "vLLM" in md
    assert "5.14" in md
    assert "OOM" in md


def test_parse_llamacpp_log(tmp_path):
    """_parse_llamacpp_log extracts pp + tg + smi from an llama-bench markdown."""
    log = tmp_path / "llamacpp_8192.txt"
    log.write_text(
        "| model | size | params | backend | ngl | test | t/s |\n"
        "|---|---|---|---|---|---|---|\n"
        "| llama 3B Q4_K_M | 1.87 GiB | 3.21 B | CUDA | 999 | pp8192      | 736.55 ± 55.81 |\n"
        "| llama 3B Q4_K_M | 1.87 GiB | 3.21 B | CUDA | 999 | tg128       | 39.60 ± 0.11   |\n"
        "---\n"
        "VRAM at run-tail (nvidia-smi):\n"
        "memory.used [MiB], memory.free [MiB]\n"
        "3543 MiB, 552 MiB\n"
    )
    rec = H._parse_llamacpp_log(log, ctx=8192, rc=0, stderr="")
    assert rec["prefill_tok_s"] == 736.55
    assert rec["decode_tok_s"] == 39.60
    assert rec["peak_vram_mib"] == 3543
    assert rec["oom"] is False


def test_parse_llamacpp_log_oom(tmp_path):
    """A non-zero rc with 'out of memory' in stderr → oom=true."""
    log = tmp_path / "fake.txt"
    log.write_text("")
    rec = H._parse_llamacpp_log(
        log, ctx=131072, rc=1,
        stderr="ggml_cuda_compute_forward: out of memory",
    )
    assert rec["oom"] is True
