"""WHT correctness + orthogonality (used by Phase 7 TurboQuant)."""
import pytest
import torch

from flashquest.kernel.wht import wht_along_head_dim


def test_wht_inverse_fp32():
    """Normalized WHT is its own inverse in fp32 (exact math, tight tolerance)."""
    torch.manual_seed(7)
    for D in (64, 128):
        x = torch.randn(2, 4, 8, D, dtype=torch.float32, device="cuda")
        y = wht_along_head_dim(x)
        x_back = wht_along_head_dim(y)
        err = (x - x_back).abs().max()
        assert err < 1e-4, f"D={D}: fp32 wht inverse err {err}"


def test_wht_inverse_bf16():
    """BF16 round-trip stays within the bf16-precision band."""
    torch.manual_seed(7)
    for D in (64, 128):
        x = torch.randn(2, 4, 8, D, dtype=torch.bfloat16, device="cuda")
        y = wht_along_head_dim(x)
        x_back = wht_along_head_dim(y)
        err = (x - x_back).float().abs().max()
        # 14 BF16 ops accumulate ~3e-3 * sqrt(N) rel error; 5e-2 is a safe band.
        assert err < 5e-2, f"D={D}: bf16 wht inverse err {err}"


def test_wht_orthogonal():
    """Inner products preserved: <wht(x), wht(y)> ≈ <x, y>."""
    torch.manual_seed(11)
    D = 128
    x = torch.randn(1, 1, 1, D, dtype=torch.float32, device="cuda")
    y = torch.randn(1, 1, 1, D, dtype=torch.float32, device="cuda")
    dot_raw = (x * y).sum(dim=-1)
    dot_rot = (wht_along_head_dim(x) * wht_along_head_dim(y)).sum(dim=-1)
    err = (dot_raw - dot_rot).abs().max()
    assert err < 1e-4, f"orthogonality err {err}"


def test_wht_requires_power_of_two():
    """Non-power-of-2 head_dim raises."""
    x = torch.randn(1, 1, 1, 96, device="cuda")
    with pytest.raises(ValueError, match="power of 2"):
        wht_along_head_dim(x)
