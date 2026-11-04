"""N-L1 bounds: the exact Bekenstein and Bekenstein-Hawking formulas, the register read from the hardware,
and the size caps they give to a context label, a density matrix and an MPO bond."""

import math

import pytest

from vam.nested import NestedConfig
from vam.nfa import bounds
from vam.realizations.vam1 import Config


def test_bekenstein_formula_matches_its_definition():
    R, E = 0.1, 1.0e3
    nats = 2 * math.pi * R * E / (bounds.HBAR * bounds.C)
    assert bounds.bekenstein_bits(R, E) == pytest.approx(nats / math.log(2), rel=1e-15)
    with pytest.raises(ValueError):
        bounds.bekenstein_bits(-1.0, 1.0)


def test_bekenstein_hawking_of_a_solar_mass_black_hole():
    """S = 4 pi G M^2 / (hbar c) nats for a Schwarzschild horizon: about 1.5e77 bits for one solar mass."""
    M = 1.98892e30
    r = bounds.schwarzschild_radius(M)
    S = bounds.bekenstein_hawking_bits(4 * math.pi * r ** 2)
    assert S == pytest.approx(4 * math.pi * bounds.G * M ** 2 / (bounds.HBAR * bounds.C) / math.log(2), rel=1e-12)
    assert 1.4e77 < S < 1.6e77


def test_a_black_hole_saturates_bekenstein():
    """For a Schwarzschild black hole, Bekenstein's bound with R = r_s and E = M c^2 equals A / 4 l_P^2."""
    M = 10.0
    r = bounds.schwarzschild_radius(M)
    assert bounds.bekenstein_bits(r, M * bounds.C ** 2) == pytest.approx(
        bounds.bekenstein_hawking_bits(4 * math.pi * r ** 2), rel=1e-12)


def test_register_is_the_binding_limit():
    hw = bounds.Hardware.detect()
    assert hw.word_bits == 64 and hw.line_words == hw.line_bits // 64
    assert bounds.silicon_bekenstein_bits() > 1e20               # physics allows far more than a register
    assert bounds.register_cap(hw) == 64
    assert bounds.register_cap(hw, physical_bits=40.7) == 40     # a tighter physical bound would win
    assert bounds.resolution(hw) == 2.0 ** -52


def test_hardware_parsing_and_fallback(tmp_path):
    for i, (lv, kind, size) in enumerate([(1, "Data", "32K"), (1, "Instruction", "32K"), (2, "Unified", "1M")]):
        d = tmp_path / f"index{i}"
        d.mkdir()
        (d / "level").write_text(f"{lv}\n")
        (d / "type").write_text(kind + "\n")
        (d / "size").write_text(size + "\n")
        (d / "coherency_line_size").write_text("128\n")
    hw = bounds.Hardware.detect(str(tmp_path))
    assert (hw.l1d_bytes, hw.l2_bytes, hw.line_bytes, hw.detected) == (32 * 1024, 1024 ** 2, 128, True)
    assert hw.l3_bytes == bounds.Hardware().l3_bytes                       # missing entry keeps the default
    missing = bounds.Hardware.detect(str(tmp_path / "nowhere"))
    assert not missing.detected and missing.l2_bytes == bounds.Hardware().l2_bytes


def test_cache_caps():
    hw = bounds.Hardware(l1d_bytes=48 * 1024, l2_bytes=2048 * 1024)
    d1 = bounds.max_density_dim(hw, 1)
    assert d1 ** 2 * 16 <= 48 * 1024 < (d1 + 1) ** 2 * 16 and d1 == 55
    assert bounds.max_density_qubits(hw, 1) == 5 and bounds.max_density_qubits(hw, 2) == 8
    D = bounds.max_mpo_bond(hw, local_dim=4, level=2)
    assert D * D * 16 * 16 <= 2048 * 1024 < (D + 1) ** 2 * 16 * 16 and D == 90
    s = bounds.summary(hw)
    assert s["register_cap_bits"] == 64 and s["max_mpo_bond"][2] == 90


def test_nested_depths_respect_the_register():
    law = Config(cells=1, context=1)
    bounds.check_depths(NestedConfig(law).depths)                # the default nested machine fits one register
    with pytest.raises(ValueError, match="register cap"):
        bounds.check_depths((8, 72))
