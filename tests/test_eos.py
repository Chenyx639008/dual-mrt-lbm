"""Tests for Peng-Robinson EOS utilities in lbm_mrt.utils.eos."""

import numpy as np
import pytest

from lbm_mrt.utils.eos import (
    alpha_PR,
    compute_ab_from_Tc_Pc,
    compute_Tc_Pc_from_ab,
    critical_from_abR,
    find_rho_c_given_Tc_Pc,
    pr_eos,
)


# ── alpha_PR ──────────────────────────────────────────────────────────────

def test_alpha_at_Tc_is_one() -> None:
    """alpha(T=Tc) must equal 1.0 exactly for any acentric factor."""
    Tc = 0.036461
    for omega in [0.0, 0.344, 0.7]:
        result = alpha_PR(Tc, Tc, omega)
        assert abs(result - 1.0) < 1e-12, f"alpha(Tc) = {result} for omega={omega}"


def test_alpha_decreases_above_Tc() -> None:
    """alpha increases for T < Tc and is < 1 for T > Tc."""
    Tc, omega = 0.03, 0.344
    assert alpha_PR(Tc * 0.5, Tc, omega) > 1.0
    assert alpha_PR(Tc * 2.0, Tc, omega) < 1.0


# ── compute_ab / compute_Tc_Pc ────────────────────────────────────────────

def test_ab_Tc_Pc_roundtrip() -> None:
    """(Tc, Pc) → (a, b) → (Tc, Pc) should be exact."""
    Tc, Pc, R = 0.036461, 0.026813, 1.0
    a, b = compute_ab_from_Tc_Pc(Tc, Pc, R)
    Tc2, Pc2 = compute_Tc_Pc_from_ab(a, b, R)
    assert abs(Tc2 - Tc) / Tc < 1e-10
    assert abs(Pc2 - Pc) / Pc < 1e-10


def test_ab_positive() -> None:
    a, b = compute_ab_from_Tc_Pc(0.036, 0.027, 1.0)
    assert a > 0.0
    assert b > 0.0


# ── pr_eos ────────────────────────────────────────────────────────────────

def test_pr_eos_returns_positive_pressure() -> None:
    """At low density and above critical temperature, pressure should be positive."""
    a, b = compute_ab_from_Tc_Pc(0.036461, 0.026813, 1.0)
    p = pr_eos(0.1, 0.04, a, b, 1.0, 0.036461, 0.344)
    assert np.isfinite(p)
    assert p > 0.0


def test_pr_eos_singularity_guard() -> None:
    """Pressure near singularity should return inf, not raise."""
    a, b = compute_ab_from_Tc_Pc(0.036461, 0.026813, 1.0)
    # rho = 1/b triggers denom1 = 0
    p = pr_eos(1.0 / b, 0.036461, a, b, 1.0, 0.036461, 0.344)
    assert p == np.inf


# ── find_rho_c / critical_from_abR ───────────────────────────────────────

def test_find_rho_c_pressure_residual_small() -> None:
    """P(Tc, rho_c) should be very close to Pc."""
    a_w = 2.0 / 49.0
    b_w = 2.0 / 21.0
    R_w = 1.0
    omega_w = 0.344
    Tc, Pc = compute_Tc_Pc_from_ab(a_w, b_w, R_w)
    rho_c, method = find_rho_c_given_Tc_Pc(a_w, b_w, R_w, omega_w, Tc, Pc)
    p_err = abs(pr_eos(rho_c, Tc, a_w, b_w, R_w, Tc, omega_w) - Pc)
    assert p_err < 1e-6, f"Pressure residual {p_err:.2e} too large (method={method})"


def test_critical_from_abR_alpha_Tc(capsys) -> None:
    """critical_from_abR should return alpha_Tc ≈ 1.0."""
    a_w = 2.0 / 49.0
    b_w = 2.0 / 21.0
    Tc, Pc, alpha_Tc, rho_c = critical_from_abR(a_w, b_w, 1.0, 0.344)
    assert abs(alpha_Tc - 1.0) < 1e-10
    assert rho_c > 0.0
