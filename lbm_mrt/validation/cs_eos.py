"""Carnahan-Starling Equation of State — with robust Maxwell construction.

Provides:
- cs_pressure(rho, a, b, R, T) → p
- cs_chemical_potential(rho, a, b, R, T) → mu
- cs_critical_point(a, b, R) → (Tc, pc, rhoc)
- maxwell_coexistence(a, b, R, T) → (rho_g, rho_l, p_eq) or None
- coexistence_curve(a, b, R, T_vals) → dict of arrays
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def cs_pressure(rho, a, b, R, T):
    """CS pressure: p = ρRT·Z(η) − aρ²,  η = bρ/4."""
    rho = np.asarray(rho, dtype=np.float64)
    eta = b * rho / 4.0
    safe = eta < 0.999
    denom = np.where(safe, (1.0 - eta) ** 3, 1e-12)
    Z = np.where(safe, (1.0 + eta + eta**2 - eta**3) / denom, 1e10)
    return rho * R * T * Z - a * rho * rho


def cs_critical_point(a, b, R):
    """CS critical constants: Tc, pc, rhoc."""
    return 0.3773 * a / (b * R), 0.0707 * a / (b * b), 0.1304 / b


def cs_chemical_potential(rho, a, b, R, T):
    """Chemical potential μ for CS-EOS.
    μ = RT·ln(ρ) + RT·(8η−9η²+3η³)/(1−η)³ − 2aρ
    """
    if rho <= 0:
        return -1e10
    eta = b * rho / 4.0
    if eta >= 0.999:
        return 1e10
    denom = (1.0 - eta) ** 3
    return (
        R * T * np.log(max(rho, 1e-30))
        + R * T * (8 * eta - 9 * eta**2 + 3 * eta**3) / denom
        - 2.0 * a * rho
    )


def maxwell_coexistence(a, b, R, T):
    """Maxwell construction via equal-μ at equal-p.

    Returns (rho_g, rho_l, p_eq) or None if T ≥ Tc or solver fails.

    Key fix (2026-05): classify p(ρ)=peq crossings by stable branch:
      - gas branch:  ρ < ρ_gas_spinodal  (dp/dρ > 0, stable)
      - liquid branch: ρ > ρ_liq_spinodal (dp/dρ > 0, stable)
      - Crossings inside the van der Waals loop (dp/dρ < 0) are REJECTED.
    This prevents the solver from converging to peq=0 and selecting
    unstable-region crossings as false coexistence points.
    """
    Tc, _, _ = cs_critical_point(a, b, R)
    if T >= Tc * 0.999:
        return None

    # High-res p−ρ scan (20k points for robust crossing detection at low ρ_g)
    N_SCAN = 20000
    rhos = np.linspace(1e-5, 0.49, N_SCAN)
    ps = np.array([cs_pressure(r, a, b, R, T) for r in rhos])

    # ── Identify stable branches via spinodal points ──
    dps = np.diff(ps) / np.diff(rhos)
    neg = np.where(dps < -1e-12)[0]
    if len(neg) < 10:
        return None

    idx_gas_spin = neg[0]  # gas spinodal: local max of p(ρ)
    idx_liq_spin = neg[-1] + 1  # liquid spinodal: local min of p(ρ)
    rho_gas_spin = rhos[idx_gas_spin]
    rho_liq_spin = rhos[idx_liq_spin]
    p_gas_spin = ps[idx_gas_spin]
    p_liq_spin = ps[idx_liq_spin]

    # Sanity: gas spinodal must be at lower ρ than liquid spinodal
    if rho_gas_spin >= rho_liq_spin:
        return None

    # ── Valid peq bracket ──
    # Gas branch (ρ < ρ_gas_spin): crossings exist for 0 < peq ≤ p_gas_spin
    # Liquid branch (ρ > ρ_liq_spin): crossings exist for peq ≥ p_liq_spin
    # → valid peq ∈ [max(p_liq_spin, ε), p_gas_spin]
    p_lo = max(p_liq_spin, 1e-15)
    p_hi = p_gas_spin
    if p_lo >= p_hi:
        return None

    def delta_mu(peq):
        """Δμ = μ_l − μ_g using ONLY stable-branch crossings."""
        # Quick rejection: peq outside valid range
        if peq <= 0 or peq >= p_gas_spin:
            return None

        all_crossings = []
        for i in range(len(rhos) - 1):
            fa, fb = ps[i] - peq, ps[i + 1] - peq
            if fa * fb <= 0:
                r = rhos[i] - fa * (rhos[i + 1] - rhos[i]) / (fb - fa + 1e-30)
                all_crossings.append(r)

        # Classify by stable branch
        gas_cross = [r for r in all_crossings if r < rho_gas_spin]
        liq_cross = [r for r in all_crossings if r > rho_liq_spin]

        if len(gas_cross) < 1 or len(liq_cross) < 1:
            return None

        # Rightmost on gas branch, leftmost on liquid branch
        rg = float(gas_cross[-1])
        rl = float(liq_cross[0])

        if rg <= 1e-10 or rl <= rg or rl > 0.49:
            return None

        return cs_chemical_potential(rl, a, b, R, T) - cs_chemical_potential(
            rg, a, b, R, T
        )

    # ── Scan for Δμ sign change (adaptive: finer near low peq) ──
    # Use log-spaced trials at low end + linear at high end for robustness
    if p_hi / max(p_lo, 1e-15) > 10:
        # Wide range: log-spaced for low peq, linear for high peq
        p_trials_log = np.logspace(max(np.log10(p_lo), -15), np.log10(p_hi * 0.5), 100)
        p_trials_lin = np.linspace(p_hi * 0.5, p_hi * 0.9999, 100)
        p_trials = np.unique(np.concatenate([p_trials_log, p_trials_lin]))
    else:
        p_trials = np.linspace(p_lo, p_hi * 0.9999, 200)

    dmu_vals = []
    for pt in p_trials:
        d = delta_mu(pt)
        if d is not None:
            dmu_vals.append((pt, d))

    if len(dmu_vals) < 2:
        return None

    sign_change = None
    for i in range(len(dmu_vals) - 1):
        if dmu_vals[i][1] * dmu_vals[i + 1][1] <= 0:
            sign_change = (dmu_vals[i][0], dmu_vals[i + 1][0])
            break
    if sign_change is None:
        return None

    # ── Brent root-find ──
    def _dmu_wrapped(p):
        d = delta_mu(p)
        return float(d) if d is not None else 1e10

    try:
        peq = brentq(
            _dmu_wrapped,
            sign_change[0],
            sign_change[1],
            xtol=1e-14,
            rtol=1e-12,
            maxiter=200,
        )
    except (ValueError, TypeError):
        return None

    # ── Final rg, rl (stable-branch filtered) ──
    final_crossings = []
    for i in range(len(rhos) - 1):
        fa, fb = ps[i] - peq, ps[i + 1] - peq
        if fa * fb <= 0:
            r = rhos[i] - fa * (rhos[i + 1] - rhos[i]) / (fb - fa + 1e-30)
            final_crossings.append(r)

    gas_final = [r for r in final_crossings if r < rho_gas_spin]
    liq_final = [r for r in final_crossings if r > rho_liq_spin]

    if gas_final and liq_final:
        rg = float(gas_final[-1])
        rl = float(liq_final[0])
        if rg > 1e-10 and rl > rg and rl < 0.49:
            return rg, rl, float(peq)
    return None


def coexistence_curve(a, b, R, T_vals):
    """Compute coexistence curve over temperature array."""
    Tc, _, _ = cs_critical_point(a, b, R)
    T_list, rg_list, rl_list, p_list = [], [], [], []
    for T in T_vals:
        if T >= Tc * 0.999:
            continue
        r = maxwell_coexistence(a, b, R, T)
        if r is None:
            continue
        rg, rl, peq = r
        T_list.append(T)
        rg_list.append(rg)
        rl_list.append(rl)
        p_list.append(peq)
    if not T_list:
        return None
    return {
        "T": np.array(T_list),
        "rho_g": np.array(rg_list),
        "rho_l": np.array(rl_list),
        "p_eq": np.array(p_list),
    }
