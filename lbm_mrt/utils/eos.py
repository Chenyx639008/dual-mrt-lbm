"""Peng-Robinson Equation of State utilities for LBM unit conversion.

Provides functions to:
- Compute alpha(T) for the PR-EOS temperature correction.
- Evaluate the PR-EOS pressure given density and temperature.
- Convert between (Tc, Pc) and (a, b) PR-EOS parameters.
- Find the critical density given (a, b, R).

These are used to map LBM lattice units to physical units for
methane-water two-phase simulations.
"""

from __future__ import annotations

from math import sqrt

import numpy as np
from scipy.optimize import brentq, minimize_scalar


def alpha_PR(T: float, Tc: float, omega: float) -> float:
    """Compute the Peng-Robinson temperature correction factor alpha(T).

    Args:
        T:     Temperature (same units as Tc).
        Tc:    Critical temperature.
        omega: Acentric factor.

    Returns:
        alpha value; equals 1.0 exactly when T == Tc.
    """
    m = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
    return (1.0 + m * (1.0 - np.sqrt(T / Tc))) ** 2


def pr_eos(
    rho: float,
    T: float,
    a: float,
    b: float,
    R: float,
    Tc: float,
    omega: float,
) -> float:
    """Evaluate the Peng-Robinson EOS pressure P(rho, T).

    Args:
        rho:   Molar density.
        T:     Temperature.
        a:     PR-EOS attractive parameter.
        b:     PR-EOS repulsive parameter (co-volume).
        R:     Gas constant (in lattice or SI units).
        Tc:    Critical temperature (for alpha_PR computation).
        omega: Acentric factor.

    Returns:
        Pressure value, or np.inf near singularities.
    """
    aT = a * alpha_PR(T, Tc, omega)
    denom1 = 1.0 - b * rho
    denom2 = 1.0 + 2.0 * b * rho - (b * b) * (rho * rho)
    if denom1 <= 0.0 or denom2 <= 0.0:
        return np.inf
    return (rho * R * T) / denom1 - (aT * rho * rho) / denom2


def compute_ab_from_Tc_Pc(Tc: float, Pc: float, R: float = 1.0) -> tuple[float, float]:
    """Compute PR-EOS parameters a and b from critical properties.

    Args:
        Tc: Critical temperature.
        Pc: Critical pressure.
        R:  Gas constant (default 1.0 for dimensionless LBM units).

    Returns:
        Tuple (a, b).
    """
    A = 0.457235
    B = 0.077796
    a = A * R**2 * Tc**2 / Pc
    b = B * R * Tc / Pc
    return a, b


def compute_Tc_Pc_from_ab(a: float, b: float, R: float = 1.0) -> tuple[float, float]:
    """Compute critical temperature and pressure from PR-EOS parameters.

    Args:
        a: PR-EOS attractive parameter.
        b: PR-EOS repulsive parameter.
        R: Gas constant (default 1.0).

    Returns:
        Tuple (Tc, Pc).
    """
    A = 0.457235
    B = 0.077796
    Tc = (B / A) * a / (b * R)
    Pc = (B**2 / A) * a / (b**2)
    return Tc, Pc


def find_rho_c_given_Tc_Pc(
    a: float,
    b: float,
    R: float,
    omega: float,
    Tc: float,
    Pc: float,
) -> tuple[float, str]:
    """Find the critical density rho_c by solving P(Tc, rho_c) = Pc.

    Uses Brent's method if a sign change is found in the search interval;
    falls back to scalar minimization of |P - Pc| otherwise.

    Args:
        a, b:   PR-EOS parameters.
        R:      Gas constant.
        omega:  Acentric factor.
        Tc, Pc: Critical temperature and pressure.

    Returns:
        Tuple (rho_c, method_str).
    """
    hi_safe = min(0.999 * (1.0 / b), 0.999 * ((1.0 + sqrt(2.0)) / b))
    lo_safe = max(1e-9, 0.0)

    def f(rho: float) -> float:
        return pr_eos(rho, Tc, a, b, R, Tc, omega) - Pc

    xs = np.linspace(lo_safe, hi_safe, 400)
    vals = [f(x) if np.isfinite(f(x)) else np.nan for x in xs]

    for i in range(len(xs) - 1):
        v1, v2 = vals[i], vals[i + 1]
        if np.isfinite(v1) and np.isfinite(v2) and v1 * v2 < 0.0:
            rho_c = brentq(f, xs[i], xs[i + 1], maxiter=200, xtol=1e-12)
            return rho_c, "brentq"

    res = minimize_scalar(
        lambda r: abs(f(r)),
        bounds=(lo_safe, hi_safe),
        method="bounded",
        options={"xatol": 1e-12, "maxiter": 500},
    )
    return float(res.x), "minimize_scalar"


def critical_from_abR(
    a: float,
    b: float,
    R: float,
    omega: float,
    fluid_name: str = "fluid",
) -> tuple[float, float, float, float]:
    """Compute and print all critical properties from PR-EOS (a, b, R, omega).

    Args:
        a, b:       PR-EOS parameters.
        R:          Gas constant.
        omega:      Acentric factor.
        fluid_name: Label for printed output.

    Returns:
        Tuple (Tc, Pc, alpha_Tc, rho_c).
    """
    Tc, Pc = compute_Tc_Pc_from_ab(a, b, R)
    alpha_Tc = alpha_PR(Tc, Tc, omega)
    rho_c, method = find_rho_c_given_Tc_Pc(a, b, R, omega, Tc, Pc)

    print(f"--- {fluid_name} ---")
    print(f"a={a:.8g}, b={b:.8g}, R={R:.8g}, omega={omega}")
    print(f"Tc = {Tc:.10f}")
    print(f"Pc = {Pc:.10f}")
    print(f"alpha(Tc) = {alpha_Tc:.10f}  (should be 1.0)")
    print(f"rho_c = {rho_c:.10f}  (method: {method})")

    p_err = pr_eos(rho_c, Tc, a, b, R, Tc, omega) - Pc
    print(f"|P_EOS(Tc,rho_c)-Pc| = {abs(p_err):.3e}")
    return Tc, Pc, alpha_Tc, rho_c


if __name__ == "__main__":
    # Demo: LBM water-like fluid (a=2/49, b=2/21, R=1, omega=0.344)
    a_w = 2 / 49
    b_w = 2.0 / 21.0
    R_w = 1.0
    omega_w = 0.344
    critical_from_abR(a_w, b_w, R_w, omega_w, fluid_name="water(LBM)")

    Tc_ref = 0.036461
    Pc_ref = 0.026813
    a_calc, b_calc = compute_ab_from_Tc_Pc(Tc_ref, Pc_ref, R_w)
    print(f"\nFrom (Tc, Pc) back-computed: a={a_calc:.8g}, b={b_calc:.8g}")
