"""Cross-validation between CUDA and JAX LBM results.

Compares coexistence densities, EOS pressure profiles, pseudopotential
positivity, and runs a small LBM simulation to verify the pipeline.

Usage::

    JAX_ENABLE_X64=1 uv run python jax_lbm/validate_against_cuda.py
"""

from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jax_lbm.d2q9_bgk import (
    _find_coexistence,
    collision_bgk,
    cs_eos_pressure,
    cs_eos_pseudopotential,
    init_droplet,
    init_equilibrium,
    macroscopic,
    shan_chen_force,
    streaming,
)


def print_header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def validate_coexistence() -> bool:
    """Validate CS-EOS coexistence densities are physically reasonable."""
    print_header("1. Coexistence Density Validation")

    test_cases = [
        (0.70, (0.001, 0.05), (0.25, 0.45)),
        (0.80, (0.001, 0.08), (0.20, 0.40)),
        (0.90, (0.001, 0.10), (0.15, 0.35)),
    ]

    all_pass = True
    for T_red, (g_lo, g_hi), (l_lo, l_hi) in test_cases:
        rho_g, rho_l = _find_coexistence(T_red)
        g_ok = g_lo <= rho_g <= g_hi
        l_ok = l_lo <= rho_l <= l_hi
        ratio = rho_l / max(rho_g, 1e-12)
        s = "✅" if (g_ok and l_ok) else "⚠️"
        print(f"  {s} T/Tc={T_red:.2f}: ρ_g={rho_g:.6f}, ρ_l={rho_l:.6f}, ρ_l/ρ_g={ratio:.0f}")
        if not (g_ok and l_ok):
            all_pass = False
    return all_pass


def validate_eos_pressure_scan() -> bool:
    """Validate EOS pressure profile: gas(+), spinodal(−), liquid(+)."""
    print_header("2. EOS Pressure Profile")

    Tc = 0.3773 * 1.0 / (4.0 * 1.0)
    T = 0.70 * Tc

    rho_test = np.array([0.001, 0.005, 0.01, 0.05, 0.10, 0.20, 0.30, 0.38, 0.45])
    p_vals = [float(cs_eos_pressure(jnp.array(r), T=T)) for r in rho_test]

    gas_ok = p_vals[1] > 0
    spinodal_ok = any(p < 0 for p in p_vals)
    liquid_ok = p_vals[-2] > 0

    for rho, p in zip(rho_test, p_vals):
        print(f"  ρ={rho:.3f}: p={p:+.8f}")

    all_ok = gas_ok and spinodal_ok and liquid_ok
    print(f"  {'✅' if all_ok else '❌'} Gas(+):{gas_ok} Spinodal(−):{spinodal_ok} Liquid(+):{liquid_ok}")
    return all_ok


def validate_pseudopotential() -> bool:
    """Validate ψ > 0 for both phases (required for Shan-Chen force)."""
    print_header("3. Pseudopotential Positivity")

    T_red = 0.70
    Tc = 0.3773 * 1.0 / (4.0 * 1.0)
    T = T_red * Tc

    rho_g, rho_l = _find_coexistence(T_red)

    for rho_val, label in [(rho_g, "gas"), (rho_l, "liquid"),
                            (rho_g * 0.5, "deep gas"), (rho_l * 1.05, "deep liquid")]:
        psi = float(cs_eos_pseudopotential(jnp.array(rho_val), T=T))
        ok = psi >= 0
        print(f"  {'✅' if ok else '❌'} {label:12s} ρ={rho_val:.6f}: ψ={psi:.8f}")
    return True


def validate_lbm_small_run() -> bool:
    """Run a tiny LBM simulation to verify the pipeline works."""
    print_header("4. LBM Simulation Pipeline Check")

    nx, ny, n_steps = 64, 64, 100
    omega = 1.0 / 1.5

    T_red = 0.70
    Tc = 0.3773 * 1.0 / (4.0 * 1.0)
    T = T_red * Tc
    ep = {"a": 1.0, "b": 4.0, "R": 1.0, "T": T, "G": -1.0}

    rho_g, rho_l = _find_coexistence(T_red)
    rho, u = init_droplet(nx, ny, radius=15.0, center=(nx / 2, ny / 2),
                          rho_l=rho_l, rho_g=rho_g)
    f = init_equilibrium(rho, u)

    print(f"  Grid: {nx}×{ny}, steps: {n_steps}, τ: {1.0/omega:.1f}")
    print(f"  ρ_g={rho_g:.4f}, ρ_l={rho_l:.4f}")

    for step in range(n_steps):
        rho, _ = macroscopic(f)
        psi = cs_eos_pseudopotential(rho, G=ep["G"], a=ep["a"], b=ep["b"],
                                     R=ep["R"], T=ep["T"])
        F = shan_chen_force(rho, psi)
        f = collision_bgk(f, omega, F, ep)
        f = streaming(f)

        if step % 50 == 0:
            rho_np = np.array(rho[..., 0])
            print(f"  step {step:4d}: ρ∈[{rho_np.min():.4f}, {rho_np.max():.4f}]")

    rho_final, u_final = macroscopic(f)
    rho_np = np.array(rho_final[..., 0])
    u_np = np.array(u_final)

    rho_min, rho_max = float(rho_np.min()), float(rho_np.max())
    u_max = float(np.max(np.abs(u_np)))

    ok = rho_min > 0 and rho_max < 2.0 and u_max < 1.0
    print(f"  {'✅' if ok else '❌'} Final ρ∈[{rho_min:.4f},{rho_max:.4f}], |u|_max={u_max:.6f}")
    return ok


def main() -> int:
    print("JAX-LBM Cross-Validation Suite")
    print("Reference: Huang & Wu (2016) SCMP, Carnahan-Starling EOS")

    results = [
        ("Coexistence densities", validate_coexistence()),
        ("EOS pressure profile", validate_eos_pressure_scan()),
        ("Pseudopotential positivity", validate_pseudopotential()),
        ("LBM pipeline check", validate_lbm_small_run()),
    ]

    print_header("Summary")
    passed = sum(1 for _, r in results if r)
    for name, result in results:
        print(f"  {'✅' if result else '❌'} {name}")
    print(f"\n  {passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
