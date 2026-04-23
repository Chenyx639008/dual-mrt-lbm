#!/usr/bin/env python3
"""Validate LBM hydrate dissociation benchmark test results.

Reads VTK output from a simulation results directory and checks physics
against analytical or expected solutions for five benchmark scenarios.

Usage::

    uv run python scripts/03_validate_benchmarks.py --bm BM1 --dir results/smoke_test
    uv run python scripts/03_validate_benchmarks.py --all --dir results/smoke_test
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from lbm_mrt.io.vtk_reader import latest_vtk, read_vtk_scalars


# ── BM-1: Pure thermal diffusion ──────────────────────────────────────────
def validate_bm1_diffusion(folder: str, T_inlet: float = 285.0, T_init: float = 278.15) -> None:
    print("\n===== BM-1: Pure Thermal Diffusion =====")
    vtk = latest_vtk(folder)
    fields, nx, ny = read_vtk_scalars(vtk)
    if "temperature" not in fields:
        print("  [SKIP] 'temperature' field not found (requires mcmp_sim_hydrate)")
        return

    T = fields["temperature"]
    T_profile = T.mean(axis=1)
    y_fluid = np.arange(1, ny - 1)
    T_fluid = T_profile[1:-1]
    coeffs = np.polyfit(y_fluid, T_fluid, 1)
    T_fit = np.polyval(coeffs, y_fluid)
    residuals = T_fluid - T_fit
    l2_err = np.sqrt(np.mean(residuals**2)) / (T_inlet - T_init + 1e-30)
    print(f"  T profile L2 relative error = {l2_err:.2e}  (pass < 1e-3)")
    if l2_err < 1e-3:
        print("  [PASS] Steady-state temperature profile is linear")
    else:
        print("  [FAIL] Residual too large; check thermal BC or run more steps")


# ── BM-2: Conjugate heat transfer ─────────────────────────────────────────
def validate_bm2_conjugate(folder: str, lambda_f: float = 0.6, lambda_s: float = 0.9) -> None:
    print("\n===== BM-2: Conjugate Heat Transfer =====")
    vtk = latest_vtk(folder)
    fields, nx, ny = read_vtk_scalars(vtk)
    if "temperature" not in fields or "flag" not in fields:
        print("  [SKIP] Required fields not found")
        return

    T = fields["temperature"]
    flag = fields["flag"]
    expected_ratio = lambda_s / lambda_f

    y_mid = ny // 2
    T_row = T[y_mid, :]
    f_row = flag[y_mid, :]
    transitions = np.where(np.diff((f_row == -2).astype(int)) != 0)[0]
    if len(transitions) < 2:
        print("  [SKIP] Insufficient solid-liquid interfaces found")
        return

    ratios = []
    for xi in transitions[:4]:
        xf = np.arange(max(0, xi - 3), xi + 1)
        xs = np.arange(xi + 1, min(nx, xi + 5))
        if len(xf) < 2 or len(xs) < 2:
            continue
        dTdx_f = np.polyfit(xf, T_row[xf], 1)[0]
        dTdx_s = np.polyfit(xs, T_row[xs], 1)[0]
        if abs(dTdx_f) > 1e-10:
            ratios.append(abs(dTdx_s / dTdx_f))

    if not ratios:
        print("  [SKIP] Cannot compute gradient ratio")
        return

    measured_ratio = np.mean(ratios)
    rel_err = abs(measured_ratio - expected_ratio) / expected_ratio
    print(f"  lambda_s/lambda_f = {expected_ratio:.3f},  dT_s/dT_f = {measured_ratio:.3f},  err = {rel_err:.2%}")
    if rel_err < 0.10:
        print("  [PASS] Conjugate heat transfer satisfies lambda ratio (error < 10%)")
    else:
        print("  [FAIL] Gradient ratio deviation too large")


# ── BM-3: Reaction-diffusion steady state ────────────────────────────────
def validate_bm3_reactive(
    folder: str,
    D_mol: float = 1.85e-9,
    k0: float = 1.0,
    dx_phys: float = 1e-5,
    dt_phys: float = 1e-6,
) -> None:
    print("\n===== BM-3: Reaction-Diffusion =====")
    vtk = latest_vtk(folder)
    fields, nx, ny = read_vtk_scalars(vtk)
    if "concentration" not in fields:
        print("  [SKIP] 'concentration' field not found")
        return

    Cm = fields["concentration"].mean(axis=1)
    D_latt = D_mol * dt_phys / dx_phys**2
    k_r_latt = k0 * dt_phys / dx_phys
    L_r_analytic = np.sqrt(D_latt / (k_r_latt + 1e-30))

    y_fluid = np.arange(1, ny // 2)
    Cm_fluid = Cm[y_fluid]
    Cm_inf = Cm_fluid.max()
    if Cm_inf < 1e-12:
        print("  [SKIP] Concentration field is zero (too few steps or reaction rate too small)")
        return

    from scipy.optimize import curve_fit
    y_b = ny // 2

    def model(y, L_r):
        return Cm_inf * (1.0 - np.exp(-(y_b - y) / L_r))

    try:
        popt, _ = curve_fit(model, y_fluid, Cm_fluid, p0=[L_r_analytic])
        L_r_sim = popt[0]
        rel_err = abs(L_r_sim - L_r_analytic) / max(L_r_analytic, 1e-6)
        print(f"  L_r analytic = {L_r_analytic:.3f} latt,  L_r sim = {L_r_sim:.3f} latt,  err = {rel_err:.2%}")
        if rel_err < 0.05:
            print("  [PASS] Reaction-diffusion length scale error < 5%")
        else:
            print("  [FAIL] Length scale deviation too large (check k0_rxn or step count)")
    except Exception as e:
        print(f"  [SKIP] Curve fit failed: {e}")


# ── BM-4: VOP mass conservation ──────────────────────────────────────────
def validate_bm4_vop(folder: str, Vm_hydrate: float = 2.274e-5, dx_phys: float = 1e-5) -> None:
    print("\n===== BM-4: VOP Mass Conservation =====")
    Vm_latt = Vm_hydrate / dx_phys**3

    vtk_files = sorted(glob.glob(os.path.join(folder, "outputdata_flow*.vtk")))
    if len(vtk_files) < 3:
        print("  [SKIP] Fewer than 3 output snapshots; need more steps for trend analysis")
        return

    masses = []
    for vf in vtk_files:
        try:
            fields, nx, ny = read_vtk_scalars(vf)
        except Exception:
            continue
        if "hydrate_Vh" not in fields or "concentration" not in fields:
            continue
        Vh = fields["hydrate_Vh"]
        Cm = fields["concentration"]
        total = Vh.sum() + Cm.sum() * Vm_latt
        masses.append(total)

    if len(masses) < 2:
        print("  [SKIP] Insufficient valid snapshots")
        return

    masses = np.array(masses)
    drift = np.abs(masses - masses[0]) / max(masses[0], 1e-12)
    max_drift = drift.max()
    print(f"  Max mass drift = {max_drift:.4%}  (pass < 0.5%)")
    if max_drift < 0.005:
        print("  [PASS] VOP mass conservation satisfied")
    else:
        print("  [FAIL] Mass drift too large")


# ── BM-5: Full coupling trend ─────────────────────────────────────────────
def validate_bm5_full_coupling(folder: str) -> None:
    print("\n===== BM-5: Full Coupling Trend =====")
    vtk_files = sorted(glob.glob(os.path.join(folder, "outputdata_flow*.vtk")))
    if len(vtk_files) < 3:
        print("  [SKIP] Insufficient snapshots")
        return

    Vh_means, T_means, Cm_means = [], [], []
    for vf in vtk_files:
        try:
            fields, nx, ny = read_vtk_scalars(vf)
        except Exception:
            continue
        if "hydrate_Vh" in fields:
            Vh_means.append(fields["hydrate_Vh"].mean())
        if "temperature" in fields:
            T_means.append(fields["temperature"].mean())
        if "concentration" in fields:
            Cm_means.append(fields["concentration"].mean())

    results = []
    if len(Vh_means) >= 2:
        mono = all(Vh_means[i] <= Vh_means[i - 1] + 1e-6 for i in range(1, len(Vh_means)))
        results.append(("Vh monotonically decreasing", mono))
    if len(T_means) >= 2:
        results.append(("T_mean changes non-trivially", abs(T_means[-1] - T_means[0]) > 1e-4))
    if len(Cm_means) >= 2:
        results.append(("Cm_mean increases from zero", Cm_means[-1] > Cm_means[0]))

    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")


# ── Main ──────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Validate LBM hydrate benchmark results")
    parser.add_argument("--bm", choices=["BM1", "BM2", "BM3", "BM4", "BM5"],
                        help="Run a specific benchmark")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--dir", default="results", help="Results directory containing VTK files")
    args = parser.parse_args()

    d = args.dir
    run_all = args.all
    if run_all or args.bm == "BM1": validate_bm1_diffusion(d)
    if run_all or args.bm == "BM2": validate_bm2_conjugate(d)
    if run_all or args.bm == "BM3": validate_bm3_reactive(d)
    if run_all or args.bm == "BM4": validate_bm4_vop(d)
    if run_all or args.bm == "BM5": validate_bm5_full_coupling(d)


if __name__ == "__main__":
    main()
