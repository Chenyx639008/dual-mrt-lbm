"""σ-decoupling sweep for Huang & Wu (2016) SCMP solver.

Verifies the headline claim of Huang & Wu (2016) Eq. 62:
  σ ∝ (1 − 6·k₁), with ρ_l and ρ_g invariant under k₁.

Public API:
- generate_decoupling_design: create design CSV for k₁ sweep
- analyze_decoupling: post-process results, verify σ ∝ (1−6k₁) and ρ drift
- plot_decoupling: double-panel figure (σ-k₁ line + ρ drift)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lbm_mrt.io.vtk_reader import latest_vtk, read_vtk_scalars
from lbm_mrt.validation.analytical import (
    detect_interface_radius,
    fit_pressure_inside_outside,
)
from lbm_mrt.validation.cs_eos import (
    cs_critical_point,
    maxwell_coexistence,
)

CS_A = 1.0
CS_B = 4.0
CS_R = 1.0
TC = cs_critical_point(CS_A, CS_B, CS_R)[0]


def _base_decoupling_params() -> dict[str, Any]:
    return {
        "pp_mode": 1,
        "init_eq": 0,
        "tau_p_a": 1.0,
        "tau_p_b": 1.0,
        "kappa": 0.0,
        "GAB": 0.0,
        "GBA": 0.0,
        "sigmaA": 0.0,
        "k2_huang": 0.0,
        "kd_huang": -1.0 / 12.0,
        "alpha_meq": 1.0,
        "cs_a": CS_A,
        "cs_b": CS_B,
        "cs_R": CS_R,
        "cs_G": -1.0,
        "huang_R0": 40.0,
        "huang_xc": 128.0,
        "huang_yc": 128.0,
        "huang_W": 3.0,
        "huang_init_mode": 1,
        "Gx": 0.0,
        "Gy": 0.0,
        "drive_mode": 1,
        "ENABLE_CKPT": False,
        "OUTPUT_EVERY": 5000,
    }


def generate_decoupling_design(
    out_path: str,
    tr: float = 0.70,
    k1_list: list[float] | None = None,
) -> str:
    """Generate design CSV for σ-decoupling sweep.

    Args:
        out_path: Output CSV path.
        tr: Reduced temperature (T/Tc). Default: 0.70 (strong density ratio).
        k1_list: k₁ values. Default: [0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15].

    Returns:
        Path to the generated CSV.
    """
    if k1_list is None:
        k1_list = [0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15]

    T_abs = tr * TC
    result = maxwell_coexistence(CS_A, CS_B, CS_R, T_abs)
    if result is None:
        raise RuntimeError(f"Maxwell coexistence failed for Tr={tr:.2f}")
    rg, rl, _ = result

    base = _base_decoupling_params()
    rows = []
    for k1 in k1_list:
        row = dict(base)
        row["case_name"] = f"decoupling_Tr{tr:.2f}_k1{k1:.4f}"
        row["cs_T"] = T_abs
        row["k1_huang"] = k1
        row["huang_rho_g"] = float(rg)
        row["huang_rho_l"] = float(rl)
        rows.append(row)

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[decoupling] Generated {out_path} ({len(rows)} cases, k₁ ∈ {k1_list})")
    return out_path


def analyze_decoupling(
    results_root: str,
    out_dir: str | None = None,
) -> pd.DataFrame:
    """Analyze σ-decoupling batch results.

    For each case, extracts σ (single-R estimate), ρ_l, ρ_g.
    Verifies:
      (a) σ vs (1−6k₁) linear fit with R² ≥ 0.99
      (b) ρ_l and ρ_g drift < 1%.

    Args:
        results_root: Path to the batch results directory.
        out_dir: If given, write decoupling_summary.csv here.

    Returns:
        DataFrame with columns: k1, sigma, rho_l, rho_g.
    """
    root = Path(results_root)
    out = Path(out_dir) if out_dir else root

    records = []
    case_dirs = sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("decoupling_"))
    if not case_dirs:
        for sub in sorted(root.iterdir()):
            if sub.is_dir():
                case_dirs.extend(sorted(d for d in sub.iterdir() if d.is_dir() and d.name.startswith("decoupling_")))
        if not case_dirs:
            raise FileNotFoundError(f"No decoupling_* case directories under {results_root}")

    for case_dir in case_dirs:
        case_name = case_dir.name
        # Parse k1: decoupling_Tr0.70_k10.0500
        try:
            k1_str = case_name.split("_k1")[-1]
            k1 = float(k1_str)
        except (ValueError, IndexError):
            print(f"[WARNING] Cannot parse k1 from: {case_name}, skipping")
            continue

        vtk_dir = case_dir / "outputdata_scmp"
        if not vtk_dir.exists():
            print(f"[WARNING] No outputdata_scmp in {case_dir}, skipping")
            continue

        try:
            vtk_path = latest_vtk(str(vtk_dir), "flow")
            fields, nx, ny = read_vtk_scalars(vtk_path)
        except Exception as e:
            print(f"[WARNING] Failed to read VTK for {case_name}: {e}")
            continue

        rho = fields["rho"]
        pressure = fields.get("pressure", np.zeros_like(rho))

        center, R_meas = detect_interface_radius(rho, nx, ny)
        if R_meas <= 0:
            print(f"[WARNING] No interface detected for {case_name}")
            continue

        p_in, p_out = fit_pressure_inside_outside(rho, pressure, center, R_meas)
        sigma = (p_in - p_out) * R_meas  # single-R estimate

        # Extract plateau densities
        rho_vals = rho.flatten()
        rho_sorted = np.sort(rho_vals)
        n = len(rho_sorted)
        rho_g = float(np.mean(rho_sorted[: max(1, int(0.05 * n))]))
        rho_l = float(np.mean(rho_sorted[-max(1, int(0.05 * n)):]))

        records.append({
            "case_name": case_name,
            "k1": k1,
            "one_minus_6k1": 1.0 - 6.0 * k1,
            "sigma": sigma,
            "rho_l": rho_l,
            "rho_g": rho_g,
            "R_meas": R_meas,
        })
        print(f"  {case_name}: k₁={k1:.4f}, σ={sigma:.6e}, ρ_l={rho_l:.4f}, ρ_g={rho_g:.4f}")

    if not records:
        raise RuntimeError(f"No valid cases found under {results_root}")

    df = pd.DataFrame(records)
    df.to_csv(out / "decoupling_summary.csv", index=False)
    print(f"[decoupling] Wrote {out / 'decoupling_summary.csv'} ({len(df)} cases)")

    # Verification (a): σ vs (1−6k₁)
    valid = df.dropna(subset=["sigma", "one_minus_6k1"])
    if len(valid) >= 3:
        x = valid["one_minus_6k1"].values
        y = valid["sigma"].values
        slope = np.sum(x * y) / np.sum(x * x)
        r2_sigma = 1.0 - np.sum((y - slope * x) ** 2) / np.sum((y - np.mean(y)) ** 2)
        print(f"[decoupling] σ vs (1−6k₁): slope={slope:.6e}, R²={r2_sigma:.4f}")
        print(f"  PASS: R² ≥ 0.99" if r2_sigma >= 0.99 else f"  FAIL: R² < 0.99")

    # Verification (b): ρ_l, ρ_g drift
    if len(valid) >= 2:
        rho_l_std_rel = float(valid["rho_l"].std() / max(valid["rho_l"].mean(), 1e-12))
        rho_g_std_rel = float(valid["rho_g"].std() / max(valid["rho_g"].mean(), 1e-12))
        print(f"[decoupling] ρ_l drift (std/mean): {rho_l_std_rel:.4f}")
        print(f"[decoupling] ρ_g drift (std/mean): {rho_g_std_rel:.4f}")
        print(f"  PASS: drift < 1%" if max(rho_l_std_rel, rho_g_std_rel) < 0.01 else f"  FAIL: drift ≥ 1%")

    return df


def plot_decoupling(
    df: pd.DataFrame,
    out_path: str,
) -> str:
    """Generate publication-quality σ-decoupling figure.

    Double panel:
      (a) σ vs (1−6k₁) — origin-constrained linear fit
      (b) ρ_l, ρ_g vs k₁ — drift bar chart

    Args:
        df: Summary DataFrame from analyze_decoupling.
        out_path: Output PDF path.

    Returns:
        Path to the generated figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lbm_mrt.viz.viz_template import init_style, format_axes, save_figure

    init_style()

    valid = df.dropna(subset=["sigma", "one_minus_6k1"]).sort_values("k1")
    if len(valid) < 2:
        raise ValueError(f"Need ≥ 2 valid points, got {len(valid)}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ── Panel (a): σ vs (1−6k₁) ──
    x = valid["one_minus_6k1"].values
    y = valid["sigma"].values
    slope = np.sum(x * y) / np.sum(x * x)
    r2 = 1.0 - np.sum((y - slope * x) ** 2) / np.sum((y - np.mean(y)) ** 2)

    x_fit = np.linspace(0, max(x) * 1.1, 50)
    ax1.plot(x_fit, slope * x_fit, "-", color="#2166AC", linewidth=1.5, alpha=0.5)
    ax1.scatter(x, y, marker="o", facecolors="white", edgecolors="#2166AC",
                s=60, linewidths=1.2, zorder=5)

    # Annotate k₁ values
    for _, row in valid.iterrows():
        ax1.annotate(f"$k_1={row['k1']:.3f}$", (row["one_minus_6k1"], row["sigma"]),
                     textcoords="offset points", xytext=(8, -4), fontsize=7, alpha=0.7)

    ax1.set_xlabel("$1 - 6k_1$")
    ax1.set_ylabel("$\\sigma$ (lu)")
    ax1.set_title(f"(a) $\\sigma \\propto (1-6k_1)$\nslope$={slope:.4e}$, $R^2={r2:.4f}$")

    # ── Panel (b): ρ_l, ρ_g drift ──
    k1_vals = valid["k1"].values
    rho_l_vals = valid["rho_l"].values
    rho_g_vals = valid["rho_g"].values
    rho_l_ref = rho_l_vals[0] if len(rho_l_vals) > 0 else 0
    rho_g_ref = rho_g_vals[0] if len(rho_g_vals) > 0 else 0

    rho_l_drift = 100.0 * (rho_l_vals - rho_l_ref) / max(rho_l_ref, 1e-12)
    rho_g_drift = 100.0 * (rho_g_vals - rho_g_ref) / max(rho_g_ref, 1e-12)

    x_pos = np.arange(len(k1_vals))
    width = 0.35
    bars1 = ax2.bar(x_pos - width / 2, rho_l_drift, width, color="#B2182B", alpha=0.7, label="$\\rho_l$")
    bars2 = ax2.bar(x_pos + width / 2, rho_g_drift, width, color="#2166AC", alpha=0.7, label="$\\rho_g$")
    ax2.axhline(y=0, color="gray", linewidth=0.5)
    ax2.axhline(y=1, color="red", linestyle="--", linewidth=0.5, alpha=0.5)
    ax2.axhline(y=-1, color="red", linestyle="--", linewidth=0.5, alpha=0.5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"{k:.3f}" for k in k1_vals], rotation=45, fontsize=8)
    ax2.set_xlabel("$k_1$")
    ax2.set_ylabel("Drift from $k_1=0$ (%)")
    ax2.set_title("(b) Density invariance")
    ax2.legend(fontsize=9)

    fig.tight_layout(pad=2.0)
    save_figure(fig, out_path)
    plt.close(fig)
    print(f"[decoupling] Figure saved to {out_path}")
    return out_path
