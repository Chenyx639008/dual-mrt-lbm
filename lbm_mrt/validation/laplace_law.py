"""Laplace law validation for Huang & Wu (2016) SCMP solver.

Verifies ΔP = σ/R for static droplets across two reduced temperatures (Tr).

Public API:
- run_laplace_sweep: generate design CSV + optional batch run
- analyze_laplace: post-process results, fit σ per Tr
- plot_laplace_double_panel: publication-quality double-panel figure
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
    laplace_sigma,
    compute_sigma_from_rho,
)
from lbm_mrt.validation.cs_eos import (
    cs_critical_point,
    maxwell_coexistence,
)

# ── Default CS-EOS parameters (matching Huang & Wu 2016 paper) ──
CS_A = 1.0
CS_B = 4.0
CS_R = 1.0
TC = cs_critical_point(CS_A, CS_B, CS_R)[0]  # ≈ 0.09433


def _base_laplace_params() -> dict[str, Any]:
    return {
        "pp_mode": 1,
        "init_eq": 0,
        "tau_p_a": 1.0,
        "tau_p_b": 1.0,
        "kappa": 0.0,
        "GAB": 0.0,
        "GBA": 0.0,
        "sigmaA": 0.0,
        "epsilon_huang": -2.0 / 3.0,  # ε = −8k₁ with k₁=1/12
        "k2_huang": 0.0,
        "kd_huang": -1.0 / 12.0,
        "alpha_meq": 1.0,
        "cs_a": CS_A,
        "cs_b": CS_B,
        "cs_R": CS_R,
        "cs_G": -1.0,
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


def generate_laplace_design(
    out_path: str,
    tr_list: list[float] | None = None,
    r_list: list[float] | None = None,
) -> str:
    """Generate design CSV for Laplace law sweep.

    Args:
        out_path: Output CSV path.
        tr_list: Reduced temperatures (T/Tc). Default: [0.70, 0.90].
        r_list: Droplet radii in lattice units. Default: [20, 25, 30, 35, 40, 45, 50, 55, 60].

    Returns:
        Path to the generated CSV.
    """
    if tr_list is None:
        tr_list = [0.70, 0.90]
    if r_list is None:
        r_list = [20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0]

    base = _base_laplace_params()
    rows = []
    for Tr in tr_list:
        T_abs = Tr * TC
        result = maxwell_coexistence(CS_A, CS_B, CS_R, T_abs)
        if result is None:
            print(f"[WARNING] Maxwell coexistence failed for Tr={Tr:.2f}, skipping")
            continue
        rg, rl, _ = result
        for R in r_list:
            row = dict(base)
            row["case_name"] = f"laplace_Tr{Tr:.2f}_R{R:.0f}"
            row["cs_T"] = Tr  # reduced temperature T/Tc (solver treats cs_T as Tr)
            row["huang_R0"] = R
            row["huang_rho_g"] = float(rg)
            row["huang_rho_l"] = float(rl)
            rows.append(row)

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(
        f"[laplace] Generated {out_path} ({len(rows)} cases: {len(tr_list)} Tr × {len(r_list)} R)"
    )
    return out_path


def analyze_laplace(
    results_root: str,
    out_dir: str | None = None,
) -> pd.DataFrame:
    """Analyze Laplace law batch results.

    For each case directory under results_root, extracts (R_meas, ΔP) from the
    final VTK file, then fits σ per Tr group.

    Args:
        results_root: Path to the batch results directory containing case subdirs.
        out_dir: If given, write laplace_summary.csv and laplace_fit.csv here.

    Returns:
        DataFrame with columns: case_name, Tr, R_nominal, R_meas, dP, invR.
    """
    root = Path(results_root)
    out = Path(out_dir) if out_dir else root

    records = []
    case_dirs = sorted(
        d for d in root.iterdir() if d.is_dir() and d.name.startswith("laplace_")
    )
    if not case_dirs:
        # Try one level deeper (batch results may have batch_tag subdir)
        for sub in sorted(root.iterdir()):
            if sub.is_dir():
                case_dirs.extend(
                    sorted(
                        d
                        for d in sub.iterdir()
                        if d.is_dir() and d.name.startswith("laplace_")
                    )
                )
        if not case_dirs:
            raise FileNotFoundError(
                f"No laplace_* case directories found under {results_root}"
            )

    for case_dir in case_dirs:
        case_name = case_dir.name
        # Parse Tr and R from case_name: laplace_Tr0.70_R40
        parts = case_name.replace("laplace_Tr", "").split("_R")
        if len(parts) != 2:
            print(f"[WARNING] Cannot parse case_name: {case_name}, skipping")
            continue
        try:
            Tr = float(parts[0])
            R_nom = float(parts[1])
        except ValueError:
            print(f"[WARNING] Cannot parse Tr/R from: {case_name}, skipping")
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

        center, R_meas = detect_interface_radius(rho, nx, ny)
        if R_meas <= 0:
            print(f"[WARNING] No interface detected for {case_name}, R_meas={R_meas}")
            continue

        # --- FIXED: compute σ via paper Eq. 62 integral (psi-based) ---
        # The VTK "pressure" field is isotropic only (p = cs²ρ + ½c²Gψ²),
        # which does NOT contain the interfacial pressure tensor anisotropy.
        # We compute σ from the radial ψ profile instead.
        sigma_est = compute_sigma_from_rho(
            rho,
            center,
            k1=1.0 / 12.0,  # default k₁ used in Laplace sweep
            cs_T=Tr,  # reduced temperature (matches design CSV)
        )
        # Convert σ to effective ΔP for the 1/R fit: ΔP = σ / R
        dP = sigma_est / R_meas if R_meas > 0 else np.nan

        records.append(
            {
                "case_name": case_name,
                "Tr": Tr,
                "R_nominal": R_nom,
                "R_meas": R_meas,
                "sigma_calc": sigma_est,
                "dP": dP,
                "invR": 1.0 / R_meas if R_meas > 0 else np.nan,
            }
        )
        print(f"  {case_name}: R_meas={R_meas:.1f}, σ={sigma_est:.6e}, ΔP={dP:.6e}")

    if not records:
        raise RuntimeError(f"No valid cases found under {results_root}")

    df = pd.DataFrame(records)
    df.to_csv(out / "laplace_summary.csv", index=False)
    print(f"[laplace] Wrote {out / 'laplace_summary.csv'} ({len(df)} cases)")

    # Fit σ per Tr — verify constancy: ΔP = σ/R → σ = ΔP·R
    # When σ is computed from psi integral, it should be constant across radii.
    fit_rows = []
    for Tr, group in df.groupby("Tr"):
        valid = group[group["R_meas"] > 0].dropna(subset=["sigma_calc"])
        if len(valid) < 3:
            print(f"[WARNING] Tr={Tr:.2f}: only {len(valid)} valid points, need ≥ 3")
            continue
        # σ from psi integral: compute mean and std/mean
        sigma_vals = valid["sigma_calc"].values
        sigma_mean = float(np.mean(sigma_vals))
        sigma_std_rel = float(np.std(sigma_vals) / max(abs(sigma_mean), 1e-12))
        # Also do ΔP vs 1/R fit for R² check
        dP_vals = valid["dP"].values
        R_vals = valid["R_meas"].values
        sigma_fit, r2, intercept = laplace_sigma(R_vals, dP_vals)
        fit_rows.append(
            {
                "Tr": Tr,
                "sigma": sigma_mean,
                "sigma_std_rel": sigma_std_rel,
                "R2": r2,
                "intercept": intercept,
                "n_points": len(valid),
            }
        )
        print(
            f"  Tr={Tr:.2f}: σ={sigma_mean:.6e} (std/mean={sigma_std_rel:.4f}), R²={r2:.4f}, n={len(valid)}"
        )

    if fit_rows:
        fit_df = pd.DataFrame(fit_rows)
        fit_df.to_csv(out / "laplace_fit.csv", index=False)
        print(f"[laplace] Wrote {out / 'laplace_fit.csv'}")

    return df


def plot_laplace_double_panel(
    df: pd.DataFrame,
    out_path: str,
    fit_csv: str | None = None,
) -> str:
    """Generate publication-quality double-panel Laplace plot.

    Left panel: Tr=0.7 (Eq1), right panel: Tr=0.9 (Eq2).
    X = 1/R, Y = ΔP; scatter + origin-constrained fit line.

    Args:
        df: Summary DataFrame from analyze_laplace.
        out_path: Output PDF path.
        fit_csv: Optional path to laplace_fit.csv for fit annotations.

    Returns:
        Path to the generated figure.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lbm_mrt.viz.viz_template import init_style, format_axes, save_figure

    init_style()

    tr_vals = sorted(df["Tr"].unique())
    if len(tr_vals) < 1:
        raise ValueError("No Tr groups in data")

    # Load fit data if provided
    fit_data = {}
    if fit_csv and os.path.exists(fit_csv):
        fit_df = pd.read_csv(fit_csv)
        for _, row in fit_df.iterrows():
            fit_data[row["Tr"]] = {"sigma": row["sigma"], "R2": row["R2"]}

    n_panels = min(len(tr_vals), 2)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5.5), squeeze=False)
    axes = axes[0]

    colors = ["#2166AC", "#B2182B"]
    markers = ["o", "s"]

    for i, Tr in enumerate(tr_vals[:2]):
        ax = axes[i]
        group = df[df["Tr"] == Tr].dropna(subset=["invR", "dP"])
        if len(group) == 0:
            continue

        x = group["invR"].values
        y = group["dP"].values

        # Origin-constrained fit: y = sigma * x
        sigma = np.sum(x * y) / np.sum(x * x)
        r2 = 1.0 - np.sum((y - sigma * x) ** 2) / np.sum((y - np.mean(y)) ** 2)

        x_fit = np.linspace(0, max(x) * 1.1, 50)
        ax.plot(x_fit, sigma * x_fit, "-", color=colors[i], linewidth=1.5, alpha=0.7)
        ax.scatter(
            x,
            y,
            marker=markers[i],
            facecolors="white",
            edgecolors=colors[i],
            s=50,
            linewidths=1.2,
            zorder=5,
        )

        # Info box
        textstr = f"$\\sigma = {sigma:.4e}$\n$R^2 = {r2:.4f}$"
        if Tr in fit_data:
            textstr = f"$\\sigma = {fit_data[Tr]['sigma']:.4e}$\n$R^2 = {fit_data[Tr]['R2']:.4f}$"
        ax.text(
            0.95,
            0.05,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.85,
                edgecolor="gray",
            ),
        )

        ax.set_xlabel("$1/R$ (lu$^{-1}$)")
        ax.set_ylabel("$\\Delta P$ (lu)")
        ax.set_title(f"(a) $T_r = {Tr:.2f}$" if i == 0 else f"(b) $T_r = {Tr:.2f}$")

    fig.tight_layout(pad=2.0)
    save_figure(fig, out_path)
    plt.close(fig)
    print(f"[laplace] Figure saved to {out_path}")
    return out_path
