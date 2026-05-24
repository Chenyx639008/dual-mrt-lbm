"""Mesh convergence analysis for Huang SCMP solver.

Quantifies numerical error ε(NY) = |Q − Q_an| / |Q_an| for Poiseuille flow
as a function of grid resolution. Uses Richardson extrapolation to estimate
observed convergence order p_obs.

NOTE (P1): Requires solver-side multi-grid support (compile-time NX/NY injection)
and wall BC + body force in SCMP path. Until then, provides the analysis
framework adapted from legacy validation/mesh_convergence/mesh_convergence.py.

Public API:
- analyze_mesh_convergence: process multi-resolution batch results
- richardson_extrapolation: estimate p_obs and grid convergence index (GCI)
- plot_mesh_convergence: log-log ε vs NY with power-law fit
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
from lbm_mrt.validation.cs_eos import (
    cs_critical_point,
    maxwell_coexistence,
)
from lbm_mrt.validation.poiseuille_sp import (
    poiseuille_analytical,
    extract_centerline_ux,
)

CS_A = 1.0
CS_B = 4.0
CS_R = 1.0
TC = cs_critical_point(CS_A, CS_B, CS_R)[0]


def analyze_mesh_convergence(
    results_root: str,
    out_dir: str | None = None,
    tr: float = 0.70,
    tau: float = 1.5,  # matches tau_huang in configs/huang_scmp.yaml (MRT τ=1.5)
    Fb_ref: float = 5e-9,
    h_ref: float = 198.0,  # reference half-channel for NY=200
) -> pd.DataFrame:
    """Analyze mesh convergence from multi-resolution batch results.

    For each NY, computes ε = |Q_lbm − Q_an| / |Q_an|.
    Then fits log(ε) vs log(NY) to estimate observed convergence order p_obs.

    Args:
        results_root: Path to batch results directory.
        out_dir: Output directory for mesh_convergence.csv.
        tr: Reduced temperature.
        tau: Relaxation time.
        Fb_ref: Reference body force (at NY=200).
        h_ref: Reference half-channel (at NY=200).

    Returns:
        DataFrame with columns: NY, eps_Q, Q_lbm, Q_an, p_obs, GCI.
    """
    root = Path(results_root)
    out = Path(out_dir) if out_dir else root

    T_abs = tr * TC
    result = maxwell_coexistence(CS_A, CS_B, CS_R, T_abs)
    if result is None:
        raise RuntimeError(f"Maxwell coexistence failed for Tr={tr:.2f}")
    _, rho_l, _ = result
    nu = (tau - 0.5) / 3.0

    records = []
    case_dirs = sorted(
        d
        for d in root.iterdir()
        if d.is_dir()
        and (d.name.startswith("mesh_") or d.name.startswith("poiseuille_"))
    )

    for case_dir in case_dirs:
        vtk_dir = case_dir / "outputdata_scmp"
        if not vtk_dir.exists():
            vtk_dir = case_dir / "outputdata_eq"
        if not vtk_dir.exists():
            print(f"[WARNING] No VTK in {case_dir}, skipping")
            continue

        try:
            vtk_path = latest_vtk(str(vtk_dir), "flow")
            y, u_profile, nx, ny = extract_centerline_ux(vtk_path)
        except Exception as e:
            print(f"[WARNING] Failed for {case_dir.name}: {e}")
            continue

        b = (ny - 2) / 2.0
        # Scale body force: Fb(NY) = Fb_ref * (h_ref / h(NY))³
        # This keeps the Reynolds number constant across resolutions
        Fb = Fb_ref * (h_ref / b) ** 3
        u_an = poiseuille_analytical(y, b, Fb, rho_l, nu)

        Q_lbm = np.sum(u_profile)
        Q_an = np.sum(u_an)
        eps_Q = abs(Q_lbm - Q_an) / max(abs(Q_an), 1e-12)

        records.append(
            {
                "case_name": case_dir.name,
                "NY": ny,
                "NX": nx,
                "b": b,
                "Fb": Fb,
                "Q_lbm": Q_lbm,
                "Q_an": Q_an,
                "eps_Q": eps_Q,
            }
        )
        print(f"  {case_dir.name}: NY={ny}, ε_Q={eps_Q:.4e}")

    if not records:
        raise RuntimeError(f"No valid cases found under {results_root}")

    df = pd.DataFrame(records).sort_values("NY")

    # Richardson extrapolation — dataset-level metadata
    p_obs = np.nan
    GCI = np.nan
    if len(df) >= 3:
        ny_vals = df["NY"].values
        eps_vals = df["eps_Q"].values
        log_ny = np.log(ny_vals)
        log_eps = np.log(np.maximum(eps_vals, 1e-15))
        coeffs = np.polyfit(log_ny, log_eps, 1)
        p_obs = -coeffs[0]
        print(
            f"[mesh] Observed convergence order p_obs = {p_obs:.2f} (theoretical MRT p=2)"
        )

        # Grid Convergence Index (GCI) for finest two grids
        if len(df) >= 2:
            r = ny_vals[-1] / ny_vals[-2]  # refinement ratio
            GCI = 1.25 * eps_vals[-1] / (r**p_obs - 1.0) if r**p_obs > 1.0 else np.nan
            print(f"[mesh] GCI (finest) = {GCI:.4e}")

    # Write per-row data + dataset-level metadata columns
    df_out = df.copy()
    df_out["p_obs"] = p_obs  # same for all rows (dataset-level property)
    df_out["GCI"] = GCI  # same for all rows
    df_out.to_csv(out / "mesh_convergence.csv", index=False)
    print(
        f"[mesh] Wrote {out / 'mesh_convergence.csv'} ({len(df_out)} cases, p_obs={p_obs:.2f}, GCI={GCI:.4e})"
    )

    # Convergence assessment
    if len(df) > 0:
        eps_finest = df["eps_Q"].iloc[-1] if len(df) > 0 else np.nan
        if eps_finest < 0.01:
            print(
                f"[mesh] SUFFICIENT: ε(NY={df['NY'].iloc[-1]}) = {eps_finest:.4e} < 1%"
            )
        elif eps_finest < 0.05:
            print(
                f"[mesh] MARGINAL: ε(NY={df['NY'].iloc[-1]}) = {eps_finest:.4e} ∈ [1%, 5%]"
            )
        else:
            print(
                f"[mesh] INSUFFICIENT: ε(NY={df['NY'].iloc[-1]}) = {eps_finest:.4e} > 5%"
            )

    return df


def plot_mesh_convergence(
    df: pd.DataFrame,
    out_path: str,
    spurious_df: pd.DataFrame | None = None,
) -> str:
    """Generate mesh convergence figure (optionally combined with spurious).

    Double panel:
      (a) NY vs ε(Poiseuille) — log-log with power-law fit
      (b) NY vs max|u| (spurious) — log-log (optional)

    Args:
        df: Mesh convergence DataFrame from analyze_mesh_convergence.
        out_path: Output PDF path.
        spurious_df: Optional spurious currents DataFrame for panel (b).

    Returns:
        Path to the generated figure.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lbm_mrt.viz.viz_template import init_style, format_axes, save_figure

    init_style()

    n_panels = 2 if spurious_df is not None and len(spurious_df) > 0 else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5), squeeze=False)
    axes = axes[0]

    # ── Panel (a): Poiseuille convergence ──
    ax = axes[0]
    valid = df.dropna(subset=["NY", "eps_Q"]).sort_values("NY")
    if len(valid) >= 2:
        ax.loglog(
            valid["NY"],
            valid["eps_Q"],
            "o-",
            color="#2166AC",
            linewidth=1.5,
            markersize=7,
            markerfacecolor="white",
            markeredgewidth=1.2,
        )

        # Power-law fit
        log_ny = np.log(valid["NY"].values)
        log_eps = np.log(np.maximum(valid["eps_Q"].values, 1e-15))
        coeffs = np.polyfit(log_ny, log_eps, 1)
        ny_fit = np.logspace(
            np.log10(valid["NY"].min()), np.log10(valid["NY"].max()), 20
        )
        ax.loglog(
            ny_fit,
            np.exp(coeffs[1]) * ny_fit ** coeffs[0],
            ":",
            color="#2166AC",
            linewidth=1.0,
            alpha=0.5,
            label=f"$p_{{obs}} = {-coeffs[0]:.2f}$",
        )

        ax.legend(fontsize=9)

    ax.set_xlabel("Grid size $N_Y$")
    ax.set_ylabel("$\\varepsilon = |Q - Q_{an}| / |Q_{an}|$")
    ax.set_title("(a) Poiseuille flow convergence")
    ax.grid(True, alpha=0.3)

    # ── Panel (b): Spurious currents decay (optional) ──
    if spurious_df is not None and len(spurious_df) > 0:
        ax2 = axes[1]
        sp_valid = spurious_df.dropna(subset=["NY", "max_u"]).sort_values("NY")
        ax2.loglog(
            sp_valid["NY"],
            sp_valid["max_u"],
            "s-",
            color="#B2182B",
            linewidth=1.5,
            markersize=7,
            markerfacecolor="white",
            markeredgewidth=1.2,
            label="Huang SCMP",
        )

        if len(sp_valid) >= 3:
            log_ny2 = np.log(sp_valid["NY"].values)
            log_u2 = np.log(sp_valid["max_u"].values)
            c2 = np.polyfit(log_ny2, log_u2, 1)
            ny_fit2 = np.logspace(
                np.log10(sp_valid["NY"].min()), np.log10(sp_valid["NY"].max()), 20
            )
            ax2.loglog(
                ny_fit2,
                np.exp(c2[1]) * ny_fit2 ** c2[0],
                ":",
                color="#B2182B",
                linewidth=1.0,
                alpha=0.5,
                label=f"slope = {c2[0]:.2f}",
            )
            ax2.legend(fontsize=9)

        ax2.set_xlabel("Grid size $N_Y$")
        ax2.set_ylabel("$\\max|\\mathbf{u}|$ (lu)")
        ax2.set_title("(b) Spurious currents decay")
        ax2.grid(True, alpha=0.3)

    fig.tight_layout(pad=2.0)
    save_figure(fig, out_path)
    plt.close(fig)
    print(f"[mesh] Figure saved to {out_path}")
    return out_path


# ── Design CSV generator (for future use after solver extension) ──


def _base_mesh_params() -> dict[str, Any]:
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
        "huang_init_mode": 3,
        "drive_mode": 1,
        "ENABLE_CKPT": False,
        "OUTPUT_EVERY": 10000,
    }


def generate_mesh_design(
    out_path: str,
    tr: float = 0.70,
    ny_list: list[int] | None = None,
) -> str:
    """Generate design CSV for mesh convergence study.

    NOTE: Requires solver-side multi-grid support (P1).

    Args:
        out_path: Output CSV path.
        tr: Reduced temperature.
        ny_list: Grid sizes. Default: [100, 200, 400, 800].

    Returns:
        Path to the generated CSV.
    """
    if ny_list is None:
        ny_list = [100, 200, 400, 800]

    T_abs = tr * TC
    result = maxwell_coexistence(CS_A, CS_B, CS_R, T_abs)
    if result is None:
        raise RuntimeError(f"Maxwell coexistence failed for Tr={tr:.2f}")
    _, rl, _ = result

    base = _base_mesh_params()
    # Fb scaling: keep Re constant across resolutions
    Fb_ref = 5e-9
    h_ref = 198.0  # NY=200 → b=99

    rows = []
    for ny in ny_list:
        b = (ny - 2) / 2.0
        Fb = Fb_ref * (h_ref / b) ** 3
        row = dict(base)
        row["case_name"] = f"mesh_NY{ny}"
        row["cs_T"] = tr  # reduced temperature T/Tc (solver treats cs_T as Tr)
        row["Gy"] = Fb
        row["huang_rho_g"] = float(rl)
        row["huang_rho_l"] = float(rl)
        rows.append(row)

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[mesh] Generated {out_path} ({len(rows)} cases, NY ∈ {ny_list})")
    return out_path
