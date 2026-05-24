"""Single-phase Poiseuille flow validation for Huang SCMP solver.

Validates u(y) parabolic profile against analytical solution in a channel
with top/bottom bounce-back walls and periodic left/right boundaries.

NOTE (P1): This module requires solver-side extensions:
  - bounce-back wall BC in SCMP path (currently periodic-only)
  - uniform body force injection (Gx/Gy) in SCMP
  - huang_init_mode=3 (uniform liquid)
  - SCMP SteadyMonitor for flow rate convergence

Until solver extensions are complete, this module provides the analysis
framework and can validate legacy MCMP Poiseuille results.

Public API:
- poiseuille_analytical: compute analytical u(y) for channel flow
- extract_centerline_ux: extract velocity profile from VTK
- analyze_poiseuille: compare LBM vs analytical, compute R²
- plot_poiseuille: normalized velocity profile plot
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

CS_A = 1.0
CS_B = 4.0
CS_R = 1.0
TC = cs_critical_point(CS_A, CS_B, CS_R)[0]


def poiseuille_analytical(
    y: np.ndarray,
    b: float,
    Fb: float,
    rho: float,
    nu: float,
) -> np.ndarray:
    """Analytical Poiseuille velocity profile.

    u(y) = Fb * (b² − y²) / (2 * rho * nu)  for |y| ≤ b
    u(y) = 0                                for |y| > b

    Args:
        y: Position array (centered at channel mid-plane).
        b: Half-channel width (in lattice units).
        Fb: Body force (acceleration).
        rho: Fluid density.
        nu: Kinematic viscosity.

    Returns:
        Analytical velocity array.
    """
    u = np.zeros_like(y)
    mask = np.abs(y) <= b
    u[mask] = Fb * (b**2 - y[mask] ** 2) / (2.0 * rho * nu)
    return u


def extract_centerline_ux(
    vtk_path: str,
    column: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Extract ux profile along the centerline (x = NX/2).

    Args:
        vtk_path: Path to the VTK file.
        column: X-column index. If None, uses NX/2.

    Returns:
        (y, ux_profile, nx, ny) where y is centered at mid-plane.
    """
    fields, nx, ny = read_vtk_scalars(vtk_path)

    ux = fields.get("ux", None)
    if ux is None and "U" in fields:
        ux = fields["U"][..., 0] if fields["U"].ndim == 3 else fields["U"]

    col = column if column is not None else nx // 2
    u_profile = ux[:, col] if ux.ndim == 2 else ux.reshape(ny, nx)[:, col]

    # Center y at channel mid-plane
    y = np.arange(ny) - (ny - 1) / 2.0
    return y, u_profile, nx, ny


def analyze_poiseuille(
    results_root: str,
    out_dir: str | None = None,
    tr: float = 0.70,
    tau: float = 1.5,  # matches tau_huang in configs/huang_scmp.yaml (MRT τ=1.5)
    Fb: float = 5e-9,
) -> pd.DataFrame:
    """Analyze Poiseuille flow results.

    Args:
        results_root: Path to results directory containing poiseuille_* cases.
        out_dir: Output directory for summary CSV.
        tr: Reduced temperature (for density reference).
        tau: Relaxation time (ν = (τ−0.5)/3).
        Fb: Body force magnitude.

    Returns:
        DataFrame with R² and flow rate comparison.
    """
    root = Path(results_root)
    out = Path(out_dir) if out_dir else root

    T_abs = tr * TC
    result = maxwell_coexistence(CS_A, CS_B, CS_R, T_abs)
    if result is None:
        raise RuntimeError(f"Maxwell coexistence failed for Tr={tr:.2f}")
    _, rho_l, _ = result

    nu = (tau - 0.5) / 3.0  # MRT D2Q9 kinematic viscosity

    records = []
    case_dirs = sorted(
        d for d in root.iterdir() if d.is_dir() and d.name.startswith("poiseuille_")
    )
    if not case_dirs:
        for sub in sorted(root.iterdir()):
            if sub.is_dir():
                case_dirs.extend(
                    sorted(
                        d
                        for d in sub.iterdir()
                        if d.is_dir() and d.name.startswith("poiseuille_")
                    )
                )

    for case_dir in case_dirs:
        vtk_dir = case_dir / "outputdata_scmp"
        if not vtk_dir.exists():
            # Try legacy outputdata_eq
            vtk_dir = case_dir / "outputdata_eq"
        if not vtk_dir.exists():
            print(f"[WARNING] No VTK output in {case_dir}, skipping")
            continue

        try:
            vtk_path = latest_vtk(str(vtk_dir), "flow")
            y, u_profile, nx, ny = extract_centerline_ux(vtk_path)
        except Exception as e:
            print(f"[WARNING] Failed for {case_dir.name}: {e}")
            continue

        b = (ny - 2) / 2.0  # half-channel (excluding wall ghost nodes)
        u_an = poiseuille_analytical(y, b, Fb, rho_l, nu)

        # R²
        ss_res = np.sum((u_profile - u_an) ** 2)
        ss_tot = np.sum((u_profile - np.mean(u_profile)) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

        # Flow rate comparison
        Q_lbm = np.sum(u_profile)
        Q_an = np.sum(u_an)
        eps_Q = abs(Q_lbm - Q_an) / max(abs(Q_an), 1e-12)

        U_max_lbm = np.max(u_profile)
        U_max_an = np.max(u_an)

        records.append(
            {
                "case_name": case_dir.name,
                "NY": ny,
                "NX": nx,
                "Tr": tr,
                "R2": r2,
                "eps_Q": eps_Q,
                "U_max_lbm": U_max_lbm,
                "U_max_an": U_max_an,
                "Q_lbm": Q_lbm,
                "Q_an": Q_an,
            }
        )
        print(f"  {case_dir.name}: R²={r2:.5f}, ε_Q={eps_Q:.4e}")

    if not records:
        raise RuntimeError(f"No valid cases found under {results_root}")

    df = pd.DataFrame(records)
    df.to_csv(out / "poiseuille_summary.csv", index=False)
    print(f"[poiseuille] Wrote {out / 'poiseuille_summary.csv'} ({len(df)} cases)")

    return df


def plot_poiseuille(
    df: pd.DataFrame,
    results_root: str,
    out_path: str,
    tr: float = 0.70,
    tau: float = 1.5,  # matches tau_huang in configs/huang_scmp.yaml
    Fb: float = 5e-9,
) -> str:
    """Generate normalized Poiseuille velocity profile plot.

    X = u/U_max (normalized), Y = y/b (normalized).
    Solid line = analytical, scatter = LBM.

    Args:
        df: Summary DataFrame.
        results_root: Path to results directory.
        out_path: Output PDF path.
        tr, tau, Fb: Physical parameters for analytical solution.

    Returns:
        Path to the generated figure.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lbm_mrt.viz.viz_template import init_style, format_axes, save_figure

    init_style()

    # Get analytical parameters
    T_abs = tr * TC
    result = maxwell_coexistence(CS_A, CS_B, CS_R, T_abs)
    if result is None:
        raise RuntimeError(f"Maxwell coexistence failed for Tr={tr:.2f}")
    _, rho_l, _ = result
    nu = (tau - 0.5) / 3.0

    root = Path(results_root)
    case_dirs = sorted(
        d for d in root.iterdir() if d.is_dir() and d.name.startswith("poiseuille_")
    )
    if not case_dirs:
        for sub in sorted(root.iterdir()):
            if sub.is_dir():
                case_dirs = sorted(
                    d
                    for d in sub.iterdir()
                    if d.is_dir() and d.name.startswith("poiseuille_")
                )

    if not case_dirs:
        raise FileNotFoundError(f"No poiseuille_* cases under {results_root}")

    fig, ax = plt.subplots(figsize=(5, 5))

    for i, case_dir in enumerate(case_dirs[:2]):  # max 2 panels
        vtk_dir = case_dir / "outputdata_scmp"
        if not vtk_dir.exists():
            vtk_dir = case_dir / "outputdata_eq"
        if not vtk_dir.exists():
            continue

        vtk_path = latest_vtk(str(vtk_dir), "flow")
        y, u_profile, nx, ny = extract_centerline_ux(vtk_path)
        b = (ny - 2) / 2.0
        u_an = poiseuille_analytical(y, b, Fb, rho_l, nu)
        U_max = max(np.max(u_an), 1e-12)

        color = ["#2166AC", "#B2182B"][i % 2]
        marker = ["o", "s"][i % 2]

        # Analytical line
        y_norm = y / b
        ax.plot(
            u_an / U_max,
            y_norm,
            "-",
            color=color,
            linewidth=1.5,
            label=f"Analytical (Tr={tr:.2f})" if i == 0 else None,
        )
        # LBM scatter
        ax.scatter(
            u_profile / U_max,
            y_norm,
            marker=marker,
            facecolors="white",
            edgecolors=color,
            s=30,
            linewidths=0.8,
            zorder=5,
            label=f"LBM (Tr={tr:.2f})" if i == 0 else None,
        )

    ax.set_xlabel("$u_x / U_{\\max}$")
    ax.set_ylabel("$y / b$")
    ax.set_title("Poiseuille velocity profile")
    ax.legend(fontsize=9)
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle=":")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)
    print(f"[poiseuille] Figure saved to {out_path}")
    return out_path


# ── Design CSV generator (for future use after solver extension) ──


def _base_poiseuille_params() -> dict[str, Any]:
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
        "huang_init_mode": 3,  # uniform liquid (needs solver extension)
        "Gx": 0.0,
        "Gy": 5e-9,
        "drive_mode": 1,
        "ENABLE_CKPT": False,
        "OUTPUT_EVERY": 5000,
    }


def generate_poiseuille_design(
    out_path: str,
    tr_list: list[float] | None = None,
) -> str:
    """Generate design CSV for Poiseuille flow.

    NOTE: Requires solver-side wall BC + body force support (P1).

    Args:
        out_path: Output CSV path.
        tr_list: Reduced temperatures. Default: [0.70, 0.90].

    Returns:
        Path to the generated CSV.
    """
    if tr_list is None:
        tr_list = [0.70, 0.90]

    base = _base_poiseuille_params()
    rows = []
    for Tr in tr_list:
        T_abs = Tr * TC
        result = maxwell_coexistence(CS_A, CS_B, CS_R, T_abs)
        if result is None:
            print(f"[WARNING] Maxwell coexistence failed for Tr={Tr:.2f}, skipping")
            continue
        _, rl, _ = result
        row = dict(base)
        row["case_name"] = f"poiseuille_Tr{Tr:.2f}"
        row["cs_T"] = Tr  # reduced temperature T/Tc (solver treats cs_T as Tr)
        row["huang_rho_g"] = float(rl)
        row["huang_rho_l"] = float(rl)
        rows.append(row)

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[poiseuille] Generated {out_path} ({len(rows)} cases)")
    return out_path
