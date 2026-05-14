"""Spurious currents analysis for Huang & Wu (2016) SCMP solver.

Extracts max|u| from static droplet simulations as a function of grid resolution.
Optionally compares Huang SCMP vs legacy Li MCMP.

Public API:
- extract_max_u: extract max|u| from a single VTK file
- analyze_spurious: process multi-resolution batch results
- plot_spurious: log-log plot of max|u| vs NY
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


def extract_max_u(vtk_path: str) -> dict[str, float]:
    """Extract max|u| and mean|u| from a VTK file.

    Args:
        vtk_path: Path to the VTK file.

    Returns:
        Dict with keys: max_u, mean_u, max_ux, max_uy.
    """
    fields, nx, ny = read_vtk_scalars(vtk_path)

    # VTK U field: shape (ny, nx, 2) or flat with components
    ux = fields.get("ux", fields.get("U", np.zeros((ny, nx, 2)))[..., 0])
    uy = fields.get("uy", np.zeros_like(ux))
    if "U" in fields and fields["U"].ndim == 3:
        ux = fields["U"][..., 0]
        uy = fields["U"][..., 1]

    u_mag = np.sqrt(ux**2 + uy**2)
    return {
        "max_u": float(np.max(u_mag)),
        "mean_u": float(np.mean(u_mag)),
        "max_ux": float(np.max(np.abs(ux))),
        "max_uy": float(np.max(np.abs(uy))),
        "nx": nx,
        "ny": ny,
    }


def _base_spurious_params() -> dict[str, Any]:
    return {
        "pp_mode": 1,
        "init_eq": 0,
        "tau_p_a": 1.0,
        "tau_p_b": 1.0,
        "kappa": 0.0,
        "GAB": 0.0,
        "GBA": 0.0,
        "sigmaA": 0.0,
        "k1_huang": 1.0 / 12.0,
        "k2_huang": 0.0,
        "kd_huang": -1.0 / 12.0,
        "alpha_meq": 1.0,
        "cs_a": CS_A,
        "cs_b": CS_B,
        "cs_R": CS_R,
        "cs_G": -1.0,
        "huang_W": 3.0,
        "huang_init_mode": 1,
        "Gx": 0.0,
        "Gy": 0.0,
        "drive_mode": 1,
        "ENABLE_CKPT": False,
        "OUTPUT_EVERY": 5000,
    }


def generate_spurious_design(
    out_path: str,
    tr: float = 0.70,
    ny_list: list[int] | None = None,
) -> str:
    """Generate design CSV for spurious currents sweep.

    Note: Multi-resolution sweep requires solver-side multi-grid support
    (compile-time NX/NY injection). For now, generates single NY=256 case
    as a placeholder for future extension.

    Args:
        out_path: Output CSV path.
        tr: Reduced temperature.
        ny_list: Grid sizes (NY=NX). Default: [256] only (single point).

    Returns:
        Path to the generated CSV.
    """
    if ny_list is None:
        ny_list = [256]

    T_abs = tr * TC
    result = maxwell_coexistence(CS_A, CS_B, CS_R, T_abs)
    if result is None:
        raise RuntimeError(f"Maxwell coexistence failed for Tr={tr:.2f}")
    rg, rl, _ = result

    base = _base_spurious_params()
    rows = []
    for ny in ny_list:
        R = int(0.2 * ny)  # R/NY = 0.2 per validation_plan §6.2
        row = dict(base)
        row["case_name"] = f"spurious_NY{ny}_R{R}"
        row["cs_T"] = T_abs
        row["huang_R0"] = float(R)
        row["huang_xc"] = float(ny / 2)
        row["huang_yc"] = float(ny / 2)
        row["huang_rho_g"] = float(rg)
        row["huang_rho_l"] = float(rl)
        rows.append(row)

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[spurious] Generated {out_path} ({len(rows)} cases)")
    return out_path


def analyze_spurious(
    results_root: str,
    out_dir: str | None = None,
) -> pd.DataFrame:
    """Analyze spurious currents from batch results.

    Args:
        results_root: Path to the batch results directory.
        out_dir: If given, write spurious_summary.csv here.

    Returns:
        DataFrame with columns: case_name, NY, R, max_u, mean_u, rho_l, rho_g.
    """
    root = Path(results_root)
    out = Path(out_dir) if out_dir else root

    records = []
    # Also scan existing laplace results for spurious data
    case_dirs = sorted(d for d in root.iterdir() if d.is_dir() and
                       (d.name.startswith("spurious_") or d.name.startswith("laplace_")))
    if not case_dirs:
        for sub in sorted(root.iterdir()):
            if sub.is_dir():
                case_dirs.extend(sorted(d for d in sub.iterdir() if d.is_dir() and
                                       (d.name.startswith("spurious_") or d.name.startswith("laplace_"))))

    for case_dir in case_dirs:
        vtk_dir = case_dir / "outputdata_scmp"
        if not vtk_dir.exists():
            continue

        try:
            vtk_path = latest_vtk(str(vtk_dir), "flow")
            result = extract_max_u(vtk_path)
        except Exception as e:
            print(f"[WARNING] Failed for {case_dir.name}: {e}")
            continue

        # Try to read rho for density info
        try:
            fields, _, _ = read_vtk_scalars(vtk_path)
            rho = fields["rho"]
            rho_vals = np.sort(rho.flatten())
            n = len(rho_vals)
            rho_g = float(np.mean(rho_vals[: max(1, int(0.05 * n))]))
            rho_l = float(np.mean(rho_vals[-max(1, int(0.05 * n)):]))
        except Exception:
            rho_l, rho_g = np.nan, np.nan

        records.append({
            "case_name": case_dir.name,
            "NY": result["ny"],
            "NX": result["nx"],
            "max_u": result["max_u"],
            "mean_u": result["mean_u"],
            "max_ux": result["max_ux"],
            "max_uy": result["max_uy"],
            "rho_l": rho_l,
            "rho_g": rho_g,
        })
        print(f"  {case_dir.name}: NY={result['ny']}, max|u|={result['max_u']:.4e}")

    if not records:
        raise RuntimeError(f"No valid cases found under {results_root}")

    df = pd.DataFrame(records)
    df.to_csv(out / "spurious_summary.csv", index=False)
    print(f"[spurious] Wrote {out / 'spurious_summary.csv'} ({len(df)} cases)")

    # Summary statistics
    if len(df) > 0:
        print(f"[spurious] max|u| range: [{df['max_u'].min():.4e}, {df['max_u'].max():.4e}]")
        print(f"[spurious] mean max|u|: {df['max_u'].mean():.4e}")

    return df


def plot_spurious(
    df: pd.DataFrame,
    out_path: str,
    li_comparison: pd.DataFrame | None = None,
) -> str:
    """Generate spurious currents plot.

    X = NY (log), Y = max|u| (log).
    Optional Li comparison data as second series.

    Args:
        df: Summary DataFrame from analyze_spurious.
        out_path: Output PDF path.
        li_comparison: Optional DataFrame with Li MCMP spurious data (NY, max_u).

    Returns:
        Path to the generated figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lbm_mrt.viz.viz_template import init_style, format_axes, save_figure

    init_style()

    fig, ax = plt.subplots(figsize=(6, 5))

    valid = df.dropna(subset=["NY", "max_u"]).sort_values("NY")
    if len(valid) == 0:
        raise ValueError("No valid data points")

    # Huang data
    ax.loglog(valid["NY"], valid["max_u"], "o-", color="#2166AC", linewidth=1.5,
              markersize=7, markerfacecolor="white", markeredgewidth=1.2,
              label="Huang SCMP (this work)")

    # Li comparison (optional)
    if li_comparison is not None and len(li_comparison) > 0:
        li_valid = li_comparison.dropna(subset=["NY", "max_u"]).sort_values("NY")
        ax.loglog(li_valid["NY"], li_valid["max_u"], "s--", color="#B2182B", linewidth=1.5,
                  markersize=7, markerfacecolor="white", markeredgewidth=1.2,
                  label="Li MCMP (legacy)")

    # Power-law fit for Huang
    if len(valid) >= 3:
        log_ny = np.log(valid["NY"].values)
        log_u = np.log(valid["max_u"].values)
        coeffs = np.polyfit(log_ny, log_u, 1)
        fit_label = f"slope = {coeffs[0]:.2f}"
        ny_fit = np.logspace(np.log10(valid["NY"].min()), np.log10(valid["NY"].max()), 20)
        ax.loglog(ny_fit, np.exp(coeffs[1]) * ny_fit ** coeffs[0], ":",
                  color="#2166AC", linewidth=1.0, alpha=0.5, label=fit_label)

    ax.set_xlabel("Grid size $N_Y$")
    ax.set_ylabel("$\\max|\\mathbf{u}|$ (lu)")
    ax.set_title("Spurious currents decay with grid resolution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)
    print(f"[spurious] Figure saved to {out_path}")
    return out_path
