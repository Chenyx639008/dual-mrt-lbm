"""Contact angle validation for Huang & Wu (2016) SCMP.

Measures contact angle θ from SCMP droplet on bottom wall (huang_init_mode=4),
using circle fitting to the droplet interface. Supports G_ads sweep and
θ-G_ads linear regression.

Usage::
    from lbm_mrt.validation.contact_angle import analyze_contact_sweep
    df = analyze_contact_sweep("results/contact_sweep", out_dir="results/contact_sweep")
"""

from __future__ import annotations

import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from lbm_mrt.io.vtk_reader import latest_vtk, read_vtk_scalars


# ═══════════════════════════════════════════════════════════════════════════
# Circle fitting
# ═══════════════════════════════════════════════════════════════════════════


def fit_circle_lstsq(
    pts: np.ndarray,
) -> tuple[float, float, float] | None:
    """Least-squares circle fit: x² + y² + A·x + B·y + C = 0.

    Returns (xc, yc, R) or None if fit fails.
    """
    if len(pts) < 5:
        return None

    x, y = pts[:, 0], pts[:, 1]
    A_mat = np.column_stack([x, y, np.ones_like(x)])
    b_vec = -(x**2 + y**2)

    try:
        sol, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
    except np.linalg.LinAlgError:
        return None

    A, B, C = sol[0], sol[1], sol[2]
    xc = -A / 2.0
    yc = -B / 2.0
    R_sq = xc**2 + yc**2 - C
    if R_sq <= 0:
        return None
    R = float(np.sqrt(R_sq))
    return float(xc), float(yc), R


def compute_theta_from_circle(xc: float, yc: float, R: float) -> float:
    """Compute contact angle (degrees) from fitted circle.

    The droplet sits on the bottom wall (y=0). The circle center is at (xc, yc).
    - yc < 0: center below wall → hydrophilic (acute angle)
    - yc ≈ 0: center at wall → 90°
    - yc > 0: center above wall → hydrophobic (obtuse angle)

    θ = arccos(|yc|/R) if yc < 0, else θ = 180° − arccos(yc/R)
    """
    ratio = np.clip(abs(yc) / R, 0.0, 1.0)
    if yc < 0:
        theta = np.degrees(np.arccos(ratio))
    else:
        theta = 180.0 - np.degrees(np.arccos(ratio))
    return float(theta)


# ═══════════════════════════════════════════════════════════════════════════
# VTK → θ
# ═══════════════════════════════════════════════════════════════════════════


def compute_theta_from_vtk(
    vtk_path: str,
    thr_frac: float = 0.5,
    min_y: int = 3,
    x_step: int = 2,
    x_margin: int = 10,
) -> dict | None:
    """Extract contact angle from a VTK file containing a droplet on bottom wall.

    Args:
        vtk_path: Path to VTK file.
        thr_frac: Density threshold fraction between min and max.
        min_y: Minimum y to consider (exclude wall ghost/boundary layers).
        x_step: Step size for interface point sampling.
        x_margin: Margin from domain edges.

    Returns:
        dict with keys: theta_deg, xc, yc, R, n_pts, thr
        or None if insufficient interface points.
    """
    fields, nx, ny = read_vtk_scalars(vtk_path)
    rho = np.array(fields["rho"]).reshape(ny, nx)

    rho_min, rho_max = float(rho.min()), float(rho.max())
    thr = rho_min + thr_frac * (rho_max - rho_min)

    # Extract interface points: for each x column, find y where rho crosses thr
    pts = []
    for x in range(x_margin, nx - x_margin, x_step):
        col = rho[:, x]
        for y in range(min_y, ny - 2):
            if col[y] >= thr and y + 1 < ny and col[y + 1] < thr:
                pts.append([float(x), float(y) + 0.5])
                break

    if len(pts) < 10:
        return None

    pts_arr = np.array(pts)

    # Filter points far from the wall to avoid bottom curvature artifacts
    pts_arr = pts_arr[pts_arr[:, 1] > min_y]

    result = fit_circle_lstsq(pts_arr)
    if result is None:
        return None

    xc, yc, R = result
    theta = compute_theta_from_circle(xc, yc, R)

    return {
        "theta_deg": theta,
        "xc": xc,
        "yc": yc,
        "R": R,
        "n_pts": len(pts_arr),
        "thr": thr,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Sweep analysis
# ═══════════════════════════════════════════════════════════════════════════


def analyze_contact_sweep(
    results_root: str,
    out_dir: str | None = None,
) -> pd.DataFrame:
    """Analyze contact angle batch results.

    Scans for case directories matching 'contact_*' under results_root.
    Parses G_ads (or theta_nominal) from params.txt and measures θ from VTK.

    Args:
        results_root: Path to directory containing contact_* case subdirs.
        out_dir: If given, write contact_summary.csv here.

    Returns:
        DataFrame with columns: case_name, G_ads, theta_deg, xc, yc, R, n_pts
    """
    root = Path(results_root)
    out = Path(out_dir) if out_dir else root

    case_dirs = sorted(
        d for d in root.iterdir() if d.is_dir() and d.name.startswith("contact_")
    )
    if not case_dirs:
        raise FileNotFoundError(f"No contact_* directories found under {results_root}")

    records = []
    for case_dir in case_dirs:
        case_name = case_dir.name
        vtk_dir = case_dir / "outputdata_scmp"
        if not vtk_dir.exists():
            print(f"[WARNING] No outputdata_scmp in {case_dir}, skipping")
            continue

        vtk_path = latest_vtk(str(vtk_dir), "flow")
        if vtk_path is None:
            print(f"[WARNING] No VTK in {vtk_dir}, skipping")
            continue

        # Try to read G_ads from params.txt
        G_ads = None
        params_path = case_dir / "params.txt"
        if params_path.exists():
            with open(params_path) as f:
                for line in f:
                    if line.startswith("GAw_quartz") or line.startswith("G_ads"):
                        try:
                            G_ads = float(line.split()[1])
                        except (ValueError, IndexError):
                            pass
                        break
        if G_ads is None:
            # Try parsing from case name: contact_G0.05 or contact_theta90
            import re

            m = re.search(r"G([\-\d.]+)", case_name)
            if m:
                G_ads = float(m.group(1))

        result = compute_theta_from_vtk(vtk_path)
        if result is None:
            print(f"[WARNING] Failed to measure θ for {case_name}")
            continue

        record = {
            "case_name": case_name,
            "G_ads": G_ads,
            **result,
        }
        records.append(record)
        print(
            f"  {case_name}: θ={result['theta_deg']:.1f}°"
            + (f" (G_ads={G_ads:.4f})" if G_ads is not None else "")
        )

    df = pd.DataFrame(records)

    # Linear regression: θ vs G_ads
    if len(df) >= 3 and df["G_ads"].notna().all():
        x_vals = df["G_ads"].values.astype(float)
        y_vals = df["theta_deg"].values.astype(float)
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        r2 = float(
            1.0
            - np.sum((y_vals - (slope * x_vals + intercept)) ** 2)
            / np.sum((y_vals - np.mean(y_vals)) ** 2)
        )
        print(f"[contact] θ = {slope:.2f}·G_ads + {intercept:.1f}, R²={r2:.4f}")

    if out_dir:
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "contact_summary.csv", index=False)
        print(f"[contact] Wrote {out / 'contact_summary.csv'} ({len(df)} cases)")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════


def plot_contact_angle(
    df: pd.DataFrame,
    out_path: str,
) -> str:
    """Plot contact angle vs G_ads with linear fit.

    Args:
        df: DataFrame from analyze_contact_sweep.
        out_path: Output figure path (.pdf).

    Returns:
        out_path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))

    valid = df["G_ads"].notna() & df["theta_deg"].notna()
    x = df.loc[valid, "G_ads"].values.astype(float)
    y = df.loc[valid, "theta_deg"].values.astype(float)

    ax.scatter(x, y, c="blue", s=40, zorder=5, label="SCMP")

    if len(x) >= 3:
        slope, intercept = np.polyfit(x, y, 1)
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = slope * x_fit + intercept
        r2 = float(
            1.0
            - np.sum((y - (slope * x + intercept)) ** 2) / np.sum((y - np.mean(y)) ** 2)
        )
        ax.plot(
            x_fit,
            y_fit,
            "r--",
            label=f"fit: θ={slope:.1f}·G+{intercept:.0f}, R²={r2:.3f}",
        )

    ax.axhline(90, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("G_ads")
    ax.set_ylabel("Contact angle θ (°)")
    ax.set_title("Huang SCMP Contact Angle vs G_ads")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[contact] Figure saved: {out_path}")
    return out_path
