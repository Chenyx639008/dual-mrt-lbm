#!/usr/bin/env python3
"""Huang & Wu (2016) SCMP validation suite — one-stop design, run & analyze.

Generates design CSVs, runs batch simulations via lbm-batch / run_batch,
post-processes results, and produces a summary validation report.

Usage::

    # All-in-one: generate + run + analyze + report
    uv run python scripts/06_run_huang_validation_suite.py --all --run

    # Single sweep
    uv run python scripts/06_run_huang_validation_suite.py --sweep laplace --run
    uv run python scripts/06_run_huang_validation_suite.py --sweep decoupling --run
    uv run python scripts/06_run_huang_validation_suite.py --sweep spurious --run

    # Manual batch run
    uv run lbm-batch --csv data/design_scmp_laplace.csv --app lbm_mrt/solver/mcmp_huang_256

    # Post-process existing results
    uv run python scripts/06_run_huang_validation_suite.py --analyze results/design_scmp_laplace_20260513

    # Generate summary report
    uv run python scripts/06_run_huang_validation_suite.py --report results/huang_validation_20260514 --out-md REPORT.md
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from lbm_mrt.core.paths import PROJ_ROOT, DATA_DIR, RESULTS_DIR
from lbm_mrt.io.vtk_reader import latest_vtk, read_vtk_scalars
from lbm_mrt.validation.analytical import (
    detect_interface_radius,
    extract_rho_l_g,
    fit_pressure_inside_outside,
    laplace_sigma,
)
from lbm_mrt.validation.cs_eos import (
    cs_critical_point,
    maxwell_coexistence,
    coexistence_curve,
)

BINARY = os.path.join(PROJ_ROOT, "lbm_mrt", "solver", "mcmp_huang_256")

# ── Unified CS-EOS parameters (matching Huang & Wu 2016 paper) ──
CS_A = 1.0
CS_B = 4.0
CS_R = 1.0
TC = cs_critical_point(CS_A, CS_B, CS_R)[0]  # ≈ 0.09433

# ═══════════════════════════════════════════════════════════════════════════
# Base parameters (unified cs_a=1.0 per validation_plan §2.1)
# ═══════════════════════════════════════════════════════════════════════════


def _base_params() -> dict:
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


def _coexistence_density(Tr: float) -> tuple[float, float]:
    """Get coexistence densities (rg, rl) for a given reduced temperature."""
    T_abs = Tr * TC
    result = maxwell_coexistence(CS_A, CS_B, CS_R, T_abs)
    if result is None:
        raise RuntimeError(f"Maxwell coexistence failed for Tr={Tr:.2f} (T={T_abs:.4f})")
    return float(result[0]), float(result[1])

# ═══════════════════════════════════════════════════════════════════════════
# Design CSV generators
# ═══════════════════════════════════════════════════════════════════════════


def generate_laplace_design(
    out_path: str,
    tr_list: list[float] | None = None,
    r_list: list[float] | None = None,
) -> str:
    if tr_list is None:
        tr_list = [0.70, 0.90]
    if r_list is None:
        r_list = [20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0]

    base = _base_params()
    rows = []
    for Tr in tr_list:
        rg, rl = _coexistence_density(Tr)
        T_abs = Tr * TC
        for R in r_list:
            row = dict(base)
            row["case_name"] = f"laplace_Tr{Tr:.2f}_R{R:.0f}"
            row["cs_T"] = T_abs
            row["huang_R0"] = R
            row["huang_rho_g"] = rg
            row["huang_rho_l"] = rl
            rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[design] {out_path} ({len(rows)} cases: {len(tr_list)} Tr × {len(r_list)} R)")
    return out_path


def generate_decoupling_design(
    out_path: str,
    tr: float = 0.70,
    k1_list: list[float] | None = None,
) -> str:
    if k1_list is None:
        k1_list = [0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15]

    base = _base_params()
    rg, rl = _coexistence_density(tr)
    T_abs = tr * TC
    rows = []
    for k1 in k1_list:
        row = dict(base)
        row["case_name"] = f"decoupling_Tr{tr:.2f}_k1{k1:.4f}"
        row["cs_T"] = T_abs
        row["k1_huang"] = k1
        row["huang_R0"] = 40.0
        row["huang_xc"] = 128.0
        row["huang_yc"] = 128.0
        row["huang_rho_g"] = rg
        row["huang_rho_l"] = rl
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[design] {out_path} ({len(rows)} cases, k₁ ∈ {k1_list})")
    return out_path


def generate_coexistence_design(
    out_path: str,
    tr_list: list[float] | None = None,
) -> str:
    if tr_list is None:
        tr_list = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    base = _base_params()
    rows = []
    for Tr in tr_list:
        rg, rl = _coexistence_density(Tr)
        T_abs = Tr * TC
        row = dict(base)
        row["case_name"] = f"coex_Tr{Tr:.2f}"
        row["cs_T"] = T_abs
        row["huang_init_mode"] = 2  # flat interface
        row["huang_yc"] = 64.0
        row["huang_W"] = 5.0
        row["huang_rho_g"] = rg
        row["huang_rho_l"] = rl
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[design] {out_path} ({len(rows)} cases, Tr ∈ {tr_list})")
    return out_path


def generate_spurious_design(
    out_path: str,
    tr: float = 0.70,
    ny_list: list[int] | None = None,
) -> str:
    """Generate spurious currents design (single NY=256 for now; multi-grid needs solver extension)."""
    if ny_list is None:
        ny_list = [256]

    base = _base_params()
    rg, rl = _coexistence_density(tr)
    T_abs = tr * TC
    rows = []
    for ny in ny_list:
        R = int(0.2 * ny)
        row = dict(base)
        row["case_name"] = f"spurious_NY{ny}_R{R}"
        row["cs_T"] = T_abs
        row["huang_R0"] = float(R)
        row["huang_xc"] = float(ny / 2)
        row["huang_yc"] = float(ny / 2)
        row["huang_rho_g"] = rg
        row["huang_rho_l"] = rl
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[design] {out_path} ({len(rows)} cases, NY ∈ {ny_list})")
    return out_path


def generate_poiseuille_design(
    out_path: str,
    tr_list: list[float] | None = None,
) -> str:
    """Generate Poiseuille design (P1: needs solver wall BC + body force)."""
    if tr_list is None:
        tr_list = [0.70, 0.90]

    base = _base_params()
    base["huang_init_mode"] = 3  # uniform liquid (needs solver extension)
    base["Gy"] = 5e-9
    rows = []
    for Tr in tr_list:
        _, rl = _coexistence_density(Tr)
        T_abs = Tr * TC
        row = dict(base)
        row["case_name"] = f"poiseuille_Tr{Tr:.2f}"
        row["cs_T"] = T_abs
        row["huang_rho_g"] = rl
        row["huang_rho_l"] = rl
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[design] {out_path} ({len(rows)} cases, Tr ∈ {tr_list}) [P1: needs solver extension]")
    return out_path


def generate_mesh_design(
    out_path: str,
    tr: float = 0.70,
    ny_list: list[int] | None = None,
) -> str:
    """Generate mesh convergence design (P1: needs multi-grid binaries)."""
    if ny_list is None:
        ny_list = [100, 200, 400, 800]

    Fb_ref = 5e-9
    h_ref = 198.0
    base = _base_params()
    base["huang_init_mode"] = 3
    _, rl = _coexistence_density(tr)
    T_abs = tr * TC
    rows = []
    for ny in ny_list:
        b = (ny - 2) / 2.0
        Fb = Fb_ref * (h_ref / b) ** 3
        row = dict(base)
        row["case_name"] = f"mesh_NY{ny}"
        row["cs_T"] = T_abs
        row["Gy"] = Fb
        row["huang_rho_g"] = rl
        row["huang_rho_l"] = rl
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[design] {out_path} ({len(rows)} cases, NY ∈ {ny_list}) [P1: needs multi-grid]")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# Batch runner
# ═══════════════════════════════════════════════════════════════════════════


def run_batch_sweep(design_csv: str, out_root: str | None = None, app: str | None = None) -> str:
    """Run all cases in a design CSV via the batch runner."""
    from lbm_mrt.runners.batch_run import run_batch

    if out_root is None:
        tag = os.path.splitext(os.path.basename(design_csv))[0]
        out_root = os.path.join(RESULTS_DIR, f"{tag}_{datetime.now():%Y%m%d_%H%M%S}")

    _app = app or BINARY
    print(f"\n[batch] {design_csv}  →  {out_root}")
    print(f"[batch] binary: {_app}")
    n_ok, n_fail = run_batch(
        design_csv=design_csv,
        out_root=out_root,
        app=_app,
        resume=False,
    )
    print(f"[batch] Done: {n_ok} ok, {n_fail} failed  |  results in {out_root}/")
    return out_root


# ═══════════════════════════════════════════════════════════════════════════
# Post-processing (delegates to validation modules)
# ═══════════════════════════════════════════════════════════════════════════


def analyze_laplace(results_dir: str) -> dict:
    """Analyze Laplace sweep results using laplace_law module."""
    from lbm_mrt.validation.laplace_law import analyze_laplace as _analyze

    df = _analyze(results_dir, out_dir=results_dir)
    fit_csv = os.path.join(results_dir, "laplace_fit.csv")
    summary = {"n_cases": len(df)}
    if os.path.exists(fit_csv):
        fit_df = pd.read_csv(fit_csv)
        for _, row in fit_df.iterrows():
            key = f"Tr{row['Tr']:.2f}"
            summary[key] = {
                "sigma": float(row["sigma"]),
                "R2": float(row["R2"]),
                "intercept": float(row["intercept"]),
                "n_points": int(row["n_points"]),
            }
    # Generate figures
    try:
        from lbm_mrt.validation.laplace_law import plot_laplace_double_panel
        fig_path = os.path.join(results_dir, "fig_laplace.pdf")
        plot_laplace_double_panel(df, fig_path, fit_csv=fit_csv)
        summary["figure"] = fig_path
    except Exception as e:
        print(f"[WARNING] Laplace plot failed: {e}")
    return summary


def analyze_decoupling(results_dir: str) -> dict:
    """Analyze σ-decoupling sweep results."""
    from lbm_mrt.validation.decoupling_sweep import analyze_decoupling as _analyze

    df = _analyze(results_dir, out_dir=results_dir)
    summary = {"n_cases": len(df)}
    # Key metrics
    valid = df.dropna(subset=["sigma", "one_minus_6k1"])
    if len(valid) >= 3:
        x = valid["one_minus_6k1"].values
        y = valid["sigma"].values
        slope = float(np.sum(x * y) / np.sum(x * x))
        r2_sigma = float(1.0 - np.sum((y - slope * x) ** 2) / np.sum((y - np.mean(y)) ** 2))
        rho_l_drift = float(valid["rho_l"].std() / max(valid["rho_l"].mean(), 1e-12))
        rho_g_drift = float(valid["rho_g"].std() / max(valid["rho_g"].mean(), 1e-12))
        summary.update({
            "sigma_slope": slope,
            "sigma_R2": r2_sigma,
            "rho_l_drift": rho_l_drift,
            "rho_g_drift": rho_g_drift,
            "pass_sigma": r2_sigma >= 0.99,
            "pass_drift": max(rho_l_drift, rho_g_drift) < 0.01,
        })
    try:
        from lbm_mrt.validation.decoupling_sweep import plot_decoupling
        fig_path = os.path.join(results_dir, "fig_decoupling.pdf")
        plot_decoupling(df, fig_path)
        summary["figure"] = fig_path
    except Exception as e:
        print(f"[WARNING] Decoupling plot failed: {e}")
    return summary


def analyze_coexistence(results_dir: str) -> dict:
    """Analyze coexistence sweep results."""
    case_dirs = sorted(glob.glob(os.path.join(results_dir, "coex_Tr*")))
    Ts, rho_ls, rho_gs = [], [], []
    for cd in case_dirs:
        vtk_dir = os.path.join(cd, "outputdata_scmp")
        vtk_path = latest_vtk(vtk_dir)
        if vtk_path is None:
            continue
        fields, nx, ny = read_vtk_scalars(vtk_path)
        rho = fields["rho"]
        rl, rg = extract_rho_l_g(rho, ny)
        try:
            T_str = os.path.basename(cd).replace("coex_Tr", "")
            Tr = float(T_str)
        except ValueError:
            continue
        Ts.append(Tr)
        rho_ls.append(rl)
        rho_gs.append(rg)
        print(f"  Tr={Tr:.2f}: ρ_l={rl:.6f}, ρ_g={rg:.6f}")

    summary = {"n_cases": len(Ts), "Tr": Ts, "rho_l": rho_ls, "rho_g": rho_gs}

    # Compare with Maxwell
    if len(Ts) >= 2:
        from lbm_mrt.validation.coexistence import compare_coexistence_curves
        try:
            maxwell = coexistence_curve(CS_A, CS_B, CS_R, np.array(Ts) * TC)
            deviations = []
            for i, Tr in enumerate(Ts):
                if maxwell and i < len(maxwell["rho_l"]):
                    dev_l = abs(rho_ls[i] - maxwell["rho_l"][i]) / max(maxwell["rho_l"][i], 1e-12)
                    dev_g = abs(rho_gs[i] - maxwell["rho_g"][i]) / max(maxwell["rho_g"][i], 1e-12)
                    deviations.append(max(dev_l, dev_g))
            if deviations:
                summary["max_deviation"] = float(max(deviations))
                summary["pass_coex"] = summary["max_deviation"] < 0.02
        except Exception as e:
            print(f"[WARNING] Maxwell comparison failed: {e}")

    return summary


def analyze_spurious(results_dir: str) -> dict:
    """Analyze spurious currents from batch results."""
    from lbm_mrt.validation.spurious_currents import analyze_spurious as _analyze

    df = _analyze(results_dir, out_dir=results_dir)
    summary = {"n_cases": len(df)}
    if len(df) > 0:
        summary["max_u_max"] = float(df["max_u"].max())
        summary["max_u_min"] = float(df["max_u"].min())
        summary["max_u_mean"] = float(df["max_u"].mean())
        # Huang spurious typically < 0.01 for well-resolved droplet
        summary["pass_spurious"] = summary["max_u_max"] < 0.05
    try:
        from lbm_mrt.validation.spurious_currents import plot_spurious
        fig_path = os.path.join(results_dir, "fig_spurious.pdf")
        plot_spurious(df, fig_path)
        summary["figure"] = fig_path
    except Exception as e:
        print(f"[WARNING] Spurious plot failed: {e}")
    return summary


def analyze_poiseuille(results_dir: str) -> dict:
    """Analyze Poiseuille flow results."""
    from lbm_mrt.validation.poiseuille_sp import analyze_poiseuille as _analyze

    df = _analyze(results_dir, out_dir=results_dir)
    summary = {"n_cases": len(df)}
    if len(df) > 0:
        summary["R2_min"] = float(df["R2"].min())
        summary["eps_Q_max"] = float(df["eps_Q"].max())
        summary["pass_poiseuille"] = df["R2"].min() >= 0.99
    try:
        from lbm_mrt.validation.poiseuille_sp import plot_poiseuille
        fig_path = os.path.join(results_dir, "fig_poiseuille.pdf")
        plot_poiseuille(df, results_dir, fig_path)
        summary["figure"] = fig_path
    except Exception as e:
        print(f"[WARNING] Poiseuille plot failed: {e}")
    return summary


def analyze_mesh(results_dir: str) -> dict:
    """Analyze mesh convergence results."""
    from lbm_mrt.validation.mesh_convergence import analyze_mesh_convergence as _analyze

    df = _analyze(results_dir, out_dir=results_dir)
    summary = {"n_cases": len(df)}
    if len(df) > 0:
        summary["eps_Q_min"] = float(df["eps_Q"].min())
        if "p_obs" in df.columns and not df["p_obs"].isna().all():
            summary["p_obs"] = float(df["p_obs"].iloc[0])
    try:
        from lbm_mrt.validation.mesh_convergence import plot_mesh_convergence
        fig_path = os.path.join(results_dir, "fig_mesh.pdf")
        plot_mesh_convergence(df, fig_path)
        summary["figure"] = fig_path
    except Exception as e:
        print(f"[WARNING] Mesh plot failed: {e}")
    return summary


ANALYZERS = {
    "laplace": analyze_laplace,
    "decoupling": analyze_decoupling,
    "coexistence": analyze_coexistence,
    "spurious": analyze_spurious,
    "poiseuille": analyze_poiseuille,
    "mesh": analyze_mesh,
}


# ═══════════════════════════════════════════════════════════════════════════
# Summary report generator
# ═══════════════════════════════════════════════════════════════════════════

def make_report_md(
    analysis_results: dict[str, dict],
    out_path: str,
    title: str = "Huang & Wu (2016) SCMP Validation Report",
) -> str:
    """Generate a Markdown summary report from analysis results.

    Args:
        analysis_results: Dict mapping sweep_name → analysis dict.
        out_path: Output .md file path.
        title: Report title.

    Returns:
        Path to the generated report.
    """
    lines = [
        f"# {title}",
        f"",
        f"> Auto-generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"> Solver: `mcmp_huang_256` (256×256, CS-EOS a={CS_A}, b={CS_B}, R={CS_R}, T_c≈{TC:.5f})",
        f"",
        "---",
        f"",
        "## Validation Summary",
        f"",
        "| Test | Status | Key Metric | Criterion | Pass? |",
        "|------|--------|------------|-----------|:-----:|",
    ]

    # ── Laplace ──
    lp = analysis_results.get("laplace", {})
    lp_pass = True
    lp_metrics = []
    for key, val in lp.items():
        if key.startswith("Tr"):
            r2 = val.get("R2", 0)
            sigma = val.get("sigma", 0)
            ok = r2 >= 0.99
            lp_pass = lp_pass and ok
            lp_metrics.append(f"Tr={key.replace('Tr', '')}: σ={sigma:.4e}, R²={r2:.4f}")
            lines.append(f"| Laplace ({key}) | {'✅' if ok else '❌'} | R²={r2:.4f} | R² ≥ 0.99 | {'✅' if ok else '❌'} |")
    if not lp_metrics:
        lines.append("| Laplace | ⚠️ | — | No data | — |")

    # ── σ-decoupling ──
    dc = analysis_results.get("decoupling", {})
    if dc:
        sigma_ok = dc.get("pass_sigma", False)
        drift_ok = dc.get("pass_drift", False)
        lines.append(f"| σ-decoupling (σ fit) | {'✅' if sigma_ok else '❌'} | R²={dc.get('sigma_R2', 0):.4f} | R² ≥ 0.99 | {'✅' if sigma_ok else '❌'} |")
        rho_l_d = dc.get("rho_l_drift", 1)
        rho_g_d = dc.get("rho_g_drift", 1)
        lines.append(f"| σ-decoupling (ρ drift) | {'✅' if drift_ok else '❌'} | ρ_l={rho_l_d:.4f}, ρ_g={rho_g_d:.4f} | < 1% | {'✅' if drift_ok else '❌'} |")
    else:
        lines.append("| σ-decoupling | ⚠️ | — | No data | — |")

    # ── Coexistence ──
    cx = analysis_results.get("coexistence", {})
    if cx:
        cx_ok = cx.get("pass_coex", False)
        lines.append(f"| Coexistence curve | {'✅' if cx_ok else '❌'} | max Δρ/ρ={cx.get('max_deviation', 1):.4f} | < 2% | {'✅' if cx_ok else '❌'} |")
    else:
        lines.append("| Coexistence curve | ⚠️ | — | No data | — |")

    # ── Spurious ──
    sp = analysis_results.get("spurious", {})
    if sp:
        sp_ok = sp.get("pass_spurious", False)
        lines.append(f"| Spurious currents | {'✅' if sp_ok else '❌'} | max\\|u\\|={sp.get('max_u_max', 1):.4e} | < 0.05 | {'✅' if sp_ok else '❌'} |")
    else:
        lines.append("| Spurious currents | ⚠️ | — | No data | — |")

    # ── Poiseuille ──
    pp = analysis_results.get("poiseuille", {})
    if pp:
        pp_ok = pp.get("pass_poiseuille", False)
        lines.append(f"| Poiseuille flow | {'✅' if pp_ok else '❌'} | min R²={pp.get('R2_min', 0):.4f} | R² ≥ 0.99 | {'✅' if pp_ok else '❌'} |")
    else:
        lines.append("| Poiseuille flow | ⚠️ | — | P1 (needs solver extension) | — |")

    # ── Mesh convergence ──
    mc = analysis_results.get("mesh", {})
    if mc:
        lines.append(f"| Mesh convergence | {'✅' if mc.get('eps_Q_min', 1) < 0.01 else '⚠️'} | ε_min={mc.get('eps_Q_min', 1):.4e} | ε < 1% | {'✅' if mc.get('eps_Q_min', 1) < 0.01 else '⚠️'} |")
    else:
        lines.append("| Mesh convergence | ⚠️ | — | P1 (needs multi-grid) | — |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Detailed results ──
    for sweep_name, data in analysis_results.items():
        lines.append(f"## {sweep_name.title()} Results")
        lines.append("")
        if isinstance(data, dict):
            for k, v in data.items():
                if k.startswith("Tr") and isinstance(v, dict):
                    lines.append(f"- **{k}**: σ={v.get('sigma', 0):.4e}, R²={v.get('R2', 0):.4f}, n={v.get('n_points', 0)}")
                elif k == "figure" and isinstance(v, str) and os.path.exists(v):
                    lines.append(f"- 📊 Figure: `{v}`")
                elif isinstance(v, (int, float, str, bool)):
                    if k not in ("n_cases", "figure"):
                        lines.append(f"- **{k}**: {v}")
        lines.append("")

    lines.append("---")
    lines.append(f"*Report generated by `scripts/06_run_huang_validation_suite.py`*")

    content = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(content)
    print(f"[report] Wrote {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    p = argparse.ArgumentParser(
        description="Huang & Wu (2016) SCMP — one-stop validation suite"
    )
    p.add_argument("--sweep", choices=list(ANALYZERS.keys()),
                   help="Generate design CSV for the specified parameter sweep")
    p.add_argument("--all", action="store_true",
                   help="Run all P0 sweeps (laplace, decoupling, coexistence, spurious)")
    p.add_argument("--run", action="store_true",
                   help="Run batch simulation after generating the design CSV")
    p.add_argument("--analyze", metavar="RESULTS_DIR",
                   help="Post-process an existing batch results directory")
    p.add_argument("--report", metavar="RESULTS_DIR",
                   help="Generate summary report from analyzed results directory")
    p.add_argument("--out-md", metavar="PATH", default=None,
                   help="Output path for the summary report .md")
    p.add_argument("--out-root", metavar="DIR",
                   help="Output root for batch runs")
    p.add_argument("--Tr-list", metavar="T1,T2,...", default=None,
                   help="Comma-separated reduced temperatures (e.g. 0.7,0.9)")
    p.add_argument("--k1-list", metavar="K1,K2,...", default=None,
                   help="Comma-separated k₁ values for decoupling sweep")
    p.add_argument("--app", metavar="PATH", default=BINARY,
                   help="Path to the SCMP binary")
    args = p.parse_args()

    # ── Report mode ──
    if args.report:
        results_dir = args.report
        # Collect analysis JSONs from subdirectories
        analysis_results: dict[str, dict] = {}
        for sweep_name in ANALYZERS:
            # Try direct analysis
            try:
                analyzer = ANALYZERS[sweep_name]
                result = analyzer(results_dir)
                analysis_results[sweep_name] = result
            except Exception as e:
                print(f"[report] {sweep_name}: skipped ({e})")

        out_md = args.out_md or os.path.join(results_dir, "VALIDATION_REPORT.md")
        make_report_md(analysis_results, out_md)
        return

    # ── Post-process mode ──
    if args.analyze:
        results_dir = args.analyze
        print(f"[analyze] {results_dir}")
        basename = os.path.basename(results_dir).lower()

        # Try each analyzer; use the one that finds data
        for sweep_name, analyzer in ANALYZERS.items():
            if sweep_name in basename or sweep_name == "laplace":
                try:
                    result = analyzer(results_dir)
                    if result.get("n_cases", 0) > 0:
                        out_json = os.path.join(results_dir, f"analysis_{sweep_name}.json")
                        with open(out_json, "w") as f:
                            json.dump(result, f, indent=2, default=str)
                        print(f"[analyze] Saved → {out_json}")
                        # Also generate report
                        report_md = os.path.join(results_dir, "VALIDATION_REPORT.md")
                        make_report_md({sweep_name: result}, report_md)
                        return
                except Exception as e:
                    continue

        print(f"[analyze] No matching data found; trying all analyzers...")
        analysis_results = {}
        for sweep_name, analyzer in ANALYZERS.items():
            try:
                result = analyzer(results_dir)
                if result.get("n_cases", 0) > 0:
                    analysis_results[sweep_name] = result
                    out_json = os.path.join(results_dir, f"analysis_{sweep_name}.json")
                    with open(out_json, "w") as f:
                        json.dump(result, f, indent=2, default=str)
            except Exception:
                pass
        if analysis_results:
            report_md = os.path.join(results_dir, "VALIDATION_REPORT.md")
            make_report_md(analysis_results, report_md)
        else:
            print("[analyze] No data found in any sweep.")
        return

    # ── Generate design CSVs ──
    os.makedirs(DATA_DIR, exist_ok=True)

    sweeps_to_run: list[str] = []
    if args.all:
        sweeps_to_run = ["laplace", "decoupling", "coexistence", "spurious"]
    elif args.sweep:
        sweeps_to_run = [args.sweep]
    else:
        p.print_help()
        print("\n┌──────────────────────────────────────────────────────────┐")
        print("│  Quick start                                             │")
        print("│  1. All-in-one P0:                                       │")
        print("│     uv run python scripts/06_run_huang_validation_suite.py --all --run  │")
        print("│  2. Single sweep:                                        │")
        print("│     uv run python scripts/06_run_huang_validation_suite.py --sweep laplace --run  │")
        print("│  3. Manual batch run:                                    │")
        print("│     uv run lbm-batch --csv data/design_scmp_laplace.csv --app lbm_mrt/solver/mcmp_huang_256  │")
        print("│  4. Analyze + report:                                    │")
        print("│     uv run python scripts/06_run_huang_validation_suite.py --analyze results/...  │")
        print("│  5. Report only:                                         │")
        print("│     uv run python scripts/06_run_huang_validation_suite.py --report results/...  │")
        print("└──────────────────────────────────────────────────────────┘")
        return

    # Parse Tr list
    tr_list = None
    if args.Tr_list:
        tr_list = [float(x.strip()) for x in args.Tr_list.split(",")]

    # Parse k1 list
    k1_list = None
    if args.k1_list:
        k1_list = [float(x.strip()) for x in args.k1_list.split(",")]

    generators = {
        "laplace": lambda p: generate_laplace_design(p, tr_list=tr_list),
        "decoupling": lambda p: generate_decoupling_design(p, tr=0.70 if tr_list is None else tr_list[0], k1_list=k1_list),
        "coexistence": lambda p: generate_coexistence_design(p, tr_list=tr_list),
        "spurious": lambda p: generate_spurious_design(p),
        "poiseuille": lambda p: generate_poiseuille_design(p, tr_list=tr_list),
        "mesh": lambda p: generate_mesh_design(p),
    }

    design_paths: dict[str, str] = {}
    for sweep_name in sweeps_to_run:
        design_path = os.path.join(DATA_DIR, f"design_scmp_{sweep_name}.csv")
        generators[sweep_name](design_path)
        design_paths[sweep_name] = design_path

    if args.run:
        out_roots: dict[str, str] = {}
        for sweep_name, design_path in design_paths.items():
            out_roots[sweep_name] = run_batch_sweep(design_path, args.out_root, app=args.app)

        # Auto-analyze all after run
        print("\n[auto-analyze] Post-processing all sweeps...")
        analysis_results: dict[str, dict] = {}
        for sweep_name, out_root in out_roots.items():
            try:
                analyzer = ANALYZERS[sweep_name]
                result = analyzer(out_root)
                analysis_results[sweep_name] = result
                out_json = os.path.join(out_root, f"analysis_{sweep_name}.json")
                with open(out_json, "w") as f:
                    json.dump(result, f, indent=2, default=str)
                print(f"  {sweep_name}: {result.get('n_cases', 0)} cases analyzed")
            except Exception as e:
                print(f"  {sweep_name}: FAILED ({e})")

        # Generate master report
        if analysis_results:
            master_results = args.out_root or RESULTS_DIR
            report_md = os.path.join(master_results, "VALIDATION_REPORT.md")
            make_report_md(analysis_results, report_md)
    else:
        for sweep_name in sweeps_to_run:
            print(f"\nTo run the {sweep_name} sweep:")
            print(f"  uv run lbm-batch --csv {design_paths[sweep_name]} --app {args.app}")


if __name__ == "__main__":
    main()
