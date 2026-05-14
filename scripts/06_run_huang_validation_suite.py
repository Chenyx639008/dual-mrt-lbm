#!/usr/bin/env python3
"""Batch SCMP runner for Huang & Wu (2016) validation.

Generates design CSVs, runs batch simulations via lbm-batch / run_batch,
and optionally post-processes results.

Usage::

    # Generate design + run batch
    uv run python scripts/06_run_huang_validation_suite.py --sweep laplace --run

    # Generate design CSV only
    uv run python scripts/06_run_huang_validation_suite.py --sweep decoupling

    # Run existing design CSV via lbm-batch CLI
    uv run lbm-batch --csv data/design_scmp_laplace.csv --app lbm_mrt/solver/mcmp_huang_256

    # Post-process existing results
    uv run python scripts/06_run_huang_validation_suite.py --analyze results/design_scmp_laplace_20260513_120000
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from lbm_mrt.core.paths import PROJ_ROOT, DATA_DIR, RESULTS_DIR
from lbm_mrt.io.vtk_reader import latest_vtk, read_vtk_scalars
from lbm_mrt.validation.analytical import (
    detect_interface_radius,
    extract_rho_l_g,
    fit_pressure_inside_outside,
    laplace_sigma,
)

BINARY = os.path.join(PROJ_ROOT, "lbm_mrt", "solver", "mcmp_huang_256")


# ═══════════════════════════════════════════════════════════════════════════
# Design CSV generators (one CSV → one batch sweep → one results/ subfolder)
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
        "k1_huang": 0.0,
        "k2_huang": 0.0,
        "kd_huang": -0.083333,
        "alpha_meq": 1.0,
        "cs_a": 0.75,
        "cs_b": 4.0,
        "cs_R": 1.0,
        "cs_T": 0.06,
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


def generate_laplace_design(out_path: str, r_list=None) -> str:
    import pandas as pd

    if r_list is None:
        r_list = [20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]
    base = _base_params()
    rows = []
    for R in r_list:
        row = dict(base)
        row["case_name"] = f"laplace_R{R:.0f}"
        row["huang_R0"] = R
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[design] {out_path} ({len(rows)} cases, R in {r_list})")
    return out_path


def generate_decoupling_design(out_path: str, k1_list=None) -> str:
    import pandas as pd

    if k1_list is None:
        k1_list = [0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.16667]
    base = _base_params()
    rows = []
    for k1 in k1_list:
        row = dict(base)
        tag = str(k1).replace(".", "p")
        row["case_name"] = f"decoupling_k1_{tag}"
        row["k1_huang"] = k1
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[design] {out_path} ({len(rows)} cases, k1 in {k1_list})")
    return out_path


def generate_coexistence_design(out_path: str, T_list=None) -> str:
    import pandas as pd

    if T_list is None:
        T_list = [0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.068]
    base = _base_params()
    rows = []
    for T in T_list:
        row = dict(base)
        tag = str(T).replace(".", "p")
        row["case_name"] = f"coex_T_{tag}"
        row["cs_T"] = T
        row["huang_init_mode"] = 2
        row["huang_yc"] = 64.0
        row["huang_W"] = 5.0
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[design] {out_path} ({len(rows)} cases, T in {T_list})")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# Batch runner
# ═══════════════════════════════════════════════════════════════════════════


def run_batch_sweep(design_csv: str, out_root: str | None = None) -> str:
    """Run all cases in a design CSV via the batch runner."""
    from lbm_mrt.runners.batch_run import run_batch

    if out_root is None:
        tag = os.path.splitext(os.path.basename(design_csv))[0]
        out_root = os.path.join(RESULTS_DIR, f"{tag}_{datetime.now():%Y%m%d_%H%M%S}")

    print(f"\n[batch] {design_csv}  →  {out_root}")
    n_ok, n_fail = run_batch(
        design_csv=design_csv,
        out_root=out_root,
        app=BINARY,
        resume=False,
    )
    print(f"[batch] Done: {n_ok} ok, {n_fail} failed  |  results in {out_root}/")
    return out_root


# ═══════════════════════════════════════════════════════════════════════════
# Post-processing
# ═══════════════════════════════════════════════════════════════════════════


def analyze_laplace(results_dir: str) -> dict:
    """Post-process Laplace sweep results directory."""
    case_dirs = sorted(glob.glob(os.path.join(results_dir, "laplace_R*")))
    Rs, dPs = [], []
    for cd in case_dirs:
        vtk_dir = os.path.join(cd, "outputdata_scmp")
        vtk_path = latest_vtk(vtk_dir)
        if vtk_path is None:
            continue
        fields, nx, ny = read_vtk_scalars(vtk_path)
        rho = fields["rho"]
        pressure = fields.get("pressure", np.zeros_like(rho))
        center, radius = detect_interface_radius(rho, nx, ny)
        p_in, p_out = fit_pressure_inside_outside(rho, pressure, center, radius)
        if radius > 0:
            Rs.append(radius)
            dPs.append(p_in - p_out)
            print(f"  {os.path.basename(cd)}: R={radius:.1f}, ΔP={p_in - p_out:.6f}")

    result = {"R": Rs, "delta_P": dPs}
    if len(Rs) >= 3:
        sigma, r2, intercept = laplace_sigma(np.array(Rs), np.array(dPs))
        result.update(sigma=sigma, r2=r2, intercept=intercept)
        print(f"  σ={sigma:.6f}, R²={r2:.4f}, intercept={intercept:.2e}")
    return result


def analyze_coexistence(results_dir: str) -> dict:
    """Post-process coexistence sweep results directory."""
    case_dirs = sorted(glob.glob(os.path.join(results_dir, "coex_T_*")))
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
            T_str = os.path.basename(cd).replace("coex_T_", "").replace("p", ".")
            T = float(T_str)
        except ValueError:
            continue
        Ts.append(T)
        rho_ls.append(rl)
        rho_gs.append(rg)
        print(f"  T={T:.4f}: ρ_l={rl:.6f}, ρ_g={rg:.6f}")
    return {"T": Ts, "rho_l": rho_ls, "rho_g": rho_gs}


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    p = argparse.ArgumentParser(
        description="Huang & Wu (2016) SCMP — batch design generator, runner & analyzer"
    )
    p.add_argument(
        "--sweep",
        choices=["laplace", "decoupling", "coexistence"],
        help="Generate design CSV for the specified parameter sweep",
    )
    p.add_argument(
        "--run",
        action="store_true",
        help="Run batch simulation after generating the design CSV",
    )
    p.add_argument(
        "--analyze",
        metavar="RESULTS_DIR",
        help="Post-process an existing batch results directory",
    )
    p.add_argument(
        "--out-root",
        metavar="DIR",
        help="Output root for batch runs (default: results/<tag>_<timestamp>)",
    )
    args = p.parse_args()

    # ── Post-process mode ──
    if args.analyze:
        results_dir = args.analyze
        print(f"[analyze] {results_dir}")
        basename = os.path.basename(results_dir).lower()
        if "laplace" in basename:
            result = analyze_laplace(results_dir)
        elif "coex" in basename:
            result = analyze_coexistence(results_dir)
        else:
            print(f"[analyze] Unknown sweep type; trying laplace analysis...")
            result = analyze_laplace(results_dir)
        out_json = os.path.join(results_dir, "analysis.json")
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"[analyze] Saved → {out_json}")
        return

    # ── Generate design CSV ──
    if args.sweep:
        os.makedirs(DATA_DIR, exist_ok=True)
        design_path = os.path.join(DATA_DIR, f"design_scmp_{args.sweep}.csv")
        generators = {
            "laplace": generate_laplace_design,
            "decoupling": generate_decoupling_design,
            "coexistence": generate_coexistence_design,
        }
        generators[args.sweep](design_path)

        if args.run:
            run_batch_sweep(design_path, args.out_root)
        else:
            print(f"\nTo run this sweep:")
            print(
                f"  uv run lbm-batch --csv {design_path} --app lbm_mrt/solver/mcmp_huang_256"
            )
    else:
        p.print_help()
        print("\n┌──────────────────────────────────────────────────────────┐")
        print("│  Quick start                                             │")
        print("│  1. Generate + run sweep:                                │")
        print(
            "│     uv run python scripts/06_run_huang_validation_suite.py --sweep laplace --run  │"
        )
        print("│  2. Manual batch run:                                    │")
        print(
            "│     uv run lbm-batch --csv data/design_scmp_laplace.csv --app lbm_mrt/solver/mcmp_huang_256  │"
        )
        print("│  3. Analyze results:                                     │")
        print(
            "│     uv run python scripts/06_run_huang_validation_suite.py --analyze results/...  │"
        )
        print("└──────────────────────────────────────────────────────────┘")


if __name__ == "__main__":
    main()
