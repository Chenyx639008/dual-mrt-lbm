#!/usr/bin/env python3
"""Batch run single-phase porous media flow with multiple Gx and geometries."""

import subprocess
import os
import sys
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOLVER_BIN = PROJECT_ROOT / "lbm_mrt/solver/mcmp_huang_porous_300x300"
GEOM_DIR = PROJECT_ROOT / "data/geometry/for_lbm"
RESULTS_DIR = PROJECT_ROOT / "results"

# Test configurations
GX_VALUES = [1e-8, 5e-8, 1e-7, 5e-7]
GEOM_FILES = sorted(GEOM_DIR.glob("geometry_case*.plt"))

# Base params
BASE_PARAMS = {
    "pp_mode": 1,
    "huang_init_mode": 5,
    "epsilon_huang": 1.7,
    "k2_huang": 0.0,
    "tau_huang": 1.5,
    "Lambda_huang": 0.08333,
    "alpha_meq": 1.0,
    "cs_a": 1.0,
    "cs_b": 4.0,
    "cs_R": 1.0,
    "cs_T": 0.9,
    "cs_G": -1.0,
    "huang_rho_l": 1.0,
    "huang_rho_g": 1.0,
    "huang_u_max": 0.15,
    "huang_psi_cut": 1.0e-3,
    "theta_contact_deg": 90.0,
    "G_ads": 0.0,
    "Gy": 0.0,
    "drive_mode": 1,
    "OUTPUT_EVERY": 20000,
    "flow_tol_rel": 1.0e-5,
    "flow_need_consec": 3,
    "flow_max_steps": 200000,
}


def run_case(geom_path: Path, gx: float) -> dict:
    """Run one case and return results."""
    case_id = geom_path.stem.replace("geometry_case", "")
    run_id = f"case{case_id}_Gx{gx:.0e}"
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write params.txt
    params = dict(BASE_PARAMS)
    params["Gx"] = gx
    params["geom_file"] = str(geom_path)
    params["file_dir"] = str(run_dir)

    params_path = run_dir / "params.txt"
    with open(params_path, "w") as f:
        for k, v in params.items():
            if isinstance(v, float):
                f.write(f"{k} {v:.8e}\n")
            else:
                f.write(f"{k} {v}\n")

    # Run solver
    print(f"\n{'=' * 60}")
    print(f"Running: {run_id}")
    print(f"  geom: {geom_path.name}, Gx: {gx:.1e}")
    print(f"{'=' * 60}")

    result = subprocess.run(
        [str(SOLVER_BIN), str(params_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )

    # Parse output
    summary_path = run_dir / run_id / "run_summary.txt"
    Q = None
    steady_step = None
    elapsed = None
    if summary_path.exists():
        with open(summary_path) as f:
            for line in f:
                if line.startswith("Q "):
                    Q = float(line.split()[1])
                elif line.startswith("steady_step "):
                    steady_step = int(line.split()[1])
                elif line.startswith("elapsed_s "):
                    elapsed = float(line.split()[1])

    print(f"  Q={Q}, steady_step={steady_step}, elapsed={elapsed}s")

    return {
        "run_id": run_id,
        "geom": geom_path.name,
        "Gx": gx,
        "Q": Q,
        "steady_step": steady_step,
        "elapsed_s": elapsed,
    }


def main():
    results = []

    # Gx sweep on geometry_case0000
    geom0 = GEOM_DIR / "geometry_case0000.plt"
    print("=== Gx Sweep on geometry_case0000 ===")
    for gx in GX_VALUES:
        r = run_case(geom0, gx)
        results.append(r)

    # Geometry sweep at Gx=1e-7
    print("\n=== Geometry Sweep at Gx=1e-7 ===")
    for geom in GEOM_FILES:
        if geom == geom0:
            continue  # Already done
        r = run_case(geom, 1e-7)
        results.append(r)

    # Summary
    print("\n" + "=" * 60)
    print("BATCH SUMMARY")
    print("=" * 60)
    print(f"{'Run':<30} {'Gx':>10} {'Q':>14} {'SteadyStep':>12} {'Time(s)':>10}")
    print("-" * 76)
    for r in results:
        print(
            f"{r['run_id']:<30} {r['Gx']:>10.1e} {r['Q'] or 'N/A':>14} {r['steady_step'] or 'N/A':>12} {r['elapsed_s'] or 'N/A':>10}"
        )

    # Check Darcy linearity (Q ∝ Gx)
    gx_results = [
        r for r in results if "case0000" in r["run_id"] and r["Q"] is not None
    ]
    if len(gx_results) >= 2:
        print("\n=== Darcy Linearity Check (case0000) ===")
        gx_vals = [r["Gx"] for r in gx_results]
        q_vals = [r["Q"] for r in gx_results]
        ratios = [q / g for q, g in zip(q_vals, gx_vals)]
        print(f"  Q/Gx ratios: {[f'{r:.1f}' for r in ratios]}")
        if len(ratios) >= 2:
            mean_ratio = sum(ratios) / len(ratios)
            max_dev = max(abs(r / mean_ratio - 1) for r in ratios)
            print(f"  Mean Q/Gx: {mean_ratio:.1f}, Max deviation: {max_dev * 100:.1f}%")
            if max_dev < 0.1:
                print("  ✓ Darcy regime confirmed (Q ∝ Gx)")
            else:
                print("  ⚠ Deviation from Darcy linearity detected")


if __name__ == "__main__":
    main()
