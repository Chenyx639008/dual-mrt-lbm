#!/usr/bin/env python3
"""Generate batch design CSVs for Huang & Wu (2016) SCMP parameter sweeps.

Produces CSV files compatible with lbm_mrt.runners.batch_run.run_batch().

Usage::

    uv run python scripts/generate_scmp_design.py --sweep laplace     # Laplace law radius sweep
    uv run python scripts/generate_scmp_design.py --sweep decoupling  # k1 sweep for σ-decoupling
    uv run python scripts/generate_scmp_design.py --sweep coexistence # T sweep for coexistence curve
    uv run python scripts/generate_scmp_design.py --sweep all         # generate all CSVs
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUT_DIR = "data"


def _base_params() -> dict:
    """Common SCMP base parameters."""
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


def generate_laplace_sweep() -> str:
    """Generate CSV for Laplace law radius sweep."""
    base = _base_params()
    radii = [20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]
    rows = []
    for R in radii:
        row = dict(base)
        row["case_name"] = f"laplace_R{R:.0f}"
        row["huang_R0"] = R
        rows.append(row)
    df = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, "design_scmp_laplace.csv")
    df.to_csv(path, index=False)
    print(f"[generate] {path} ({len(rows)} cases, R ∈ {radii})")
    return path


def generate_decoupling_sweep() -> str:
    """Generate CSV for σ-decoupling k1 sweep."""
    base = _base_params()
    k1_vals = [0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.16667]
    rows = []
    for k1 in k1_vals:
        row = dict(base)
        row["case_name"] = f"decoupling_k1{k1:.4f}".replace(".", "p")
        row["k1_huang"] = k1
        rows.append(row)
    df = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, "design_scmp_decoupling.csv")
    df.to_csv(path, index=False)
    print(f"[generate] {path} ({len(rows)} cases, k1 ∈ {k1_vals})")
    return path


def generate_coexistence_sweep() -> str:
    """Generate CSV for coexistence curve T sweep."""
    base = _base_params()
    # CS-EOS: a=0.75, b=4, R=1 → Tc≈0.0707
    T_vals = [0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.068]
    rows = []
    for T in T_vals:
        row = dict(base)
        row["case_name"] = f"coex_T{T:.4f}".replace(".", "p")
        row["cs_T"] = T
        row["huang_init_mode"] = 2  # flat interface
        row["huang_yc"] = 64.0
        row["huang_W"] = 5.0
        rows.append(row)
    df = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, "design_scmp_coexistence.csv")
    df.to_csv(path, index=False)
    print(f"[generate] {path} ({len(rows)} cases, T ∈ {T_vals})")
    return path


def generate_spurious_sweep() -> str:
    """Generate CSV for spurious currents comparison (single case)."""
    base = _base_params()
    row = dict(base)
    row["case_name"] = "spurious_currents"
    df = pd.DataFrame([row])
    path = os.path.join(OUT_DIR, "design_scmp_spurious.csv")
    df.to_csv(path, index=False)
    print(f"[generate] {path} (1 case)")
    return path


def main() -> None:
    p = argparse.ArgumentParser(description="Generate SCMP design CSVs for batch runs")
    p.add_argument(
        "--sweep",
        choices=["laplace", "decoupling", "coexistence", "spurious", "all"],
        default="all",
        help="Which sweep CSV to generate",
    )
    args = p.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    sweeps = {
        "laplace": generate_laplace_sweep,
        "decoupling": generate_decoupling_sweep,
        "coexistence": generate_coexistence_sweep,
        "spurious": generate_spurious_sweep,
    }

    if args.sweep == "all":
        for fn in sweeps.values():
            fn()
    else:
        sweeps[args.sweep]()

    print(
        "\nUsage: uv run lbm-batch --csv data/design_scmp_<sweep>.csv --app lbm_mrt/solver/mcmp_huang_256"
    )


if __name__ == "__main__":
    main()
