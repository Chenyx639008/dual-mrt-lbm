#!/usr/bin/env python3
"""Generate a parametric design matrix CSV for MRT-LBM geometry sweeps.

Sweeps r_mid (morph=1, pore-fill) and coat_thick (morph=2, coating) over
a fixed set of driving and wettability conditions.

Output: data/design_geom_scan.csv
"""

import os
import sys

# Allow running as `uv run python scripts/01_make_design.py` from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from lbm_mrt.core.paths import DATA_DIR

# ── Configuration ────────────────────────────────────────────────────────
SEED = 2025

R_MID_M1 = list(range(0, 17, 1))   # morph=1: r_mid = 0..16
COAT_M2  = list(range(0, 9, 1))    # morph=2: coat_thick = 0..8

FIX_INIT_EQ   = 1
FIX_GX        = 2e-5
FIX_Sw        = 0.5
FIX_THETA_HYD = 60
WATER_SEED_FIXED = 1234567

R_OBS = 20
L_GAP = 20
THETA_QUARTZ = 30
GBW_QUARTZ = 0.0
GBW_HYDRATE = 0.0
GY = 0.0
DRIVE_MODE_FIXED = 1

OUT_CSV = os.path.join(DATA_DIR, "design_geom_scan.csv")

# ── Build design ──────────────────────────────────────────────────────────

def _gx_tag(x: float) -> str:
    return f"{x:.0e}"


def build_design() -> pd.DataFrame:
    rows = []

    for r_mid in R_MID_M1:
        rows.append(dict(
            morph=1, r_obs=R_OBS, l_gap=L_GAP, coat_thick=0, r_mid=int(r_mid),
            Sw=float(FIX_Sw), thetaA_hydrate_deg=int(FIX_THETA_HYD),
            water_seed=int(WATER_SEED_FIXED), thetaA_quartz_deg=THETA_QUARTZ,
            GBw_quartz=GBW_QUARTZ, GBw_hydrate=GBW_HYDRATE,
            Gx=float(FIX_GX), Gy=float(GY), drive_mode=int(DRIVE_MODE_FIXED),
            init_eq=int(FIX_INIT_EQ),
        ))

    for coat in COAT_M2:
        rows.append(dict(
            morph=2, r_obs=R_OBS, l_gap=L_GAP, coat_thick=int(coat), r_mid=0,
            Sw=float(FIX_Sw), thetaA_hydrate_deg=int(FIX_THETA_HYD),
            water_seed=int(WATER_SEED_FIXED), thetaA_quartz_deg=THETA_QUARTZ,
            GBw_quartz=GBW_QUARTZ, GBw_hydrate=GBW_HYDRATE,
            Gx=float(FIX_GX), Gy=float(GY), drive_mode=int(DRIVE_MODE_FIXED),
            init_eq=int(FIX_INIT_EQ),
        ))

    df = pd.DataFrame(rows).reset_index(drop=True)
    df["exp_id"] = df.index + 1
    df["exp_id_str"] = df["exp_id"].apply(lambda i: f"EXP{i:04d}")

    slice_id = f"eq{FIX_INIT_EQ}_gx{_gx_tag(FIX_GX)}"
    df["slice_id"] = df["morph"].apply(lambda m: f"{slice_id}_m{int(m)}")
    df["slice_idx"] = df.groupby("slice_id").cumcount() + 1

    def _case_name(r):
        return (
            f"{r['exp_id_str']}"
            f"_eq{int(r['init_eq'])}"
            f"_gx{_gx_tag(float(r['Gx']))}"
            f"_m{int(r['morph'])}"
            f"_Sw{float(r['Sw']):.1f}"
            f"_th{int(r['thetaA_hydrate_deg'])}"
            f"_ct{int(r['coat_thick'])}"
            f"_rm{int(r['r_mid'])}"
        )

    df["case_name"] = df.apply(_case_name, axis=1)
    front_cols = ["exp_id", "exp_id_str", "slice_id", "slice_idx", "case_name"]
    df = df[front_cols + [c for c in df.columns if c not in front_cols]]
    return df


if __name__ == "__main__":
    np.random.seed(SEED)
    df = build_design()
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"[OK] Design written: {OUT_CSV}  ({len(df)} rows)")
    print(df.head(5).to_string())
