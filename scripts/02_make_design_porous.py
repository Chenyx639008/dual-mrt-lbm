#!/usr/bin/env python3
"""Generate a design matrix CSV for porous geometry cases.

Scans the data/geometry/ directory for case subdirectories containing
Tecplot .plt files, then expands each geometry across a range of water
saturations (Sw). Also appends the clean (no-hydrate) baseline case.

Output: data/design_from_geometry.csv
"""

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from lbm_mrt.core.paths import DATA_DIR, GEOMETRY_DIR

# ── Configuration ────────────────────────────────────────────────────────
FIX_GX        = 8e-5
FIX_THETA_HYD = 60
FIX_INIT_EQ   = 1
WATER_SEED_FIXED = 1234567

SW_LIST = np.round(np.linspace(0.0, 1.0, 11), 2)

R_OBS = 20
L_GAP = 20
THETA_QUARTZ = 30
GBW_QUARTZ = 0.0
GBW_HYDRATE = 0.0
GY = 0.0
DRIVE_MODE_FIXED = 1

GEOM_ROOT = GEOMETRY_DIR
OUT_CSV   = os.path.join(DATA_DIR, "design_from_geometry.csv")

APPEND_CLEAN_BASELINE = True
CLEAN_CASE_DIRNAME    = "case0000_clean"
CLEAN_GEOM_ID         = 0
CLEAN_MODE_NAME       = "clean"
CLEAN_SH              = 0.0
CLEAN_FULL_SW_SCAN    = True


def _read_Sh_clean_from_meta(meta_path: str) -> float:
    Sh_clean = -1.0
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("Sh_clean_final"):
                    try:
                        Sh_clean = float(line.split(":")[1].strip())
                    except Exception:
                        pass
    return Sh_clean


def _scan_geometry_cases(geom_root: str) -> list[dict]:
    rows = []
    exp_idx = 1

    case_dirs = sorted(
        d for d in os.listdir(geom_root)
        if d.startswith("case") and os.path.isdir(os.path.join(geom_root, d))
    )

    for d in case_dirs:
        if APPEND_CLEAN_BASELINE and d == CLEAN_CASE_DIRNAME:
            continue
        case_dir = os.path.join(geom_root, d)

        geom_files = [
            f for f in os.listdir(case_dir)
            if f.startswith("geometry_case") and f.endswith(".plt")
        ]
        if not geom_files:
            continue

        geom_file = sorted(geom_files)[0]
        m = re.search(r"geometry_case(\d+)\.plt", geom_file)
        geom_id = int(m.group(1)) if m else None

        parts = d.split("_")
        geom_mode = parts[1] if len(parts) >= 2 else "unknown"

        meta_path = os.path.join(case_dir, "meta.txt")
        Sh_clean = _read_Sh_clean_from_meta(meta_path)

        for Sw_val in SW_LIST:
            case_name = f"geom{geom_id:04d}_{geom_mode}_Sh{Sh_clean:.2f}_Sw{Sw_val:.2f}"
            rows.append(dict(
                exp_id=exp_idx, exp_id_str=f"EXP{exp_idx:04d}", case_name=case_name,
                geom_id=geom_id, geom_mode=geom_mode, Sh_clean=Sh_clean,
                geometry_src=os.path.join(case_dir, geom_file),
                morph=0, r_obs=R_OBS, l_gap=L_GAP, coat_thick=0, r_mid=0,
                Sw=float(Sw_val), thetaA_hydrate_deg=int(FIX_THETA_HYD),
                water_seed=int(WATER_SEED_FIXED), thetaA_quartz_deg=THETA_QUARTZ,
                GBw_quartz=GBW_QUARTZ, GBw_hydrate=GBW_HYDRATE,
                Gx=float(FIX_GX), Gy=float(GY), drive_mode=int(DRIVE_MODE_FIXED),
                init_eq=int(FIX_INIT_EQ),
            ))
            exp_idx += 1

    return rows


def _append_clean_baseline(rows: list[dict], geom_root: str) -> list[dict]:
    case_dir = os.path.join(geom_root, CLEAN_CASE_DIRNAME)
    if not os.path.isdir(case_dir):
        print(f"[WARN] Clean case dir not found, skipping: {case_dir}")
        return rows

    geom_files = [
        f for f in os.listdir(case_dir)
        if f.startswith("geometry_case") and f.endswith(".plt")
    ]
    if not geom_files:
        print(f"[WARN] Clean geometry_case*.plt not found in: {case_dir}")
        return rows

    geom_path = os.path.join(case_dir, sorted(geom_files)[0])
    exp_idx = (max(r["exp_id"] for r in rows) + 1) if rows else 1
    sw_list = SW_LIST if CLEAN_FULL_SW_SCAN else np.array([0.0, 1.0])

    for Sw_val in sw_list:
        case_name = f"geom{CLEAN_GEOM_ID:04d}_{CLEAN_MODE_NAME}_Sh{CLEAN_SH:.2f}_Sw{Sw_val:.2f}"
        rows.append(dict(
            exp_id=exp_idx, exp_id_str=f"EXP{exp_idx:04d}", case_name=case_name,
            geom_id=int(CLEAN_GEOM_ID), geom_mode=CLEAN_MODE_NAME, Sh_clean=float(CLEAN_SH),
            geometry_src=geom_path,
            morph=0, r_obs=R_OBS, l_gap=L_GAP, coat_thick=0, r_mid=0,
            Sw=float(Sw_val), thetaA_hydrate_deg=int(FIX_THETA_HYD),
            water_seed=int(WATER_SEED_FIXED), thetaA_quartz_deg=THETA_QUARTZ,
            GBw_quartz=GBW_QUARTZ, GBw_hydrate=GBW_HYDRATE,
            Gx=float(FIX_GX), Gy=float(GY), drive_mode=int(DRIVE_MODE_FIXED),
            init_eq=int(FIX_INIT_EQ),
        ))
        exp_idx += 1

    print(f"[OK] Appended clean baseline: {CLEAN_CASE_DIRNAME} ({len(sw_list)} Sw values)")
    return rows


if __name__ == "__main__":
    np.random.seed(2025)

    rows = _scan_geometry_cases(GEOM_ROOT)
    if not rows:
        raise SystemExit(
            f"[ERR] No caseXXXX_* directories or geometry_caseXXXX.plt found in: {GEOM_ROOT}"
        )

    if APPEND_CLEAN_BASELINE:
        rows = _append_clean_baseline(rows, GEOM_ROOT)

    df = pd.DataFrame(rows)
    front_cols = ["exp_id", "exp_id_str", "case_name", "geom_id", "geom_mode", "Sh_clean", "geometry_src"]
    df = df[front_cols + [c for c in df.columns if c not in front_cols]]

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"[OK] Design written: {OUT_CSV}  ({len(df)} rows)")
    print(df.tail(5).to_string())
