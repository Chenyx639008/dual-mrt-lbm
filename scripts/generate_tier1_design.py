#!/usr/bin/env python3
"""
生成 Tier 1 验证子集的设计 CSV（含 baseline 无hydrate案例）。

筛选逻辑:
  - 5 个 bg 组 (bg00-bg04)，各取第一个 gs 子组
  - 3 种物理模式 (dp/dr=5, gc/gsf=0.05, pf/cha=1.0)
  - 4 个 Sh (0=baseline, 0.10, 0.25, 0.40)
  - seed=0 (第一个种子)
  - 2 个 Gx (1e-5, 5e-5)
  → 5×(baseline×1 + 3mode×3Sh)×2Gx = 5×10×2 = 100 条设计

用法:
  cd /home/server/projects/lbm_twoflow/complex-porous-media/worktrees/huang_mrt_2d
  uv run python scripts/generate_tier1_design.py [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

# ── 路径配置 ──────────────────────────────────────────────────
HYDRATE_CASES_DIR = Path(
    "/home/server/projects/lbm_twoflow/complex-porous-media/hydrate_structure"
    "/data/cases/hydrate_growth_massive"
)
RESULTS_BASE = Path(
    "/home/server/projects/lbm_twoflow/complex-porous-media/hydrate_structure"
    "/results/20260525"
)
BASELINE_GEOM_DIR = RESULTS_BASE / "baseline_geom"
OUT_CSV = Path("data/design_tier1_validation.csv")

# ── 筛选参数 ──────────────────────────────────────────────────
TARGET_BG = ["bg00", "bg01", "bg02", "bg03", "bg04"]
TARGET_GS = {
    "bg00": "gs001000",
    "bg01": "gs001100",
    "bg02": "gs001200",
    "bg03": "gs001300",
    "bg04": "gs001400",
}
TARGET_SH = [0.0, 0.10, 0.25, 0.40]  # 0.0 = baseline (no hydrate)
TARGET_PARAMS = {"pf": "cha1.0", "gc": "gsf0.05", "dp": "dr5"}
GX_VALUES = [1e-5, 5e-5]

# LBM 固定参数
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
    "theta_contact_deg": 30,
    "thetaA_quartz_deg": 30,
    "thetaA_hydrate_deg": 30,
    "G_ads": 0.0,
    "Gy": 0.0,
    "drive_mode": 1,
    "OUTPUT_EVERY": 5000,
    "flow_tol_rel": 1.0e-4,
    "flow_need_consec": 3,
    "flow_max_steps": 500000,
    "morph": 0,
    "init_eq": 0,
}

DESIGN_COLS = [
    "case_name",
    "geom_file",
    "Gx",
    "tau_huang",
    "thetaA_quartz_deg",
    "thetaA_hydrate_deg",
    "OUTPUT_EVERY",
    "file_dir",
    "hs_case_tag",
    "hs_mode",
    "hs_target_Sh",
]


def parse_case_name(name: str) -> dict | None:
    """解析案例目录名。"""
    parts = name.split("_")
    if len(parts) < 6:
        return None
    bg = parts[0]
    gs = parts[1]
    mode = parts[2]
    sh_str = [p for p in parts if p.startswith("Sh")]
    if not sh_str:
        return None
    try:
        sh = float(sh_str[0][2:])
    except ValueError:
        return None
    param = ""
    if mode == "dp":
        p = [x for x in parts if x.startswith("dr")]
        param = p[0] if p else ""
    elif mode == "gc":
        p = [x for x in parts if x.startswith("gsf")]
        param = p[0] if p else ""
    elif mode == "pf":
        p = [x for x in parts if x.startswith("cha")]
        param = p[0] if p else ""
    hs = parts[-1]
    seed_digit = hs[-1] if hs.startswith("hs") else "?"
    return {"name": name, "bg": bg, "gs": gs, "mode": mode, "Sh": sh, "param": param, "seed": seed_digit}


def generate_baseline_plt(src_plt: Path, dst_plt: Path) -> bool:
    """将 phase=0.5 (hydrate) → 0.0 (pore)，生成无hydrate基线几何。"""
    if dst_plt.exists():
        return True
    dst_plt.parent.mkdir(parents=True, exist_ok=True)
    with open(src_plt, "r") as fin:
        lines = fin.readlines()
    header = lines[:3]
    with open(dst_plt, "w") as fout:
        fout.writelines(header)
        for line in lines[3:]:
            parts = line.strip().split(",")
            if len(parts) == 3:
                x, y, val = parts
                if abs(float(val) - 0.5) < 1e-6:
                    val = "0.000000"
                fout.write(f"{x},{y},{val}\n")
            else:
                fout.write(line)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 Tier 1 验证设计 CSV (含 baseline)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-csv", default=str(OUT_CSV))
    args = parser.parse_args()

    cases = sorted(os.listdir(HYDRATE_CASES_DIR))

    # Step 1: 收集匹配的现有案例
    matched: dict[tuple, str] = {}  # (bg, gs, mode, Sh) → case_dir_name
    for c in cases:
        info = parse_case_name(c)
        if info is None:
            continue
        if info["bg"] not in TARGET_BG:
            continue
        if info["gs"] != TARGET_GS.get(info["bg"], ""):
            continue
        if info["param"] != TARGET_PARAMS.get(info["mode"], ""):
            continue
        if info["seed"] != "0":
            continue
        key = (info["bg"], info["gs"], info["mode"], info["Sh"])
        if key not in matched:
            matched[key] = c

    print(f"匹配现有案例: {len(matched)} 个")

    # Step 2: 生成 baseline .plt
    baseline_plts: dict[str, Path] = {}
    for bg in TARGET_BG:
        gs = TARGET_GS[bg]
        bg_gs = f"{bg}_{gs}"
        src_case = None
        for (b, g, m, sh), cname in matched.items():
            if b == bg and g == gs:
                src_case = cname
                break
        if src_case is None:
            print(f"  ⚠ {bg_gs}: 无现有案例，跳过 baseline")
            continue
        src_plt = HYDRATE_CASES_DIR / src_case / "geometry_case.plt"
        dst_plt = BASELINE_GEOM_DIR / f"{bg_gs}_baseline.plt"
        if generate_baseline_plt(src_plt, dst_plt):
            baseline_plts[bg_gs] = dst_plt
            print(f"  ✅ baseline: {dst_plt.name}")

    # Step 3: 生成设计行
    rows = []
    for bg in TARGET_BG:
        gs = TARGET_GS[bg]
        bg_gs = f"{bg}_{gs}"

        # Baseline (Sh=0): 每个 bg 组 1 个 baseline（所有 mode 共用同一个无hydrate几何）
        baseline_geom = baseline_plts.get(bg_gs)
        if baseline_geom:
            for gx in GX_VALUES:
                run_name = f"{bg_gs}_baseline_Sh0_Gx{gx:.0e}"
                rows.append({
                    "case_name": run_name,
                    "geom_file": str(baseline_geom.resolve()),
                    "Gx": f"{gx:.8e}",
                    "tau_huang": BASE_PARAMS["tau_huang"],
                    "thetaA_quartz_deg": BASE_PARAMS["thetaA_quartz_deg"],
                    "thetaA_hydrate_deg": BASE_PARAMS["thetaA_hydrate_deg"],
                    "OUTPUT_EVERY": BASE_PARAMS["OUTPUT_EVERY"],
                    "file_dir": str(RESULTS_BASE / run_name),
                    "hs_case_tag": f"{bg_gs}_baseline_Sh0",
                    "hs_mode": "baseline",
                    "hs_target_Sh": 0.0,
                })

        # 有 hydrate 的 case
        for mode in ["dp", "gc", "pf"]:
            for sh in [s for s in TARGET_SH if s > 0]:
                key = (bg, gs, mode, sh)
                case_dir = matched.get(key)
                if case_dir is None:
                    continue
                geom = HYDRATE_CASES_DIR / case_dir / "geometry_case.plt"
                for gx in GX_VALUES:
                    run_name = f"{case_dir}_Gx{gx:.0e}"
                    rows.append({
                        "case_name": run_name,
                        "geom_file": str(geom.resolve()),
                        "Gx": f"{gx:.8e}",
                        "tau_huang": BASE_PARAMS["tau_huang"],
                        "thetaA_quartz_deg": BASE_PARAMS["thetaA_quartz_deg"],
                        "thetaA_hydrate_deg": BASE_PARAMS["thetaA_hydrate_deg"],
                        "OUTPUT_EVERY": BASE_PARAMS["OUTPUT_EVERY"],
                        "file_dir": str(RESULTS_BASE / run_name),
                        "hs_case_tag": case_dir,
                        "hs_mode": mode,
                        "hs_target_Sh": sh,
                    })

    print(f"\nTier 1: {len(rows)} 条设计")
    for sh_val in TARGET_SH:
        label = f"Sh={sh_val:.2f}" if sh_val > 0 else "baseline"
        print(f"  {label}: {sum(1 for r in rows if abs(r['hs_target_Sh'] - sh_val) < 0.001)}")
    print(f"  参数: OUTPUT_EVERY={BASE_PARAMS['OUTPUT_EVERY']}, flow_max_steps={BASE_PARAMS['flow_max_steps']}")

    if args.dry_run:
        print("\n前 8 条预览:")
        for row in rows[:8]:
            print(f"  {row['case_name']}")
            print(f"    Gx={row['Gx']}, Sh={row['hs_target_Sh']}, mode={row['hs_mode']}")
            print(f"    output={row['file_dir']}")
        print(f"  ...（共 {len(rows)} 条）")
        return

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DESIGN_COLS)
        w.writeheader()
        w.writerows(rows)

    print(f"\n✅ 已写入 {out_csv}")
    print(f"下一步: uv run python scripts/darcy_batch_manager.py run --csv {out_csv} --max-parallel 2")


if __name__ == "__main__":
    main()
