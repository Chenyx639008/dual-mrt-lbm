#!/usr/bin/env python3
"""
v4 论文专用 — LBM 单相 Darcy 批量运行器
=========================================

扫描 hydrate_structure/data/cases/ 下所有 v4_* 目录，对其中的 geometry_case.plt
运行单相 SCMP Darcy 渗流模拟。

特性:
  - 固定 Gx = 5e-5（单值，不再扫两组）
  - 输出到 hydrate_structure/results/<v4_xxx>/ 对应的子目录
  - 自动跳过已完成 case（有 run_summary.txt）
  - 支持断点续跑
  - GPU 并发控制

用法:
  cd /home/server/projects/lbm_twoflow/complex-porous-media/worktrees/huang_mrt_2d

  # 预览（不运行）
  uv run python scripts/v4_paper_batch.py --dry-run

  # 正式运行
  uv run python scripts/v4_paper_batch.py --max-parallel 2

  # 仅运行指定目录
  uv run python scripts/v4_paper_batch.py --only v4_s5_seed_variance

  # 强制重跑已完成
  uv run python scripts/v4_paper_batch.py --force
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# ── 路径配置 ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOLVER_BIN = PROJECT_ROOT / "lbm_mrt/solver/mcmp_huang_porous_300x300"

HYDRATE_ROOT = Path(
    "/home/server/projects/lbm_twoflow/complex-porous-media/hydrate_structure"
)
CASES_BASE = HYDRATE_ROOT / "data/cases"
RESULTS_BASE = HYDRATE_ROOT / "results"

# ── 日志 ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("v4_batch")

# ── v4 源目录 ─────────────────────────────────────────────────
V4_DIRS = [
    "v4_s5_seed_variance",
    "v4_s1_dense_sh",
    "v4_s1_cgan_dense_sh",
    "v4_s3_mode_param",
    "v4_s2_background_contrast",
    "v4_s2_cgan_background_contrast",
    "v4_s4_growth_cgan",
    "v4_s4_cgan_wang",
    "v4_s4_cgan_ct",
    "v4_stage2_cgan_anchors",
    "v4_stage2_growth_anchors",
    "v4_stage3_cgan",
    "v4_stage2_ct_growth",
    "v4_stage3_growth",
    "v4_baseline_clean",
]

# ── LBM 参数（与 darcy_batch_manager.py 一致）─────────────────
GX = 5.0e-5  # 固定体力

BASE_PARAMS: dict[str, Any] = {
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
    "Gx": GX,
    "Gy": 0.0,
    "drive_mode": 1,
    "OUTPUT_EVERY": 5000,
    "flow_tol_rel": 1.0e-4,
    "flow_need_consec": 3,
    "flow_max_steps": 500000,
    "morph": 0,
    "init_eq": 0,
}

PARAM_KEYS = list(BASE_PARAMS.keys()) + ["geom_file", "file_dir"]


# ═══════════════════════════════════════════════════════════════
#  Utility
# ═══════════════════════════════════════════════════════════════


def write_params(path: Path, geom_file: str, file_dir: str) -> None:
    """写 params.txt。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    params = dict(BASE_PARAMS)
    params["geom_file"] = geom_file
    params["file_dir"] = file_dir
    with open(path, "w") as f:
        for k in PARAM_KEYS:
            v = params.get(k)
            if v is None:
                continue
            if isinstance(v, float):
                f.write(f"{k} {v:.8e}\n")
            else:
                f.write(f"{k} {v}\n")


def is_completed(case_dir: Path) -> bool:
    """检查 case 是否已完成（有 run_summary.txt 且 status ok）。"""
    summary = case_dir / "run_summary.txt"
    if summary.exists():
        if "status ok" in summary.read_text():
            return True
    # 也检查 outputdata_scmp 下的 run_summary.txt（solver 写的位置）
    alt_summary = case_dir / "outputdata_scmp" / "run_summary.txt"
    if alt_summary.exists():
        return True
    return False


def check_binary() -> bool:
    if SOLVER_BIN.exists():
        return True
    log.error("LBM 二进制未找到: %s", SOLVER_BIN)
    return False


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(description="v4 论文 LBM 批量运行器")
    parser.add_argument("--dry-run", action="store_true", help="仅预览，不运行")
    parser.add_argument(
        "--max-parallel", type=int, default=2, help="GPU 并发数 (默认 2)"
    )
    parser.add_argument("--force", action="store_true", help="强制重跑已完成 case")
    parser.add_argument("--only", type=str, help="仅运行指定 v4_* 目录（逗号分隔）")
    args = parser.parse_args()

    if not check_binary():
        sys.exit(1)

    # 确定要扫描的目录
    if args.only:
        target_dirs = [d.strip() for d in args.only.split(",")]
    else:
        target_dirs = V4_DIRS

    # ── 收集所有待运行 case ──
    all_runs: list[dict[str, Any]] = []
    skipped = 0
    missing_plt = 0

    for v4_dir in target_dirs:
        src_dir = CASES_BASE / v4_dir
        if not src_dir.exists():
            log.warning("源目录不存在，跳过: %s", src_dir)
            continue

        for case_name in sorted(os.listdir(src_dir)):
            case_path = src_dir / case_name
            if not case_path.is_dir():
                continue

            geom_plt = case_path / "geometry_case.plt"
            if not geom_plt.exists():
                missing_plt += 1
                continue

            # 输出路径: results/<v4_dir>/<case_name>
            out_dir = RESULTS_BASE / v4_dir / case_name

            if not args.force and is_completed(out_dir):
                skipped += 1
                continue

            all_runs.append(
                {
                    "v4_dir": v4_dir,
                    "case_name": case_name,
                    "geom_file": str(geom_plt.resolve()),
                    "out_dir": out_dir,
                }
            )

    # ── 预览模式 ──
    if args.dry_run:
        print(f"\n{'=' * 70}")
        print(f"v4 论文批量预览")
        print(f"{'=' * 70}")
        print(f"总案例数:       {len(all_runs) + skipped}")
        print(f"已完成(跳过):   {skipped}")
        print(f"待运行:         {len(all_runs)}")
        print(f"Gx:             {GX:.1e}")
        print(f"output:         {RESULTS_BASE}/<v4_xxx>/<case_name>/")
        print()
        for d in target_dirs:
            n = sum(1 for r in all_runs if r["v4_dir"] == d)
            s = sum(1 for r in all_runs if r["v4_dir"] == d)  # already counted
            if n > 0 or any(r["v4_dir"] == d for r in all_runs):
                # count skipped for this dir
                sd = CASES_BASE / d
                if sd.exists():
                    total = len(
                        [
                            x
                            for x in os.listdir(sd)
                            if (sd / x).is_dir()
                            and (sd / x / "geometry_case.plt").exists()
                        ]
                    )
                    done = total - sum(1 for r in all_runs if r["v4_dir"] == d)
                    print(
                        f"  {d:<35s} {done:>4d} done + {sum(1 for r in all_runs if r['v4_dir'] == d):>4d} todo = {total:>4d} total"
                    )
        print()
        if missing_plt > 0:
            print(f"⚠ 缺少 .plt: {missing_plt} 个")
        return

    # ── 运行 ──
    print(f"\n{'=' * 70}")
    print(f"v4 论文批量运行")
    print(f"{'=' * 70}")
    log.info("待运行: %d 个 case (已完成跳过: %d)", len(all_runs), skipped)
    log.info("Gx = %.1e, max_parallel = %d", GX, args.max_parallel)

    if not all_runs:
        log.info("所有 case 已完成！")
        return

    running: list[subprocess.Popen] = []
    t_start = time.perf_counter()

    for i, run in enumerate(all_runs):
        # 等待槽位
        while len(running) >= args.max_parallel:
            for p in list(running):
                if p.poll() is not None:
                    running.remove(p)
                    break
            else:
                time.sleep(2)

        out_dir = run["out_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        params_path = out_dir / "params.txt"
        write_params(params_path, run["geom_file"], str(out_dir))

        label = f"{run['v4_dir']}/{run['case_name']}"
        log.info("[%3d/%3d] %s", i + 1, len(all_runs), label)

        log_path = out_dir / "log.txt"
        with open(log_path, "w") as log_f:
            p = subprocess.Popen(
                [str(SOLVER_BIN), str(params_path)],
                cwd=PROJECT_ROOT,
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )
        running.append(p)

    # 等待全部完成
    for p in running:
        p.wait()

    elapsed = time.perf_counter() - t_start
    log.info("=" * 50)
    log.info(
        "全部 %d 个 case 完成！耗时 %.0f 秒 (%.1f 分)",
        len(all_runs),
        elapsed,
        elapsed / 60,
    )


if __name__ == "__main__":
    main()
