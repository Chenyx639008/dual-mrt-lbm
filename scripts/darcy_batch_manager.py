#!/usr/bin/env python3
"""
多孔介质 SCMP 达西流 — 批量管理器
===================================

跨项目协调 `hydrate_structure` (几何生成) 与 `huang_mrt_2d` (LBM 求解) 的完整工具。

功能:
  design   — 扫描 hydrate_structure 案例目录，生成设计 CSV
  run      — 按设计 CSV 批量运行 LBM 模拟
  collect  — 后处理提取渗透率，回写 flow.csv 到 hydrate_structure
  clean    — 清理指定运行目录的 VTK 文件（保留 params + summary）

用法:
  cd /home/server/projects/lbm_twoflow/complex-porous-media/worktrees/huang_mrt_2d
  uv run python scripts/darcy_batch_manager.py design [--dry-run]
  uv run python scripts/darcy_batch_manager.py run [--csv design.csv] [--max-parallel 2]
  uv run python scripts/darcy_batch_manager.py collect [--results-dir results/hydrate_batch]
  uv run python scripts/darcy_batch_manager.py clean [--results-dir results/hydrate_batch] [--keep-last]

参考:
  hydrate_structure/reserach/LBM_COORDINATION_GUIDE.md  — 跨项目协调指南
  research/POROUS_DARCY_DEVLOG_20260524.md               — 开发日志
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# ── 路径配置 ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOLVER_BIN = PROJECT_ROOT / "lbm_mrt/solver/mcmp_huang_porous_300x300"
HYDRATE_ROOT = Path(
    "/home/server/projects/lbm_twoflow/complex-porous-media/hydrate_structure"
)
HYDRATE_CASES_DIR = HYDRATE_ROOT / "data/cases/hydrate_growth_massive"
HYDRATE_TABLES_DIR = HYDRATE_ROOT / "data/tables"

# ── 日志 ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("darcy_batch")


# ═══════════════════════════════════════════════════════════════
# §1  默认 LBM 参数
# ═══════════════════════════════════════════════════════════════

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
    "Gx": 1.0e-5,
    "Gy": 0.0,
    "drive_mode": 1,
    "OUTPUT_EVERY": 5000,
    "flow_tol_rel": 1.0e-4,
    "flow_need_consec": 3,
    "flow_max_steps": 500000,
    "morph": 0,
    "init_eq": 0,
}

# 所有会被写入 params.txt 的参数键（保持顺序）
PARAM_KEYS = list(BASE_PARAMS.keys()) + ["geom_file", "file_dir"]

# CSV 设计表列名
DESIGN_COLS = [
    "exp_id",
    "exp_id_str",
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

FLOW_HEADER = [
    "case_tag",
    "k",
    "k_abs",
    "k_rel",
    "QA",
    "QB",
    "Qmix",
    "bg_id",
    "mode",
    "target_Sh",
    "actual_Sh",
    "run_status",
    "message",
    "lbm_case_dir",
]


# ═══════════════════════════════════════════════════════════════
# §2  Utility
# ═══════════════════════════════════════════════════════════════


def _parse_meta(meta_path: Path) -> dict[str, str]:
    """解析 hydrate_structure 的 meta.txt。"""
    info: dict[str, str] = {}
    if not meta_path.exists():
        return info
    for line in meta_path.read_text().splitlines():
        line = line.strip()
        if ":" in line:
            k, v = line.split(":", 1)
            info[k.strip()] = v.strip()
    return info


def _write_params(path: Path, params: dict[str, Any]) -> None:
    """写 params.txt。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for k in PARAM_KEYS:
            v = params.get(k)
            if v is None:
                continue
            if isinstance(v, float):
                f.write(f"{k} {v:.8e}\n")
            else:
                f.write(f"{k} {v}\n")


def _check_binary() -> bool:
    """检查 LBM 二进制是否存在并可执行。"""
    if SOLVER_BIN.exists():
        return True
    log.error("LBM 二进制未找到: %s", SOLVER_BIN)
    log.error("请先编译: uv run lbm-build --porous --plt <某个.plt>")
    return False


# ═══════════════════════════════════════════════════════════════
# §3  design — 生成设计 CSV
# ═══════════════════════════════════════════════════════════════


def cmd_design(args: argparse.Namespace) -> None:
    """扫描 hydrate_structure 案例，生成设计 CSV。"""
    out_csv = Path(args.out_csv)
    results_base = Path(args.results_base)

    rows: list[dict[str, Any]] = []
    case_dirs = sorted(HYDRATE_CASES_DIR.iterdir())

    if args.limit:
        case_dirs = case_dirs[: args.limit]

    for idx, case_dir in enumerate(case_dirs, start=1):
        if not case_dir.is_dir():
            continue

        geom_plt = case_dir / "geometry_case.plt"
        if not geom_plt.exists():
            log.warning("跳过(无 .plt): %s", case_dir.name)
            continue

        meta = _parse_meta(case_dir / "meta.txt")
        case_tag = case_dir.name
        mode = meta.get("mode", "unknown")
        try:
            target_sh = float(meta.get("sat_max", -1))
        except ValueError:
            target_sh = -1.0

        # 曝光参数扫描 — 在此处扩展
        gx_list = args.gx if args.gx else [1e-7]
        theta_pairs = []
        if args.theta_quartz and args.theta_hydrate:
            theta_pairs = list(zip(args.theta_quartz, args.theta_hydrate))
        else:
            theta_pairs = [
                (BASE_PARAMS["thetaA_quartz_deg"], BASE_PARAMS["thetaA_hydrate_deg"])
            ]

        for gx in gx_list:
            for th_qz, th_hy in theta_pairs:
                run_id = f"{case_tag}_Gx{gx:.0e}_qz{th_qz}_hy{th_hy}"
                rows.append(
                    {
                        "exp_id": idx,
                        "exp_id_str": f"EXP{idx:04d}",
                        "case_name": run_id,
                        "geom_file": str(geom_plt.resolve()),
                        "Gx": gx,
                        "tau_huang": BASE_PARAMS["tau_huang"],
                        "thetaA_quartz_deg": th_qz,
                        "thetaA_hydrate_deg": th_hy,
                        "OUTPUT_EVERY": BASE_PARAMS["OUTPUT_EVERY"],
                        "file_dir": str(results_base / run_id),
                        "hs_case_tag": case_tag,
                        "hs_mode": mode,
                        "hs_target_Sh": target_sh,
                    }
                )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DESIGN_COLS)
        w.writeheader()
        w.writerows(rows)

    log.info("生成 %d 条设计 → %s", len(rows), out_csv)

    if args.dry_run:
        log.info("Dry-run: 不执行后续步骤。前 5 条预览:")
        for row in rows[:5]:
            log.info(
                "  %s  |  Gx=%.1e  |  geom=%s",
                row["case_name"],
                row["Gx"],
                Path(row["geom_file"]).name,
            )


# ═══════════════════════════════════════════════════════════════
# §4  run — 批量运行 LBM
# ═══════════════════════════════════════════════════════════════


def _run_one(row: dict[str, Any]) -> dict[str, Any]:
    """运行单个 LBM case，返回结果字典。"""
    case_dir = Path(row["file_dir"])
    case_dir.mkdir(parents=True, exist_ok=True)

    # 组装参数
    params = dict(BASE_PARAMS)
    params["geom_file"] = row["geom_file"]
    params["file_dir"] = str(case_dir)
    for k in [
        "Gx",
        "tau_huang",
        "thetaA_quartz_deg",
        "thetaA_hydrate_deg",
        "OUTPUT_EVERY",
    ]:
        if k in row and row[k] is not None:
            params[k] = row[k]

    params_path = case_dir / "params.txt"
    _write_params(params_path, params)

    log.info(
        "[run] %s  Gx=%.1e  geom=%s",
        row["case_name"],
        params["Gx"],
        Path(row["geom_file"]).name,
    )

    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            [str(SOLVER_BIN), str(params_path)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=args.run_timeout if hasattr(args, "run_timeout") else 900,
        )
        elapsed = time.perf_counter() - t0

        # 保存完整日志
        log_path = case_dir / "log.txt"
        log_path.write_text(result.stdout + "\n" + result.stderr)

        # 解析 Done 行，提取 MLUPS
        mlups = None
        for line in result.stdout.splitlines():
            if "Done." in line and "MLUPS" in line:
                m = re.search(r"([\d.]+)\s*MLUPS", line)
                if m:
                    mlups = float(m.group(1))

        # 写运行摘要
        summary_path = case_dir / "run_summary.txt"
        with open(summary_path, "w") as f:
            f.write(f"status ok\n")
            f.write(f"elapsed_s {elapsed:.1f}\n")
            if mlups:
                f.write(f"mlups {mlups:.1f}\n")
            f.write(f"returncode {result.returncode}\n")

        log.info(
            "  ✓ %s  耗时 %.1fs  MLUPS=%s",
            row["case_name"],
            elapsed,
            f"{mlups:.1f}" if mlups else "N/A",
        )

        return {
            "case_name": row["case_name"],
            "status": "ok",
            "elapsed_s": elapsed,
            "mlups": mlups,
        }

    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        log.warning("  ⚠ %s  超时 (%.0fs)", row["case_name"], elapsed)
        return {
            "case_name": row["case_name"],
            "status": "timeout",
            "elapsed_s": elapsed,
        }


def cmd_run(args: argparse.Namespace) -> None:
    """按设计 CSV 批量运行 LBM。"""
    if not _check_binary():
        sys.exit(1)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        log.error("设计 CSV 不存在: %s", csv_path)
        log.error("请先运行: uv run python scripts/darcy_batch_manager.py design")
        sys.exit(1)

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    log.info("共 %d 个 case，最大并发 %d", len(rows), args.max_parallel)

    # 跳过已完成（有 run_summary.txt 且 status ok）
    pending = []
    for row in rows:
        case_dir = Path(row["file_dir"])
        summary = case_dir / "run_summary.txt"
        if summary.exists() and not args.force:
            content = summary.read_text()
            if "status ok" in content:
                log.info("[skip] %s (已完成)", row["case_name"])
                continue
        pending.append(row)

    if not pending:
        log.info("所有 case 已完成，无需运行。")
        return

    log.info("待运行: %d 个", len(pending))

    # 简单并发调度
    running: list[subprocess.Popen] = []
    results: list[dict] = []

    for row in pending:
        # 等待一个槽位
        while len(running) >= args.max_parallel:
            for p in list(running):
                if p.poll() is not None:
                    running.remove(p)
                    break
            else:
                time.sleep(2)

        # 启新进程
        case_dir = Path(row["file_dir"])
        case_dir.mkdir(parents=True, exist_ok=True)
        params = dict(BASE_PARAMS)
        params["geom_file"] = row["geom_file"]
        params["file_dir"] = str(case_dir)
        for k in [
            "Gx",
            "tau_huang",
            "thetaA_quartz_deg",
            "thetaA_hydrate_deg",
            "OUTPUT_EVERY",
        ]:
            if k in row and row[k] is not None:
                params[k] = row[k]

        params_path = case_dir / "params.txt"
        _write_params(params_path, params)

        log.info("[run] %s", row["case_name"])
        log_path = case_dir / "log.txt"

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

    log.info("全部 %d 个 case 完成。", len(pending))


# ═══════════════════════════════════════════════════════════════
# §5  collect — 后处理提取渗透率
# ═══════════════════════════════════════════════════════════════


def _read_vtk_quick(vtk_path: Path) -> dict[str, np.ndarray] | None:
    """快速读取 VTK legacy binary 中的 flag 和 ux。"""
    try:
        from lbm_mrt.io.vtk_reader import read_vtk_scalars  # type: ignore[import-untyped]

        fields, _nx, _ny = read_vtk_scalars(str(vtk_path))
        return fields
    except Exception:
        pass

    # 备用：裸读
    try:
        with open(vtk_path, "rb") as f:
            while True:
                line = f.readline()
                if b"DATASET" in line:
                    dims = [int(x) for x in f.readline().split()[1:]]
                    f.readline()
                    f.readline()
                    f.readline()
                    break
            npts = dims[0] * dims[1]
            results: dict[str, np.ndarray] = {}
            while True:
                line = f.readline()
                if not line:
                    break
                line_str = line.decode("ascii", errors="ignore").strip()
                if line_str.startswith("SCALARS"):
                    name = line_str.split()[1]
                    f.readline()
                    if name == "flag":
                        raw = f.read(npts * 4)
                        results[name] = np.frombuffer(raw, dtype=">i4").copy()
                    else:
                        raw = f.read(npts * 8)
                        results[name] = np.frombuffer(raw, dtype=">f8").copy()
                if len(results) >= 3:  # flag, ux, rho 就够了
                    break
            return results
    except Exception as e:
        log.warning("读取 VTK 失败 %s: %s", vtk_path.name, e)
        return None


def _compute_k(vtk_path: Path, Gx: float, nu: float = 1.0 / 3.0) -> float | None:
    """从 VTK 提取 Darcy 渗透率 k = Q/N * ν/Gx。"""
    fields = _read_vtk_quick(vtk_path)
    if fields is None:
        return None
    flag = fields.get("flag")
    ux = fields.get("ux")
    rho = fields.get("rho")
    if flag is None or ux is None:
        return None

    fluid = flag == 1
    N_fluid = fluid.sum()
    if N_fluid == 0:
        return 0.0

    if rho is not None:
        Q = (ux[fluid] * rho[fluid]).sum()
    else:
        Q = ux[fluid].sum()

    u_darcy = Q / N_fluid
    k = u_darcy * nu / Gx
    return float(k)


def cmd_collect(args: argparse.Namespace) -> None:
    """后处理：提取渗透率并回写 flow.csv。"""
    results_dir = Path(args.results_dir)
    flow_csv = Path(args.flow_csv)

    if not results_dir.exists():
        log.error("结果目录不存在: %s", results_dir)
        sys.exit(1)

    rows: list[dict[str, Any]] = []
    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        params_path = run_dir / "params.txt"
        summary_path = run_dir / "run_summary.txt"

        # 读 Gx
        Gx = 1.0e-7
        if params_path.exists():
            for line in params_path.read_text().splitlines():
                if line.startswith("Gx "):
                    try:
                        Gx = float(line.split()[1])
                    except (ValueError, IndexError):
                        pass

        # 找最后的 VTK
        vtk_dir = run_dir / "outputdata_scmp"
        vtk_files = sorted(vtk_dir.glob("flow*.vtk")) if vtk_dir.exists() else []
        last_vtk = vtk_files[-1] if vtk_files else None

        k = None
        status = "no_vtk"
        message = ""

        if last_vtk:
            k = _compute_k(last_vtk, Gx)
            status = "ok" if k is not None else "vtk_read_error"
            step_str = last_vtk.stem.replace("flow", "")
            message = f"step={step_str}"
        elif summary_path.exists():
            summary = summary_path.read_text()
            if "status ok" in summary:
                status = "no_vtk_but_ok"
                message = "VTK 缺失但运行成功"

        rows.append(
            {
                "case_tag": run_dir.name,
                "k": f"{k:.6e}" if k else "",
                "k_abs": f"{k:.6e}" if k else "",
                "k_rel": "",
                "QA": "",
                "QB": "",
                "Qmix": "",
                "bg_id": "",
                "mode": "",
                "target_Sh": "",
                "actual_Sh": "",
                "run_status": status,
                "message": message,
                "lbm_case_dir": str(run_dir),
            }
        )

    # 写 flow.csv
    flow_csv.parent.mkdir(parents=True, exist_ok=True)
    existed = flow_csv.exists()
    with open(flow_csv, "w" if args.overwrite else "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FLOW_HEADER, extrasaction="ignore")
        if not existed or args.overwrite:
            w.writeheader()
        for row in rows:
            w.writerow(row)

    ok_count = sum(1 for r in rows if r["run_status"] == "ok")
    log.info("提取 %d/%d 个 case 的渗透率 → %s", ok_count, len(rows), flow_csv)

    # 简要汇总
    k_vals = []
    for r in rows:
        if r["k"]:
            try:
                k_vals.append(float(r["k"]))
            except ValueError:
                pass
    if k_vals:
        log.info(
            "k 范围: [%.3e, %.3e]  均值: %.3e",
            min(k_vals),
            max(k_vals),
            np.mean(k_vals),
        )


# ═══════════════════════════════════════════════════════════════
# §6  clean — 清理 VTK 文件
# ═══════════════════════════════════════════════════════════════


def cmd_clean(args: argparse.Namespace) -> None:
    """清理 VTK 文件以释放磁盘空间。"""
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        log.error("结果目录不存在: %s", results_dir)
        sys.exit(1)

    removed_bytes = 0
    for run_dir in results_dir.iterdir():
        if not run_dir.is_dir():
            continue
        vtk_dir = run_dir / "outputdata_scmp"
        if not vtk_dir.exists():
            continue

        vtk_files = sorted(vtk_dir.glob("flow*.vtk"))
        if not vtk_files:
            continue

        if args.keep_last:
            # 只保留最后一步
            keep = {vtk_files[-1]}
            to_remove = [f for f in vtk_files if f not in keep]
        elif args.dry_run:
            to_remove = []
            total_size = sum(f.stat().st_size for f in vtk_files)
            log.info(
                "[dry-run] %s: %d VTK, %.1f MB",
                run_dir.name,
                len(vtk_files),
                total_size / 1e6,
            )
            continue
        else:
            # 删除所有 VTK（保留 params + summary）
            to_remove = list(vtk_files)

        for f in to_remove:
            removed_bytes += f.stat().st_size
            f.unlink()

        if not args.dry_run and to_remove:
            log.info("清理 %s: 删除 %d 个 VTK", run_dir.name, len(to_remove))

    if removed_bytes > 0:
        log.info("共释放 %.1f MB", removed_bytes / 1e6)


# ═══════════════════════════════════════════════════════════════
# §7  CLI
# ═══════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="多孔介质 SCMP 达西流批量管理器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  uv run python scripts/darcy_batch_manager.py design --dry-run
  uv run python scripts/darcy_batch_manager.py run --max-parallel 2
  uv run python scripts/darcy_batch_manager.py collect
  uv run python scripts/darcy_batch_manager.py clean --keep-last
        """,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── design ──
    p_design = sub.add_parser("design", help="扫描 hydrate_structure 生成设计 CSV")
    p_design.add_argument("--out-csv", default="data/design_hydrate_batch.csv")
    p_design.add_argument("--results-base", default="results/hydrate_batch")
    p_design.add_argument("--gx", type=float, nargs="+", help="Gx 扫描列表 (默认 1e-7)")
    p_design.add_argument(
        "--theta-quartz", type=float, nargs="+", help="石英接触角扫描列表"
    )
    p_design.add_argument(
        "--theta-hydrate", type=float, nargs="+", help="水合物接触角扫描列表"
    )
    p_design.add_argument("--limit", type=int, help="限制扫描 case 数（调试用）")
    p_design.add_argument("--dry-run", action="store_true", help="仅预览不写入")

    # ── run ──
    p_run = sub.add_parser("run", help="按设计 CSV 批量运行 LBM")
    p_run.add_argument("--csv", default="data/design_hydrate_batch.csv")
    p_run.add_argument(
        "--max-parallel", type=int, default=2, help="GPU 最大并发数 (默认 2)"
    )
    p_run.add_argument("--force", action="store_true", help="强制重新运行已完成 case")
    p_run.add_argument(
        "--run-timeout", type=int, default=900, help="单 case 超时秒数 (默认 900)"
    )

    # ── collect ──
    p_collect = sub.add_parser("collect", help="提取渗透率回写 flow.csv")
    p_collect.add_argument("--results-dir", default="results/hydrate_batch")
    p_collect.add_argument("--flow-csv", default=str(HYDRATE_TABLES_DIR / "flow.csv"))
    p_collect.add_argument(
        "--overwrite", action="store_true", help="覆盖而非追加 flow.csv"
    )

    # ── clean ──
    p_clean = sub.add_parser("clean", help="清理 VTK 文件释放空间")
    p_clean.add_argument("--results-dir", default="results/hydrate_batch")
    p_clean.add_argument(
        "--keep-last", action="store_true", help="每个 case 仅保留最后一步 VTK"
    )
    p_clean.add_argument("--dry-run", action="store_true", help="仅预览")

    args = parser.parse_args()

    cmd_map = {
        "design": cmd_design,
        "run": cmd_run,
        "collect": cmd_collect,
        "clean": cmd_clean,
    }
    cmd_map[args.command](args)


if __name__ == "__main__":
    main()
