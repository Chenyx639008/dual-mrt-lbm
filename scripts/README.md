# Scripts 脚本索引

> **最后更新**: 2026-07-17
> **文件数**: 10（活跃）
> **AI 入口**: 根目录 [CLAUDE.md](../CLAUDE.md)

---

## 工作流脚本（按编号）

| 脚本 | 用途 | 输入 → 输出 |
|------|------|------------|
| `01_make_design.py` | 生成 MCMP 实验设计 CSV | 手动参数 → `data/design_*.csv` |
| `02_make_design_porous.py` | 生成多孔介质实验设计 CSV | 手动参数 → `data/design_*.csv` |
| `03_validate_benchmarks.py` | 运行批量 benchmark 验证 | `data/design_*.csv` → `results/` |
| `04_run_hydrate_sphere_case.py` | 运行球形水合物案例 | `configs/hydrate_sphere_hot.yaml` → `results/` |
| `05_run_porous_hydrate_case.py` | 运行多孔介质水合物案例 | `configs/hydrate_porous.yaml` → `results/` |
| `06_run_huang_validation_suite.py` | 🆕 Huang SCMP 验证套件 | `configs/huang_scmp.yaml` → `results/` |

## 批量/工具脚本

| 脚本 | 用途 |
|------|------|
| `batch_gx_sweep.py` | Gx 参数扫描批量运行 |
| `darcy_batch_manager.py` | 多孔介质达西流批量管理 |
| `generate_tier1_design.py` | Tier 1 验证实验设计生成 |
| `run_porous_batch.py` | 多孔介质批量运行入口 |
| `v4_paper_batch.py` | V4 论文批量运行 |

---

## 脚本依赖关系

```
01_make_design.py ──→ data/design_*.csv
                            │
02_make_design_porous.py ──→ data/design_*.csv
                            │
              ┌─────────────┴──────────────┐
              ↓                            ↓
   03_validate_benchmarks.py        批量运行脚本:
   (验证 + 基准测试)                04/05/06 + batch_*.py
              ↓                            ↓
         results/                     results/
```

## 注意事项

- 脚本 01-02 依赖 `data/geometry/` 中的 `.plt` 几何文件
- 脚本 03 依赖已编译的 CUDA 二进制文件（`mcmp_sim` / `mcmp_sim_hydrate`）
- 批量脚本默认使用 `uv run lbm-batch` 底层命令
- 新统一 CLI (`uv run lbm run <model>`) 可替代大部分脚本的 `lbm-run` 调用
