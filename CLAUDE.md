# dual-mrt-lbm — AI 入口

> **当前状态**: 双轨架构 Phase 6 完成 · 98 项 JAX 自动化测试 · MCMP + 水合物已注册 | 2026-08-08 | **相场阶段 0/1/2 ✅ + CUDA pf_ns_2d ✅ + 阶段 3 根因定位**：MRT-LB 保守 AC 液滴溶解根因=平衡矩矩序错乱（已数值验证）；实施计划 `research/phasefield_mrt_ac_implementation_plan.md`（BGK→MRT→耦合→湿壁→CUDA 五步）；文献库 `phasefield_LBM/`（Liang 2018 BGK 母本 + Yang 2024 MRT）
> **GitHub**: https://github.com/Chenyx639008/dual-mrt-lbm
> **一键入口**: `uv run lbm run scmp_cs_huang_256`
> **框架手册**: [`research/FRAMEWORK_GUIDE.md`](research/FRAMEWORK_GUIDE.md) | **研究索引**: [`research/INDEX.md`](research/INDEX.md) | **实施记录**: [`research/implementation/`](research/implementation/) | **测试**: [`jax_lbm/tests/README.md`](jax_lbm/tests/README.md) | **相场计划**: [`research/phasefield_development_plan.md`](research/phasefield_development_plan.md)

双轨 MRT-LBM 两相流 + 甲烷水合物分解 — C++17/CUDA 生产轨 + JAX 镜像验证轨 + Python 统一调度。

---

## 快速命令

```bash
uv sync --all-groups
uv run lbm models                             # 9 个注册模型 (5 SCMP + 2 MCMP + 2 Hydrate)
uv run lbm run scmp_cs_huang_256 --steps 50000
JAX_ENABLE_X64=1 uv run pytest jax_lbm/tests/ -v  # 98 项自动化测试 (含相场 pf)
uv run pytest && uv run ruff check . && uv run ruff format .
```

---

## 核心约定


- **配置流**: YAML → `load_config()` → flat dict → `params.txt` → `load_params_txt()` → `__constant__` 设备内存。水合物运行时叠加 `hydrate.yaml`。SCMP 统一路径：`ModelDefinition.to_params_dict()` → `params.txt`。
- **网格编译期常量**: `NX`/`NY` 是 `LBM.h` 中 `constexpr`，改网格需重编译。统一 Huang 二进制用 `d_nx_active`/`d_ny_active` 运行时网格（Phase 2 进行中）。
- **params.txt 是唯一边界**: Python → C++ 通信仅通过 `params.txt`。key 名必须精确匹配 `RuntimeParams` 字段名，不匹配的被静默忽略。
- **VTK Legacy Binary**: 输出为 big-endian 二进制 VTK（`results/<case>/outputdata_eq/flow*.vtk`），读入需用 `lbm_mrt/io/vtk_reader.py`。
- **JAX 独立于 CUDA**: `jax_lbm/` 是独立 D2Q9 BGK 实现，不依赖 CUDA 编译，用于 autograd 敏感度分析。
- **Python**: uv 管理（pyproject.toml + uv.lock），GPU 目标 sm_120（RTX 5090 / H100）。

---

## 目录导航

| 目录 | 说明 | 详细索引 |
|------|------|---------|
| `lbm_mrt/solver/` | CUDA C++17 核心（内核、EOS、边界、水合物） | [`CLAUDE.md`](lbm_mrt/solver/CLAUDE.md) |
| `lbm_mrt/unified/` | 🆕 统一框架（models / CLI / runner） | — |
| `jax_lbm/` | 🆕 JAX D2Q9 BGK + MRT + Huang-Zhang 力 + 伴随梯度 | [`tests/README.md`](jax_lbm/tests/README.md) |
| `jax_lbm/tests/` | 🆕 98 项自动化测试 (streaming/力/碰撞/EOS/液滴/BC/伴随/vmap/相场 pf) | [`README.md`](jax_lbm/tests/README.md) |
| `phasefield_LBM/` | 🆕 相场文献库（4 PDF 转换 MD+JSON，Liang 2018 / Yang 2024 / JCP 2023 / CMWA 2021 + 对比总结） | [`phasefield_literature_summary.md`](phasefield_LBM/phasefield_literature_summary.md) |
| `configs/` | YAML 配置（MCMP 默认 + 水合物叠加 + SCMP） | [`INDEX.md`](configs/INDEX.md) |
| `scripts/` | 工作流入口（01-06）+ 批量脚本 | [`README.md`](scripts/README.md) |
| `research/` | 科学文档、水合物指南、Phase 6 实施记录 | [`INDEX.md`](research/INDEX.md) |
| `research/implementation/` | 🆕 Phase 6 逐任务实施文档 (01-07) | [`README.md`](research/implementation/README.md) |
| `validation/` | contact_angle / mesh_convergence / surface | — |
| `data/` | 几何 (.plt)、benchmark、design CSV | — |
| `results/` | 模拟输出（gitignored） | — |

---

## 数据流

| 流 | 路径 |
|----|------|
| **SCMP 统一** | `ModelDefinition` → `to_params_dict()` → `params.txt` → `mcmp_huang*` → VTK |
| **MCMP 两相** | `configs/default.yaml` → `load_config()` → `params.txt` → `mcmp_sim` → VTK |
| **MCMP 水合物** | `default.yaml` + `hydrate*.yaml`（叠加）→ `params.txt` → `mcmp_sim_hydrate` → VTK |
| **JAX 验证** | `d2q9_bgk.py` → `collision_mrt(Guo+C)` → `compute_Q_huang` → autograd 梯度 |
| **JAX 多GPU** | `sharded_lbm.py` → shard_map 域分解 (需 GPU 实测) |

---

## 常见陷阱（Top 5）

1. **ε 默认值**: 论文 ε=−2/3 (k₁=1/12) 在此代码中不稳定。必须用经验值 **ε=1.7**。参数覆盖时用 `epsilon_huang` 而非 `k1_huang`（后者不被 params.txt 识别）。
2. **网格改完没重编译**: `NX`/`NY` 在 `LBM.h` 中是 `constexpr`，改数值后必须 `lbm-build` 重新编译。
3. **params.txt key 命名**: 必须精确匹配 C++ `RuntimeParams` 字段名（如 `tau_p_a` 不是 `tauA`）。不匹配的 key 被 `load_params_txt()` 静默忽略，不报错。
4. **水合物 `#ifdef` 隔离**: hydrate 相关代码全部在 `#ifdef HYDRATE_ENABLE` 内。用 `mcmp_sim` 跑 hydrate 参数会静默忽略。
5. **VTK 字节序**: 输出为 big-endian legacy binary。用 `vtk_reader.py` 读取，不要直接 parse。

---

## 相关 AI 工具

- **记忆管理**: 使用 `cyx_memory` skill → 更新 CLAUDE.md + 子索引 + 交叉引用校验
- **代码分析**: 使用 `/understand` → 生成交互式知识图谱分析 CUDA/Python 代码结构
- **论文写作**: 使用 `nature-polishing` / `cyx_*_review` skills → 学术写作润色与审稿
- **版本发布**: 使用 `cyx_git_push` skill → 自动提交 + tag + push
