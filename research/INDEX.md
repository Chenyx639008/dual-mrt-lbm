# Research 文档索引

> **最后更新**: 2026-07-22
> **文件数**: 14（活跃）+ 3（归档）
> **AI 入口**: 根目录 [CLAUDE.md](../CLAUDE.md)

---

## 📌 快速入门

| 文件 | 内容 | 适合 |
|------|------|------|
| [USAGE_GUIDE.md](USAGE_GUIDE.md) | 端到端使用手册（编译、运行、配置） | 新用户 |
| [FRAMEWORK_GUIDE.md](FRAMEWORK_GUIDE.md) | 🆕 **双轨架构设计·开发·使用手册 v2.0** — 架构、参数、工作流、陷阱 | 所有人 |
| [Research.md](Research.md) | 多物理场 LBM 模型完整文档 | 理解物理模型 |

---

## 🔧 双轨统一框架（当前开发 ⭐）

| 文件 | 内容 | 日期 |
|------|------|------|
| [FRAMEWORK_GUIDE.md](FRAMEWORK_GUIDE.md) | 🆕 **双轨架构手册 v2.0** — CUDA 生产轨 ↔ JAX 镜像轨、Debug/Loop/Adjoint 三大价值 | 2026-07-22 |
| [unified_framework_feasibility.md](unified_framework_feasibility.md) | 🆕 **双轨架构可行性评估 v2.0** — 双轨核心价值评估、多 GPU 可行性、价值实现矩阵 | 2026-07-22 |
| [implementation_summary.md](implementation_summary.md) | **统一框架实施总结** — Phase 1-5 已完成工作、文件清单、CLI 使用指南 | 2026-07-17 |
| [phase2_runtime_grid_summary.md](phase2_runtime_grid_summary.md) | **Phase 2 CUDA 运行时网格** — 35 处内核改造、统一二进制设计、验证清单 | 2026-07-17 |
| [phase5_mcmp_integration_plan.md](phase5_mcmp_integration_plan.md) | **Phase 5 MCMP 纳入** — 2 个 MCMP 模型注册、工厂方法、运行验证 | 2026-07-17 |
| [phase6_dual_track_deepening_plan.md](phase6_dual_track_deepening_plan.md) | 🆕 **Phase 6 双轨深化计划** — P0-P6 任务矩阵、JAX-LaB 借鉴、时间线 | 2026-07-22 |

> 双轨架构的核心思路：CUDA 生产轨极致性能 / JAX 镜像轨可微可调试可 AI 迭代。不是"JAX 替代 CUDA"，而是"JAX 为 CUDA 提供安全网和金标准"。

---

## 💧 水合物 (Hydrate / MCMP)

| 文件 | 内容 |
|------|------|
| [HYDRATE_FULL_DEBUG_VALIDATION_GUIDE.md](HYDRATE_FULL_DEBUG_VALIDATION_GUIDE.md) | 水合物版调试与验证完整指南 |
| [hydrate_cu_scientific_guide.md](hydrate_cu_scientific_guide.md) | `hydrate.cu` 逐函数科学说明 |
| [hydrate_vop_scientific_guide.md](hydrate_vop_scientific_guide.md) | `hydrate_vop.cu` VOP 固相更新指南 |
| [hydrate_sphere_case_design.md](hydrate_sphere_case_design.md) | 300×300 水合物球 benchmark 设计 |
| [hydrate_porous_case_design.md](hydrate_porous_case_design.md) | 多孔介质水合物分解算例设计 |
| [hydrate_validation_design.md](hydrate_validation_design.md) | 参数换算 & YAML 验证链设计 |
| [hydrate_dissociation_space_analysis.md](hydrate_dissociation_space_analysis.md) | 分解后孔隙占据机制分析 |

---

## 🧪 多孔介质达西流

| 文件 | 内容 |
|------|------|
| [POROUS_DARCY_DEVLOG_20260524.md](POROUS_DARCY_DEVLOG_20260524.md) | 多孔介质 SCMP 达西流开发与调试总结 |

---

## 📦 已归档

| 文件 | 说明 |
|------|------|
| `_archive/BUG_FIX_LOG.md` | 2026-05-19 bug 修复记录 |
| `_archive/change_plan.md` | 初始重构计划（已执行完毕） |
| `_archive/validation_plan.md` | 验证方法学设计（已执行完毕） |

---

## 🗑️ 最近清理

> 2026-07-17 移除了 10 个已过时的文档（Huang SCMP 验证文档 7 份 + 旧规划文档 3 份）。
> 这些文档服务于 2026-05 的 Huang & Wu (2016) SCMP 验证阶段，该阶段已完成，统一框架工作已取代其规划功能。
