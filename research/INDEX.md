# Research 文档索引

> **更新**: 2026-07-26 | **AI 入口**: [CLAUDE.md](../CLAUDE.md)

## 核心文档

| 文件 | 说明 |
|------|------|
| [FRAMEWORK_GUIDE.md](FRAMEWORK_GUIDE.md) | 双轨架构设计·开发·使用手册 |
| [USAGE_GUIDE.md](USAGE_GUIDE.md) | 端到端使用手册 |
| [Research.md](Research.md) | 多物理场 LBM 模型文档 |
| [phase6_dual_track_deepening_plan.md](phase6_dual_track_deepening_plan.md) | Phase 6 双轨深化总计划 |

## 架构文档

| 文件 | 说明 |
|------|------|
| [unified_framework_feasibility.md](unified_framework_feasibility.md) | 双轨可行性评估 |
| [implementation_summary.md](implementation_summary.md) | Phase 1-5 实施总结 |
| [phase2_runtime_grid_summary.md](phase2_runtime_grid_summary.md) | CUDA 运行时网格 |
| [phase5_mcmp_integration_plan.md](phase5_mcmp_integration_plan.md) | MCMP 纳入计划 |

## Phase 6 实施记录 → [implementation/](implementation/)

| # | 文件 | 内容 |
|:--:|------|------|
| 01 | `01_huang_zhang_force.md` | P0a: JAX Huang-Zhang 力 + streaming 修复 |
| 02 | `02_mrt_verification.md` | P0b: MRT 碰撞验证 |
| 03 | `03_cross_validation.md` | P0c: 吸附力 + 接触角 + Laplace |
| 04 | `04_ai_sandbox_plan.md` | P1-P2: vmap + 回归测试 + 伴随 计划 |
| 05 | `05_ai_sandbox_done.md` | P1-P2: vmap + 回归测试 + 伴随 总结 |
| 06 | `06_python_hydrate_plan.md` | P3-P6: params + 多GPU + 水合物 计划 |
| 07 | `07_python_hydrate_done.md` | P3-P6: params + 多GPU + 水合物 总结 |

## 水合物

| 文件 | 说明 |
|------|------|
| [HYDRATE_FULL_DEBUG_VALIDATION_GUIDE.md](HYDRATE_FULL_DEBUG_VALIDATION_GUIDE.md) | 水合物调试与验证 |
| [hydrate_cu_scientific_guide.md](hydrate_cu_scientific_guide.md) | hydrate.cu 科学说明 |
| [hydrate_vop_scientific_guide.md](hydrate_vop_scientific_guide.md) | VOP 固相更新 |
| [hydrate_sphere_case_design.md](hydrate_sphere_case_design.md) | 球形案例设计 |
| [hydrate_porous_case_design.md](hydrate_porous_case_design.md) | 多孔介质案例设计 |
| [hydrate_validation_design.md](hydrate_validation_design.md) | 验证链设计 |
| [hydrate_dissociation_space_analysis.md](hydrate_dissociation_space_analysis.md) | 分解机制分析 |

## 论文规划

| 文件 | 说明 |
|------|------|
| [next_paper_plan.md](next_paper_plan.md) | 下一篇论文计划 |
| [single_phase_water_design.md](single_phase_water_design.md) | 单相水模拟设计 |

## 历史参考

Huang & Wu (2016) SCMP 验证阶段（已完成）、多孔介质达西流开发日志等。详见各文件。

## 已归档 → `_archive/`

旧版 bug 修复记录、重构计划、验证方法学设计。
