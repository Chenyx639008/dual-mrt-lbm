# Research 文档索引

> **更新**: 2026-08-08 | **AI 入口**: [CLAUDE.md](../CLAUDE.md)

## 相场开发（最新）

| 文件 | 说明 |
|------|------|
| [phasefield_development_plan.md](phasefield_development_plan.md) | **相场 LBM 开发计划（v4）**：单相 NS → 保守 AC → 两相耦合 → 润湿 → 两相流动 → hydrate 集成。阶段 0/1/2 ✅ + CUDA pf_ns_2d ✅；**阶段 3 定量湿壁阻塞于 MRT-LB AC** |
| [phasefield_mrt_ac_implementation_plan.md](phasefield_mrt_ac_implementation_plan.md) | 🔴 **MRT-LB 保守 AC 实施计划（2026-08-08）**：液滴溶解根因=平衡矩矩序错乱（6 矩错误）；BGK 立真值→MRT 对标→耦合→湿壁→CUDA 五步；文献精确公式已数值验证 |
| [phasefield_stage2_summary.md](phasefield_stage2_summary.md) | 阶段 2 总结：AC+NS 耦合、Laplace R²=0.9996、共存、伪速度 1.1e-4 |
| [phasefield_stage3_summary.md](phasefield_stage3_summary.md) | 阶段 3 总结：NS 无滑移壁 ✅ + S17 公式 ✅ + 直接 FD AC 湿壁阻塞确认（MRT-LB AC 硬前置） |
| [phasefield_execution_summary.md](phasefield_execution_summary.md) | 3 天自主执行总结（阶段 0-3 + CUDA 交叉验证） |
| [literature/phase_field_2d_implementation_plan.md](literature/phase_field_2d_implementation_plan.md) | 相场 LB + 无量纲框架 2D 实现（Yang SI 逐字公式 S10-S41，含 MRT 矩定义） |
| [literature/SC_vs_phasefield_and_3D_migration_guide.md](literature/SC_vs_phasefield_and_3D_migration_guide.md) | Yang SC→相场转变 + 2D→3D 迁移 + 自研路线（复制自 hydrate 仓库） |
| [literature/yang_papers_pdf_deep_analysis.md](literature/yang_papers_pdf_deep_analysis.md) | Yang 五篇 PDF 直读权威分析（复制自 hydrate 仓库） |
| [literature/hydrate_formation_thermodynamics.md](literature/hydrate_formation_thermodynamics.md) | 水合物形成热力学总纲（双场表示/软钉扎/自限终态）（复制自 hydrate 仓库，集成用） |
| [HUANG_SCMP_VALIDATION_GUIDE.md](HUANG_SCMP_VALIDATION_GUIDE.md) | Huang SCMP 验证指南（相场对齐 SC 的验证任务清单） |

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
