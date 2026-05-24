# Research 文档索引

> 最后更新: 2026-05-24
> 文件数: 19

---

## 📌 新用户看这里

1. **[USAGE_GUIDE.md](USAGE_GUIDE.md)** — 端到端使用手册（编译、运行、配置）
2. **[Research.md](Research.md)** — 多物理场 LBM 模型完整文档
3. **[DEBUG_SUMMARY_20260524.md](DEBUG_SUMMARY_20260524.md)** — 最新验证 Bug 修复总结

---

## 🔬 黄 SCMP 验证 (Huang & Wu 2016)

| 文件 | 内容 | 状态 |
|------|------|------|
| [DEBUG_SUMMARY_20260524.md](DEBUG_SUMMARY_20260524.md) | **最新** 全部验证 debug 总结 (2026-05-21~24) | ⭐ 当前 |
| [HUANG_2016_VALIDATION_STRATEGY.md](HUANG_2016_VALIDATION_STRATEGY.md) | ε/k₁ 参数策略与论文 §6.3 验证方案 | 参考 |
| [HUANG_PRESSURE_TENSOR_GUIDE.md](HUANG_PRESSURE_TENSOR_GUIDE.md) | 压力张量推导与 Eq.(55-56) 解读 | 参考 |
| [CODE_PAPER_AUDIT_20260515.md](CODE_PAPER_AUDIT_20260515.md) | 论文公式到代码的逐行审计 | 参考 |
| [VALIDATION_CODE_GUIDE.md](VALIDATION_CODE_GUIDE.md) | 验证代码结构与工作流 | 参考 |
| [SCMP_CONTACT_ANGLE_PSI_BC.md](SCMP_CONTACT_ANGLE_PSI_BC.md) | 接触角控制方案探索（结论：G_ads 不可行） | 参考 |
| [HUANG_MRT_REFACTOR.md](HUANG_MRT_REFACTOR.md) | 编译变体对照 + 方程-代码映射 | 速查 |

<details>
<summary>📦 已归档 (4 个)</summary>

| 文件 | 说明 |
|------|------|
| `_archive/validation_summary_20260514.md` | 早期 256² 验证 |
| `_archive/change_plan.md` | 初始重构计划 (已执行完毕) |
| `_archive/validation_plan.md` | 验证方法学设计 (已执行完毕) |
| `_archive/BUG_FIX_LOG.md` | 2026-05-19 bug 修复记录 |

</details>

---

## 💧 水合物 (Hydrate)

| 文件 | 内容 |
|------|------|
| [hydrate_cu_scientific_guide.md](hydrate_cu_scientific_guide.md) | `hydrate.cu` 逐函数科学指南 |
| [hydrate_vop_scientific_guide.md](hydrate_vop_scientific_guide.md) | `hydrate_vop.cu` VOP 固相更新指南 |
| [HYDRATE_FULL_DEBUG_VALIDATION_GUIDE.md](HYDRATE_FULL_DEBUG_VALIDATION_GUIDE.md) | 水合物全面调试与验证指南 |
| [hydrate_sphere_case_design.md](hydrate_sphere_case_design.md) | 300×300 水合物球 benchmark 设计 |
| [hydrate_porous_case_design.md](hydrate_porous_case_design.md) | 多孔介质水合物分解算例设计 |
| [hydrate_validation_design.md](hydrate_validation_design.md) | 参数换算 & YAML 验证链 |
| [hydrate_dissociation_space_analysis.md](hydrate_dissociation_space_analysis.md) | 分解后孔隙占据机制分析 |

---

## 🔮 规划与设计

| 文件 | 内容 |
|------|------|
| [next_paper_plan.md](next_paper_plan.md) | 第二篇论文计划（甲烷水合物/CO₂ 置换） |
| [single_phase_water_design.md](single_phase_water_design.md) | 单相水批量流设计（待实施） |

---

## 🗑️ 文件状态

| 状态 | 数量 | 说明 |
|------|------|------|
| ⭐ 活跃 | 15 | 当前使用中 |
| 📦 已归档 | 4 | 移至 `_archive/` |
| 📚 通用 | 2 | USAGE_GUIDE + Research |
| 🔮 规划 | 2 | 待实施 |

> 无重复文件。所有文件服务于不同目的（验证 vs 水合物 vs 规划）。
>
> 其他归档位置: `logs/_archive/` (5个旧日志), `data/_archive/` (19个过时CSV)
