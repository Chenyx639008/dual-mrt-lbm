# P0c: 双轨交叉验证增强 — 实施记录

> **日期**: 2026-07-25
> **状态**: ✅ **P0c 完成** — 吸附力 + 接触角 BC 集成 + Laplace 定律验证
> **验证**: 31/31 自动化测试通过（含 9 项 P0c 新增）
> **依赖**: P0a + P0b 完成

---

## 完成内容

### 吸附力集成

| 项目 | 说明 |
|------|------|
| 公式 | F_ads = -G_ads·ψ·Σ w_F_k·s(x+c_k)·c_k |
| CUDA 对照 | `compute_adsorption_force_scmp` → JAX `adsorption_force()` |
| 集成位置 | `_step_scmp_huang_core` Step 6（Q_m 之后，确保 F_mol 不受污染） |
| JIT 兼容 | 移除 `if G_ads==0` 早期返回，改为始终计算（G_ads=0 时自然为零） |

### 接触角 BC 集成

| 项目 | 说明 |
|------|------|
| 方法 | Scheme IV ψ-based ghost BC |
| 公式 | ψ_ghost = ψ_ref + |∇ψ|·cot(θ) |
| CUDA 对照 | `update_ghost_psi_bc` → JAX `apply_psi_ghost_bc()` |
| 集成位置 | `_step_scmp_huang_core` Step 3（ψ 计算后、分子力计算前） |
| 支持角度 | θ=0°~180°（已验证 60°/90°/120°） |

### Laplace 定律验证

| 项目 | 结果 |
|------|------|
| 物理 | ΔP = σ/R，液滴内外压差与曲率半径成反比 |
| 验证 | R=30 vs R=40 → ΔP(30) > ΔP(40) ✅ |
| 条件 | T/Tc=0.90, ε=-0.48, MRT Guo+C, 128² |

---

## CUDA → JAX 对照（最终）

| CUDA 内核 | JAX 函数 | 状态 |
|-----------|---------|:----:|
| `compute_molecular_force_scmp` | `shan_chen_force()` | ✅ P0a |
| `compute_Q_huang_gpu` | `compute_Q_huang()` | ✅ P0a |
| `compute_S_huang_gpu` | `compute_S_guo()` | ✅ P0a |
| `compute_velocity_scmp` | 内联 (UMAX=0.15) | ✅ P0a |
| `compute_p_psi_scmp_cs` | 内联 ψ² = clip(...) | ✅ P0a |
| `mrt_collide_single_component_gpu` | `collision_mrt(S_guo=, C=)` | ✅ P0b |
| `stream_single_component_gpu` | `streaming()` | ✅ P0a 修复 |
| `compute_adsorption_force_scmp` | `adsorption_force()` | ✅ P0c |
| `update_ghost_psi_bc` | `apply_psi_ghost_bc()` | ✅ P0c |

> 🎯 **9/9 CUDA 内核已全部完成 JAX 等价实现**

---

## 自动化测试 (31/31)

```
P0a sanity (22 项):
  Streaming方向(8) 力均匀性(2) 碰撞守恒(3) MRT矩阵(1)
  液滴稳定性(1) Q_m修正(2) 伪势(2) MRT平衡态(2) MRT液滴(1)

P0c cross-validation (9 项):
  吸附力(3) 接触角BC(3) Laplace定律(1) 完整步进(2)
```

---

## 🔴 P0 全面闭环

| P0 子任务 | 状态 | 关键交付 |
|-----------|:----:|------|
| P0a: Huang-Zhang 力实现 | ✅ | 力模型 + 碰撞改造 + 步进函数 |
| P0b: MRT 系统验证 | ✅ | MRT 逐项验证 + Guo+C 液滴 |
| **P0c: 交叉验证增强** | **✅** | **吸附力 + 接触角 + Laplace 定律** |

> **JAX 轨现已完整复现 CUDA SCMP 的全部 9 个内核。**

## 下一步

Phase 6 剩余任务：
- P1a: vmap 批量参数扫描（无依赖，可随时启动）
- P1b: 自动化回归测试套件（依赖 P0 完成 ✅）
- P2a: 伴随金标准 Demo（依赖 P0 完成 ✅）
- P3a+P3b: MCMP 工厂方法 + params 校验
