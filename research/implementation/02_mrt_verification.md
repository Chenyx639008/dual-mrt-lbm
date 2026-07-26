# P0b: JAX MRT 系统验证 — 实施记录

> **日期**: 2026-07-25
> **状态**: ✅ **P0b 完成** — MRT 碰撞公式逐项验证 + Guo+C 液滴稳定
> **验证**: 22/22 自动化测试通过 · MRT Guo+C 液滴 200 步无发散
> **依赖**: P0a 完成

---

## 验证结果

### CUDA → JAX MRT 逐项对照

| 组件 | CUDA | JAX | 状态 |
|------|------|-----|:----:|
| M 矩阵 | `M[9][9]` in LBM.h | `MRT_M` in collision.py | ✅ 逐元素一致 |
| M⁻¹ 矩阵 | `Minv[9][9]` in LBM.h | `MRT_MINV` in collision.py | ✅ 逐元素一致 |
| meq 公式 | `meq_gpu(k, ρ, u)` | `meq_mrt(ρ, u)` | ✅ 逐槽位验证通过 |
| α-meq 修正 | `meq[2] = (α−3u²)ρ` | `m_eq[...,2] = (α−3u²)ρ` | ✅ |
| 碰撞公式 | `m − s(m−meq) + (1−0.5s)S_guo + C` | 同左 (Δt=1) | ✅ |
| 松弛参数 | s=[1,1/τ,1/τ,1,s_q,1,s_q,1/τ,1/τ] | 同左 | ✅ |
| s_q 公式 | 1/(0.5+Λ/(τ−0.5)) | 同左 | ✅ |

### 稳定性验证

| 模式 | 条件 | 网格 | 步数 | 状态 |
|------|------|:----:|:----:|:----:|
| MRT + EDM | ε=1.7, T/Tc=0.70 | 64² | 300 | ✅ |
| MRT + Guo | ε=1.7, T/Tc=0.70 | 64² | 200 | ✅ |
| MRT + C (Q_m) | ε=1.7, T/Tc=0.70 | 64² | 200 | ✅ |
| **MRT + Guo+C** | **ε=1.7, T/Tc=0.70** | **64²** | **200** | ✅ |

### 自动化测试 (22/22)

```
TestStreamingDirection (8)  ✅
TestForceUniformField (2)   ✅
TestCollisionConservation (3) ✅
TestMRTMatrix (1)           ✅
TestDropletStability (1)    ✅
TestQmCorrection (2)        ✅
TestPseudopotential (2)     ✅
TestMRTEquilibrium (2)  🆕  ✅
TestMRTGuoCDroplet (1)  🆕  ✅
```

---

## 关键发现

1. **MRT Guo+C 模式稳定**: Guo 源项和 Q_m 修正均可安全应用于 MRT 碰撞，200 步无发散
2. **公式完全等价**: JAX collision_mrt 与 CUDA mrt_collide_single_component_gpu 在数学上等价
3. **EDM 和 Guo 可互换**: 对于 BGK 使用 EDM、对于 MRT 使用 Guo+C 均可正常工作
4. **速度裁剪不是必需的**: 移除速度裁剪后 MRT Guo+C 仍然稳定（之前的 NaN 是测试代码 bug）

---

## P0 闭环状态

| P0 子任务 | 状态 |
|-----------|:----:|
| P0a: JAX Huang-Zhang 力实现 | ✅ |
| **P0b: JAX MRT 系统验证** | **✅** |
| P0c: 双轨交叉验证增强 | 🔜 |

> **P0 核心目标已达成**: JAX 轨现在拥有与 CUDA 物理等价的力模型 (`compute_Q_huang` + `compute_S_guo`) 和碰撞算子 (`collision_mrt` Guo+C 模式)。JAX 能完整复现 CUDA 的 SCMP 行为。
