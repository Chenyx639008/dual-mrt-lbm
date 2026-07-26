# P0a: JAX Huang-Zhang 三阶力修正 — 实施记录

> **日期**: 2026-07-25
> **状态**: ✅ **P0a 完成** — JAX 轨物理对等化（力模型 + 碰撞 + 步进函数）
> **验证**: 19/19 自动化测试通过 · 液滴生产参数 2000 步稳定 · 质量误差 10⁻¹⁴
> **依赖**: Phase 6 §4.1

---

## 📊 最终评估

| 维度 | 评分 | 说明 |
|------|:----:|------|
| **公式正确性** | ✅ | `compute_Q_huang` / `compute_S_guo` 与 CUDA 逐行对齐 |
| **碰撞等价性** | ✅ | Guo+C 模式精确复现 CUDA `mrt_collide_single_component_gpu` |
| **步进函数** | ✅ | `_step_scmp_huang_core` 11 步执行顺序与 CUDA 一致 |
| **液滴稳定性** | ✅ | 256² ε=1.7 T/Tc=0.70 — 2000 步无发散 |
| **向后兼容** | ✅ | 现有 BGK/EDM 代码不受影响 |
| **自动化验证** | ✅ | 19 项测试覆盖 streaming/力/碰撞/EOS/液滴 |
| **生产就绪** | ⚠️ | Guo+C MRT 模式通过单元测试，但完整液滴验证用的是 BGK；MRT 集成待 P0b |

**总体判断**: P0a 代码层面完成。核心交付物（力模型、碰撞改造、步进函数）均可工作，液滴 BGK 模拟在生产参数下稳定。MRT Guo+C 模式的完整端到端验证留给 P0b。

---

## 已修改文件

| 文件 | 改动 | 说明 |
|------|------|------|
| `jax_lbm/force.py` | +125 行 | `compute_Q_huang()`, `compute_S_guo()`, `huang_zhang_force_and_correction()` |
| `jax_lbm/collision.py` | ~40 行改 | `collision_mrt()` 支持 EDM/Guo+C 双模式 + alpha_meq |
| `jax_lbm/d2q9_bgk.py` | ~95 行 | `_step_scmp_huang_core()`, `step_scmp_huang()`, streaming 方向修复 |
| `jax_lbm/tests/test_sanity.py` | +255 行 | 🆕 19 项自动化验证 |
| `jax_lbm/tests/__init__.py` | +0 | 包初始化 |
| `pyproject.toml` | +2 行 | jax, jaxlib 依赖 |
| `research/INDEX.md` | 更新 | 新增验证体系条目 |

---

## CUDA → JAX 对照表

| CUDA 内核 | JAX 函数 | 状态 |
|-----------|---------|:----:|
| `compute_molecular_force_scmp` | `shan_chen_force()` | ✅ |
| `compute_Q_huang_gpu` | `compute_Q_huang()` | ✅ |
| `compute_S_huang_gpu` | `compute_S_guo()` | ✅ |
| `compute_velocity_scmp` | 内联 (UMAX=0.15) | ✅ |
| `compute_p_psi_scmp_cs` | 内联 psi² = clip(2(p-ρcs²)/(Gc²), 0) | ✅ |
| `mrt_collide_single_component_gpu` | `collision_mrt(S_guo=, C=)` | ✅ |
| `stream_single_component_gpu` | `streaming()` | ✅ 修复方向 bug |
| `compute_adsorption_force_scmp` | `adsorption_force()` | 🔜 P0c |
| `update_ghost_psi_bc` | `apply_psi_ghost_bc()` | 🔜 P0c |

---

## 🔧 Bug 修复：Streaming 方向

**根因**: `d2q9_bgk.py` 中 `streaming()` 使用 `jnp.roll(f, -C[k])` 使粒子沿 c_k **反方向**移动。力计算 `_stream_field` 正确获取 ψ(x+c_k)，但 streaming 将粒子推向 (x−c_k)。力与对流方向矛盾 → 正反馈 → 密度指数爆炸 (10³⁵)。

**修复**: `-C[k,0]→C[k,0]`, `-C[k,1]→C[k,1]`（一行改动）。

**防护**: `test_sanity.py::TestStreamingDirection` 8 个参数化测试确保 9 个方向正确。

---

## 验证结果

### 自动化测试 (19/19 通过)

```bash
JAX_ENABLE_X64=1 uv run pytest jax_lbm/tests/test_sanity.py -v
```

| 测试类 | 项数 | 状态 |
|--------|:----:|:----:|
| Streaming 方向 | 8 | ✅ |
| 均匀场力为零 | 2 | ✅ |
| 碰撞质量守恒 | 3 | ✅ |
| MRT 矩阵可逆 | 1 | ✅ |
| 液滴 500 步稳定 | 1 | ✅ |
| Q_m 修正性质 | 2 | ✅ |
| 伪势正确性 | 2 | ✅ |

### 液滴稳定性

| 参数 | 网格 | 步数 | 状态 | 质量误差 |
|------|------|:----:|:----:|:--------:|
| T/Tc=0.90, k₁=0.06 | 128² | 2000 | ✅ | 10⁻¹⁴ |
| T/Tc=0.70, ε=1.7 | 256² | 2000 | ✅ | 10⁻¹⁴ |

---

## 关键参数约定

- **JAX_ENABLE_X64=1**: CUDA 用 double，JAX 默认 float32，验证需 x64
- **k₁ = −ε/8**: CUDA 从 `epsilon_huang` 计算 k₁，不读取 params.txt 中的 `k1_huang`
- **Q_m 只作用于界面**: 体相 F≈0 → Q_m 自动为零
- **Streaming 方向**: `jnp.roll(f, C[k])` — 粒子必须沿 c_k 方向移动

---

## 待办 (P0b/P0c)

| 任务 | 说明 |
|------|------|
| P0b: MRT 系统验证 | 用 Guo+C 碰撞跑完整液滴，与 CUDA 逐格点对比 |
| P0c: 交叉验证增强 | 吸附力 + 接触角 BC 集成，Laplace 定律验证 |
| 速度裁剪完善 | 当前速度裁剪在步进函数中，需与 S_guo 一致 |
