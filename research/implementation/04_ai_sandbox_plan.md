# P1a + P1b + P2a — 详细实施计划

> **日期**: 2026-07-26
> **状态**: 📋 计划阶段
> **前提**: P0 (P0a+P0b+P0c) 全部完成 ✅
> **总工作量**: 4-5 天

---

## 总体依赖关系

```
P0 完成 ✅
  ├── P1a (vmap) — 无依赖，可立即启动
  ├── P1b (回归测试) — 依赖 P0 ✅，可立即启动
  └── P2a (伴随Demo) — 依赖 P0 ✅，可立即启动
```

三个任务互不依赖，可以并行推进。

---

## 📋 P1a: JAX vmap 批量参数扫描

> **目标**: 一键并发扫描 LBM 参数空间，利用 JAX 自动向量化
> **状态**: 全新实现
> **文件**: `jax_lbm/parameter_sweep.py`
> **测试**: `jax_lbm/tests/test_p1a_vmap.py`

### Step 1: 实现 vmap 扫描核心

**文件**: `jax_lbm/parameter_sweep.py`

核心函数签名：
```python
@partial(jax.jit, static_argnames=('nx', 'ny', 'n_steps'))
def batch_sweep_scmp(params_batch, nx=128, ny=128, n_steps=500):
    """vmap 批量 SCMP LBM 运行，返回每个参数组的质量/密度统计量。"""

def sweep_surface_tension(T_range, eps_range, nx=128, ny=128):
    """扫描 (T, ε) 参数空间，返回表面张力 σ 矩阵。"""

def sweep_contact_angle(theta_range, G_ads=0.1, nx=256, ny=128):
    """扫描接触角参数空间，返回 θ 矩阵。"""
```

### Step 2: 验证

| 验证项 | 方法 |
|--------|------|
| vmap 结果与串行一致 | 同一参数组对比 |
| mass 守恒 | 所有参数组 mass_err < 1e-12 |
| 无 NaN | 全参数空间检查 |
| 性能提升 | vmap 耗时 vs for 循环 |

### Step 3: 示例用法

```python
from jax_lbm.parameter_sweep import sweep_surface_tension

# 扫描 10×10=100 组 (T, ε)，一键完成
sigma_matrix = sweep_surface_tension(
    T_range=(0.55, 0.90, 10),
    eps_range=(1.0, 2.5, 10),
    nx=128, ny=128,
)
```

---

## 📋 P1b: JAX 自动化回归测试套件

> **目标**: AI 改完代码后一键 `pytest` 验证所有物理正确性
> **状态**: 部分已有（test_sanity.py 22 项, test_p0c 9 项），需拆分+补全
> **文件**: `jax_lbm/tests/` 目录重组
> **总测试数目标**: ~50 项

### 现有测试 → 拆分映射

| 现有文件 | 拆分为 | 项数 |
|----------|--------|:----:|
| `test_sanity.py` — Streaming | `test_streaming.py` | 8 |
| `test_sanity.py` — Collision | `test_collision.py` | 3+3 |
| `test_sanity.py` — Force/Q_m | `test_force.py` | 4 |
| `test_sanity.py` — EOS/ψ | `test_eos.py` | 2+1 |
| `test_sanity.py` — Droplet | `test_droplet.py` | 1+1 |
| `test_p0c_cross_validation.py` — Adsorption | `test_force.py` | 3 |
| `test_p0c_cross_validation.py` — Wetting | `test_wetting.py` | 3 |
| `test_p0c_cross_validation.py` — Laplace | `test_physics.py` | 1 |
| `test_p0c_cross_validation.py` — Full step | `test_integration.py` | 2 |

### 新增测试

| 文件 | 测试内容 | 项数 |
|------|---------|:----:|
| `test_boundary.py` | 5 种 BC（周期/bounce-back/Zou-He/equilibrium/ghost） | 5 |
| `test_eos.py` | Maxwell 共存密度构造 vs 参考值 | 2 |
| `test_conservation.py` | 10,000 步质量/动量守恒 | 2 |
| `test_regression.py` | 与 CUDA 基准快照对比 | 3 |
| `conftest.py` | 共享 fixtures (网格/参数/EOS) | — |

### 目标目录结构

```
jax_lbm/tests/
├── __init__.py
├── conftest.py              # 共享 fixtures
├── test_streaming.py        # 8 项 — 迁移方向
├── test_collision.py        # 6 项 — BGK/MRT 守恒+一致性
├── test_force.py            # 7 项 — SC力/Q_m/吸附力
├── test_eos.py              # 5 项 — 5 种 EOS + Maxwell
├── test_wetting.py          # 3 项 — 接触角 BC
├── test_boundary.py         # 5 项 — 5 种 BC
├── test_droplet.py          # 2 项 — 液滴稳定性
├── test_physics.py          # 1 项 — Laplace 定律
├── test_conservation.py     # 2 项 — 长期守恒
├── test_integration.py      # 2 项 — 完整步进函数
└── test_regression.py       # 3 项 — CUDA 基准对比
```

### Step 1: 拆分现有测试到独立文件

将 `test_sanity.py` 和 `test_p0c_cross_validation.py` 中的测试按功能拆分。

### Step 2: 创建 `conftest.py`

```python
# 共享 fixtures
@pytest.fixture
def scmp_production_params():
    return dict(rho_l=0.305202, rho_g=0.009170, T=0.70, tau=1.5)

@pytest.fixture
def small_grid():
    return 64, 64

@pytest.fixture
def mrt_relaxation():
    tau = 1.5; s_p = 1.0/tau
    Lambda = 1.0/12.0; s_q = 1.0/(0.5+Lambda/(tau-0.5))
    return jnp.array([1.0, s_p, s_p, 1.0, s_q, 1.0, s_q, s_p, s_p])
```

### Step 3: 新增边界/守恒/回归测试

### Step 4: 运行全量验证

```bash
JAX_ENABLE_X64=1 uv run pytest jax_lbm/tests/ -v  # ~50 tests
```

---

## 📋 P2a: 伴随金标准验证管线

> **目标**: 用 `jax.grad` 计算 LBM 参数的精确梯度，建立可复现的伴随验证
> **状态**: 全新实现
> **文件**: `jax_lbm/adjoint.py` + `jax_lbm/tests/test_adjoint.py`
> **文档**: `research/P2a_adjoint_gold_standard.md`

### Step 1: 实现可微 LBM 模拟器

**文件**: `jax_lbm/adjoint.py`

```python
@partial(jax.jit, static_argnames=('nx', 'ny', 'n_steps'))
def differentiable_droplet(epsilon, T_reduced, nx=64, ny=64, n_steps=200):
    """端到端可微: (ε, T) → LBM 模拟 → 质量损失率。

    全程纯 JAX 操作，jax.grad 可穿透 200 步 lax.scan。
    """
```

### Step 2: 计算敏感度

| 敏感度 | 物理意义 |
|--------|---------|
| ∂(mass_loss)/∂ε | 表面张力对质量保持的影响 |
| ∂(mass_loss)/∂T | 温度对质量保持的影响 |
| ∂σ/∂ε | 表面张力对 ε 的敏感度 |
| ∂σ/∂T | 表面张力对温度的敏感度 |

### Step 3: 伴随金标准验证流程

```
Step 1: JAX 自动微分 → 理论梯度 g*（机器精度 ~1e-8）
Step 2: 中心有限差分 → 近似梯度 g_FD (h=1e-5)
Step 3: 对比误差 ‖g* − g_FD‖ / ‖g*‖
Step 4: 验证伴随梯度优于有限差分（论文亮点）
Step 5: 输出 research/P2a_adjoint_gold_standard.md
```

### Step 4: 验证

| 验证项 | 方法 |
|--------|------|
| 梯度非零 | ∂loss/∂ε ≠ 0, ∂loss/∂T ≠ 0 |
| 梯度方向合理 | ε↑ → 表面张力↑ → mass_loss↓ |
| 有限差分一致性 | ‖g* − g_FD‖ < 1e-4 |
| 多步稳定性 | 200 步 lax.scan 可微 |

---

## 执行优先级排序

| 优先级 | 任务 | 理由 |
|:------:|------|------|
| 🔴 1 | **P1b 测试拆分** | P0 刚完成，现有 31 项测试急需按功能组织；拆分后更易维护 |
| 🔴 2 | **P2a 伴随 Demo** | 论文方法论亮点，尽早验证 `jax.grad` 穿透性 |
| 🟡 3 | **P1a vmap 扫描** | 独立模块，可最后做；需要 P1b 验证其正确性 |

### 推荐执行顺序

```
Day 1: P1b Step 1-2 (拆分现有测试 + conftest)
Day 2: P1b Step 3 (新增测试) + P2a Step 1 (可微模拟器)
Day 3: P2a Step 2-3 (梯度计算 + 有限差分对比)
Day 4: P1a Step 1-2 (vmap 实现 + 验证)
Day 5: P2a Step 4-5 (全量测试 + 文档输出)
```

---

## 完成标准

| 任务 | 完成标准 |
|------|---------|
| P1a | `parameter_sweep.py` 可运行, vmap 结果与串行一致, 测试通过 |
| P1b | `jax_lbm/tests/` 目录重组, ~50 项测试全通过, `make test` 一键运行 |
| P2a | `adjoint.py` 可计算梯度, 与有限差分误差 < 1e-4, 文档输出 |
