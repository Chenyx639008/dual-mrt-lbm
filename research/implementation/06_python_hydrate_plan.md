# P3 + P4 + P5 + P6 — 详细实施计划

> **日期**: 2026-07-26
> **状态**: 📋 计划 + 🔴 P3 执行中
> **前提**: P0+P1+P2 全部完成 ✅

---

## 现状分析

### P3: Python 抽象层 — 已完成 vs 待做

| 子任务 | 状态 | 说明 |
|--------|:----:|------|
| MCMP 工厂方法 | ✅ 已存在 | `CollisionParams.mcmp_mrt()`, `ForceParams.shan_chen()`, `WettingParams.material_mapped()` |
| MCMP 模型注册 | ✅ 已存在 | `mcmp_pr_baseline`, `mcmp_pr_wet` 在 `MCMP_MODELS` |
| params.txt 校验 | ❌ 缺失 | `__post_init__` 仅有 EOS 校验，缺少碰撞/力/润湿校验 |

### P4: JAX 多GPU — 现状

| 项目 | 状态 |
|------|:----:|
| JAX shard_map 代码 | ❌ 不存在 |
| 域分解设计 | ❌ 不存在 |
| 多GPU LBM 步进 | ❌ 不存在 |
| 单GPU LBM 步进 | ✅ 已有 (`_step_scmp_huang_core`) |

### P5: CUDA 多GPU — 现状

| 项目 | 状态 |
|------|:----:|
| MPI/NCCL 代码 | ❌ 不存在 |
| CUDA-aware MPI | ❌ 不存在 |
| Halo 交换 | ❌ 不存在 |
| 现有 CUDA 代码 | ✅ 单GPU SCMP/MCMP/Hydrate |

### P6: 水合物纳入 — 现状

| 项目 | 状态 |
|------|:----:|
| 水合物 CUDA 代码 | ✅ 已有 (`hydrate.cu`, `hydrate_vop.cu`) |
| 水合物 YAML 配置 | ✅ 已有 (`hydrate.yaml`, `hydrate_porous.yaml`, `hydrate_sphere_hot.yaml`) |
| 水合物模型注册 | ❌ 未纳入 `ModelDefinition` |
| 统一 CLI 水合物支持 | ❌ 不支持 `uv run lbm run mcmp_hydrate_*` |

---

## 📋 P3: Python 抽象层完善

### Step 1: params.txt 前置校验

**文件**: `lbm_mrt/unified/components.py`

在 `CollisionParams.__post_init__` 中添加：
- τ > 0.5 检查（负粘度防护）
- MCMP 对称性警告（tau_p_a vs tau_p_b 差异过大时）
- kappa 物理范围检查

在 `ForceParams.__post_init__` 中添加：
- GAB == GBA 对称性校验
- G 符号检查（SCMP 必须负值）
- sigmaA 非负检查

在 `WettingParams.__post_init__` 中添加：
- theta 角度范围 [0, 180]
- GAw 参数正数检查

### Step 2: 测试

`jax_lbm/tests/test_p3_validation.py` — 验证校验逻辑正确触发

---

## 📋 P4: JAX shard_map 多GPU 验证轨

> **核心理念**: 先在 JAX 沙盒中验证通信模式正确，再翻译到 CUDA MPI
> **关键约束**: 用户目前只有单GPU，需支持 CPU 模拟多设备

### Step 1: 域分解设计

D2Q9 1D X方向域分解：
- N 个 GPU，每个负责 NX/N × NY 子域
- Halo 深度 = 1（D2Q9 仅最近邻）
- 需交换的方向：右邻居 k=1,5,8（cx>0），左邻居 k=3,6,7（cx<0）

### Step 2: JAX shard_map 实现

**文件**: `jax_lbm/sharded_lbm.py`

```python
def setup_1d_domain_decomposition(n_devices=2):
    """配置 1D X 方向域分解 mesh."""

def step_sharded(f, s_relax, eos_params, k1, ...):
    """一个完整时步，JAX 自动插入 Halo 交换。"""
```

### Step 3: 验证

- 单GPU基准 vs 多GPU分片（CPU模拟）结果一致
- shard_map 不引入 NaN

### Step 4: 测试

`jax_lbm/tests/test_p4_sharded.py`

---

## 📋 P5: CUDA 多GPU — 暂不执行

> **原因**: 无 MPI/NCCL 环境，当前无多GPU硬件。P4 JAX 验证正确后，P5 的 CUDA 翻译是机械性工作。
> **计划**: 待 P4 完成 + 多GPU 硬件就绪后启动。

---

## 📋 P6: MCMP 水合物纳入统一框架

### Step 1: 水合物模型注册

**文件**: `lbm_mrt/unified/models.py`

添加 hydrate 模型到 MCMP_MODELS 或独立 HYDRATE_MODELS：
```python
MCMP_HYDRATE_MODELS = {
    "mcmp_hydrate_sphere": ModelDefinition(
        name="mcmp_hydrate_sphere",
        model_family="mcmp",
        n_components=2,
        eos=EOSParams.peng_robinson(T_reduced=0.80),
        collision=CollisionParams.mcmp_mrt(tau_p_a=0.593, tau_p_b=0.515),
        force=ForceParams.shan_chen(GAB=0.24, GBA=0.24, sigmaA=0.11),
        wetting=WettingParams.material_mapped(...),
        cuda_binary="mcmp_sim_hydrate",
        initial={"hydrate_enable": True, ...},
    ),
    "mcmp_hydrate_porous": ModelDefinition(...),
}
```

### Step 2: hydrate.yaml 参数叠加

确保 `hydrate.yaml` 参数能通过 `ModelDefinition` 的 `**overrides` 机制正确叠加。

### Step 3: CLI 集成

```bash
uv run lbm run mcmp_hydrate_sphere
uv run lbm run mcmp_hydrate_porous
```

### Step 4: 测试

`jax_lbm/tests/test_p6_hydrate.py` — 验证模型注册和参数生成

---

## 执行顺序

```
Day 1 (今天):  P3 params校验 → P6 水合物模型注册
Day 2:         P4 JAX shard_map 域分解设计 + 实现
Day 3:         P4 验证 + 测试
```

P5 (CUDA MPI) 暂缓，等待多GPU硬件。
