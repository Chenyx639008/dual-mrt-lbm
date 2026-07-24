# Phase 5: MCMP 模型纳入统一框架 — 实施计划

> **日期**: 2026-07-17
> **状态**: ✅ Phase 5a 完成 — 2 个 MCMP 模型已注册并通过运行验证
> **依赖**: Phase 1 (Python 抽象层) 完成

---

## 目标

将 MCMP (Multi-Component Multi-Phase) 模型纳入 `lbm_mrt/unified/` 框架，与 SCMP Huang 模型共用 `ModelDefinition`、注册表、CLI。

---

## MCMP vs SCMP 对照

| 维度 | SCMP Huang | MCMP |
|------|-----------|------|
| `model_family` | `"scmp"` | `"mcmp"` |
| `n_components` | 1 | 2 |
| `pp_mode` | 1 | 0 |
| EOS | CS-EOS | PR-EOS (A相) + 理想气体 (B相) |
| 碰撞 | `tau_huang` | `tau_p_a`, `tau_p_b` |
| 力 | `epsilon_huang`, `k2_huang` | `GAB`, `GBA`, `sigmaA`, `kappa` |
| 润湿 | `G_ads`, `theta_contact_deg` | `thetaA_quartz_deg`, `thetaA_hydrate_deg`, `GAw_m`, `GAw_c` |
| 初始化 | `huang_init_mode`, `huang_R0` | `Sw`, `init_eq`, 共存密度 (rhoA/B_hi/lo) |
| 几何 | 无 (periodic) | `morph`, `r_obs`, `l_gap`, `geom_file` |
| 二进制 | `mcmp_huang_256` | `mcmp_sim` / `mcmp_sim_hydrate` |
| 内核序列 | `evolution_scmp` (~12) | `evolution_all` (14 + 两组分) |

---

## 实施步骤

### Step 1: `components.py` 扩展

- [x] `CollisionParams.to_params_dict()` — MCMP 已支持 (`tau_p_a`, `tau_p_b`, `kappa`)
- [x] `ForceParams.to_params_dict()` — MCMP 已支持 (`GAB`, `GBA`, `sigmaA`)
- [x] `WettingParams.to_params_dict()` — MCMP 已支持 (`thetaA_quartz_deg`, `thetaA_hydrate_deg`)
- [ ] 新增 `CollisionParams.mcmp_mrt()` 工厂方法
- [ ] 新增 `WettingParams.material_mapped()` 工厂方法

### Step 2: `models.py` 扩展

- [ ] 新增 `MCMP_MODELS` 字典
- [ ] `ModelRegistry._registry` 合并 `MCMP_MODELS`
- [ ] `to_params_dict()` — MCMP 初始条件处理
- [ ] `resolve_binary()` — MCMP 正确选择 `mcmp_sim` / `mcmp_sim_hydrate`

### Step 3: `runner.py` — 无需修改

`run_model()` 通过 `model.resolve_binary()` 自动选择二进制，已兼容 MCMP。

### Step 4: 验证

- [ ] `uv run lbm models` 显示 MCMP 模型
- [ ] `uv run lbm info mcmp_pr_baseline` 显示正确参数
- [ ] `uv run lbm run mcmp_pr_baseline` 可运行

---

## MCMP 模型设计

```python
MCMP_MODELS = {
    # 基线：PR-EOS + MRT + Shan-Chen, Sw=0.3
    "mcmp_pr_baseline": ModelDefinition(
        name="mcmp_pr_baseline",
        model_family="mcmp", n_components=2,
        eos=EOSParams.peng_robinson(T_reduced=0.80),
        collision=CollisionParams.mcmp_mrt(tau_p_a=0.593, tau_p_b=0.515),
        force=ForceParams.shan_chen(GAB=0.24, GBA=0.24, sigmaA=0.11),
        wetting=WettingParams.material_mapped(
            theta_by_material={1: 30.0, 2: 80.0},
        ),
        cuda_binary="mcmp_sim",
        initial={"Sw": 0.3, "init_eq": 1},
    ),
}
```

## 不纳入的部分

以下 MCMP 特性暂不纳入统一框架（保持 YAML + 脚本方式）：

- 水合物扩展 (`HYDRATE_ENABLE`) — 待 Phase 5b
- 复杂几何生成 (`geom_file` + Tecplot) — 脚本路径，不在 params 中
- 稳态检测 (`SteadyMonitor`) — 编译期固定行为
