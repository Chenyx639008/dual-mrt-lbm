# Configs 配置文件索引

> **最后更新**: 2026-07-17
> **文件数**: 5
> **AI 入口**: 根目录 [CLAUDE.md](../CLAUDE.md)

---

## 配置文件清单

### MCMP 两相流（主配置）

| 文件 | 用途 | 加载方式 |
|------|------|---------|
| [default.yaml](default.yaml) | **主控配置**：MCMP 两相流所有参数基线 | `load_config()` 自动加载，所有运行的基础 |
| [hydrate.yaml](hydrate.yaml) | 通用水合物参数基线（保守值） | `load_config(hydrate.yaml)` 叠加到 default 上 |

### 水合物专用案例

| 文件 | 用途 | 运行入口 |
|------|------|---------|
| [hydrate_sphere_hot.yaml](hydrate_sphere_hot.yaml) | 300×300 球形水合物，底边强升温（T_inlet=323.15 K） | `scripts/04_run_hydrate_sphere_case.py` |
| [hydrate_porous.yaml](hydrate_porous.yaml) | 339×212 多孔介质，右边界升温（T_inlet=298.15 K） | `scripts/05_run_porous_hydrate_case.py` |

### SCMP Huang 单组分

| 文件 | 用途 | 加载方式 |
|------|------|---------|
| [huang_scmp.yaml](huang_scmp.yaml) | Huang & Wu (2016) SCMP 配置（CS-EOS, k₁=1/12, T/Tc=0.70） | Legacy `lbm-run` 或统一框架 `ModelDefinition` 自动生成 |

---

## 配置加载链

```
configs/default.yaml  ─────────────→ load_config() → flat dict
configs/hydrate.yaml  ── (叠加) ──→                  → params.txt → CUDA RuntimeParams
configs/huang_scmp.yaml ─ (Legacy) ─→ lbm-run --config
                                            或
unified/models.py ────→ ModelDefinition.to_params_dict() → params.txt (SCMP 统一路径)
```

## 参数分类速查

| 类别 | 关键参数 | 所在文件 |
|------|---------|---------|
| 网格/几何 | `NX`, `NY`, `geomPath` | `default.yaml` |
| EOS | `omega`, `a_P`, `b_P`, `Tc`, `pc` | `default.yaml` |
| 碰撞 | `tau_p_a`, `tau_p_b`, `alpha_p_a` | `default.yaml` |
| 分子力 | `GAB`, `GBA`, `sigmaA`, `kappa` | `default.yaml` |
| 热/浓度 | `alpha_T_a`, `DTB`, `Sh_number`, `Sc350` | `hydrate.yaml` |
| 达西 | `K0`, `sigma_porous`, `G_darcy` | `hydrate_porous.yaml` |
| SCMP | `k1_huang`, `T_reduced`, `tau_mrt` | `huang_scmp.yaml` |
| 润湿性 | `thetaA_quartz_deg`, `thetaA_hydrate_deg`, `GAw_m`, `GAw_c` | `default.yaml` |

---

## 注意事项

- **叠加逻辑**：`hydrate*.yaml` 只包含与 `default.yaml` 不同的参数，其余继承 default
- **SCMP vs MCMP**：`huang_scmp.yaml` 仅用于 Huang SCMP 二进制（`mcmp_huang*`），不可与 MCMP 混用
- **参数命名**：所有 key 必须匹配 C++ `RuntimeParams` 字段名，否则被 `load_params_txt()` 忽略
