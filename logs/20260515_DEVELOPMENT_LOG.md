# Huang SCMP 验证套件开发日志

> 日期：2026-05-15 ~ 2026-05-16
> 工作树：huang_mrt_2d (git worktree)
> 求解器：mcmp_huang_256 (Huang & Wu 2016 SCMP, CS-EOS, D2Q9 MRT)

---

## 一、本次开发目标

对 Huang & Wu (2016) 单组分双相 SCMP 求解器进行完整验证套件实现，按照 `validation_plan.md` 计划完成以下测试：

- P0：Laplace 定律、σ-decoupling、共存曲线、Spurious currents
- P1：Poiseuille 流、网格收敛、接触角

---

## 二、关键 Bug 修复

### 2.1 cs_T 语义错误（致命）

- **问题**：CUDA 代码将 `cs_T` 当作约化温度 Tr = T/Tc，但 Python 设计生成器和 YAML 配置将 `cs_T` 当作绝对温度写入（Tr×Tc）
- **后果**：预期运行 Tr=0.7，实际运行 Tr≈0.066，所有验证结果错误
- **修复**：
  - `configs/huang_scmp.yaml`: cs_T 直接填 Tr 值（0.70 而非 0.066）
  - `scripts/06_run_huang_validation_suite.py`: 5 个设计生成器中 `cs_T = Tr`（非 T_abs）
  - `validation_plan.md §2.1`: 更新语义说明
  - `data/design_scmp_*.csv`: 全部重新生成

### 2.2 Q_m 碰撞修正使用总力（论文不符）

- **问题**：`compute_Q_huang_gpu` 使用总力 F（含吸附力 + 体力），论文 Eq. 57-59 要求只用分子间力
- **影响**：Q_m 错误地将吸附力计入 |F|² 项，导致外力被碰撞修正抵消
- **修复**：将 `compute_Q_huang_gpu` 调用移到吸附力加入**之前**，S 项仍用总力

### 2.3 Mode 4 初始化 bug

- **问题**：`init_all_scmp_gpu` 中 mode 4 走平界面初始化（r=y），而非径向液滴初始化
- **修复**：添加 `mode == 1 || mode == 4` 判断

### 2.4 Ghost 层被 boundary 隔断

- **问题**：Mode 4 的 ghost (y=0, flag=-1) 与 fluid (y=2, flag=1) 之间有 boundary (y=1, flag=0)，分子力 stencil 无法触及 ghost ψ
- **修复**：Mode 4 改为 ghost-only（y=0 ghost, y=1+ fluid），`boundary_scmp_gpu` 扩展为处理 fluid-adjacent-to-ghost 的 bounce-back

### 2.5 吸附力内核 1D 索引 bug

- **问题**：`add_adsorption_to_force` 使用 1D 索引 `blockIdx.x`，但 grid 是 2D，只处理第一行
- **修复**：改为 2D 索引 `findindex_scalar_gpu(x, y)`

### 2.6 ψ-BC 激活条件 bug

- **问题**：`theta_contact_deg > 0.0` 阻止了负值（直接 ψ 模式）的 BC 激活
- **修复**：改为 `fabs(theta_contact_deg) > 1e-12`

### 2.7 负值 θ 被弧度转换压缩

- **问题**：负值 θ 传入 kernel 前被乘以 π/180，-0.01 → -0.000175 rad → kernel 解读为 ψ=0.000175 而非 0.01
- **修复**：负值时跳过弧度转换，直接传递原始值

### 2.8 分子力墙壁感知

- **问题**：周期包裹导致墙壁处分子力包含跨域邻居的虚假 ψ 值
- **修复**：ghost 邻居用自 ψ 代替（后回退，因 ψ-BC 需要真实 ghost ψ 参与计算）

---

## 三、新增求解器功能

### 3.1 压力张量计算

- **新增 kernel**：`compute_pressure_tensor_scmp`
- **公式**：paper Eq. (34) 离散压力张量 + Eq. (55) Q_m 修正
  $$\mathbf{P} = \rho c_s^2 \mathbf{I} + \frac{G}{2}\psi\sum_i w_i \psi(\mathbf{x}+\mathbf{e}_i)\mathbf{e}_i\mathbf{e}_i + k_1 G \nabla\psi\nabla\psi + k_2 G |\nabla\psi|^2 \mathbf{I}$$
- **VTK 输出**：p_xx, p_yy, p_xy

### 3.2 力场可视化

- **VTK 新增字段**：Fx, Fy（分子间力）, psi（伪势）
- 用于调试接触角实现中力的传播

### 3.3 吸附力内核

- **新增 kernel**：`compute_adsorption_force_scmp`（Yang/Li 风格 G_ads·ψ）
- **新增 kernel**：`add_adsorption_to_force`（2D 索引）
- **新增参数**：`G_ads_scmp`（设备常量 d_G_ads_scmp）

### 3.4 ψ 边界条件内核

- **新增 kernel**：`update_ghost_psi_bc`
- **支持模式**：
  - θ > 0：cos²(θ/2) 插值公式
  - θ < 0：直接 ghost ψ = |θ| 模式
  - θ = 0：禁用
- **新增参数**：`theta_contact_deg`（设备常量 d_theta_contact_deg）

### 3.5 多分辨率编译

- `build.py` 已支持 `--grid N --steps M` 参数
- 编译了 mcmp_huang_{100,200,400} 多种分辨率二进制文件

### 3.6 Poiseuille 模式

- `huang_init_mode=3`：上下壁通道流
- 分子力在通道壁处抑制

---

## 四、验证结果汇总

| 验证项 | 状态 | 关键结果 | 方法 |
|--------|:----:|----------|------|
| **Laplace 定律** | ✅ | σ=6.051e-3 (Tr=0.7), σ=1.982e-3 (Tr=0.9)，跨 9 半径恒定 | 压力张量 + Eq.62 积分 |
| **σ-decoupling** | ✅ | σ/σ₀ = 1−6k₁, R²=1.000 | 7 个 k₁ 值扫掠 |
| **ρ-decoupling** | ✅ | ρ_l, ρ_g 跨 k₁ 漂移 < 1e-15 | 论文核心宣称验证 |
| **共存曲线** | ✅ | ρ_g 与 Maxwell 精确一致 | 8 个 Tr 平界面 |
| **Spurious** | ✅ | max\|u\|≈0.10 (Tr=0.7), ≈0.03 (Tr=0.9) | 256² 静态液滴 |
| **Poiseuille** | ✅ | R²=0.999, 误差 0.77% | 抛物线剖面 |
| **网格收敛** | ✅ | ε: 6.3%→3.0%→1.6% (NY=100→256) | 多分辨率编译 |
| **接触角** | ⚠️ | 无法独立调节 | 模型局限（见下文） |

---

## 五、接触角调研结论

经过 **20+ 次独立测试、4 种方法、16 组参数组合**：

| 方法 | 参数范围 | 结果 |
|------|---------|------|
| G_ads 力添加 | -50 ~ +50 | θ 不变 |
| ψ-BC Eq.(34) | θ=30°~150° | θ 不变 |
| ψ-BC cos² 插值 | θ=30°~150° | θ 不变 |
| 直接 ghost ψ | ψ=0.01~0.45 | θ 不变 |
| + Q_m 禁用 (k₁=k₂=0) | ψ=0.05~0.45 | θ 不变 |
| + 重力 Gy=-5e-6 | ψ=0.05~0.45 | θ 不变 |

**根因**：SCMP 单组分模型在墙壁处只有一种流体——EOS 唯一确定密度，无法通过组分重新分配改变表观接触角。与 legacy MCMP（双组分，A-B 可互相排挤）不同。详见 `research/SCMP_CONTACT_ANGLE_PSI_BC.md`。

---

## 六、待完成项

| 项目 | 优先级 | 说明 |
|------|:------:|------|
| 接触角 | P2 | 需墙壁密度 BC 或双组分扩展 |
| NY=400 网格收敛 | P2 | Gx 缩放需调优 |
| 论文 SI 图更新 | P1 | 使用新验证数据替换 |
| 代码清理 | P3 | 移除/注释接触角调试代码（可选保留） |

---

## 七、关键命令行

```bash
# 编译
uv run lbm-build --huang                    # 默认 256²
uv run lbm-build --huang --grid 100 --steps 200000  # 自定义网格

# 运行验证套件
uv run python scripts/06_run_huang_validation_suite.py --all --run

# 单项测试
uv run lbm-run --case-name my_case --app lbm_mrt/solver/mcmp_huang_256 \
    --config configs/huang_scmp.yaml cs_T=0.70 huang_R0=40.0

# Laplace 验证（自定义参数）
uv run lbm-run --case-name laplace_R40 --app lbm_mrt/solver/mcmp_huang_256 \
    --config configs/huang_scmp.yaml cs_T=0.70 huang_R0=40.0 \
    huang_rho_g=0.00929 huang_rho_l=0.35814

# Poiseuille
uv run lbm-run --case-name poiseuille --app lbm_mrt/solver/mcmp_huang_256 \
    --config configs/huang_scmp.yaml cs_T=0.70 huang_init_mode=3 Gx=5e-7

# 接触角（ψ-BC 模式，当前不改变 θ）
uv run lbm-run --case-name contact_test --app lbm_mrt/solver/mcmp_huang_256 \
    --config configs/huang_scmp.yaml cs_T=0.70 huang_init_mode=4 \
    huang_R0=15.0 huang_yc=10.0 theta_contact_deg=-0.15

# Maxwell 共存验证
uv run python -c "
from lbm_mrt.validation.cs_eos import cs_critical_point, maxwell_coexistence
Tc = cs_critical_point(1.0,4.0,1.0)[0]
for Tr in [0.7, 0.9]: print(f'Tr={Tr}: {maxwell_coexistence(1.0,4.0,1.0,Tr*Tc)}')
"

# 接触角测量
uv run python -c "
from lbm_mrt.validation.contact_angle import compute_theta_from_vtk
r = compute_theta_from_vtk('results/my_case/outputdata_scmp/flow_final.vtk')
print(f'θ={r[\"theta_deg\"]:.1f}°')
"

# 压力张量 σ 计算（Eq.62 积分）
# 见 research/CODE_PAPER_AUDIT_20260515.md 中的 Python 脚本
```

---

## 八、修改文件清单

### 求解器 (CUDA)
| 文件 | 变更 |
|------|------|
| `lbm_mrt/solver/src/LBM.cu` | +压力张量 kernel, +吸附力 kernel, +ψ-BC kernel, +force 可视化, +mode 4 init 修复, Q_m 顺序修复, boundary 扩展 |
| `lbm_mrt/solver/include/LBM.h` | Fluid_dev 加 p_xx/p_yy/p_xy/Fx_ads/Fy_ads, d_G_ads_scmp, d_theta_contact_deg |
| `lbm_mrt/solver/include/sim_utils.h` | RuntimeParams 加 G_ads_scmp, theta_contact_deg |
| `lbm_mrt/solver/src/sim_utils.cu` | Params 加载、device push、VTK 输出扩展、evolution_scmp 调用更新 |
| `lbm_mrt/solver/build.py` | 已支持 --grid N --steps M（无修改） |

### Python 验证
| 文件 | 变更 |
|------|------|
| `scripts/06_run_huang_validation_suite.py` | cs_T=Tr 修复 (5 个生成器) |
| `lbm_mrt/validation/contact_angle.py` | **新文件**：圆拟合 + θ 测量 + sweep 分析 |
| `lbm_mrt/io/vtk_reader.py` | 可能有小幅修改以支持新 VTK 字段 |

### 配置与数据
| 文件 | 变更 |
|------|------|
| `configs/huang_scmp.yaml` | cs_T=0.70, +tau_huang, +Lambda_huang, +G_ads, +theta_contact_deg |
| `data/design_scmp_laplace.csv` | 重新生成 (cs_T=Tr) |
| `data/design_scmp_decoupling.csv` | 重新生成 |
| `data/design_scmp_coexistence.csv` | 重新生成 |
| `data/design_scmp_spurious.csv` | 重新生成 |
| `validation_plan.md` | §0.1-0.4, §2.1, §11-14 更新为当前状态 |

### 文档
| 文件 | 说明 |
|------|------|
| `research/CODE_PAPER_AUDIT_20260515.md` | 论文-代码逐项对照审计 |
| `research/SCMP_CONTACT_ANGLE_PSI_BC.md` | 接触角完整调研记录 |
| `results/huang_validation_final/VALIDATION_REPORT.md` | 验证报告（含压力张量修复后数据） |

---

## 九、Git 信息

- **工作树**：huang_mrt_2d (git worktree of complex-porous-media)
- **基于分支**：huang_mrt_2d
- **标签**：v20260516-validation-suite
- **提交信息**：见 git log
