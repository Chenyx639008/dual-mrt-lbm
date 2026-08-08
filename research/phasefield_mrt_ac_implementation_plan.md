# MRT-LB 保守 AC 实施计划（解决液滴溶解 + 发散，定量湿壁）

> **日期**: 2026-08-08 · **状态**: 📋 待批准（用户审阅后执行）
> **前置文献**（已转 LaTeX 精确版）:
> - Liang et al. 2018, PRE 97, 033309 → `phasefield_LBM/output/liang2018.md`（**D2Q9 BGK 母本，公式权威**）
> - Yang et al. 2024, Innov Energy → `phasefield_LBM/output/Yang ...md`（**D3Q19 MRT，SI S5/S10-S17**）
> - 2D 适配已有草案 → `research/literature/phase_field_2d_implementation_plan.md` §3.2 Path B

---

## 0. 结论先行：液滴溶解的确定性根因已找到

**之前的 MRT-LB AC 失败不是"反扩散源结构性无效"，而是平衡矩 $m^{eq}$ 的矩序完全错乱。**

用 `jax_lbm/collision.py` 的 Lallemand D2Q9 变换矩阵（矩序 `[ρ, e, ε, jx, qx, jy, qy, pxx, pxy]`）数值验证：

| 矩 | 正确值 (M·f_eq) | 之前失败实现 | 判定 |
|----|----------------|-------------|------|
| ρ   | φ              | φ           | ✅ |
| **e** | **−2φ**      | **+φ**      | ❌ |
| **ε** | **+φ**       | **0**       | ❌ |
| jx  | φu_x           | φu_x        | ✅ |
| **qx** | **−φu_x**   | **0**       | ❌ |
| jy  | φu_y           | φu_y        | ✅ |
| **qy** | **−φu_y**   | **0**       | ❌ |
| **pxx** | **0**      | **φu_x/3**  | ❌ |
| **pxy** | **0**      | **φu_y/3**  | ❌ |

6 个矩错误 → $f^{eq}=M^{-1}m^{eq}\neq\omega_i\phi(1+c_i\cdot u/c_s^2)$ → 恢复的 AC 方程错误 → 扩散-反扩散失衡 → **界面扩散 → 液滴溶解（φ_max 1.0→0.16）**。这与阶段 3 总结"反扩散源结构性未恢复锐化（不是强度/弛豫问题）"的诊断完全吻合。

---

## 1. 文献级精确公式（已逐项数值验证 ✅）

### 1.1 保守 AC 目标方程（Yang S10 = Liang 式1，两文一致）

$$\frac{\partial\phi}{\partial t}+\nabla\cdot(\phi\mathbf u)=\nabla\cdot M\left[\nabla\phi-\frac{4\phi(1-\phi)}{W}\,\mathbf n\right],\quad \mathbf n=\frac{\nabla\phi}{|\nabla\phi|}$$

反扩散系数：$\lambda=\frac{4\phi(1-\phi)}{W}$；等价 tanh 形式：$\frac{1}{W}[1-\tanh^2(\frac12\ln\frac{\phi}{1-\phi})]=\frac{4\phi(1-\phi)}{W}$（恒等，已验证）。

### 1.2 BGK-LB 保守 AC（Liang 2018 式 8–12，D2Q9 母本）

**演化**（式8）：
$$f_i(\mathbf x+\mathbf c_i\delta_t,t+\delta_t)-f_i(\mathbf x,t)=-\frac{1}{\tau_f}[f_i-f_i^{eq}]+\delta_t F_i$$

**平衡分布**（式10，⚠️ 只到一阶矩）：
$$f_i^{eq}=\omega_i\phi\left(1+\frac{\mathbf c_i\cdot\mathbf u}{c_s^2}\right),\quad \omega_0=\frac49,\ \omega_{1-4}=\frac19,\ \omega_{5-8}=\frac1{36}$$

**源项**（式11，⚠️ 必须含 $\partial_t(\phi\mathbf u)$ 时间导数项）：
$$F_i=\left(1-\frac{1}{2\tau_f}\right)\frac{\omega_i\,\mathbf c_i\cdot\left[\partial_t(\phi\mathbf u)+c_s^2\lambda\mathbf n\right]}{c_s^2}$$

**序参数**（式12）：$\phi=\sum_i f_i$

**迁移率**（式13）：$M=c_s^2(\tau_f-0.5)\delta_t \Rightarrow \tau_f=M/c_s^2+0.5$

> **为什么 $\partial_t(\phi\mathbf u)$ 必须**：Liang 明确指出，不含此项的 AC-LB（如 Fakhari-Bolster 2017）恢复的方程偏离目标 AC。此项需存上一步 $(\phi\mathbf u)^{n-1}$，一阶后差。

### 1.3 MRT-LB 保守 AC（Yang S11，2D 降维——已逐矩验证）

**碰撞**（Yang S11）：
$$\mathbf f^*=\mathbf f-\mathbf M^{-1}\left[\boldsymbol\Lambda_\phi(\mathbf m-\mathbf m^{eq})+\Delta t\left(\mathbf I-\frac{\boldsymbol\Lambda_\phi}{2}\right)\mathbf R\right]$$

**变换矩阵**：复用 `jax_lbm/collision.py` 的 `MRT_M`（Lallemand & Luo 2000，矩序 `[ρ,e,ε,jx,qx,jy,qy,pxx,pxy]`）。**实现第一步：打印 M 每行确认矩序**（`MRT_M` 已验证与此处一致）。

**平衡矩**（= M·f_eq，一阶平衡，数值验证）：
$$\mathbf m^{eq}=\left[\phi,\ -2\phi,\ \phi,\ \phi u_x,\ -\phi u_x,\ \phi u_y,\ -\phi u_y,\ 0,\ 0\right]^T$$

**矩空间源项**（= M·F_vec，数值验证，零阶矩=0 保质量守恒）：
$$\mathbf R=\left[0,\ 0,\ 0,\ g_x,\ -g_x,\ g_y,\ -g_y,\ 0,\ 0\right]^T,\qquad \mathbf g=\partial_t(\phi\mathbf u)+c_s^2\lambda\mathbf n$$

**松弛矩阵**（只有动量矩 jx(m3), jy(m5) 松弛；e/ε/qx/qy/pxx/pxy 不松弛=1）：
$$\boldsymbol\Lambda_\phi=\mathrm{diag}(1,\ 1,\ 1,\ \omega_\phi,\ 1,\ \omega_\phi,\ 1,\ 1,\ 1)$$

**迁移率-松弛关系**（Yang S37）：
$$M=c_s^2\left(\frac{1}{\omega_\phi}-\frac12\right)\frac{\Delta x^2}{\Delta t}\Rightarrow\omega_\phi=\frac{1}{M/c_s^2+0.5}$$

**MRT↔BGK 一致性（已验证）**：$(I-\frac{\Lambda_\phi}{2})\mathbf R$ 在 jx/jy 上给出 $(1-\omega_\phi/2)g=(1-\frac{1}{2\tau_f})g$，当 $\omega_\phi=1/\tau_f$ 时两格式物理等价。

---

## 2. 分步实施（每步有明确验证基准，全绿才进下一步）

### Step 1：BGK-LB 保守 AC（`ac_bgk_lb_step`）——先立真值

**实现**（`jax_lbm/pf/phase_field.py` 新增）：
1. `f_eq = w_i * phi * (1 + c_i·u/c_s²)`（phi 截断到 [ε, 1−ε]，ε=1e-6 防 ln 奇异）
2. 各向同性梯度 `∇φ`（复用 `gradient_isotropic`）→ `n = ∇φ/|∇φ|`
3. `λ = 4φ(1−φ)/W`，`g = (φu)ⁿ − (φu)ⁿ⁻¹ + c_s²·λ·n`（**需双缓冲存上一步 φu**）
4. `F_i = (1−1/(2τ_f))·ω_i·(c_i·g)/c_s²`
5. 碰撞 + streaming + `φ = Σf`
6. `τ_f = M/c_s² + 0.5`

**验证**（新增 `tests/test_pf_ac_lb.py`）：
- V1 静态液滴（u=0, W=4, 2000 步）：**φ_max ≥ 0.999（不溶解）**、∫φ 守恒机器精度、界面保持 tanh
- V2 平界面（x 方向周期性 tanh 剖面）静止不变
- V3 质量守恒：Σf_i 每步守恒
- V4 与直接 FD `conservative_ac_step`（W=5）界面剖面一致（同一初始条件）

**预期**：解决液滴溶解。W=4 稳定（直接 FD 需要 W≥5）。

### Step 2：MRT-LB 保守 AC（`ac_mrt_lb_step`）——与 BGK 对标

**实现**：
1. `m = M·f`；`m_eq = [φ, −2φ, φ, φu_x, −φu_x, φu_y, −φu_y, 0, 0]`
2. `R = [0,0,0,g_x,−g_x,g_y,−g_y,0,0]`（同 Step 1 的 g）
3. `Λ_φ = diag(1,1,1,ω_φ,1,ω_φ,1,1,1)`，`ω_φ = 1/(M/c_s²+0.5)`
4. `f* = f − M⁻¹[Λ_φ(m−m_eq) + (I−Λ_φ/2)R]`，再 streaming

**验证**（同一测试文件）：
- V5 静态液滴同 V1（不溶解 + 守恒）
- V6 **MRT ≡ BGK**：相同初始/参数下 φ 场逐格一致（|Δφ|<1e-10）
- V7 W=3 稳定（MRT 稳定性优势：解决直接 FD 的 W≥5 限制）

### Step 3：耦合 NS（替换直接 FD AC）

**实现**：`coupled_ac_ns_step` 中 AC 部分用 `ac_mrt_lb_step` 替换 `conservative_ac_step`（保留 wall 参数接口）。

**验证**（重跑阶段 2 全套 `test_pf_stage2.py` + 基准）：
- V8 Laplace 定律：ΔP vs 1/R 线性，R² 接近 1（对标 R²=0.9996）
- V9 共存：φ→1.0/0.0，伪速度量级
- V10 **阶段 4 发散复测**：body-force gx 驱动液滴 >5000 步不发散（原 ~1460 步发散）
- V11 无回归：阶段 0/1 测试全绿

### Step 4：S17 湿壁定量接触角（阶段 3 收口）

**实现**：`conservative_ac_step_walls` 的 LB 版——MRT-LB AC + y 方向 S17 ghost（`surface_energy_phi_s` 已就绪），x 周期。

**验证**（`test_pf_stage3.py` 解 skip）：
- V12 θ ∈ {30°, 60°, 90°, 120°, 150°}，液滴在底壁弛豫至稳态，实测接触角误差 < 2°
- V13 对照 Yang S5.2：球形颗粒 θ=60° 案例（2D 圆颗粒版）

### Step 5：CUDA 落地（`pf_ns_2d.cu`）

- 新增 MRT-LB AC kernel（矩空间碰撞 + streaming），复用现有 M/M_inv
- JAX↔CUDA 交叉验证（对标 phi=3e-10 机器精度）
- 参数键：`pf_mode` 扩展 + `pf_M`/`pf_W`/`pf_sigma`/`pf_omega_phi`

---

## 3. 参数（文献锚定）

| 参数 | 值 | 来源 |
|------|-----|------|
| 界面宽度 W | 4（验证 3–5） | Liang: W=5；MRT 目标 W=3-4 |
| 表面张力 σ | 0.001（或沿用阶段 2 标定） | Liang |
| 迁移率 M | 0.1 | Liang |
| τ_f / ω_φ | τ_f=0.8 / ω_φ=1.25（M=0.1, c_s²=1/3） | Liang M=c_s²(τ_f−0.5) |
| 密度比 | 阶段 3 先 ρ_l/ρ_g=10；阶段 4 测 100/1000 | Liang 10–1000 |
| φ 截断 ε | 1e-6 | 实现计划 |
| 格子 | D2Q9，δx=δt=1，c_s²=1/3 | 统一 |

---

## 4. 风险与对策

| 风险 | 对策 |
|------|------|
| ∂_t(φu) 时间导数项实现复杂 | Step 1 先用 u=0 验证（此时 ∂_t(φu)=0 可忽略），再开 u≠0 双缓冲 |
| MRT 与 BGK 不完全等价（非正交矩） | 先 V6 对标；若差异大，以 BGK 为真值，检查 R 布置 |
| 反扩散方向 n 在液滴内部奇异 | ∇φ 用各向同性梯度 + 分母 +1e-12；φ∈(ε,1−ε) 截断 |
| 与 NS 耦合后 W 需要增大 | W 与 σ、M 一起标定（σ=√(κβ/18) 自洽关系） |
| 阶段 3 湿壁仍需 MRT 稳定性 | 已由 Step 2 提供 |

---

## 5. 文件改动清单

| 文件 | 改动 |
|------|------|
| `jax_lbm/pf/phase_field.py` | 新增 `ac_bgk_lb_step`, `ac_mrt_lb_step`, `ac_lb_step_walls`；`coupled_ac_ns_step` 换 AC |
| `jax_lbm/tests/test_pf_ac_lb.py` | 🆕 Step 1/2 验证（V1–V7） |
| `jax_lbm/tests/test_pf_stage2.py` | 回归（V8–V11） |
| `jax_lbm/tests/test_pf_stage3.py` | 解 skip，定量接触角（V12–V13） |
| `validation/phasefield/` | 更新 laplace/共存脚本引用新 AC |
| `lbm_mrt/solver/src/pf_ns_2d.cu` | Step 5：MRT-LB AC kernel |
| 本计划 + `phasefield_development_plan.md` | 进度更新 |

**预计测试规模**：阶段 3 相关 + 回归 ≈ 25–30 项 JAX 测试。

---

## 6. 一句话总结

> 液滴溶解的根因是**平衡矩矩序错乱（6 个矩错误）**，已从 Liang 2018（D2Q9 BGK 母本）+ Yang 2024（D3Q19 MRT）提取出文献级精确的平衡矩/源项/弛豫定义并逐项数值验证。实施路径：**BGK-LB AC 立真值 → MRT-LB AC 对标 → 耦合 NS（解决发散）→ S17 定量湿壁 → CUDA 落地**。
