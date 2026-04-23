# 甲烷水合物分解多物理场 LBM 数值模型文档

> 参考文献：
> **Yang 2024** — Yang et al., *Three-dimensional pore-scale study of methane hydrate dissociation mechanisms based on micro-CT images* (2024)
> **Zhang 2019** — Zhang et al., *Pore-Scale Investigation of Methane Hydrate Dissociation Using the Lattice Boltzmann Method* (2019)

---

## 1. 物理过程概述

```
甲烷水合物（固体，mat=2）
       ↓  温度升高 / 压力降低
   CH4·nH2O  →  CH4(气) + nH2O(液)
       ↓
   ① 溶解甲烷进入水相（浓度场 Cm）
   ② 气-水界面处 CH4 挥发（CST Henry 平衡）
   ③ 反应潜热（吸热）注入热场
   ④ 水合物节点体积分数 Vh 减小
   ⑤ Vh → 0 时节点翻转为流体（VOP）
```

耦合顺序（Yang 2024 Figure S2）：

```
Flow（MRT 伪势 D2Q9）→ Conc（D2Q5 MRT-CST）→ LatentHeat → Thermal（D2Q5 MRT-DDF）→ VOP
```

---

## 2. 流场：MRT 伪势 Shan-Chen 双相 LBM（现有）

### 2.1 控制方程

两组分分布函数 $f_k^A$、$f_k^B$，D2Q9 MRT 碰撞：

$$\mathbf{f}^{\alpha}(x+e_k\Delta t,\,t+\Delta t) = \mathbf{f}^{\alpha}(x,t) - \mathbf{M}^{-1}\mathbf{\Lambda}(\mathbf{m}^{\alpha}-\mathbf{m}_{\text{eq}}^{\alpha}) + \mathbf{F}^{\alpha}$$

松弛矩阵 $\mathbf{\Lambda} = \mathrm{diag}(1, s_e, s_\varepsilon, 1, s_q, 1, s_q, 1/\tau_\alpha, 1/\tau_\alpha)$

### 2.2 Shan-Chen 分子力

$$\mathbf{F}^{\alpha}_{\text{mol}}(x) = -\psi^\alpha(x)\sum_k w_k\,G_{\alpha\beta}\,\psi^\beta(x+e_k)\,e_k$$

伪势：$\psi^A = \sqrt{2(p_{\text{EOS}}-\rho^A/3)/G_{AA}}$（PR-EOS），$\psi^B = \rho^B$（理想气体）

### 2.3 吸附力（润湿性）

$$\mathbf{F}^{\alpha}_{\text{ads}}(x) = -G_{\alpha w}\,\psi^\alpha(x)\sum_k w_k\,s(x+e_k)\,e_k$$

$G_{\alpha w}$ 从 `d_GAw_map[x]` 按节点材质查表，实现多材料空间可变润湿性。

---

## 3. 热场：D2Q5 MRT DDF（Double Distribution Function）

### 3.1 分布函数方程

$$h_k(x+e_k^{(5)}\Delta t,\,t+\Delta t) = h_k - \mathbf{M}_5^{-1}\mathbf{\Lambda}_T(\mathbf{m}_h - \mathbf{m}_{h,\text{eq}}) + \delta_{k0}\,S_{\text{latent}}$$

矩向量：$\mathbf{m} = [T,\;T u_x,\;T u_y,\;\frac{3}{4}T,\;0]$（Yang 2024 D3Q7 退化到 2D）

弛豫矩阵：$\mathbf{\Lambda}_T = \mathrm{diag}(1,\,\omega_T,\,\omega_T,\,1,\,1)$

热弛豫率：$\omega_T = \dfrac{1}{0.5 + \alpha_{\text{mat}}/c_{s,5}^2}$，$\alpha = \lambda/(\rho c_p)$（按材质 `d_wall_mat` 选取）

### 3.2 共轭热传递

通过**空间可变** $\omega_T$ 自然实现（Zhang 2019 / Karani & Huber 2015 方法）：

- 流体节点：$\omega_T^f = 1/(0.5 + \alpha_f/c_s^2)$
- 固体/水合物节点：$\omega_T^s = 1/(0.5 + \alpha_s/c_s^2)$

界面连续条件 $\lambda_f\partial_n T_f = \lambda_s\partial_n T_s$ 通过弛豫自动满足，无需显式界面处理。

### 3.3 边界条件

| 边界 | 条件 | 实现 |
|------|------|------|
| $y=0$ ghost | Dirichlet $T = T_{\text{inlet}}$ | $h_k = w_5[k]\cdot T_{\text{inlet}}$ |
| $y=N_y-1$ ghost | 全展开（Neumann $\partial_y T=0$） | $h_k(y=N_y-1) = h_k(y=N_y-2)$ |
| 固体/水合物 ghost | 全反弹 | $h_k = h_{\text{out}}[\bar{k}]$ |

### 3.4 格子单位换算

$$\alpha_{\text{latt}} = \frac{\lambda}{\rho c_p}\cdot\frac{\Delta t}{\Delta x^2}, \quad T_{\text{latt}} = T_{\text{phys}}\;[\text{K, 直接使用}]$$

---

## 4. 浓度场：D2Q5 MRT CST（Conjugate Species Transfer）

### 4.1 分布函数方程

$$g_k(x+e_k^{(5)}\Delta t,\,t+\Delta t) = g_k - \mathbf{M}_5^{-1}\mathbf{\Lambda}_D(\mathbf{m}_g - \mathbf{m}_{g,\text{eq}}) + \delta_{k0}\,S_{\text{CST}}$$

平衡矩：$\mathbf{m}_{\text{eq}} = [C_m,\;C_m u_x,\;C_m u_y,\;\frac{3}{4}C_m,\;0]$

弛豫率：$\omega_D = 1/(0.5 + D_{\text{latt}}/c_{s,5}^2)$，$D_{\text{latt}} = D_{\text{mol}}\cdot\Delta t/\Delta x^2$

### 4.2 CST 源项（气-水界面 Henry 平衡）

气-水界面判据：$\rho_B/(\rho_A+\rho_B) > 0.3$

$$S_{\text{CST}} = C_{\text{eq}} - C_m, \quad C_{\text{eq}} = K_H\cdot\exp\!\left(e_1 + \frac{e_2}{T}\right)$$

（$\omega_{\text{CST}}=1$：单步强制平衡，Yang 2024 §S3）

### 4.3 Kang 方案反应边界（水合物面）

对水合物 ghost 节点，修正浓度满足一阶反应边界（Kang 2002）：

$$C_{\text{bc}} = \frac{D_{\text{latt}}\,C_{\text{nbr}} + k_r\,C_{\text{sat}}}{D_{\text{latt}} + k_r}$$

同时输出分解速率：$\dot{m} = k_r\,\max\!\left(0,\,1-\dfrac{C_{\text{nbr}}}{C_{\text{sat}}}\right)$

---

## 5. 反应动力学：Kim-Bishnoi 模型

$$k_r(T) = k_0\exp\!\left(-\frac{E_a}{RT}\right)\cdot\max\!\left(0,\;p_L - p_{\text{eq}}(T)\right)$$

平衡压力（实验关联式，Yang 2024 Eq. S29）：

$$p_{\text{eq}}(T) = \exp\!\left(e_1 + \frac{e_2}{T}\right), \quad e_1=33.12,\;e_2=-9005.5\;\text{K}$$

**格子单位换算：**

$$k_{r,\text{latt}} = k_{0,\text{phys}}\cdot\frac{\Delta t}{\Delta x}, \quad \frac{E_a}{R}\;\text{直接用物理值（K）}$$

---

## 6. 潜热反馈

水合物分解为吸热过程，潜热源项注入热场：

$$S_{\text{latent}}(x_f) = -\Delta H_{\text{latt}}\sum_{x_g\in\text{邻居ghost}}\dot{m}(x_g)$$

$$\Delta H_{\text{latt}} = \frac{\Delta H_{\text{phys}}}{\rho c_p^f \cdot \Delta x}, \quad \text{单位：K·格子}$$

---

## 7. VOP 固相动态更新

### 7.1 体积分数更新

$$V_h^{n+1}(x) = V_h^n(x) - V_{m,\text{latt}}\cdot\dot{m}(x)$$

$$V_{m,\text{latt}} = \frac{V_{m,\text{phys}}}{\Delta x^3}$$

当 $V_h \leq 0$：节点翻转 `flag: -3 → 1`，`d_wall_mat → 0`

### 7.2 新流体节点重初始化

$$\rho_A^{\text{new}} = \overline{\rho_A}^{\text{nbr}},\quad f_k^{\text{new}} = f_k^{\text{eq}}(\rho^{\text{new}}, \mathbf{u}=0)$$

$$h_k^{\text{new}} = w_5[k]\cdot\bar{T}^{\text{nbr}},\quad g_k^{\text{new}} = w_5[k]\cdot\bar{C}_m^{\text{nbr}}$$

### 7.3 Ghost 层重建

翻转后全域运行：`init_wall_mat_from_flag` → `mark_boundary` → `mark_ghost`，确保新流体节点与其固体邻居正确建立 ghost 关系。

---

## 8. D2Q5 MRT 变换矩阵

$$\mathbf{M}_5 = \begin{pmatrix} 1&1&1&1&1 \\ 0&1&-1&0&0 \\ 0&0&0&1&-1 \\ -4&1&1&1&1 \\ 0&1&1&-1&-1 \end{pmatrix}, \quad \mathbf{M}_5^{-1} = \begin{pmatrix} 1/5&0&0&-1/5&0 \\ 1/5&1/2&0&1/20&1/4 \\ 1/5&-1/2&0&1/20&1/4 \\ 1/5&0&1/2&1/20&-1/4 \\ 1/5&0&-1/2&1/20&-1/4 \end{pmatrix}$$

矩向量次序：$[\rho,\,j_x,\,j_y,\,e,\,p_{xx}]$；速度次序：$0=(0,0)$，$1=(+x)$，$2=(-x)$，$3=(+y)$，$4=(-y)$

---

## 9. 参数换算表（典型值）

| 物理量 | 物理值 | 格子换算 | 格子值（默认参数） |
|--------|--------|---------|----------------|
| 格子尺寸 | $\Delta x = 10\,\mu\text{m}$ | $\Delta x_l = 1$ | 1 |
| 时间步 | $\Delta t = 1\,\mu\text{s}$ | $\Delta t_l = 1$ | 1 |
| 温度 | $T$ [K] | 直接使用 | 278–285 K |
| 热扩散率（水） | $\alpha_f = \lambda/\rho c_p = 1.43\times10^{-7}\,\text{m}^2/\text{s}$ | $\alpha_l = \alpha_f\Delta t/\Delta x^2$ | $1.43\times10^{-3}$ |
| 扩散系数（CH4/水） | $D = 1.85\times10^{-9}\,\text{m}^2/\text{s}$ | $D_l = D\Delta t/\Delta x^2$ | $1.85\times10^{-5}$ |
| 反应速率 | $k_0 = 3.6\times10^4\,\text{mol/(m}^2\text{sPa)}$ | $k_{0,l} = k_0\Delta t/\Delta x$ | $3.6\times10^{-3}$ |
| 摩尔体积 | $V_m = 2.274\times10^{-5}\,\text{m}^3/\text{mol}$ | $V_{m,l} = V_m/\Delta x^3$ | $2.274\times10^{4}/\text{mol}$ |
| 弛豫时间（水热） | $\tau_T = 0.5 + \alpha_l/c_{s,5}^2$ | — | $\approx 0.504$ |
| 弛豫时间（扩散） | $\tau_D = 0.5 + D_l/c_{s,5}^2$ | — | $\approx 0.500055$ |

---

## 10. 基准测试概览

| 编号 | 文件 | 物理 | 验证指标 | 参考 |
|------|------|------|---------|------|
| BM-1 | `benchmark_diffusion.txt` | 纯热扩散 | $T(y)$ 线性，L2 < $10^{-3}$ | 解析解 |
| BM-2 | `benchmark_conjugate_heat.txt` | 共轭热传递（石英圆柱） | 界面梯度比 $\approx\lambda_s/\lambda_f=1.5$，误差 < 10% | Zhang 2019 §4.2 |
| BM-3 | `benchmark_reactive.txt` | 反应-扩散 | $L_r$ 特征长度误差 < 5% | Yang 2024 §S4 |
| BM-4 | `benchmark_vop.txt` | VOP 质量守恒 | $\Sigma V_h + \Sigma C_m V_m$ 漂移 < 0.5%/1000步 | — |
| BM-5 | `benchmark_full_coupling.txt` | 全耦合趋势 | $V_h$ 单调减、$T$ 吸热下降、$C_m$ 梯度 | Yang 2024 Fig. S2 |

运行方式：
```bash
# 编译
bash compile.sh hydrate

# 运行基准（以 BM-1 为例）
mkdir -p results_bm1
./mcmp_sim_hydrate benchmark_diffusion.txt
# 验证
python3 validate_benchmarks.py --bm BM1 --dir results_bm1
```

---

## 11. 实验验证流程（微流控对比）

```
实验 CT / 光学图像
        ↓
    Tecplot .plt 格式导出几何（现有 read_tecplot_to_flag）
        ↓
    匹配物理参数（T_inlet, Gx, dx_phys, dt_phys）
        ↓
    运行 ./mcmp_sim_hydrate experiment_case.txt
        ↓
    对比：分解前沿位置 vs 实验时间序列
           水合物饱和度随时间 vs 实验测量
           孔隙尺度 Cm/T 分布 vs μ-CT 相场
```

---

## 12. 已知局限与未来拓展

| 局限 | 说明 | 可能解决方案 |
|------|------|------------|
| 2D 简化 | 忽略 z 方向流动 | 扩展到 D3Q19 + D3Q7 |
| 单一压力 | $p_L = p_{\text{mix}}$ 未考虑毛管压差 | 加入 Laplace 压差修正 |
| 等温近似（BM-3） | 关闭 Arrhenius 便于验证 | 全耦合时应打开 |
| Ghost 重建代价 | 每次翻转全域遍历 O(NX·NY) | 局部重建（仅翻转节点邻域） |
| 无盐效应 | 未考虑 NaCl 对平衡压的影响 | 修改 $p_{\text{eq}}$ 经验式 |
