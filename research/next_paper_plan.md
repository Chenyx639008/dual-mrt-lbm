# 下一篇论文研究计划

> **定位**：基于 Zhang et al. (2026) *Advances in Colloid and Interface Science* 353, 103895 对孔隙尺度甲烷水合物 / CO₂ 置换领域的系统综述，结合当前 `lbm_mrt` 求解器（MRT-LBM + MCMP + D2Q5 热/浓度 + VOP 相变）已具备的物理与软件能力，识别本课题可发表的学术突破口，形成一篇可执行的论文规划。
>
> **撰写日期**：2026-04-30
>
> **目标期刊层级**：WRR / IJHMT / Chemical Engineering Journal / Fuel / Energy（依最终 scope 与验证深度而定）

---

## 一、综述核心论点梳理（作为研究动机的来源）

Zhang (2026) 在该文中把孔隙尺度水合物研究的现状与缺口收敛到以下几个"可攻击"的点，这是选题策略的核心输入：

### 1.1 孔隙尺度方法的现状分层

| 方法 | 优势 | 局限（综述明确指出的缺陷） |
|------|------|---------------------------|
| Volume-Averaging / Darcy | 大尺度高效 | 把孔隙结构抽象成 φ, K, S 等均值，丢失形态信息 |
| Pore-Network Model (PNM) | 能处理复杂拓扑、低算力 | 把孔喉简化为理想几何，丢失真实接触关系和毛管各向异性 |
| VOF / Level-Set CFD | 界面追踪好 | 相场耦合与反应界面处常出现数值失真，长时计算昂贵 |
| **LBM (MCMP + MRT)** | **边界条件灵活、多物理自然耦合、易并行** | **仍然难以在同一框架下同时解决：(i) 多形态水合物赋存、(ii) 反应表面积随形态演化、(iii) 润湿非均质、(iv) CO₂–CH₄ 置换的双相互换动力学** |

### 1.2 综述给出的"未解科学问题"清单（Sec. 5、Sec. 3.3）

1. **形态-反应面-产能的耦合**：传统 RSSA (reaction-specific surface area) 模型忽略水合物孔隙形态演化，难以精确刻画 RSSA 动态轨迹。Wu et al. (2023) 将 RSSA 演化分为四种"由 pore-wall coating / pore-filling 主导"的阶段，但机理模型仍需要高分辨孔隙尺度数据校准。
2. **热-质耦合的 Péclet 控制**：Yang et al. (2021) 指出分解过程存在"wormhole vs. planar"两种形态，过渡由 Pe 决定。当前 LBM 模型很少把 Pe 扫面做系统化。
3. **多润湿 / 非均质润湿对分解路径的控制**：Ouyang et al. (2023) 与 Kang et al. (2025) 通过微流控实验揭示：亲水孔加速分解、疏水孔减缓，但同一计算域同时承载异质润湿的 LBM 建模几乎空白。
4. **CO₂–CH₄ 置换的"shell/armor 效应"**：CO₂ 注入在 CH₄ 水合物外层形成 mixed-hydrate 外壳 (Sec. 3.1)，阻碍后续置换，综述称为 "armor effect"。需要孔隙尺度模型去定量"外壳厚度 vs. 置换效率"。
5. **毛细-分解前沿的耦合**：Li et al. & Dreyer et al. (2023-2025) 指出毛细力在气-水-水合物三相界面调控质量/热传递；但多数 LBM 分解模型把毛细封为静态。
6. **分解后再成核（secondary hydrate / reformation）**：Zhang X. et al. (2022) 强调 CO₂ 注入会诱导二次水合物堵塞，这是 LBM 从未系统建模的缺口。
7. **微流控实验与孔隙尺度模拟的定量耦合**：综述第 4 节明确提出需要 "rock-on-a-chip" 的跨尺度闭环（模拟 ↔ 芯片）。

### 1.3 综述列的"未来方向"（Sec. 5 结尾）

- 跨尺度耦合（pore → REV → reservoir）
- 非均质矿物/润湿的真实岩心 digital rock
- 脉冲注入 / huff-and-puff 等注入策略的孔隙尺度验证
- 砂迁移 / 地质力学 + 水合物相变的耦合（DEM + LBM）
- AI for Science：用代理/可解释模型加速 RSSA 拟合

---

## 二、当前代码能做什么：能力盘点

基于 `lbm_mrt/solver/CLAUDE.md` 与 `research/hydrate_cu_scientific_guide.md` 的内容：

### 2.1 已具备能力

| 能力 | 实现位置 | 物理完备度 |
|------|----------|-----------|
| 两组分伪势 MCMP (Shan-Chen) | `LBM.cu : mrt_collide_two_components_gpu` | 成熟；含 MRT 弛豫 |
| PR-EOS 非理想气相 | `eos::` 命名空间（`LBM.h`） | 水+CH₄，可直接扩 CO₂ |
| 几何节点材质系统 | `pointsflag` + `d_wall_mat` (`mat∈{0,1,2}`) | 已支持 石英 / 水合物两种固体 |
| 润湿各向异性 | `GAw_by_mat_gpu[256]`, `push_wettability_and_maps()` | 支持 per-material 接触角 |
| 共轭热传递 | D2Q5 DDF + 节点相关 `omegaT` (`hydrate.cu : step_thermal`) | 流体/石英/水合物导热率独立 |
| 反应界面（Kim-Bishnoi） | `hydrate.cu : step_conc`, `diss_rate` | 单反应通道（CH₄ 水合物分解） |
| VOP 固→液翻转 | `hydrate_vop.cu : kernel_apply_vop_conversion` | 含 `pore_origin` 诊断场 |
| 多孔介质发生器 | `LBM.cu : build_circle_array` | 可配置 `r_obs, l_gap, morph, coat_thick, r_mid` |
| 两形态案例 | `configs/hydrate_porous.yaml` (morph=1/2) | coating vs. pore-fill 已对照 |
| 稳态 / limiter 监控 | `SteadyMonitor` | 稳态判据 + 锁死检测 |
| 断点/VTK 流程 | `sim_utils.cu : run_stage_hydrate` | 完整 |

### 2.2 不足（对照综述缺口）

| 综述缺口 | 当前代码缺少的模块 |
|----------|---------------------|
| CO₂–CH₄ 置换 | 只有 CH₄ 分解单通道，无 CO₂ 水合物生成通道、无双反应 |
| 非均质润湿 | 润湿仅按 mat 区分（全域 mat=1/2 使用同一接触角）；尚无"同材质但空间异质"的机制 |
| RSSA 动态诊断 | `pore_origin` 已有雏形，但缺少"有效反应比表面积"的在线计算字段 |
| Pe-扫参 | 缺乏自动化无量纲扫参工具（只能手动多次运行） |
| 二次成核 / reformation | 无逆向 VOP（液→固） |
| 毛细-分解耦合定量化 | 毛细由 Shan-Chen 自然产生，但缺乏"毛细数/邦德数 vs. 分解前沿形态"系统化 |
| 微流控验证 | 无与真实微模型照片/CT 的对照管道 |
| 砂迁移（DEM） | 全固体，无颗粒运动 |

---

## 三、论文选题候选（三选一 / 组合）

以下每一条都对应综述的一个明确缺口，并且其"新增代码工作量"在本框架内可控（6-12 周）。按**学术新颖度 × 工程可行度**我给出排序建议。

### 🔴 候选 A（**推荐**）：Pore-scale LBM unification of hydrate morphology and reaction-specific surface area (RSSA) evolution during methane hydrate dissociation

**科学问题**：
Wu et al. (2023) 把 RSSA 演化分成四阶段，但其经验 logistic 曲线并不能预测**任意孔隙几何 + 初始水合物形态**下的 RSSA(t)。能否用 LBM 直接产生 RSSA(t) 的第一性原理曲线，并反推 REV 级的 logistic 参数？

**方法**：
1. 在现有 `build_circle_array` 上扩展第三种形态 `morph=3`：patchy/random 分布（综述 Fig. 9 的 point/vein/block）——用 Poisson 盘采样生成。
2. 新增 `RSSA_dev` 诊断字段：每步扫描 `pointsflag==-3` 与流体的接触面（沿 D2Q9 方向），用 D2Q9 各向异性加权得到实际反应面积；输出 VTK 时间序列。
3. 扫参：初始饱和度 Sh ∈ {0.3, 0.5, 0.7}，三种形态；每组产出 RSSA(Sh(t))。
4. 用对数斯蒂克/分段拟合出四阶段参数，给出 morphology-aware RSSA 模型的经验系数并与 Wu et al. 对比。

**新增代码**（预估）：
- `hydrate.h / hydrate_vop.cu`：`kernel_compute_RSSA`（~80 行）
- `build.py`：编译开关 `--diag-rssa`
- Python 后处理：`lbm_mrt/viz/rssa_curves.py`
- 配置 morph=3 Poisson 生成器：`sim_utils.cu : build_patchy_hydrate`（~150 行）

**创新点**：
- 首次在 LBM 框架中把 RSSA 作为一级诊断场而非后处理推断。
- 把 Wu et al. 的宏观 logistic 公式做**几何-可验证**的第一性原理推导。
- 输出物：morphology → RSSA(Sh) 曲线族 + 可直接用于 REV-scale 的系数查表。

**对照论文**：Wu et al. 2023 *Chem. Eng. J.* 463, 142464; Yang et al. 2021 *Chem. Eng. J.* 422, 130206; Zhang et al. 2019 *WRR* 55, 8422.

**期刊定位**：*Water Resources Research* / *International Journal of Heat and Mass Transfer*（与原有 Zhang et al. 2019 WRR 同条线，升级版）。

---

### 🟠 候选 B：Péclet-regime phase diagram of dissociation front morphology in heterogeneously wettable hydrate-bearing porous media

**科学问题**：
Yang et al. (2021) 发现 Pe 调控 wormhole ↔ planar 前沿；Ouyang et al. (2023) 发现润湿影响分解路径。但**"Pe × 润湿异质度"二维相图**在文献中缺失。

**方法**：
1. 引入"空间变接触角"机制：新增 `theta_by_node_gpu[NX*NY]` 字段代替 `GAw_by_mat_gpu[256]`，允许在同一材质上画亲水/疏水 patch（Voronoi 或条带）。
2. 定义 Péclet：`Pe = U*L/D_CH4`；扫参 Pe ∈ [0.01, 10]，润湿异质度 χ ∈ {0, 0.3, 0.6}。
3. 对每组 (Pe, χ) 用新诊断字段（前沿曲率、前沿粗糙度 fractal dimension）做后处理。
4. 输出 4×4 相图 + 机理解释。

**新增代码**：
- `sim_utils.cu : push_wettability_spatial()`（~100 行）
- `hydrate_vop.cu : kernel_extract_front_curvature`（~80 行）
- Python：`lbm_mrt/viz/phase_diagram.py`

**创新点**：
- 首个同时覆盖 Pe 和润湿异质的 LBM 相图。
- 为综述 Sec. 3.2 "wettability controls phase transition" 提供定量相图。

**期刊定位**：*Chemical Engineering Journal* / *Fuel*。

**风险**：空间变接触角在 Shan-Chen 体系需要验证数值稳定性（接触角跃变处的 spurious currents），工作量可能比 A 高 30%。

---

### 🟢 候选 C：A first-principles LBM simulation of the "armor effect" in CO₂-CH₄ hydrate exchange at the pore scale

**科学问题**：
综述 Sec. 3.1 明确提出"armor effect"（mixed-hydrate 外壳阻碍置换）是置换效率卡点。目前**没有在同一孔隙尺度模型里同时分解 CH₄ 水合物 + 生成 CO₂ 水合物的 LBM 工作**。

**方法**：
1. 扩展组件：`Fluid_dev` 由 A/B 双组分变为 A/B/C 三组分（H₂O、CH₄、CO₂）。
2. VOP 增加两反应通道：
   - Kim-Bishnoi 分解 CH₄ 水合物 (mat=2)
   - 逆向 Kim-Bishnoi 形成 CO₂ 水合物 (mat=3)
3. 引入热力学相图判据：当 T, P_CO₂, x_CO₂ 满足 CO₂-hydrate 稳定区 → 翻转流体→ 固（mat=3）。
4. 研究 "shell thickness vs. exchange rate"：同一几何扫参 P_inlet, T_inlet, 初始 Sh。

**新增代码**（大）：
- `LBM.h`：扩三组分 `Fluid_dev` → 增一套 fin/fout 数组；内存翻倍
- `hydrate.cu`：新增 `step_conc_co2`，两反应核
- `hydrate_vop.cu`：新增 `kernel_form_co2_hydrate`（液→固逆翻转，需要小心处理 `pointsflag` 与 `d_wall_mat` 一致性）
- 配置：`configs/hydrate_co2_exchange.yaml`
- 新几何材质：`mat=3` (CO₂ hydrate)

**创新点**：
- 综述明确为"缺口"的 armor 效应的首个 LBM 定量研究。
- "分解-形成"双 VOP 可复用到水合物再成核（candidate B 附带）。

**期刊定位**：*Energy* / *Chemical Engineering Journal* / *Energy & Fuels* 头部。

**风险**：工作量最大（~4-6 周额外编码 + 验证），三组分伪势模型需要仔细验证热力学一致性；另外 PR-EOS 扩 CO₂ 参数、CO₂ 水合物的 Kim-Bishnoi 常数选取需做文献调研。

---

## 四、推荐路线：**候选 A 作为主线，候选 B 作为扩展，候选 C 作为下一篇**

**理由**：
1. A 的工程量最小（估计 4 周代码 + 4 周跑算 + 4 周写作），与当前仓库成熟度最匹配。
2. A 的输出（RSSA 查表）可以成为 B、C 的上游输入，形成一条"RSSA→Pe相图→CO₂置换"三篇的发表链。
3. A 本身可单独投稿 WRR / IJHMT，成功率较高。

### 4.1 分阶段计划（甘特）

```
阶段 1 [Week 1-2]  方法准备
  ├─ 文献补充：Wu 2023 / Yang 2021 / 综述中表 1 所有 LBM 条目全文获取
  ├─ 代码：新增 RSSA 诊断字段（hydrate_vop.cu）
  └─ 代码：新增 morph=3 patchy 生成器

阶段 2 [Week 3-4]  验证
  ├─ 单球案例（script 04）回归：RSSA(t) 与几何解析解对照（球面积递减）
  ├─ 多孔 morph=1/2（script 05）扫 Sh ∈ {0.3, 0.5, 0.7}
  └─ 数值敏感性：网格细化（300² → 600²）、τ 扫描

阶段 3 [Week 5-8]  生产跑
  ├─ morph ∈ {1, 2, 3} × Sh ∈ {0.3, 0.5, 0.7} = 9 组
  ├─ 每组跑到分解完毕
  └─ RSSA(t), Sh(t), dissRate(t) 曲线入库

阶段 4 [Week 9-10] 分析
  ├─ 四阶段拟合（对比 Wu 2023 logistic）
  ├─ 提取 morphology-aware RSSA 公式
  └─ 误差棒与不确定度量化

阶段 5 [Week 11-12] 写作 + 投稿
  ├─ 图 1: 概念 + 几何
  ├─ 图 2: LBM 模型示意（参考 Zhang 2019 WRR 风格）
  ├─ 图 3: RSSA(t) 曲线族
  ├─ 图 4: 四阶段拟合
  ├─ 图 5: morphology-aware 修正公式 vs. Wu et al.
  └─ 讨论: 与微流控数据定性对照（综述 Table 2 的 Yang et al. 2024 [135]）
```

### 4.2 里程碑验收标准

- [ ] `kernel_compute_RSSA` 通过解析解测试（单球水合物：`RSSA = 2πr`，误差 < 3%）
- [ ] morph=3 patchy 分布的孔隙率与设定值一致，RSSA 初值与 morph=1/2 在相同 Sh 下存在统计可区分
- [ ] 9 组跑完均稳态到 Sh → 0 或触碰最大步数，无 limiter 锁死
- [ ] 拟合 RSSA(Sh) 的 R² > 0.95
- [ ] 论文 5 张主图能 1:1 对应上文 4.1 的图示

---

## 五、具体待解决的科学问题（论文章节草案）

### 5.1 Introduction 可用的三句话

1. 综述 (Zhang 2026) 指出 RSSA 是"链接孔隙尺度与 REV 尺度的决定性参数"，但其随形态的动态演化目前只能靠经验 logistic 拟合（Wu 2023）。
2. 现有 LBM 研究（Zhang 2019, Yang 2021-2022）给出 RSSA 的**瞬时值**，但缺少跨形态的统一曲线族。
3. 本文在 MRT-LBM + VOP 框架下直接把 RSSA 作为一级诊断场，首次给出 (morphology, Sh) → RSSA 的第一性原理数据库。

### 5.2 方法章节骨架

- 5.2.1 MCMP-MRT 流场（已有）
- 5.2.2 D2Q5 共轭热场（已有）
- 5.2.3 D2Q5 MRT-CST 浓度场 + Kim-Bishnoi 反应（已有）
- 5.2.4 VOP 翻转（已有）
- 5.2.5 **新增**：RSSA 的格点级计算（D2Q9 方向加权的水合物-流体界面长度）
- 5.2.6 **新增**：三种形态 (coating / pore-fill / patchy) 的几何生成器
- 5.2.7 无量纲数（Pe, Da, Ca）

### 5.3 关键方法创新

**RSSA 的格点定义**：

对每个 `pointsflag == -3` 节点 $\mathbf{x}$，

$$A_{\rm react}(\mathbf{x}) = \sum_{\alpha=1}^{8} w_\alpha \cdot \mathbb{1}\big[{\rm flag}(\mathbf{x}+\mathbf{e}_\alpha) > 0 \text{ 或 }= 0\big] \cdot |\mathbf{e}_\alpha|^{-1}$$

其中 $w_\alpha$ 为 D2Q9 权重，$|\mathbf{e}_\alpha|$ 补偿对角方向。

RSSA 定义：
$$\text{RSSA}(t) = \frac{\sum_{\mathbf{x} : {\rm flag}=-3} A_{\rm react}(\mathbf{x})}{\sum_{\mathbf{x} : {\rm flag} \neq -2} 1}$$

分子 = 水合物-流体有效接触边界长度，分母 = 孔隙总面积。

这是 Wu et al. (2023) logistic 公式所没有的、几何可验证的一级定义。

### 5.4 待回答的 sub-questions

1. RSSA(Sh) 在 coating / pore-fill / patchy 三形态下是否差异超过 2×？
2. RSSA(t) 的"转折点"是否对应 Wu 2023 的相态切换？
3. 是否存在一条 morphology-invariant 的主曲线 (universal law)？
4. 当初始 Sh < 0.1 时 patchy 形态 RSSA 是否如微流控观测那样陡增？

---

## 六、工程风险与对策

| 风险 | 后果 | 对策 |
|------|------|------|
| morph=3 patchy 生成器孔隙连通性差 | 分解前沿被孤岛阻断，物理失真 | 生成后执行连通分量检查；孤立水合物岛移除或合并 |
| RSSA 诊断字段 VTK 输出占空间 | 长时计算磁盘爆 | 间隔输出（每 50000 步）+ 提供 `--rssa-every` 参数 |
| 不同 morph 在同一 Sh 下有效 RSSA 偏差大，影响归一化 | 相图无法叠合 | 用 `RSSA / RSSA(t=0)` 的归一化曲线代替原始值 |
| limiter 在高 Pe 下命中率激增 | 求解不稳定 | 保留 `SteadyMonitor` 的 limiter 统计，>5% 命中直接报错停机 |
| 物理参数换算误差 | 结论不可重复 | 强制输出 `params.txt` + `log.txt`，配合 `research/hydrate_validation_design.md` 的换算表 |

---

## 七、为这篇文章新建的文档 / 数据路径

```
research/
├── next_paper_plan.md          # 本文档
├── rssa_scientific_derivation.md   # [阶段 1 创建] RSSA 格点定义的数学推导
└── rssa_validation_protocol.md     # [阶段 2 创建] 验证单球/多孔案例的 checklist

configs/
├── hydrate_rssa_coating.yaml
├── hydrate_rssa_porefill.yaml
└── hydrate_rssa_patchy.yaml

scripts/
└── 06_run_rssa_sweep.py        # 9 组自动化运行 + 合并 CSV

data/
└── rssa_curves/                # 输出存放位置（.gitignore）
```

---

## 八、下一步动作（立即可做）

1. **确认选题**：用户在 A / B / C 中确认主线（推荐 A）。
2. **文献补充**：把综述中编号 **[57, 62, 74, 96, 108, 127, 135, 155, 167]** 的原文下载到 `references/` 作为方法对照基线（这些是综述中最强关联 LBM 的工作）。
3. **代码实验**：在 feature branch `feat/rssa-diagnostics` 上先实现 `kernel_compute_RSSA` 的最小原型（~1 天工作量），用单球 case 04 先回归。
4. **写作准备**：在 `research/rssa_scientific_derivation.md` 里把 §5.3 的格点 RSSA 定义展开到完整推导（含 D2Q9 权重修正的几何证明）。

---

## 附录 A：综述中与本工作方向强相关的参考文献（供深读）

| 编号 | 作者 年份 | 贡献 | 与本工作的接口 |
|------|-----------|------|----------------|
| [57] | Yang 2022 | LBM 得到渗透率 & 表面积修正，支持 REV 上尺度 | 本工作的 RSSA 是其方法的升级 |
| [62] | Zhang 2019 WRR | 耦合 LBM 二维多物理场 | 直接对比基线 |
| [74] | Wang 2022 | Pore-scale CFD：颗粒大小、曲折、有效孔隙 | 借用它的 RSSA 拟合形式 |
| [96] | Wu 2023 | Logistic RSSA 模型（4 阶段） | 本工作要验证/修正的原始公式 |
| [108] | Yang 2024 | Micro-CT 3D 与 LBM 对照 | 为验证提供真实数据 |
| [127] | Wang 2022 | CO₂-CH₄ pore-scale + 初始含水饱和度 | 候选 C 的上游 |
| [135] | Yang 2024 | 微流控 CH₄ 水合物分解形态观察（membrane / crystal） | 定性对照 |
| [155] | Ouyang 2023 | 润湿差异实验 | 候选 B 的上游 |
| [167] | Zhang 2019 (Wiley) | Pore-scale LBM CH₄ 分解 | 直接对标 |

---

## 附录 B：与综述图 4 的对应关系

综述 Fig. 4 把 pore-scale 水合物分解研究分为 (a-d) pore structure 与 (e-g) saturation-time series。本工作输出：
- 对应 (a-c)：三种 morph 的几何示意 + LBM 计算域
- 对应 (d)：分解界面放大（来自 VTK → ParaView 截图）
- 对应 (e-g)：两种形态在三个 Sh 值下的时间序列（候选 A 的主要数据图）

这种一对一对应性是本文 Introduction 引综述图、Discussion 回应综述缺口的关键"闭环结构"。
