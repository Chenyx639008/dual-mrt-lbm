# hydrate.cu 科学说明与函数级导读

本文面向 `lbm_mrt/solver/src/hydrate.cu`，目标不是重复代码注释，而是把它还原成一条清晰的“物理模型 → 数值方法 → 函数职责 → 调试检查点”的链路，方便你理解每个函数在做什么、为什么这样做，以及出错时该看哪里。

适用范围：水合物编译版 `mcmp_sim_hydrate`，即启用 `-DHYDRATE_ENABLE` 后的求解路径。

---

## 1. 这个文件在整个求解器中的位置

`hydrate.cu` 负责水合物耦合里的三个核心物理模块：

1. 热场 `Therm_dev`
2. 浓度场 `Conc_dev`
3. 水合物相变与翻转 `VOP_dev`

它不是主程序入口。真正的时间推进由 `main.cu` 和 `sim_utils.cu` 调度，而 `hydrate.cu` 提供具体的物理演化函数。

在全耦合时，实际顺序是：

1. 流场先由已有的 `evolution_all()` 推进
2. 再算浓度场 `step_conc()`
3. 再从分解速率生成潜热源项 `compute_latent_heat_source()`
4. 再推进热场 `step_thermal()`
5. 最后做 VOP 更新 `step_vop()`

这个顺序在代码里对应 `step_hydrate_physics()` 和 `run_stage_hydrate()`。

---

## 2. 先理解它用的数值思想

这个文件的核心方法可以概括为三类：

1. D2Q5 MRT 的对流-扩散格式
- 用于温度场和浓度场。
- 不是简单 BGK，而是 MRT（Multiple Relaxation Time）矩空间弛豫。

2. 空间可变热扩散率
- 不同材料节点使用不同热弛豫率 `omegaT`。
- 这是共轭传热的关键：流体、石英、水合物可以有不同导热性质。

3. 反应边界 + VOP 翻转
- 浓度场在水合物 ghost 节点上施加反应边界。
- 当水合物体积分数 `Vh` 降到 0 时，节点从水合物转为流体。

---

## 3. 文件顶部条件编译为什么重要

文件一开始就有：

```cpp
#ifdef HYDRATE_ENABLE
```

这意味着：

1. 如果编译时没有 `-DHYDRATE_ENABLE`，这个文件整体不参与编译。
2. 如果你在编辑器里看到很多行发暗，通常就是因为编辑器没有看到这个宏。
3. 但你已经确认 `uv run lbm-build --hydrate` 成功，所以从编译角度它是有效的。

这类条件编译不是“死代码”，而是“编译模式切换”。

---

## 4. 这个文件里的主要数据结构

这些结构体和常量不是在 `hydrate.cu` 中定义的正文，但理解它们是读懂后续函数的前提。

### 4.1 `Therm_dev`

热场设备数据，一般包含：

- `h_in`, `h_out`：D2Q5 分布函数
- `T`：宏观温度场

### 4.2 `Conc_dev`

浓度场设备数据，一般包含：

- `g_in`, `g_out`：D2Q5 分布函数
- `Cm`：宏观浓度场

### 4.3 `VOP_dev`

水合物相变与翻转相关数据，一般包含：

- `Vh`：水合物体积分数
- `diss_rate`：分解速率
- `S_latent`：潜热源项
- `new_fluid_flag`：新翻转流体节点标记

### 4.4 `pointsflag` 和 `d_wall_mat`

这是调试时最重要的空间标记：

- `pointsflag`：节点类型，通常区分流体、边界、ghost、固体、水合物等
- `d_wall_mat`：材料编号，决定某个 ghost 节点是石英还是水合物

如果结果异常，往往不是方程本身，而是这两个图谱没构好。

---

## 5. 函数级说明

下面按 `hydrate.cu` 中函数出现顺序解释。

---

### 5.1 `init_device_variable_hydrate(const RuntimeParams& P)`

#### 功能
把物理参数换算成格子单位，并上传到 GPU 的常量区。

#### 它做了什么
主要包括：

- 计算热扩散率 `alpha_fluid`、`alpha_hydrate`、`alpha_solid`
- 计算扩散系数 `D_latt`
- 计算反应速率前因子 `k0_latt`
- 计算 `Ea_over_R`
- 计算摩尔体积格子量 `Vm_latt`
- 计算潜热换算量 `latent_H_latt`
- 把这些量通过 `cudaMemcpyToSymbol` 上传到设备常量

#### 使用的方法
这是典型的“物理量无量纲化/格子化”步骤。

你可以把它理解为：

- 输入是物理参数
- 输出是 GPU 内核可直接用的格子参数

#### 调试重点
如果温度场、浓度场、反应速率不对，第一步就查这里：

- `dx_phys` 和 `dt_phys` 是否合理
- `alpha_latt` 是否过大或过小
- `D_latt` 是否数量级异常
- `k0_latt` 是否太激进

#### 常见问题
1. 单位写错
2. 传入参数和 YAML 不一致
3. 物理量级导致数值不稳定

---

### 5.2 `alloc_therm(Therm_dev& TH)` / `free_therm(Therm_dev& TH)`

#### 功能
为热场分配和释放 GPU 显存。

#### 方法
- `alloc_therm` 使用 `cudaMalloc`
- `free_therm` 使用 `cudaFree`

#### 调试重点
如果程序一开始就崩：

- 查显存是否足够
- 查 `mem_size_D2Q5` 和 `mem_size_scalar` 是否正确
- 查是否有重复释放或未初始化指针

---

### 5.3 `alloc_conc(Conc_dev& CN)` / `free_conc(Conc_dev& CN)`

#### 功能
为浓度场分配和释放 GPU 显存。

#### 方法
与热场完全类似，只是操作对象变成 `g_in`, `g_out`, `Cm`。

#### 调试重点
若浓度场出现 NaN 或全零，先确认：

- `CN.g_in` 是否正确初始化
- `CN.Cm` 是否被宏观更新覆盖
- `step_conc()` 是否真的被调用

---

### 5.4 `alloc_vop(VOP_dev& VP)` / `free_vop(VOP_dev& VP)`

#### 功能
为 VOP 模块分配和释放显存。

#### 说明
这里分配的是：

- `Vh`
- `diss_rate`
- `S_latent`
- `new_fluid_flag`

#### 调试重点
如果出现“水合物完全不分解”或“节点翻转不发生”：

- 查 `VP.diss_rate` 是否有值
- 查 `VP.Vh` 是否在初始化时被置对
- 查 `new_fluid_flag` 是否正确清零和写入

---

### 5.5 `get_omega_T(unsigned char mat)`

#### 功能
根据材料编号返回热场弛豫率 `omegaT`。

#### 材料逻辑
- `mat == 0`：流体
- `mat == 1`：石英/固体
- `mat == 2`：水合物

#### 方法
先计算材料对应热扩散率 `alpha`，再用：

$$
\omega_T = \frac{1}{0.5 + \alpha / c_s^2}
$$

#### 为什么这样做
这就是共轭热传递的核心做法之一：不同材料用不同弛豫率，界面热通量连续性由空间变系数自然体现。

#### 调试重点
若界面温度不连续，先查：

- `d_wall_mat` 是否正确
- 水合物和石英的 `lambda` / `rhocp` 是否被正确上传
- `get_omega_T()` 是否按材料编号走到了预期分支

---

### 5.6 `kernel_init_thermal(double* h_in, double* T, const int* pointsflag)`

#### 功能
初始化热场。

#### 做法
对所有节点：

- `T[s] = T0_init`
- `h_in[k] = w5[k] * T0_init`

#### 数值意义
这是把温度场初始化成均匀状态，并把 D2Q5 分布函数设成对应的平衡态。

#### 调试重点
如果初场不对：

- 看是否调用了 `init_thermal_field()`
- 看 `T0_init` 是否为你预期值
- 看 `h_in` 是否被后续边界核覆盖

---

### 5.7 `kernel_update_T(const double* h_in, double* T, const int* pointsflag)`

#### 功能
从分布函数重建宏观温度 `T`。

#### 方法
直接求和：

$$
T = \sum_k h_k
$$

#### 说明
这是 D2Q5 标量模型的宏观量恢复步骤。

#### 调试重点
如果 `T` 不对：

- 先看 `h_in` 是否被正确流动和碰撞
- 再看是否有边界分布被写坏

---

### 5.8 `kernel_collide_thermal(...)`

#### 功能
热场 MRT 碰撞。

#### 做法
它做了四件事：

1. 从 `h_in` 变换到矩空间 `m`
2. 构造平衡矩 `m_eq`
3. 用材料对应的 `omegaT` 进行弛豫
4. 变回 `h_out`

#### 关键点
- `m0` 对应温度守恒量
- `m1`、`m2` 和对流有关
- `m3`、`m4` 用快速弛豫简化实现
- `S_latent` 只加到守恒通道上

#### 为什么这样做
MRT 比 BGK 更稳定，尤其是你这种多物理耦合、边界复杂、材料多变的场景。

#### 调试重点
如果热场发散或梯度怪：

- 查 `omegaT` 是否过于接近 2 或 0
- 查 `S_latent` 是否异常大
- 查 `ux_mix`, `uy_mix` 是否本身就有问题

---

### 5.9 `kernel_stream_thermal(...)`

#### 功能
热场流步，把 `h_out` 按离散速度搬运到 `h_in`。

#### 方法
标准拉格朗日流动：

$$
 h_k(x,t+1) = h_k^{out}(x-e_k,t)
$$

#### 调试重点
如果边界层乱：

- 查周期包裹是否符合你的模型假设
- 查 ghost 层是否在后续边界核中被覆盖

---

### 5.10 `kernel_boundary_thermal(...)`

#### 功能
热场边界处理。

#### 处理逻辑
1. `y==0` 且 ghost：入口定温 `T = T_inlet`
2. `y==NY-1` 且 ghost：出口全展开
3. `fl==-1` 且 `mat>0`：固体/水合物 ghost 全反弹

#### 这说明什么
热边界不是统一的，而是按节点角色区分：

- 入口是 Dirichlet
- 出口是 Neumann / outflow
- 材料内部界面由反弹和弛豫共同承担共轭效应

#### 调试重点
如果温度场在固-流界面失真，优先看这里。

---

### 5.11 `init_thermal_field(Therm_dev& TH, const int* pointsflag)`

#### 功能
热场宿主封装：调用初始化核并同步。

#### 方法
- 启动 `kernel_init_thermal`
- 检查 `cudaGetLastError`
- `cudaDeviceSynchronize`

#### 调试重点
如果程序在初始化阶段就出错，通常是这里之前的显存、常量上传或网格配置问题。

---

### 5.12 `step_thermal(...)`

#### 功能
热场单步推进。

#### 顺序
1. `kernel_update_T`
2. `kernel_collide_thermal`
3. `kernel_stream_thermal`
4. `kernel_boundary_thermal`

#### 为什么这个顺序重要
这是典型的“先宏观更新，再碰撞，再流动，再边界”流程。

#### 调试重点
如果只改一个地方，优先检查：

- 边界是否被正确覆盖
- `S_latent` 是否在正确时刻传入
- 温度更新是在碰撞前，而不是碰撞后

---

### 5.13 `kernel_compute_latent_heat(...)`

#### 功能
从分解速率生成潜热源项 `S_latent`。

#### 方法
它不是直接在水合物节点上加热，而是：

- 遍历流体节点
- 检查邻居是否是水合物 ghost
- 把相邻水合物 ghost 的分解速率累加成流体侧热源

#### 物理意义
水合物分解吸热，所以流体热场里应出现负源项。

#### 调试重点
如果分解了但温度没降：

- 看 `diss_rate` 是否非零
- 看流体节点是否真的邻接到了水合物 ghost
- 看 `d_latent_H_latt` 是否太小

---

### 5.14 `compute_latent_heat_source(VOP_dev& VP, const Conc_dev& CN, const Therm_dev& TH, const int* pointsflag)`

#### 功能
潜热源项的宿主封装。

#### 说明
它当前只负责发射核函数并检查错误，真正计算在 `kernel_compute_latent_heat()`。

#### 调试重点
如果热场没有响应相变，优先追 `VP.diss_rate -> VP.S_latent` 这条链。

---

### 5.15 `kernel_update_Cm(const double* g_in, double* Cm, const int* pointsflag)`

#### 功能
从浓度分布函数恢复宏观浓度 `Cm`。

#### 方法
直接求和：

$$
C_m = \sum_k g_k
$$

#### 调试重点
如果浓度一直是零：

- 查 `g_in` 是否被初始化
- 查反应边界是否写入了 `g_in`

---

### 5.16 `kernel_init_conc(double* g_in, double* Cm, const int* pointsflag)`

#### 功能
初始化浓度场为均匀值 `Cm_init`。

#### 方法
- `Cm[s] = Cm_init`
- `g_in[k] = w5[k] * Cm_init`

#### 调试重点
如果浓度初值不对，先查 `Cm_init` 和初始化调用链。

---

### 5.17 `kernel_collide_conc(...)`

#### 功能
浓度场 MRT 碰撞，并在气-水界面施加 CST 源项。

#### 方法
它和热场碰撞结构类似，但多了一个反应驱动分支：

1. 构造 `m` 和 `m_eq`
2. 在 `fl == 1` 且气-水界面判断成立时，向 `m0` 注入 Henry 平衡修正
3. 其余矩量按扩散率弛豫

#### 界面判据
它用的是：

$$
\frac{\rho_B}{\rho_A + \rho_B} > 0.3
$$

说明这个节点被视为界面附近。

#### 调试重点
如果浓度场不增长或者增长位置不对：

- 查界面判据是否把节点筛错了
- 查 `T[s]` 是否合理，因为平衡浓度依赖温度
- 查 `d_Henry_KH`, `d_e1_peq`, `d_e2_peq` 是否正确

---

### 5.18 `kernel_stream_conc(...)`

#### 功能
浓度场流步。

#### 方法
和热场流步一样，把 `g_out` 搬运到 `g_in`。

#### 调试重点
如果反应边界刚写进去的值下一步就没了，查流步和边界覆盖顺序。

---

### 5.19 `kernel_boundary_conc_reaction(...)`

#### 功能
这是浓度场里最重要的边界核。

#### 它处理三类情况
1. `y==0` 入口 ghost：固定 `Cm = Cm_init`
2. `y==NY-1` 出口 ghost：全展开
3. 水合物 ghost：Kang 反应边界

#### Kang 反应边界的意思
对水合物 ghost 节点，浓度不再用普通反弹，而是根据邻近流体浓度和反应动力学修正边界值：

$$
C_{bc} = \frac{D C_{nbr} + k_r C_{sat}}{D + k_r}
$$

同时写入：

$$
\dot{m} = k_r \max(0, 1 - C_{nbr}/C_{sat})
$$

#### 这一步的物理意义
它把“分解速率”从边界条件中自然带出来，这样后续 VOP 才知道该削减多少 `Vh`。

#### 调试重点
这是最常见的错误源之一：

- `pointsflag` 是否真的把水合物边界标成了 `-1`
- `d_wall_mat[s]` 是否等于 2
- 最近流体邻居是否找到了
- `diss_rate[s]` 是否写进去了

如果这里错了，后面的热场和 VOP 都会跟着错。

---

### 5.20 `init_conc_field(Conc_dev& CN, const int* pointsflag)`

#### 功能
浓度场初始化封装。

#### 调试重点
若浓度场完全空白，先确认这个函数确实被 `main.cu` 在水合物模式下调用。

---

### 5.21 `step_conc(...)`

#### 功能
浓度场单步推进。

#### 顺序
1. `kernel_update_Cm`
2. `kernel_collide_conc`
3. `kernel_stream_conc`
4. `kernel_boundary_conc_reaction`

#### 为什么这样排
先更新宏观浓度，再碰撞，再流动，最后写入反应边界和分解速率。

#### 调试重点
如果 `diss_rate` 不更新，通常就是最后一步没执行或界面判定失效。

---

### 5.22 `init_vop(VOP_dev& VP, const int* pointsflag)`

#### 功能
初始化 VOP 相关设备数组。

#### 做法
- `Vh` 清零
- `diss_rate` 清零
- `S_latent` 清零
- `new_fluid_flag` 清零

#### 重要说明
这里的注释写的是“Phase 3 实现：水合物节点 Vh=1，其他=0”，但实际代码只做了清零。也就是说，真正的 `Vh` 初值通常需要结合更上层的几何/初始化流程来理解。

#### 调试重点
这是一个非常值得你注意的点：
- 如果你期待这里直接把水合物体积分数设成 1，但实际没有发生，那不是 GPU 计算错，而是初始化策略要从上层追踪。
- 你要去看 `main.cu`、几何初始化和 `pointsflag` 的来源。

---

### 5.23 `step_hydrate_physics(...)`

#### 功能
这是水合物耦合的总入口。

#### 条件
- 如果 `P.hydrate_enable == false`，直接返回
- 如果 `current_step < P.hydrate_start_step`，也直接返回

#### 耦合顺序

1. `step_conc(...)`
2. `compute_latent_heat_source(...)`
3. `step_thermal(...)`
4. `step_vop(...)`

#### 为什么这是核心函数
因为它把所有水合物物理串成一个闭环：

- 浓度决定分解速率
- 分解速率决定潜热
- 潜热影响温度
- 温度又反过来影响浓度与反应
- 最后 Vh 减小并可能触发翻转

#### 调试重点
如果你要抓耦合 bug，先看这里：

- 这四步是否都执行了
- `current_step` 是否已经达到 `hydrate_start_step`
- `n_conv` 是否合理

---

## 6. 这份文件的数值逻辑可以怎么整体理解

你可以把 `hydrate.cu` 理解成五层：

### 第一层：参数换算
`init_device_variable_hydrate()` 把物理量变成格子量。

### 第二层：热场
`init_thermal_field()`、`step_thermal()` 负责温度传播与潜热反馈。

### 第三层：浓度场
`init_conc_field()`、`step_conc()` 负责溶质扩散、界面反应和分解速率。

### 第四层：潜热
`compute_latent_heat_source()` 把反应速率翻译成热源项。

### 第五层：VOP
`step_vop()` 负责水合物减少、节点翻转和几何重建。

---

## 7. 你查错时建议优先看的顺序

如果结果不对，别一上来就盯最复杂的 VOP，建议按下面顺序查：

1. 先确认 `init_device_variable_hydrate()` 的量纲是否正确
2. 再确认 `step_conc()` 是否真的产生了 `diss_rate`
3. 再看 `compute_latent_heat_source()` 是否把热源写进去了
4. 再看 `step_thermal()` 的温度是否被拉低
5. 最后看 `step_vop()` 有没有把 `Vh` 推到 0

这个顺序最省时间，因为它符合因果链。

---

## 8. 常见故障模式与定位建议

### 8.1 水合物完全不分解
优先检查：

- `hydrate_enable` 是否为真
- `current_step >= hydrate_start_step`
- `kernel_boundary_conc_reaction()` 是否真的命中水合物 ghost
- `diss_rate` 是否始终为零
- `k0_rxn` 是否太小

### 8.2 几步内全部分解
优先检查：

- `k0_latt` 是否过大
- `dx_phys` / `dt_phys` 是否把反应速率放大了
- `Csat` 是否过小
- `Vh` 初始量是否太低

### 8.3 温度场不响应分解
优先检查：

- `VP.S_latent` 是否非零
- `d_latent_H_latt` 是否合理
- `step_thermal()` 是否确实传入了 `VP.S_latent`

### 8.4 VOP 不翻转
优先检查：

- `VP.Vh` 是否真的在下降
- `diss_rate` 是否写到水合物节点
- `d_Vm_latt` 是否正确
- `new_fluid_flag` 是否被置位

### 8.5 界面温度/浓度很噪
优先检查：

- `pointsflag` 和 `d_wall_mat` 是否一致
- 邻居搜索是否找到了正确的流体节点
- 入口/出口边界是否覆盖了你本来想要的界面信号

---

## 9. 如何把这个文件和主程序一起读

如果你要真正搞懂运行路径，建议按这个顺序联读：

1. [lbm_mrt/solver/src/main.cu](../lbm_mrt/solver/src/main.cu)
2. [lbm_mrt/solver/src/sim_utils.cu](../lbm_mrt/solver/src/sim_utils.cu)
3. [lbm_mrt/solver/src/hydrate.cu](../lbm_mrt/solver/src/hydrate.cu)
4. [lbm_mrt/solver/src/hydrate_vop.cu](../lbm_mrt/solver/src/hydrate_vop.cu)
5. [lbm_mrt/solver/include/hydrate.h](../lbm_mrt/solver/include/hydrate.h)

这样你会明白：

- 主程序如何初始化
- 每一步是谁在调用谁
- 哪个函数负责哪个物理量
- 出问题时应该从哪一层下钻

---

## 10. 一句话总结

`hydrate.cu` 的本质不是“一个大函数文件”，而是水合物多物理耦合的中间层：它把参数换算、热传导、浓度扩散、反应速率、潜热回馈和 VOP 翻转这些环节串成了一个可执行的闭环。

如果你理解了这条闭环，你就能快速判断：某个异常是出在物理参数、边界条件、耦合顺序，还是 VOP 重建。
