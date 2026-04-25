# hydrate_vop.cu 科学说明与函数级导读

本文面向 `lbm_mrt/solver/src/hydrate_vop.cu`，目标是把这个文件从“代码块”还原成一条清晰的物理与数值链路：

1. 水合物体积分数如何随分解速率下降
2. 什么时候把水合物节点翻转成流体节点
3. 新翻转节点如何重初始化分布函数与宏观量
4. 为什么翻转后要重建几何、边界和 ghost 层
5. 出问题时应该从哪一步查起

适用范围：水合物编译版 `mcmp_sim_hydrate`，即启用 `-DHYDRATE_ENABLE` 后的求解路径。

---

## 1. 这个文件在整个求解器中的位置

`hydrate_vop.cu` 负责水合物耦合中的“相变后半段”，也就是 VOP（Volume of Phase change / Volume of Pore?，项目中主要体现为水合物体积分数驱动的节点翻转）更新。

它不负责浓度场和热场的计算本身，而是消费前一阶段已经算出来的结果：

- `VP.diss_rate`：来自浓度边界反应的分解速率
- `VP.Vh`：当前水合物体积分数
- `pointsflag`：节点类型
- `d_wall_mat`：材料类型
- `TH`、`CN`、`A`、`B`、`MX`：用于新流体节点的重初始化和几何重建

在全耦合流程里，它通常位于：

1. `step_conc()` 之后，得到 `diss_rate`
2. `compute_latent_heat_source()` 之后，得到潜热源项
3. `step_thermal()` 之后，温度已经更新
4. 最后调用 `step_vop()`，根据 `diss_rate` 推进固相体积分数并判断是否翻转

也就是说，`hydrate_vop.cu` 是“物理状态变化后，修改拓扑和初值”的模块。

---

## 2. 先理解它的数值思想

这个文件的核心方法有三层：

1. 体积分数递减
- 用分解速率 `diss_rate` 驱动 `Vh` 下降。
- 这是最直接的相变演化。

2. 阈值翻转
- 当 `Vh <= 0` 时，把节点从水合物态改成流体态。
- 这一步是离散拓扑变化的开关。

3. 翻转后重初始化与重建
- 用邻域流体状态估计新节点初值。
- 重新构建 `pointsflag`、`d_wall_mat`、boundary、ghost。

这意味着它不是连续方程求解器的一部分，而是“离散几何更新器”。

---

## 3. 条件编译为什么重要

文件顶部有：

```cpp
#ifdef HYDRATE_ENABLE
```

所以：

1. 没有 `-DHYDRATE_ENABLE` 时，这个文件整体不编译。
2. 你在编辑器里看到发暗，通常也是因为宏条件没被语言服务识别。
3. 你已经用 `uv run lbm-build --hydrate` 成功编译，所以实际运行时它是生效的。

这类条件编译不是注释，而是编译模式切换。

---

## 4. 先看它依赖哪些外部对象

### 4.1 `VOP_dev`

这个文件的核心数据容器。

通常包含：

- `Vh`：水合物体积分数
- `diss_rate`：分解速率
- `S_latent`：潜热源项
- `new_fluid_flag`：本步需要翻转的节点

### 4.2 `Therm_dev`

翻转后要把新流体节点的热场分布函数和温度重置到合理值。

### 4.3 `Conc_dev`

翻转后新节点的浓度场也要重初始化，否则会出现数值空洞。

### 4.4 `Fluid_dev A, B`

双相流场的分布函数和宏观密度。

### 4.5 `Mix_dev MX`

提供：

- `pointsflag`
- `ux`, `uy`
- 以及几何和混合场相关信息

### 4.6 `pointsflag` 和 `d_wall_mat`

这是 VOP 逻辑里最敏感的两个全局标记。

- `pointsflag` 决定节点是不是流体、边界、ghost、固体、水合物
- `d_wall_mat` 决定这个节点属于哪种材料

如果这两个图谱不对，后面所有翻转、边界和重初始化都会偏。

---

## 5. 函数级说明

下面按 `hydrate_vop.cu` 中函数出现顺序解释。

---

### 5.1 `feq_vop(int k, double rho, const double u[2])`

#### 功能
计算 D2Q9 速度组上的平衡分布函数。

#### 方法
它使用的是标准二阶平衡态形式：

$$
 f_k^{eq} = w_k \rho \left(1 + \frac{e_k \cdot u}{c_s^2} + \frac{(e_k \cdot u)^2}{2c_s^4} - \frac{u^2}{2c_s^2}\right)
$$

这里的 `cs2 = 1/3`，与 D2Q9 伪势流场一致。

#### 为什么单独写在这里
因为这个文件要给“新翻转的流体节点”重设流场分布函数，而又不想跨文件调用 `LBM.cu` 里的设备内联函数，避免链接和可见性问题。

#### 调试重点
如果新节点翻转后流场异常：

- 查这个函数是否与 `LBM.cu` 中的平衡态定义一致
- 查 `rho` 和 `u` 的初值是否合理

---

### 5.2 `kernel_update_vop(...)`

#### 功能
更新水合物体积分数 `Vh`，并标记需要翻转的节点。

#### 输入
- `Vh`：当前水合物体积分数
- `new_fluid_flag`：标记数组
- `diss_rate`：分解速率
- `pointsflag`：节点类型

#### 方法
只处理 `pointsflag[s] == -3` 的水合物内部节点：

$$
V_h^{n+1} = V_h^n - V_{m,latt} \cdot \dot{m}
$$

如果更新后 `Vh <= 0`：

- `Vh = 0`
- `new_fluid_flag = 1`

#### 为什么这样做
这是把连续分解过程离散化成“体积分数递减 + 阈值触发翻转”的实现。

#### 调试重点
如果你发现水合物不消失，先看这里：

- `pointsflag` 是否真的把水合物内部标成 `-3`
- `diss_rate` 是否非零
- `d_Vm_latt` 是否被正确上传

#### 常见问题
1. `diss_rate` 始终为零，`Vh` 不变
2. `Vh` 更新太快，几个步就全没了
3. 节点类型没标对，导致根本没进入该核函数分支

---

### 5.3 `kernel_apply_vop_conversion(...)`

#### 功能
把已经标记为翻转的节点真正改成流体节点。

#### 方法
对于 `new_fluid_flag[s] == 1` 的节点：

- `pointsflag[s] = 1`
- `d_wall_mat[s] = 0`
- `atomicAdd(n_conv, 1)` 统计翻转数

#### 为什么要分两步
这很重要：

1. `kernel_update_vop()` 负责判断“该不该翻转”
2. `kernel_apply_vop_conversion()` 负责“真正改几何状态”

这样做的好处是，更新物理量和修改拓扑被分离，逻辑更清楚。

#### 调试重点
如果 `Vh` 已经到 0，但节点没变成流体：

- 看 `new_fluid_flag` 是否写成功
- 看 `atomicAdd` 后的 `n_conv` 是否为 0
- 看 `pointsflag` 是否被其他几何重建逻辑覆盖

---

### 5.4 `kernel_reinit_new_fluid(...)`

#### 功能
对刚刚翻转成流体的节点重新初始化全部分布函数和宏观场。

#### 输入输出内容
它同时接收和写入：

- `fin_A`, `fout_A`, `rho_A`
- `fin_B`, `fout_B`, `rho_B`
- `h_in`, `T`
- `g_in`, `Cm`

也就是说，它一次性把双相流、热场和浓度场都补齐。

#### 方法
先在 D2Q9 邻域中找周围可用节点，只统计：

- `fn == 1`：流体
- `fn == 0`：边界

然后对邻居场量做平均：

- `rho_A` 平均
- `rho_B` 平均
- `T` 平均
- `Cm` 平均

如果邻居数为 0，则使用保底值：

- `rA_new = 0.05`
- `rB_new = 0.05`
- `T_new = d_T0_init`
- `Cm_new = d_Cm_init`

再做：

- `fin_A / fout_A = feq_vop(rA_new, u=0)`
- `fin_B / fout_B = feq_vop(rB_new, u=0)`
- `h_in = w5 * T_new`
- `g_in = w5 * Cm_new`

#### 为什么这样做
翻转节点本质上是一个“新生成的流体点”，它不能空着，必须立刻继承邻居状态，否则流场、热场和浓度场都会出现数值空洞。

#### 调试重点
如果翻转后局部出现尖峰或空洞：

- 查邻居统计是否正确
- 查新节点平均值是否过于离谱
- 查 `pointsflag` 的邻域是否已经被几何重建改变

#### 这是最容易出 bug 的地方之一
因为它同时处理双相流、热场、浓度场三套场，任何一套初始化出错都会造成后续不稳定。

---

### 5.5 `kernel_clear_new_fluid_flag(int* new_fluid_flag)`

#### 功能
把本步的翻转标记清零，为下一步准备。

#### 为什么需要它
因为 `new_fluid_flag` 是“一步性的事件标记”，不是长期状态。如果不清零，节点会在后续每步都被当成新翻转节点，造成重复重初始化。

#### 调试重点
如果你发现某些节点每步都像“刚翻转一样”重新初始化，优先查这个核函数是否执行，或者 `new_fluid_flag` 是否被其他路径污染。

---

### 5.6 `step_vop(VOP_dev& VP, Therm_dev& TH, Conc_dev& CN, Fluid_dev& A, Fluid_dev& B, Mix_dev& MX)`

#### 功能
这是 VOP 模块的总入口，负责完整的一次水合物节点更新。

#### 顺序
1. `kernel_update_vop()` 更新 `Vh` 并标记翻转
2. `kernel_apply_vop_conversion()` 执行翻转并统计数量
3. 如果本步翻转数大于 0：
   - `kernel_reinit_new_fluid()` 重初始化新流体节点
   - `init_wall_mat_from_flag()` 重新构建材料图
   - `mark_boundary()` 重新标注边界
   - `mark_ghost()` 重新标注 ghost
4. `kernel_clear_new_fluid_flag()` 清掉标记
5. 返回 `n_conv`

#### 这个顺序为什么关键
VOP 的本质不是单纯更新一个标量，而是带着几何变化的状态演化。顺序错了会出大问题：

- 先重建再更新，可能把翻转节点漏掉
- 先清标记再重初始化，会让新节点丢失识别
- 不重建 ghost，后续热场和浓度场的边界会错

#### 调试重点
如果你只想抓 VOP bug，优先看这个函数：

- `n_conv` 是否正常
- 翻转后是否真的触发了几何重建
- `printf("[VOP] 本步翻转 ...")` 是否出现

#### 常见问题
1. `n_conv > 0` 但后续边界没变，说明几何重建链条有问题
2. `n_conv == 0` 但理论上应翻转，说明 `kernel_update_vop()` 或 `diss_rate` 有问题
3. `step_vop()` 在逻辑上正确，但主循环没调用它，说明上层调度错了

---

## 6. 这个文件的整体物理逻辑可以怎么理解

你可以把 `hydrate_vop.cu` 理解为“相变后处理器”：

### 第一步：消耗分解速率
从浓度边界反应得到 `diss_rate`。

### 第二步：削减体积分数
用 `diss_rate` 推动 `Vh` 下降。

### 第三步：触发离散翻转
当 `Vh <= 0`，把水合物节点改成流体节点。

### 第四步：补全新流体状态
给新节点赋予合理的流场、热场、浓度场初值。

### 第五步：重建几何拓扑
重新划分边界和 ghost，确保下一步热-质-流耦合继续正确工作。

---

## 7. 查错时建议优先看的顺序

如果 VOP 结果不对，按下面顺序查，最快：

1. `step_conc()` 里有没有产生 `diss_rate`
2. `kernel_update_vop()` 有没有把 `Vh` 推到 0
3. `kernel_apply_vop_conversion()` 有没有真正把 `pointsflag` 改成 1
4. `kernel_reinit_new_fluid()` 是否把新节点补齐
5. `init_wall_mat_from_flag()`、`mark_boundary()`、`mark_ghost()` 是否重建成功

这个顺序符合因果链，不要反过来查。

---

## 8. 常见故障模式与定位建议

### 8.1 水合物体积分数一直不变
优先检查：

- `diss_rate` 是否始终为 0
- `pointsflag` 是否没有水合物内部节点 `-3`
- `d_Vm_latt` 是否被正确上传

### 8.2 节点已经翻转但场量发散
优先检查：

- `kernel_reinit_new_fluid()` 的邻居平均是否异常
- `rho_A`、`rho_B`、`T`、`Cm` 是否用了保底值
- 翻转节点周围是否已经没有足够的流体邻居

### 8.3 翻转后边界失效
优先检查：

- `init_wall_mat_from_flag()` 是否执行
- `mark_boundary()` 是否执行
- `mark_ghost()` 是否执行
- `pointsflag` 是否被错误覆盖

### 8.4 同一个节点反复翻转
优先检查：

- `kernel_clear_new_fluid_flag()` 是否成功清零
- `pointsflag` 是否没有被永久更新
- 是否外部还有别的路径在重置节点状态

### 8.5 VOP 翻转数总是 0，但理论上应该翻转
优先检查：

- `Vh` 的初值是否正确
- `diss_rate` 是否传到了水合物节点
- 是否 `pointsflag[s] != -3`，导致 `kernel_update_vop()` 直接跳过

---

## 9. 它和 `hydrate.cu` 的关系

这两个文件是互补的：

- `hydrate.cu` 负责“热场 + 浓度场 + 潜热 + 耦合入口”
- `hydrate_vop.cu` 负责“体积分数变化 + 节点翻转 + 几何重建”

如果只看其中一个，你会觉得水合物模型是不完整的；两个一起看，才是完整的相变闭环。

---

## 10. 如何与主程序一起读

如果你要真正搞懂运行路径，建议这样联读：

1. [lbm_mrt/solver/src/main.cu](../lbm_mrt/solver/src/main.cu)
2. [lbm_mrt/solver/src/sim_utils.cu](../lbm_mrt/solver/src/sim_utils.cu)
3. [lbm_mrt/solver/src/hydrate.cu](../lbm_mrt/solver/src/hydrate.cu)
4. [lbm_mrt/solver/src/hydrate_vop.cu](../lbm_mrt/solver/src/hydrate_vop.cu)
5. [lbm_mrt/solver/include/hydrate.h](../lbm_mrt/solver/include/hydrate.h)

读完这五个文件后，你应该能回答三个问题：

- 水合物是怎么分解的
- 分解之后怎么改变几何
- 新生成流体点怎么继续参与后续耦合

---

## 11. 一句话总结

`hydrate_vop.cu` 的本质是一个“离散拓扑更新器”：它把由反应驱动的水合物体积分数衰减，转换成节点翻转、场量重初始化和几何重建，从而让水合物分解过程可以在 LBM 框架里持续演化。

如果你理解了这条链路，你就能更快判断：某个异常是出在分解速率、翻转条件、重初始化，还是几何重建。
