# 300x300 水合物球案例设计说明

这个案例的目标是给水合物编译版提供一个简单、稳定、可复现的验证场景，用来观察三件事：

1. 水合物如何分解
2. 温度如何响应潜热吸收
3. 节点如何从水合物翻转成流体

---

## 1. 几何定义

- 计算域：`300 x 300`
- 中心点：`(150, 150)`
- 水合物球半径：`50`
- 其余区域：水

Tecplot 几何文件中使用的标记规则是：

- `0.0` = 水/流体
- `0.5` = 水合物

求解器会把 `0.5` 读成水合物节点，把 `0.0` 读成流体节点。

---

## 2. 热边界与初值

为了触发分解，采用“初始低温 + 顶边升温”的方式：

- `T0_init = 278.15 K`
- `T0_inlet = 323.15 K`
- `Sw = 1.0`
- `hydrate_enable = true`
- `hydrate_start_step = 0`

这会形成一个从热边界向域内传播的温度梯度，进而驱动水合物分解。

---

## 3. 结果输出规范

这个案例的标准输出目录是：

`results/hydrate_sphere_300x300_r50_hot/`

推荐保留的文件如下：

- `geometry_case.plt`
- `params.txt`
- `log.txt`
- `case_manifest.json`
- `ckpt/`
- `outputdata_eq*.vtk`
- `outputdata_flow*.vtk`

其中 `case_manifest.json` 是这次新增的规范文件，用来明确记录：

- 几何尺寸与半径
- 热边界温度
- 运行步数
- 实际生成的 VTK 文件列表

---

## 4. 运行方式

直接运行脚本：

```bash
uv run python scripts/04_run_hydrate_sphere_case.py
```

如果想先看计划、不真正开跑：

```bash
uv run python scripts/04_run_hydrate_sphere_case.py --skip-run
```

如果你想调整热边界或输出频率，也可以通过参数覆盖：

```bash
uv run python scripts/04_run_hydrate_sphere_case.py \
  --inlet-temp 333.15 \
  --flow-max-steps 50000 \
  --output-every 5000
```

---

## 5. 你应该重点看什么结果

跑完之后，优先看这些内容：

1. `log.txt`
- 先确认是否真的进入 `hydrate_enable` 路径。
- 再看 `step_vop()` 的翻转统计是否出现。

2. `outputdata_flow*.vtk`
- 看 `Vh` 是否随时间下降。
- 看 `T` 是否在热边界下升高。
- 看 `diss_rate` 是否从水合物球外缘开始出现。

3. `case_manifest.json`
- 用来确认结果文件是否齐全，且是否按标准命名保存。

---

## 6. 为什么这个案例适合入门验证

这个场景只有一个中心球，没有复杂孔隙网络和多颗粒边界，所以更容易把现象分开：

- 若温度不升，问题更可能在热边界或热场初始化。
- 若浓度场不动，问题更可能在反应边界或扩散系数。
- 若 `Vh` 不下降，问题更可能在 `diss_rate` 到 `step_vop()` 的链路。

因此它适合用作你后续更复杂几何前的基准案例。
