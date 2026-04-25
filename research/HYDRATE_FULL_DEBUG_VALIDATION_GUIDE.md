# 水合物版完整理解、运行、验证与调试指南

作者：GitHub Copilot（GPT-5.3-Codex）
日期：2026-04-23
适用仓库：lbm_mrt

---

## 0. 这份指南解决什么问题

你的目标是对“水合物版（热场 + 浓度场 + 相变）”做到三件事：

1. 完整理解：知道代码入口、耦合顺序、关键参数如何流到 CUDA 内核。
2. 稳定运行：可以稳定复现实验算例与基准算例。
3. 可验证可调试：出现异常时，能定位是热场、浓度场、VOP 还是耦合顺序问题。

这份文档按“代码认知路径 + 操作步骤 + 验证标准 + 故障排查”给出可执行流程。

---

## 1. 当前代码结构评价

### 1.1 主要优点

1. Python 编排与 CUDA 求解解耦清晰。
- Python 层负责参数、路径、批量与日志。
- CUDA 层负责数值推进与并行内核。

2. 配置体系很成熟。
- YAML -> 扁平参数 -> params.txt -> C++ RuntimeParams 的链路明确。
- hydrate.yaml 作为覆盖层，便于做水合物专项实验。

3. 编译开关设计合理。
- 通过 -DHYDRATE_ENABLE 构建双版本二进制，避免污染流场基线。
- mcmp_sim 与 mcmp_sim_hydrate 并存，方便 A/B 对照。

4. 运行器与结果落盘规范。
- 单算例、批处理、检查点、日志、VTK 输出都形成标准流程。
- 对于“重跑、复查、追溯”非常友好。

5. 水合物耦合主链已经接上，不是纯占位。
- 主循环中有真实的 Conc -> LatentHeat -> Thermal -> VOP 顺序。
- 并且带有 VOP 翻转后的 ghost 重建和润湿性重上传。

### 1.2 需要重点注意的事项

1. 单元测试主要覆盖 Python 层。
- 当前 pytest 几乎不覆盖 CUDA 水合物物理正确性。
- 你后续“完整求证”应以基准测试 + 结果统计为主，不要误以为 pytest 通过就代表物理正确。

2. hydrate.cu 文件顶部注释含“桩/后续填充”历史文字。
- 但实际代码里多处已实现（尤其浓度场、潜热、耦合入口）。
- 调试时要以函数体实现为准，不要只看早期注释。

3. 热-质-相变耦合是强耦合问题，对参数尺度敏感。
- dx_phys、dt_phys、k0_rxn、Ea_rxn、Henry_KH、Vm_hydrate 会直接影响稳定性和趋势。
- 建议先做分阶段验证，再做全耦合长程运行。

4. 输出与检查点有“流场主状态”和“水合物附加状态”两条线。
- 检查点主要保存流场主状态。
- 水合物字段验证要依赖 VTK 附加标量和日志中的水合物诊断量。

---

## 2. 你必须先建立的代码心智图

从入口到内核的推荐理解顺序：

1. 编译入口
- lbm_mrt/solver/build.py
- 看清 hydrate 版本如何加入 hydrate.cu 与 hydrate_vop.cu，以及 -DHYDRATE_ENABLE 如何注入。

2. 运行入口
- lbm_mrt/runners/single_run.py
- 理解 params.txt 生成、LBM_FILE_DIR 与 LBM_CKPT_DIR 的传递。

3. 参数总线
- lbm_mrt/core/config.py
- 从 configs/default.yaml 与 configs/hydrate.yaml 到扁平键值的映射。

4. C++ 主程序
- lbm_mrt/solver/src/main.cu
- 看 hydrate 模式下：初始化热/浓/VOP场，eq 阶段 + flow 阶段切换。

5. 耦合主循环
- lbm_mrt/solver/src/sim_utils.cu
- 重点看 run_stage_hydrate 的每步顺序和输出逻辑。

6. 三个核心物理模块
- lbm_mrt/solver/src/hydrate.cu：热场、浓度场、潜热源、耦合入口。
- lbm_mrt/solver/src/hydrate_vop.cu：VOP 更新、节点翻转、重初始化、ghost 重建。
- lbm_mrt/solver/include/hydrate.h：数据结构、常量与函数声明。

---

## 3. 完整执行路线图（建议按阶段推进）

## 阶段 A：环境与编译可重复

目标：任何时候都能稳定重建可运行二进制。

执行：

1. 激活环境

```bash
source .venv/bin/activate
```

2. 检查命令入口

```bash
uv run lbm-build --help
uv run lbm-run --help
uv run lbm-batch --help
```

3. 编译水合物版

```bash
uv run lbm-build --hydrate
```

4. 检查产物

```bash
ls -lh lbm_mrt/solver/mcmp_sim_hydrate
```

通过标准：
- mcmp_sim_hydrate 存在且可执行。
- 编译日志中包含 -DHYDRATE_ENABLE。

---

## 阶段 B：最小可运行烟雾测试

目标：先跑通，再谈对错。

执行（先用一个最小几何案例）：

```bash
uv run lbm-run \
  --case-name hydrate_smoke_01 \
  --geom data/geometry/case0001_pf/geometry_case0001.plt \
  --config configs/hydrate.yaml \
  --app lbm_mrt/solver/mcmp_sim_hydrate \
  flow_max_steps=20000 OUTPUT_EVERY=2000
```

检查：

```bash
ls results/hydrate_smoke_01
ls results/hydrate_smoke_01/*.vtk | head
tail -n 80 results/hydrate_smoke_01/log.txt
```

通过标准：
- 进程返回码为 0。
- 有 outputdata_flow*.vtk。
- 日志出现 hydrate 初始化信息，且无明显 NaN/崩溃。

---

## 阶段 C：分物理验证（不要直接上全耦合长算）

目标：把问题切成可定位的小块。

建议顺序：

1. BM1 先验热扩散
- 验证温度场离散与边界是否正确。

```bash
uv run python scripts/03_validate_benchmarks.py --bm BM1 --dir results/hydrate_smoke_01
```

2. BM2 共轭热传
- 验证流固/水合物材料热参数与界面导热比。

```bash
uv run python scripts/03_validate_benchmarks.py --bm BM2 --dir results/hydrate_smoke_01
```

3. BM3 反应扩散
- 验证浓度场 + 反应边界项量级是否合理。

```bash
uv run python scripts/03_validate_benchmarks.py --bm BM3 --dir results/hydrate_smoke_01
```

4. BM4 VOP 质量守恒
- 验证 Vh 与 Cm 组合质量漂移。

```bash
uv run python scripts/03_validate_benchmarks.py --bm BM4 --dir results/hydrate_smoke_01
```

5. BM5 全耦合趋势
- 仅在前四步不失真后做趋势验证。

```bash
uv run python scripts/03_validate_benchmarks.py --bm BM5 --dir results/hydrate_smoke_01
```

通过标准：
- BM1/BM2/BM3/BM4/BM5 至少达到脚本给出的 PASS 或“可解释偏差”。

---

## 阶段 D：全耦合求证（你的核心目标）

目标：得到可信的水合物分解过程证据链。

推荐做法：

1. 固定一个标准 case 作为主求证案例。
- 不要一开始批量扫描参数。

2. 固定随机种子与关键参数。
- 保证可复现。

3. 输出固定频率快照。
- 建议 OUTPUT_EVERY 先小后大。

4. 建立三组对照实验：
- 组 A：hydrate_enable=true（完整耦合）
- 组 B：hydrate_start_step 推迟（验证耦合触发效应）
- 组 C：调低 k0_rxn 或关闭部分驱动（做敏感性）

5. 提取并记录核心诊断量：
- 平均 Vh、总 diss_rate、平均 T、平均 Cm、节点翻转累计数。
- 与 run_summary、log、VTK 三方互相交叉验证。

---

## 4. 调试时的定位框架（非常重要）

当结果异常时，用下面顺序排查：

1. 编译态检查
- 是否真的用 mcmp_sim_hydrate 运行。
- log 中是否有 hydrate 常量上传打印。

2. 配置检查
- params.txt 中 hydrate_enable、dx_phys、dt_phys、k0_rxn 等是否是预期值。
- 是否误用了 default.yaml 导致 hydrate 关闭。

3. 物理模块隔离
- 先看温度场是否合理，再看浓度场，再看 VOP。
- 不要在三者全开时盲目调参。

4. 量纲与尺度检查
- 重点核对 D_latt、alpha_latt、k0_latt、Vm_latt 数量级。
- 出现“几步内全部分解”或“完全不分解”，通常是尺度问题。

5. 边界与几何检查
- 看入口出口 ghost 行为是否符合预期。
- 看 VOP 翻转后是否重建 ghost 成功（日志中翻转计数是否连续合理）。

---

## 5. 你可以直接复制执行的“完整求证清单”

1. 编译

```bash
uv run lbm-build --hydrate
```

2. 跑主案例

```bash
uv run lbm-run \
  --case-name hydrate_verify_main \
  --geom data/geometry/case0001_pf/geometry_case0001.plt \
  --config configs/hydrate.yaml \
  --app lbm_mrt/solver/mcmp_sim_hydrate \
  flow_max_steps=120000 OUTPUT_EVERY=5000
```

3. 跑全基准验证

```bash
uv run python scripts/03_validate_benchmarks.py --all --dir results/hydrate_verify_main
```

4. 收集证据文件

```bash
ls results/hydrate_verify_main > results/hydrate_verify_main/_manifest.txt
cp results/hydrate_verify_main/log.txt results/hydrate_verify_main/_log_snapshot.txt
```

5. 记录本次关键参数

```bash
cp results/hydrate_verify_main/params.txt results/hydrate_verify_main/_params_snapshot.txt
```

---

## 6. 建议你下一步的工作节奏

第 1 天：阶段 A + 阶段 B，保证稳定可运行。
第 2 天：阶段 C（BM1-BM3），打通热场和浓度场。
第 3 天：阶段 C（BM4-BM5），验证 VOP 和全耦合趋势。
第 4 天：阶段 D 对照实验 + 参数敏感性。
第 5 天：整理证据链，形成“可复现实验报告”。

---

## 7. 结论

你现在这套代码已经具备“工程化运行 + 水合物耦合主链 + 自动化基准验证”的基础能力，结构上是可持续迭代的。

要完成你说的“完整调试和求证”，关键不是再堆功能，而是严格按本指南分阶段做：
先跑通、再分物理验证、最后做全耦合对照与证据归档。
