# dual-mrt-lbm

双轨 MRT-LBM 框架：CUDA 生产轨 + JAX 镜像验证轨 + Python 统一调度。当前仓库主要面向 SCMP / MCMP 两相流、甲烷水合物分解，以及统一的模型注册、运行和验证流程。

## 当前状态

- 统一 CLI 已可用，入口为 `uv run lbm`。
- 已注册模型可通过 `uv run lbm models` 查看，支持 `info`、`run`、`validate`、`build` 子命令。
- CUDA 生产轨主要在 `lbm_mrt/solver/`，JAX 验证轨在 `jax_lbm/`。
- 配置仍以 `params.txt` 为 Python 和 C++ 之间的唯一边界。
- 现阶段推荐入口模型为 `scmp_cs_huang_256`。

## 快速开始

```bash
uv sync --all-groups
uv run lbm models
uv run lbm info scmp_cs_huang_256
uv run lbm validate scmp_cs_huang_256
uv run lbm run scmp_cs_huang_256 --geom data/geometry/droplet.plt
uv run lbm build --huang-unified
uv run pytest && uv run ruff check . && uv run ruff format .
```

## 统一 CLI

```bash
uv run lbm models
uv run lbm info <model_name>
uv run lbm run <model_name> [--geom PATH] [--steps N] [--case-name NAME] [--output DIR] [KEY=VALUE ...]
uv run lbm validate <model_name>
uv run lbm build --huang-unified
```

常见模型示例：

- `scmp_cs_huang_256`
- `scmp_cs_huang_256_highT`
- `scmp_cs_huang_256_lowT`
- `scmp_cs_huang_256_theta120`
- `scmp_cs_huang_256_theta60`
- `mcmp_pr_baseline`
- `mcmp_pr_wet`

## 目录导航

| 目录 | 说明 |
|------|------|
| `lbm_mrt/solver/` | CUDA C++17 核心（内核、EOS、边界、水合物） |
| `lbm_mrt/unified/` | 统一框架（模型注册、CLI、runner） |
| `jax_lbm/` | 独立的 D2Q9 BGK / Shan-Chen 验证实现 |
| `configs/` | YAML 配置（MCMP 默认、水合物叠加、SCMP） |
| `scripts/` | 工作流脚本与批量脚本 |
| `research/` | 研究说明、框架文档、验证策略 |
| `validation/` | 接触角、网格收敛、表面等验证案例 |
| `data/` | 几何、设计表、benchmark 数据 |
| `results/` | 模拟输出（通常不纳入版本控制） |

## 关键约定

- 配置流为 `YAML -> load_config() -> flat dict -> params.txt -> load_params_txt() -> __constant__` 设备内存。
- `params.txt` 是 Python 到 C++ 的唯一边界，字段名必须和 `RuntimeParams` 完全匹配。
- 旧版 Huang SCMP 二进制的 `NX/NY` 仍是编译期常量；统一构建支持运行时网格覆盖。
- 输出 VTK 为 big-endian legacy binary，读取时应使用仓库内的专用读取器。
- `HYDRATE_ENABLE` 相关逻辑只在水合物构建时启用。

## 常见注意事项

1. Huang SCMP 默认建议使用 `epsilon_huang = 1.7`，不是论文里的理论默认值。
2. 修改网格后需要重新构建对应二进制。
3. `params.txt` 里的 key 如果拼错，不会自动报错，只会被静默忽略。
4. 水合物参数对非水合物二进制不会生效。
5. 生成的 VTK 不能直接按文本文件解析。

## 文档索引

- [框架手册](research/FRAMEWORK_GUIDE.md)
- [研究索引](research/INDEX.md)
- [统一框架可行性评估](research/unified_framework_feasibility.md)
- [CUDA 内核说明](lbm_mrt/solver/CLAUDE.md)
- [配置索引](configs/INDEX.md)

## 开发命令

```bash
uv sync --all-groups
uv run pytest
uv run ruff check .
uv run ruff format .
```
