# JAX LBM 自动化测试套件

> **命令**: `JAX_ENABLE_X64=1 uv run pytest jax_lbm/tests/ -v`
> **测试数**: 48 项 (含 2 项 GPU-only skip)
> **运行时间**: ~70s (CPU)

---

## 测试文件

| 文件 | 项数 | 测试内容 |
|------|:----:|---------|
| `test_streaming.py` | 8 | **Streaming 方向** — 粒子必须沿 c_k 移动 |
| `test_collision.py` | 6 | BGK/MRT 质量守恒 + τ=1 等价性 + MRT 矩阵 + meq |
| `test_force.py` | 7 | SC 力均匀场为零 + Q_m 修正 + 吸附力 |
| `test_eos.py` | 2 | CS-EOS 伪势：液相处非零 + 随温度递减 |
| `test_droplet.py` | 2 | BGK/MRT Guo+C 液滴 200-500 步稳定性 |
| `test_wetting.py` | 3 | ψ-ghost BC 接触角：90°中性/60°亲水/120°疏水 |
| `test_physics.py` | 1 | **Laplace 定律** — ΔP ∝ 1/R |
| `test_integration.py` | 2 | 完整步进函数（吸附+BC 集成） |
| `test_boundary.py` | 7 | 5 种 BC + wall/ghost mask 工具 |
| `test_conservation.py` | 2 | 10000 步均匀场 + 1000 步液滴质量守恒 |
| `test_adjoint.py` | 4 | **伴随梯度** — jax.grad 穿透 200 步 + 有限差分验证 |
| `test_p1a_vmap.py` | 3 | **vmap 批量扫描** — 向量化 vs 串行一致 |
| `test_p4_sharded.py` | 3 | **shard_map 多GPU** — mesh 创建 + 分片步进 (需GPU) |
| `conftest.py` | — | 共享 fixtures (参数/网格/EOS) |

## 运行方式

```bash
# 全量
JAX_ENABLE_X64=1 uv run pytest jax_lbm/tests/ -v

# 单模块
JAX_ENABLE_X64=1 uv run pytest jax_lbm/tests/test_streaming.py -v

# 按关键字
JAX_ENABLE_X64=1 uv run pytest jax_lbm/tests/ -v -k "droplet"
JAX_ENABLE_X64=1 uv run pytest jax_lbm/tests/ -v -k "adjoint"
```

## 开发约定

- 所有测试强制 `jax_enable_x64` (与 CUDA double precision 一致)
- 新增功能必须先写测试
- `conftest.py` 中定义共享参数，避免重复
- GPU-dependent 测试使用 `pytest.skip()` 在 CPU 上优雅跳过
