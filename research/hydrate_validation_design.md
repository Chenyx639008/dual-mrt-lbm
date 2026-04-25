# hydrate.yaml 参数转换与水合物验证设计

本文说明如何验证 `configs/hydrate.yaml` 的参数转换链路，以及如何把这条链路和 `hydrate.cu` / `hydrate_vop.cu` 的物理功能对应起来。

目标有三个：

1. 验证 `hydrate.yaml` 是否只覆盖水合物相关参数，而不破坏默认参数。
2. 验证 YAML 中的嵌套键是否正确转换为 `load_config()` 返回的扁平参数字典。
3. 验证这些参数在 `hydrate.cu` 中被换算成格子单位后，数量级和公式都是合理的。

---

## 1. 参数转换链路

这套工程里，`hydrate.yaml` 并不是直接给 CUDA 使用的输入文件，它只是一个覆盖层。真正的链路是：

`configs/hydrate.yaml` → `lbm_mrt.core.config.load_config()` → 扁平 `params` 字典 → `write_params_txt()` → `params.txt` → CUDA 端 `RuntimeParams`

其中：

- `load_config()` 负责把 YAML 的嵌套结构展平成 C++ 需要的键名。
- `write_params_txt()` 负责把扁平字典写成求解器能读的文本。
- `hydrate.cu` 中的 `init_device_variable_hydrate()` 负责把物理量变成格子量。

因此，验证必须分两层做：

1. 配置层验证：YAML → 扁平参数 → params.txt
2. 物理层验证：物理参数 → 格子参数 → 内核使用值

---

## 2. 这个验证应该检查什么

### 2.1 默认参数是否保持不变

`hydrate.yaml` 只应该修改 `hydrate` 分组，不应该改变流场默认值，例如：

- `Sw`
- `tau_p_a`
- `tau_p_b`
- `Gx`
- `drive_mode`
- `ENABLE_CKPT`

如果这些值被意外改动，说明 overlay 逻辑或默认值逻辑有问题。

### 2.2 hydrate 分组是否正确展平

`hydrate.yaml` 的嵌套项在 `load_config()` 后应变成这些扁平键：

- `hydrate_enable`
- `hydrate_start_step`
- `T0_init`
- `T0_inlet`
- `lambda_fluid`
- `lambda_hydrate`
- `lambda_solid`
- `rhocp_fluid`
- `rhocp_hydrate`
- `rhocp_solid`
- `D_mol_water`
- `Henry_KH`
- `Cm_init`
- `k0_rxn`
- `Ea_rxn`
- `e1_peq`
- `e2_peq`
- `latent_heat`
- `Vm_hydrate`
- `Vh_init`
- `vop_terminate_frac`
- `dx_phys`
- `dt_phys`

这一步要确认的是：值没有丢、键名没写错、数值类型没被破坏。

### 2.3 格子单位换算是否正确

`hydrate.cu` 中最关键的换算公式是：

$$
\alpha_{latt} = \frac{\lambda}{\rho c_p} \cdot \frac{\Delta t}{\Delta x^2}
$$

$$
D_{latt} = D_{phys} \cdot \frac{\Delta t}{\Delta x^2}
$$

$$
k_{0,latt} = k_{0,phys} \cdot \frac{\Delta t}{\Delta x}
$$

$$
\frac{E_a}{R} \text{ 直接以 K 存储}
$$

$$
V_{m,latt} = \frac{V_{m,phys}}{\Delta x^3}
$$

$$
\Delta H_{latt} = \frac{\Delta H_{phys}}{(\rho c_p)_{fluid} \cdot \Delta x}
$$

这些公式必须和 `hydrate.cu::init_device_variable_hydrate()` 一致。

---

## 3. 推荐的测试文件设计

建议在 `tests/` 目录下增加专门的验证文件：

[tests/test_hydrate_validation.py](../tests/test_hydrate_validation.py)

它建议包含四类测试：

1. 默认参数保持测试
2. hydrate.yaml 展平映射测试
3. 格子单位换算测试
4. 参数转换图生成测试

其中第 4 类测试会调用 `lbm_mrt.viz.viz_template`，保存一个转换报告图，便于人工查看数量级是否合理。

---

## 4. 可视化应该看什么

可视化不是为了“好看”，而是为了迅速判断参数是否在合理量级。

建议图中分成两部分：

### 4.1 原始物理参数

看这些量是否和实验/文献预期一致：

- `T0_init`
- `T0_inlet`
- `lambda_fluid`
- `lambda_hydrate`
- `lambda_solid`
- `D_mol_water`
- `k0_rxn`
- `Vm_hydrate`

### 4.2 格子/派生量

看这些量是否数量级稳定：

- `alpha_fluid`
- `alpha_hydrate`
- `alpha_solid`
- `D_latt`
- `k0_latt`
- `Ea_over_R`
- `Vm_latt`
- `latent_H_latt`

如果派生量里出现特别离谱的数量级，通常说明 `dx_phys`、`dt_phys`、或者单位换算有问题。

---

## 5. 这个验证和 hydrate.cu / hydrate_vop.cu 的关系

虽然这个测试主要验证的是配置转换，但它直接服务于这两个文件：

### 对 `hydrate.cu`

它验证的就是 `init_device_variable_hydrate()` 依赖的参数链路是否正确。

### 对 `hydrate_vop.cu`

它验证 `Vh_init`、`vop_terminate_frac`、`diss_rate` 相关的输入是否合理。VOP 的行为虽然要靠运行时结果判断，但它的初始参数必须先正确。

换句话说，这个验证不是替代运行测试，而是把运行测试的前置条件先校正好。

---

## 6. 运行建议

先跑配置层验证：

```bash
uv run pytest tests/test_hydrate_validation.py -v
```

如果要只看参数转换图，可以单独运行带图的测试：

```bash
uv run pytest tests/test_hydrate_validation.py -k plot -v
```

测试会在临时目录里保存 `hydrate_conversion_report.png`。

---

## 7. 结论

最稳妥的做法不是直接从全耦合大算例开始，而是先把 `hydrate.yaml` 的参数转换链路验证干净，再去跑 `mcmp_sim_hydrate` 的功能验证和物理验证。

这样做的好处是：

1. 能快速排除配置错误
2. 能把物理错误和参数错误分开
3. 后续看 `hydrate.cu` / `hydrate_vop.cu` 的结果时，证据链更完整
