# Validation Code Structure & Workflow

> Huang & Wu (2016) SCMP solver — full validation suite
> Date: 2026-05-22

---

## 1. Overview

The validation suite verifies four physical benchmarks for the SCMP (single-component multiphase) solver `mcmp_huang_256`:

| # | Validation | Paper Reference | Type |
|---|-----------|----------------|------|
| 1 | Laplace Law | Fig. 5 (scatter) | Static droplet, multi-$\varepsilon$ + multi-$R$ |
| 2 | $\sigma$-Decoupling ($k_2 \neq 0$) | Fig. 5 (body) | Static droplet, $\varepsilon$=1,2 with 3 $k_1/k_2$ combos each |
| 3 | Coexistence Curve | Fig. 4 | Flat interface, $T_r \in [0.60, 0.95]$ |
| 4 | Poiseuille Flow | — | Single-phase channel, bounce-back walls, body force $G_x$ |

All output goes to `results/20260521/`.

---

## 2. File Map

```
lbm_mrt/
├── solver/
│   ├── src/LBM.cu              # SCMP collision, streaming, BCs, init
│   └── src/sim_utils.cu         # param loading, device constant push
├── validation/
│   ├── analytical.py            # detect_interface_radius, compute_sigma_from_rho, extract_rho_l_g
│   ├── cs_eos.py                # Carnahan-Starling EOS, Maxwell coexistence, critical point
│   ├── coexistence.py           # Flat-interface profile extraction, Maxwell comparison
│   ├── poiseuille_sp.py         # poiseuille_analytical, extract_centerline_ux
│   ├── decoupling_sweep.py      # k₁-sweep analysis (used as reference)
│   └── laplace_law.py           # Laplace fit (used as reference)
├── io/
│   └── vtk_reader.py            # read_vtk_scalars, latest_vtk
├── viz/
│   └── viz_template.py          # Unified journal/thesis plotting template
├── core/
│   ├── config.py                # YAML → flat params dict
│   └── paths.py                 # PROJ_ROOT, DATA_DIR, RESULTS_DIR, DEFAULT_BINARY
└── runners/
    ├── single_run.py            # run_one(): spawn solver subprocess
    └── batch_run.py             # run_batch(): iterate design CSV rows

scripts/
├── analyze_laplace_paper.py     # Laplace Law analysis & visualization (run after batch)
├── run_full_validation_suite.py # Decoupling + Coexistence + Poiseuille (gen/run/analyze)
├── 06_run_huang_validation_suite.py  # Legacy: full sweep generator (laplace + decoupling + ...)
└── generate_scmp_design.py      # SCMP design CSV generator

configs/
└── huang_scmp.yaml              # SCMP base config (ε, τ, CS-EOS, viscosity, ...)

data/
├── design_scmp_laplace.csv              # Laplace validation design (Tr × R sweep)
├── design_scmp_decoupling_k2.csv        # σ-decoupling design (k₁/k₂ combos)
├── design_scmp_coexistence_paper.csv    # Coexistence design (Tr sweep, flat interface)
├── design_scmp_poiseuille.csv           # Poiseuille design (Gx sweep, channel mode)

results/20260521/
├── VALIDATION_REPORT.md                 # Summary report
├── laplace_paper_fig5.pdf              # ΔP vs 1/R (5 ε groups)
├── laplace_paper_sigma_vs_1minus6k1.pdf # σ vs (1-6k₁) constancy
├── decoupling_k2_sigma_vs_1minus6k1.pdf # σ vs (1-6k₁) with k₂≠0
├── coexistence_curve.pdf               # Tr vs ρ (log scale)
├── coexistence_deviation.pdf           # Deviation from Maxwell
├── poiseuille_velocity_profile.pdf     # Two-panel (normalized + absolute)
├── laplace_paper_summary.csv           # Per-case Laplace data
├── laplace_paper_fits.csv              # Per-ε linear fit results
├── decoupling_k2_summary.csv           # Decoupling per-case data
├── coexistence_summary.csv             # Coexistence per-Tr data
└── poiseuille_summary.csv              # Poiseuille per-case data
```

---

## 3. Key Scripts & Workflow

### 3.1 `scripts/run_full_validation_suite.py`

**The main orchestrator** for validations 2–4. Three-phase design:

```bash
# Phase 1: Generate design CSVs
uv run python scripts/run_full_validation_suite.py --generate

# Phase 2: Run batch simulations
uv run python scripts/run_full_validation_suite.py --run

# Phase 3: Post-process & visualize
uv run python scripts/run_full_validation_suite.py --analyze

# All-in-one
uv run python scripts/run_full_validation_suite.py --all
```

**Key functions:**

| Function | Purpose |
|----------|---------|
| `generate_decoupling_k2_design()` | Create CSV with $\varepsilon$=1,2 × 3 $k_1/k_2$ combos ($R_0$=64, $T_r$=0.90) |
| `generate_coexistence_design()` | Create CSV with 8 $T_r$ values, `huang_init_mode=2` (flat interface) |
| `generate_poiseuille_design()` | Create CSV with 2 $G_x$ values, `huang_init_mode=3` (channel) |
| `analyze_decoupling_k2()` | Read VTK → detect interface → compute $\sigma$ from $\psi$-integral |
| `analyze_coexistence()` | Read VTK → extract $\rho_l$/$\rho_g$ plateaus → compare to Maxwell |
| `analyze_and_plot_poiseuille()` | Read VTK → extract $u_x(y)$ centerline → compare to analytical |
| `plot_decoupling_k2()` | $\sigma$ vs $(1-6k_1)$, colored by $\varepsilon$, linear fit through origin |
| `plot_coexistence()` | $\rho$ (log) vs $T_r$, Maxwell shaded region + LBM scatter |
| `generate_report()` | Update `VALIDATION_REPORT.md` |

**Design CSV format:** All columns in the CSV that match keys in `configs/huang_scmp.yaml` flattened params are passed as overrides to `params.txt`. Special columns (`case_name`, `app`, `geometry_src`) are metadata.

### 3.2 `scripts/analyze_laplace_paper.py`

Standalone analysis for Laplace Law validation (validation #1). Already run; just re-run for figures:

```bash
uv run python scripts/analyze_laplace_paper.py \
    --results-root results/20260521 \
    --output-dir results/20260521
```

Analysis: reads 20 cases (5 $\varepsilon$ × 4 $R$), computes $\sigma$ via $\psi$-integral, fits Laplace law $\Delta P = \sigma/R$ through origin.

### 3.3 Batch Runner (`lbm_mrt/runners/batch_run.py`)

```bash
uv run lbm-batch \
    --csv data/design_scmp_laplace.csv \
    --config configs/huang_scmp.yaml \
    --out-root results/20260521 \
    --app lbm_mrt/solver/mcmp_huang_256 \
    --resume 0
```

For each CSV row:
1. Override base config with matching CSV columns
2. Write `params.txt`
3. Spawn `mcmp_huang_256` subprocess
4. Solver reads `params.txt`, runs SCMP time loop, writes VTK to `outputdata_scmp/`

### 3.4 Visualization Template (`lbm_mrt/viz/viz_template.py`)

All figures use the unified template:
```python
init_style(mode="journal", base_fontsize=20, bold=True, axis_linewidth=2.0)
fig, ax = create_figure_ax(figsize=(7, 5.5))
# ... plot data ...
set_axis_labels(ax, "$x$-label", "$y$-label", fontsize=20, xlang="en", ylang="en")
set_title(ax, "Title", fontsize=16)
set_legend(ax, loc="best", lang="en", fontsize=11)
set_axis_limits(ax, xlim=(0, None), ylim=(0, None))
format_axes(ax, tick_font="en", tick_labelsize=16)
save_figure(fig, "output.pdf", dpi=300)
```

**Key rules:**
- Labels use LaTeX math mode (`$k_1$`, `$\rho_g$`, `$\sigma$`) for proper subscripts
- `set_legend()` instead of raw `ax.legend()`
- `save_figure()` handles `tight_layout` internally; do NOT call `fig.tight_layout()` before it
- Output: PDF (vector), 300 DPI
- No Chinese characters; all labels in English

---

## 4. Parameter Flow

```
configs/huang_scmp.yaml
        │
        ▼
lbm_mrt/core/config.py::_flatten()
        │  huang_scmp section → flat dict keys:
        │    epsilon_huang, k2_huang, cs_T, cs_a, cs_b, ...
        │    huang_R0, huang_init_mode, tau_huang, Lambda_huang, ...
        │
        ▼
design CSV columns matching flat keys → overrides
        │
        ▼
lbm_mrt/io/params_writer.py::write_params_txt()
        │  "key value" per line
        ▼
params.txt
        │
        ▼
sim_utils.cu::load_params_txt()
        │  get("epsilon_huang", r.epsilon_huang)
        │  k1_computed = -epsilon_huang / 8.0 - k2_huang
        │
        ▼
cudaMemcpyToSymbol → __constant__ device memory
        │
        ▼
LBM.cu kernels read via get_epsilon_huang(), get_k1_huang(), etc.
```

**Important:** The solver computes $k_1$ internally from $\varepsilon$ and $k_2$:
$$k_1 = -\varepsilon/8 - k_2$$
The CSV column `k1_huang` is written but **ignored** by the solver; use `epsilon_huang` and `k2_huang` instead.

---

## 5. Solver Initialization Modes

| `huang_init_mode` | Description | Use Case |
|-------------------|-------------|----------|
| 1 | Droplet (periodic BC) | Laplace, $\sigma$-decoupling |
| 2 | Flat interface (periodic BC) | Coexistence curve |
| 3 | Uniform liquid + channel walls | Poiseuille flow |
| 4 | Droplet on bottom wall | Contact angle (future) |

Mode 3 sets `pointsflag`:
- $y=0$, $y=N_Y-1$: ghost (-1)
- $y=1$, $y=N_Y-2$: bounce-back boundary (0)
- $y \in [2, N_Y-3]$: fluid (1)

---

## 6. Post-Processing Methods

### $\sigma$ Computation (Laplace + Decoupling)

Uses $\psi$-based integral (Huang & Wu 2016 Eq. 62):
$$\sigma = -\frac{G}{6}(1-6k_1) \int \left(\frac{d\psi}{dr}\right)^2 dr$$

Implemented in `lbm_mrt/validation/analytical.py::compute_sigma_from_rho()`.

### Coexistence Density Extraction

`extract_rho_l_g()`: finds interface via max gradient, takes plateau medians away from interface.

### Poiseuille Analysis

`poiseuille_analytical(y, b, Fb, rho, nu)`: $u(y) = F_b (b^2 - y^2) / (2 \rho \nu)$

**Note:** $F_b$ is force **density** ($\rho \cdot G_x$), not acceleration ($G_x$). The solver applies $F_x = G_x \cdot \rho$, so pass $F_b = G_x \cdot \rho_l$ to the analytical function.

---

## 7. Common Issues & Fixes

| Symptom | Cause | Fix |
|---------|-------|-----|
| All cases have same $\sigma$ regardless of $\varepsilon$ | `epsilon_huang` not in design CSV; solver uses default | Add `epsilon_huang` column to CSV |
| Poiseuille R² negative | `poiseuille_analytical` expects force density, got acceleration | Pass `Fb = Gx * rho_l` |
| Figure has Chinese or ? characters | Unicode subscripts in labels (e.g., `₁` U+2081) | Use LaTeX `$k_1$` instead |
| Coexistence plot slow (3+ min) | 200 Maxwell constructions for fine curve | Use 40 points: `linspace(0.56, 0.99, 40)` |
| Scatter hides analytical line | Too many points plotted on top | Subsample (`[::step]`) + set analytical line `zorder` higher |
