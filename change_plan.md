# Refactor LBM solver to Huang & Wu (2016) MRT framework + automated validation

## Context

The current 2D MRT-LBM solver (`lbm_mrt/solver/`) implements a Li-style pseudopotential model with surface-tension control entangled in two places: a `compute_C_gpu_A` kernel that builds an anisotropic stress correction from an 8-neighbor ψ-stencil (LBM.cu:1425-1463), and a `sigmaA·|F_m|²/(ψ²·(τ_e−0.5))` term inside the moment-space force `S_A` (LBM.cu:1489-1490). With this design, σ tuning leaks into the EOS — the user can't change interfacial tension without also shifting (ρ_l, ρ_g).

Huang & Wu (2016) resolve this by deriving an extra moment-space term **Q_m** from a third-order Taylor expansion of the discrete force scheme. Their σ-knob `k₁` enters only Q_m, leaving the EOS untouched. The paper's Section 6.2 + Eq. 59-62 show σ ∝ (1−6k₁) with (ρ_l, ρ_g) invariant under k₁ — this is the headline claim we want to reproduce.

The refactor also adds an SCMP path (single-component multiphase, Carnahan-Starling EOS) that the current code lacks entirely — only MCMP with Peng-Robinson (water) + ideal gas (methane) is supported. SCMP lets us reproduce Huang's validation cases directly. Hydrate physics (`-DHYDRATE_ENABLE`) is orthogonal to this work and stays MCMP-only.

Outcome: the SCMP build (`mcmp_huang_256`, 256×256 grid) reproduces Huang's Laplace law, coexistence curve, spurious-currents reduction, and σ-decoupling claim within published tolerances; the legacy MCMP build (`mcmp_sim` / `mcmp_sim_hydrate`) is functionally unchanged.

## Decisions locked with user

| Question | Decision |
|---|---|
| SCMP EOS | Carnahan-Starling only |
| Refactor scope | Replace + new SCMP kernel; no runtime A/B switch between Li & Huang (build-time variant) |
| Validation cases | All four: Laplace, coexistence, spurious currents, σ-decoupling |
| Automation | One-shot suite + separate iterative tuner |

## Architecture summary

- **New build variant** `mcmp_huang_256` via `uv run lbm-build --huang`. Defines `HUANG_256_BUILD`; sets `NX=NY=256`, `NSTEPS=200000`, `NOUTPUT=5000`. Compile-time `#error` if combined with `HYDRATE_ENABLE`.
- **New `pp_mode` runtime param**: `0`=legacy MCMP (PR + ideal gas), `1`=Huang SCMP (Carnahan-Starling). `mcmp_sim` only honors `pp_mode=0`; `mcmp_huang_256` only honors `pp_mode=1`.
- **Surface-tension knob `k1_huang`** (and `k2_huang`, `kd_huang`, `alpha_meq` for completeness) replaces `kappa`/`sigmaA` in the SCMP path.
- **Q_m kernel** (`compute_Q_huang_gpu`) consumes the molecular force `F_mol` (already computed at LBM.cu:1303-1304) — no ψ-stencil — and bakes the relaxation rates `s_e, s_t, s_p` into its output, so the existing collision call `+ C·δt` at LBM.cu:1635 needs **no edit**.
- **Validation suite** (`lbm_mrt/validation/`) is greenfield Python that imports existing infra: `runners/batch_run.py`, `io/vtk_reader.py`, `viz/viz_template.py`, `core/config.py`, `utils/eos.py`.

## Implementation phases

Each phase is independently testable with a smoke test before moving on.

### Phase 1 — Parameter exposure (no behavior change)

Edits, no new logic:

1. `lbm_mrt/solver/include/sim_utils.h` (`RuntimeParams` struct, lines 11-124) — add fields: `int pp_mode=0`, `double k1_huang=1.0/6.0`, `k2_huang=0.0`, `kd_huang=-1.0/12.0`, `alpha_meq=1.0`, `cs_a=1.0`, `cs_b=4.0`, `cs_R=1.0`, `cs_T=0.5`, `cs_G=-1.0`, `huang_R0`, `huang_xc`, `huang_yc`, `huang_W`, `huang_init_mode` (1=droplet, 2=flat).
2. `lbm_mrt/solver/include/LBM.h` (lines 107-150) — add `__constant__` mirrors `d_pp_mode`, `d_k1_huang`, `d_k2_huang`, `d_kd_huang`, `d_alpha_meq`, `d_cs_a/b/R/T/G`.
3. `lbm_mrt/solver/src/sim_utils.cu` — `load_params_txt` (line ~340), `push_device_constants` (line ~600), `print_params_summary` (line ~482) for new fields.
4. `configs/default.yaml` — defaults that preserve legacy behavior (`pp_mode: 0`).

**Smoke test (no recompile of hydrate)**: `uv run lbm-build && uv run python scripts/03_validate_benchmarks.py --all --dir results/smoke_test` — all existing benchmarks pass.

### Phase 2 — CS-EOS kernel + SCMP scaffolding (no σ change yet)

All new code gated by `#ifdef HUANG_256_BUILD`.

1. `lbm_mrt/solver/include/LBM.h` lines 14-18 — wrap NX/NY/NSTEPS/NOUTPUT in `#ifdef HUANG_256_BUILD` with 256/256/200000/5000.
2. `lbm_mrt/solver/include/LBM.h` top — add `#if defined(HUANG_256_BUILD) && defined(HYDRATE_ENABLE)` `#error` guard.
3. `lbm_mrt/solver/src/LBM.cu` — new kernels:
   - `compute_p_psi_scmp_cs(rho, pressure, psi, pointsflag)` — Carnahan-Starling EOS, pattern from `compute_p_psi_A_all` at LBM.cu:1049-1081. p = ρRT(1+η+η²−η³)/(1−η)³ − aρ², η = bρ/4. ψ = sqrt(2(p−ρcs²)/(G·dx²)).
   - `compute_molecular_force_scmp(psi, rho, Fx, Fy, pointsflag)` — single-fluid version of `compute_molecular_force_gpu` (LBM.cu:1255-1305), AA branch only, G read from `d_cs_G`.
   - `compute_velocity_scmp(rho, fin, Fx, Fy, ux, uy, pointsflag)` — single-fluid analogue of `compute_velocity_gpu_AB` (LBM.cu:1347-1400) with the half-force shift `ρu = Σe_i f_i + (δt/2) F`.
   - `compute_S_huang_gpu(ux, uy, rho, Fx, Fy, S, pointsflag)` — clean Guo form per paper Eq. 6: S[1]=6·u·F/c², S[2]=−6·u·F/c², S[3..8] standard. **No σ-tuning hack** (replaced by Q_m in phase 3).
   - `mrt_collide_single_component_gpu` — analogue of `mrt_collide_two_components_gpu` (LBM.cu:1596-1690) minus the B half. Same `+ C·δt` convention so phase 3's Q_m kernel slots in cleanly.
   - `stream_single_component_gpu` — pattern from LBM.cu:1693-1710.
4. `lbm_mrt/solver/src/LBM.cu` — new high-level driver `evolution_scmp` mirroring `evolution_all` (LBM.cu:713-745); skips B-related calls and adsorption force; periodic BCs only.
5. `lbm_mrt/solver/src/LBM.cu` — `init_all_scmp` writes tanh droplet (Eq. 46a) or flat-interface depending on `d_huang_init_mode`.
6. `lbm_mrt/solver/src/sim_utils.cu` — `run_scmp_huang(const RuntimeParams&)` allocates only `Fluid_dev`, sets all `pointsflag=1`, runs single-stage time loop.
7. `lbm_mrt/solver/src/main.cu` line ~49 — dispatch on `P.pp_mode`: `if (pp_mode==1) { run_scmp_huang(P); return 0; }`.
8. `lbm_mrt/solver/build.py` lines 24-25, 50-55 — add `--huang` flag producing `mcmp_huang_256` with `-DHUANG_256_BUILD`.

**Smoke test**: `uv run lbm-build --huang` succeeds; run with `pp_mode=1, k1_huang=0` (degenerates to plain Guo-MRT) for 50k steps — droplet stays circular, no NaN, max|u|<1e-2.

### Phase 3 — Wire in Q_m (the actual surface-tension scheme)

1. `lbm_mrt/solver/src/LBM.cu` — new kernel `compute_Q_huang_gpu(rho, psi, Fx_mol, Fy_mol, C, pointsflag)`:
   ```
   F2 = Fx² + Fy²; psi2 = psi² + PSI_CUT²
   denom = G · psi2 · c²
   Qm1 = 3·(k1 + 2·k2)·F2/denom
   Qm7 = k1·(Fx² − Fy²)/denom
   Qm8 = k1·Fx·Fy/denom
   C[1] = s_e·Qm1; C[2] = −s_t·Qm1; C[7] = s_p·Qm7; C[8] = s_p·Qm8
   ```
   The `s_e/s_t/s_p` baking matches the existing `+ C·δt` collision convention, so `mrt_collide_single_component_gpu` does NOT need editing.
2. `evolution_scmp` calls `compute_Q_huang_gpu` between force computation and collision.
3. `lbm_mrt/solver/src/LBM.cu` `meq_gpu` (lines 429-444) — patch slot 1 to use `d_alpha_meq` per paper Eq. 5 (only when `pp_mode==1`; legacy keeps α=1 hardcoded). **Read paper Eq. 5 carefully** before editing — the α term enters only the |u|² coefficient in `m^eq[1]`.

**Smoke test**: at fixed T=0.7, R=40, run k₁=0 and k₁=0.15. Measure σ from Laplace law at each. Expect σ(k₁=0)/σ(k₁=0.15) ≈ 1/(1−6·0.15) = 10× per Eq. 62. ρ_l, ρ_g across the two runs must match within 1%.

### Phase 4 — Python validation suite

New module dir `lbm_mrt/validation/`. Each module ~150 LOC, single-purpose.

- `cs_eos.py` — host-side CS-EOS, Maxwell-construction solver via `scipy.optimize.fsolve`. Public: `cs_pressure`, `cs_critical_point`, `maxwell_coexistence`, `coexistence_curve`.
- `analytical.py` — shared post-processing. Public: `detect_interface_radius` (skimage contours), `fit_pressure_inside_outside`, `laplace_sigma` (linear regression), `centerline_profile`, `extract_rho_l_g` (plateau finder).
- `static_droplet.py` — droplet sweep driver. Reuses `runners/batch_run.run_batch`. Public: `run_droplet_sweep(R_list, T, params_base, out_root, app)`, `analyze_droplet`.
- `laplace_law.py` — Laplace-law analyzer + plot. Reuses `viz/viz_template`. Public: `fit_laplace`, `plot_laplace`.
- `coexistence.py` — flat-interface T-sweep + comparison to Maxwell. Public: `run_coexistence_sweep`, `analyze_flat`.
- `spurious_currents.py` — max|u| extractor + Huang-vs-Li comparison plot. Public: `extract_max_u`, `time_series_max_u`, `plot_spurious_compare`.
- `decoupling_sweep.py` — headline test. Public: `run_decoupling_sweep`, `plot_decoupling`.
- Reference data under `data/validation_reference/huang_2016/` (Maxwell curve cached as JSON, paper Table 1 σ values).

Tolerances:
- Laplace fit R² ≥ 0.99, intercept |b| < 1e-4
- Coexistence |Δρ|/ρ < 0.02 across T/Tc∈[0.6,0.95]
- Spurious max|u|_Huang ≤ 0.5×max|u|_Li
- Decoupling: ρ_l std/mean < 0.01 AND σ-vs-(1−6k₁) R² ≥ 0.99

### Phase 5 — Driver scripts & documentation

1. `scripts/06_run_huang_validation_suite.py` — one-shot runner: generates sweep CSVs, calls `run_batch`, post-processes all four cases, emits `results/huang_validation_<ts>/REPORT.md` with embedded plots and pass/fail tables. Pattern from `scripts/04_run_hydrate_sphere_case.py`.
2. `scripts/07_tune_huang_parameters.py` — closed-loop tuner: `scipy.optimize.brentq` on k₁ to match a target σ from Huang's published table. Each evaluation = one full SCMP simulation (~30 min). Hold k_d=−1/12, vary k₁ only.
3. `configs/huang_scmp.yaml` — new config for SCMP cases (defaults: `pp_mode: 1, cs_T: 0.5, k1_huang: 0.0`).
4. `research/HUANG_MRT_REFACTOR.md` — equation-to-code mapping table (paper Eq. # ↔ file:line ↔ moment slot), verified parameter table (filled after first successful runs), CLI usage, comparison plots.
5. Update `lbm_mrt/solver/CLAUDE.md` — current text says NX=NY=300 (incorrect; actual is 339×212). Document the new `mcmp_huang_256` variant.

## Critical files

| File | Lines | Action |
|---|---|---|
| `lbm_mrt/solver/include/LBM.h` | 14-18, 26-47, 49, 107-150 | Add `HUANG_256_BUILD` ifdef; add `__constant__` mirrors; add hydrate-conflict `#error` |
| `lbm_mrt/solver/include/sim_utils.h` | 11-124 | Add ~15 new `RuntimeParams` fields |
| `lbm_mrt/solver/src/LBM.cu` | 429-444, 1049-1081, 1255-1305, 1467-1497, 1596-1690, 713-745 | Add 6 new kernels + `evolution_scmp` + `init_all_scmp`; patch `meq_gpu` for α |
| `lbm_mrt/solver/src/sim_utils.cu` | ~296, ~568, ~482 | Extend `load_params_txt`, `push_device_constants`, `print_params_summary`; add `run_scmp_huang` |
| `lbm_mrt/solver/src/main.cu` | ~49 | Dispatch on `pp_mode` |
| `lbm_mrt/solver/build.py` | 24-25, 50-55 | Add `--huang` variant |
| `lbm_mrt/validation/*.py` | new | 7 modules |
| `scripts/06_run_huang_validation_suite.py` | new | One-shot suite |
| `scripts/07_tune_huang_parameters.py` | new | brentq tuner |
| `configs/huang_scmp.yaml` | new | SCMP defaults |
| `research/HUANG_MRT_REFACTOR.md` | new | Documentation |
| `lbm_mrt/solver/CLAUDE.md` | top | Fix NX/NY claim, add huang variant |

## Reused functions (do not reimplement)

| Function | Source | Use in |
|---|---|---|
| `read_vtk_scalars(path) → (fields, nx, ny)` | `lbm_mrt/io/vtk_reader.py` | All validation modules |
| `latest_vtk(folder, tag)` | `lbm_mrt/io/vtk_reader.py` | All analyzers |
| `run_batch(design_csv, ...)` | `lbm_mrt/runners/batch_run.py` | Sweep drivers |
| `run_one(params, case_name, geometry_src, out_root, app)` | `lbm_mrt/runners/single_run.py` | Tuner per-iteration |
| `load_config(path)`, `override(p, **kw)` | `lbm_mrt/core/config.py` | All scripts |
| `init_style`, `create_figure_ax`, `set_xlabel/ylabel`, `format_axes`, `save_figure` | `lbm_mrt/viz/viz_template.py` | All plots |
| `pr_eos`, `compute_Tc_Pc_from_ab`, `critical_from_abR` | `lbm_mrt/utils/eos.py` | Pattern for new `cs_eos.py` |
| `compute_molecular_force_gpu` | `LBM.cu:1255-1305` | Pattern for `compute_molecular_force_scmp` |
| `compute_velocity_gpu_AB` | `LBM.cu:1347-1400` | Pattern for `compute_velocity_scmp` |
| `mrt_collide_two_components_gpu` | `LBM.cu:1596-1690` | Pattern for `mrt_collide_single_component_gpu` |
| `evolution_all` | `LBM.cu:713-745` | Pattern for `evolution_scmp` |

## Verification

End-to-end test sequence (run after Phase 4 complete):

```bash
# Build both variants
uv run lbm-build              # legacy → mcmp_sim
uv run lbm-build --huang      # new   → mcmp_huang_256
uv run lbm-build --hydrate    # hydrate untouched, must still build

# Regression: legacy benchmarks
uv run python scripts/03_validate_benchmarks.py --all --dir results/smoke_test
# Pass criterion: every benchmark within its existing tolerance

# New: Huang validation suite
uv run python scripts/06_run_huang_validation_suite.py --T 0.7 --k1 0.0833
# Pass criterion: REPORT.md shows green on all four cases (Laplace, coexistence, spurious, decoupling)

# Optional: parameter tuning to a Huang Table-1 target
uv run python scripts/07_tune_huang_parameters.py --target-label "T097_eps1"
# Pass criterion: brentq converges in ≤ 10 evals; final σ within 1% of target

# Lint + tests
uv run ruff check . && uv run ruff format --check .
uv run mypy lbm_mrt/
uv run pytest
```

The decoupling claim is verified iff: across k₁ ∈ {0, 0.05, 0.10, 0.15, 0.20, 1/6}, σ varies by ≥10× **AND** ρ_l, ρ_g stay within 1%.
