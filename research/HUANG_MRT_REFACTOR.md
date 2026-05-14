# Huang & Wu (2016) MRT-LBM SCMP Refactor — Equation-to-Code Mapping

> Ref: Huang & Wu, *Third-order analysis of pseudopotential lattice Boltzmann model for multiphase flow*, 2016.

## Build Variants

| Binary | Macro | Grid | Purpose |
|--------|-------|------|---------|
| `mcmp_sim` | (none) | 339×212 | Legacy MCMP (PR-EOS + ideal gas) |
| `mcmp_sim_hydrate` | `HYDRATE_ENABLE` | 339×212 | Hydrate dissociation physics |
| `mcmp_huang_256` | `HUANG_256_BUILD` | 256×256 | Huang SCMP (Carnahan-Starling) |

Build commands:
```bash
uv run lbm-build              # legacy → mcmp_sim
uv run lbm-build --hydrate    # hydrate → mcmp_sim_hydrate
uv run lbm-build --huang      # huang  → mcmp_huang_256
```

## Equation → Code Mapping

| Paper Eq. | Description | File:Line | Implementation |
|-----------|-------------|-----------|----------------|
| Eq. 5 | α-modified m_eq[1] | `LBM.cu` `mrt_collide_single_component_gpu` | `alpha_meq` parameter |
| Eq. 6 | Guo force S moments | `LBM.cu` `compute_S_huang_gpu` | Clean Guo form, no σ hack |
| Eq. 46a | Tanh droplet init | `LBM.cu` `init_all_scmp_gpu` | `huang_init_mode=1` |
| Eq. 59 | Q_m surface-tension moments | `LBM.cu` `compute_Q_huang_gpu` | k₁/k₂ → Qm1/Qm7/Qm8 |
| Eq. 62 | σ ∝ (1−6k₁) | Verified by `scripts/06_run_huang_validation_suite.py` | Decoupling sweep |

## CS-EOS Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cs_a` | 0.75 | Attraction parameter |
| `cs_b` | 4.0 | Hard-sphere volume |
| `cs_R` | 1.0 | Gas constant |
| `cs_T` | 0.06 | Absolute temperature (Tc≈0.0707, Tr≈0.85) |
| `cs_G` | -1.0 | Interaction strength |

## MRT Moment Slots (D2Q9)

| Slot | Moment | Relaxation | Q_m contribution |
|------|--------|------------|------------------|
| 0 | ρ | conserved | — |
| 1 | e | s_e | +s_e·Qm1 |
| 2 | ε | s_t | −s_t·Qm1 |
| 3 | j_x | conserved | — |
| 4 | q_x | s_q | — |
| 5 | j_y | conserved | — |
| 6 | q_y | s_q | — |
| 7 | p_xx | s_p | +s_p·Qm7 |
| 8 | p_xy | s_p | +s_p·Qm8 |

## Validation Tolerances

| Test | Criterion |
|------|-----------|
| Laplace fit | R² ≥ 0.99, intercept ≤ 1e-4 |
| Coexistence | Δρ/ρ ≤ 0.02 for T/Tc ∈ [0.6, 0.95] |
| Spurious currents | max\|u\|_Huang ≤ 0.5×max\|u\|_Li |
| Decoupling | ρ_l std/mean ≤ 0.01; σ vs (1−6k₁) R² ≥ 0.99 |

## CLI Usage

```bash
# Smoke test
uv run lbm-build --huang
echo "pp_mode=1" > params.txt && ./lbm_mrt/solver/mcmp_huang_256 params.txt

# Full validation
uv run python scripts/06_run_huang_validation_suite.py --all --T 0.06 --k1 0.0833

# Parameter tuning
uv run python scripts/07_tune_huang_parameters.py --target-label "T097_eps1"
```

## Key Design Decisions

1. **Build-time variant, not runtime switch**: SCMP and MCMP are separate binaries. No runtime overhead.
2. **C baked into moment space**: Q_m includes relaxation rates, so collision uses uniform `+ C·δt` convention.
3. **Periodic domain only**: SCMP supports periodic BCs; no solid boundaries, no adsorption.
4. **Single Fluid_dev**: SCMP reuses the existing `Fluid_dev` struct (only one component needed).
5. **psi formula**: Uses `fmax(-diff, 0)` to handle CS-EOS where p < ρcs² in both phases at the chosen parameters.
