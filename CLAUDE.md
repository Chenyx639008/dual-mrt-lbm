# CLAUDE.md

CUDA-based 2D Multicomponent MRT-LBM solver for two-phase flow and methane hydrate dissociation in porous media.

## Tech Stack

- **CUDA solver** (`lbm_mrt/solver/`): C++17 + CUDA, compiled with `nvcc`; two binaries (`mcmp_sim` / `mcmp_sim_hydrate`)
- **Python wrapper** (`lbm_mrt/`): `uv`-managed; build automation, config management, run orchestration
- GPU target: sm_120 (RTX 5090 / H100) by default

## Project Structure

```
lbm_mrt/
├── lbm_mrt/solver/    # CUDA core — see solver/CLAUDE.md for internals
├── lbm_mrt/core/      # paths.py, config.py
├── lbm_mrt/io/        # params_writer.py, vtk_reader.py
├── lbm_mrt/runners/   # single_run.py, batch_run.py
├── lbm_mrt/utils/     # eos.py (Peng-Robinson EOS)
├── lbm_mrt/viz/       # viz_template.py (publication matplotlib)
├── configs/           # default.yaml, hydrate.yaml
├── data/              # geometry (.plt), benchmarks, design CSVs
├── results/           # simulation output (gitignored)
├── scripts/           # 01-04 workflow scripts
└── research/          # scientific guides and design notes (agent_docs)
```

## Commands

### Setup
```bash
uv sync --all-groups
uv run pre-commit install
```

### Build
```bash
uv run lbm-build                     # flow-only → mcmp_sim
uv run lbm-build --hydrate           # hydrate → mcmp_sim_hydrate
uv run lbm-build --arch sm_89        # override GPU arch
uv run lbm-build --debug             # -g -G device debugging
uv run lbm-build --hydrate --dry-run # print nvcc command only
```

### Run
```bash
uv run lbm-run --case-name my_run --geom data/geometry/.../geometry.plt
uv run lbm-run --case-name my_run --geom <plt> Sw=0.5 Gx=2e-8
uv run lbm-batch --csv data/design_from_geometry.csv --strict-geometry
uv run python scripts/04_run_hydrate_sphere_case.py
```

### Test & Lint
```bash
uv run pytest
uv run ruff check . && uv run ruff format .
uv run mypy lbm_mrt/
uv run python scripts/03_validate_benchmarks.py --all --dir results/smoke_test
```

## Key Conventions

- **Two-binary model**: hydrate physics are compiled in only via `-DHYDRATE_ENABLE`; all hydrate code is `#ifdef`-guarded. See `lbm_mrt/solver/CLAUDE.md`.
- **Config flow**: `configs/default.yaml` → `load_config()` → flat dict → `params.txt` → C++ `load_params_txt()` → `__constant__` device memory. Hydrate overlay: load `configs/hydrate.yaml` on top.
- **Wettability**: contact angles (`thetaA_quartz_deg`, `thetaA_hydrate_deg`) → `GAw = GAw_m * (theta - GAw_c)` → `GAw_by_mat_gpu[256]`.
- **Grid size is compile-time**: `NX`/`NY` are `constexpr` in `LBM.h`; changing them requires recompile.
- **Output**: `results/<case_name>/` contains `params.txt`, `log.txt`, `outputdata_eq/flow*.vtk`, `ckpt/`. VTK files are Legacy Binary big-endian.
- **Param overrides**: CSV columns or inline `KEY=VALUE` args in `lbm-run` map directly to `RuntimeParams` field names.

## Reference Docs (`research/`)

| File | Contents |
|------|----------|
| `research/Research.md` | Full multi-physics LBM model documentation |
| `research/USAGE_GUIDE.md` | End-to-end usage manual |
| `research/HYDRATE_FULL_DEBUG_VALIDATION_GUIDE.md` | Debug & validation guide for hydrate mode |
| `research/hydrate_cu_scientific_guide.md` | `hydrate.cu` function-level scientific guide |
| `research/hydrate_vop_scientific_guide.md` | `hydrate_vop.cu` VOP scientific guide |
| `research/hydrate_sphere_case_design.md` | 300×300 hydrate sphere case design |
| `research/hydrate_validation_design.md` | Parameter conversion & validation design |
| `research/hydrate_dissociation_space_analysis.md` | Post-dissociation pore occupation analysis |
