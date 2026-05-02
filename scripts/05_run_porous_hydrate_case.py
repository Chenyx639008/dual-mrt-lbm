#!/usr/bin/env python3
"""Run a porous-media hydrate dissociation benchmark case.

Two morphology variants are supported and can be selected via --morph:
  coating   (morph=2): hydrate film of thickness coat_thick=4 wrapping solid grains
  porefill  (morph=1): hydrate spheres of radius r_mid=8 filling pore throats

Domain configuration
--------------------
  NX = 339, NY = 212  (matches the constexpr comment in LBM.h)
  Geometry: build_circle_array with r_obs=16, l_gap=4, SCALE=1

Physical scenario
-----------------
  1. Initial state: uniform T0_init=274.15 K (1 °C) — hydrate is stable.
  2. Eq stage: two-phase flow equilibrates (no hydrate physics).
  3. Flow+hydrate stage: right boundary (x=NX-1) is held at T0_inlet=298.15 K.
     Heat diffuses left → hydrates near right edge begin to dissociate first →
     dissociation front propagates leftward as temperature rises across domain.

Usage
-----
  uv run python scripts/05_run_porous_hydrate_case.py --morph coating
  uv run python scripts/05_run_porous_hydrate_case.py --morph porefill
  uv run python scripts/05_run_porous_hydrate_case.py --morph both   # run both sequentially
  uv run python scripts/05_run_porous_hydrate_case.py --morph coating --skip-run
  uv run python scripts/05_run_porous_hydrate_case.py --morph coating \\
      --inlet-temp 310.0 --flow-max-steps 1000000 --output-every 10000
  uv run python scripts/05_run_porous_hydrate_case.py --morph coating --gx 1e-5
  uv run python scripts/05_run_porous_hydrate_case.py --morph both --flow-max-steps 2000000 \\
      --no-early-stop   # run full 2M steps regardless of Vh_frac
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

from lbm_mrt.core.config import load_config, override
from lbm_mrt.core.paths import HYDRATE_BINARY, RESULTS_DIR
from lbm_mrt.runners.single_run import run_one

REPO_ROOT = Path(__file__).resolve().parents[1]
POROUS_CONFIG = REPO_ROOT / "configs" / "hydrate_porous.yaml"

# ── Domain dimensions (must match LBM.h constexpr NX/NY at compile time) ──────
DOMAIN_NX = 339
DOMAIN_NY = 212

# ── Porous-media geometry parameters ──────────────────────────────────────────
R_OBS      = 20   # solid grain radius [lu]
L_GAP      = 20   # inter-grain gap [lu]
COAT_THICK = 4    # hydrate film thickness for coating morph [lu]
R_MID      = 8    # hydrate sphere radius for pore-fill morph [lu]

MORPH_COATING  = 2   # build_circle_array morph id
MORPH_POREFILL = 1


def _case_name(morph_str: str, gx: float = 0.0) -> str:
    suffix = f"_gx{gx:.0e}" if gx != 0.0 else ""
    return f"porous_hydrate_{DOMAIN_NX}x{DOMAIN_NY}_{morph_str}{suffix}"


def _write_case_manifest(
    *,
    case_dir: Path,
    morph_str: str,
    morph_id: int,
    init_temp: float,
    inlet_temp: float,
    sw: float,
    gx: float = 0.0,
    eq_max_steps: int,
    flow_max_steps: int,
    output_every: int,
    return_code: int,
    elapsed_s: float,
) -> None:
    vtk_files = sorted(glob.glob(str(case_dir / "outputdata_*.vtk")))
    manifest = {
        "case_name": case_dir.name,
        "geometry": {
            "type": "build_circle_array",
            "morph": morph_id,
            "morph_label": morph_str,
            "nx": DOMAIN_NX,
            "ny": DOMAIN_NY,
            "r_obs": R_OBS,
            "l_gap": L_GAP,
            "coat_thick": COAT_THICK if morph_id == MORPH_COATING else 0,
            "r_mid": R_MID if morph_id == MORPH_POREFILL else 0,
        },
        "physics": {
            "hydrate_enable": True,
            "hydrate_start_step": 0,
            "thermal_bc_side": 3,
            "Sw": sw,
            "Gx": gx,
            "T0_init": init_temp,
            "T0_inlet": inlet_temp,
        },
        "runtime": {
            "eq_max_steps": eq_max_steps,
            "flow_max_steps": flow_max_steps,
            "OUTPUT_EVERY": output_every,
            "return_code": return_code,
            "elapsed_seconds": round(elapsed_s, 3),
        },
        "files": {
            "params": str(case_dir / "params.txt"),
            "log": str(case_dir / "log.txt"),
            "checkpoints": str(case_dir / "ckpt"),
            "vtk_outputs": vtk_files,
        },
    }
    path = case_dir / "case_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    print(f"[porous-hydrate] manifest: {path}")


def _run_single_morph(
    *,
    morph_str: str,
    morph_id: int,
    args: argparse.Namespace,
    out_root: Path,
) -> int:
    """Run one morphology variant. Returns process return code."""
    case_name = _case_name(morph_str, args.gx)
    case_dir = out_root / case_name

    eq_need_consec = 2
    flow_need_consec = 3
    # --no-early-stop: disable both the flow steady-state monitor AND the VOP frac cutoff
    vop_stop_frac = args.vop_stop_frac
    if args.no_early_stop:
        eq_need_consec   = max(args.eq_max_steps   + 1, 10**9)
        flow_need_consec = max(args.flow_max_steps  + 1, 10**9)
        vop_stop_frac    = 0.0   # 0 means never terminate on Vh_frac

    params = load_config(POROUS_CONFIG)
    params = override(
        params,
        # geometry: build_circle_array — no Tecplot file; the binary reads morph/r_obs/…
        morph=morph_id,
        r_obs=R_OBS,
        l_gap=L_GAP,
        coat_thick=COAT_THICK if morph_id == MORPH_COATING else 0,
        r_mid=R_MID if morph_id == MORPH_POREFILL else 0,
        # hydrate physics
        hydrate_enable=1,
        hydrate_start_step=0,
        thermal_bc_side=3,         # right boundary (x==NX-1) is the hot side
        thermal_init_mode=args.thermal_init_mode,
        T0_init=args.init_temp,
        T0_inlet=args.inlet_temp,
        vop_terminate_frac=vop_stop_frac,
        # flow
        Sw=args.sw,
        Gx=args.gx,
        Gy=0.0,
        # run control
        eq_max_steps=args.eq_max_steps,
        eq_need_consec=eq_need_consec,
        flow_max_steps=args.flow_max_steps,
        flow_need_consec=flow_need_consec,
        OUTPUT_EVERY=args.output_every,
    )

    print(f"\n{'='*60}")
    print(f"[porous-hydrate] morph     : {morph_str} (morph_id={morph_id})")
    print(f"[porous-hydrate] case dir  : {case_dir}")
    print(f"[porous-hydrate] T0_init   : {args.init_temp} K")
    print(f"[porous-hydrate] T0_inlet  : {args.inlet_temp} K  (right boundary)")
    print(f"[porous-hydrate] T_init_mode: {'linear gradient' if args.thermal_init_mode else 'uniform'}")
    print(f"[porous-hydrate] Gx        : {args.gx:.2e}")
    print(f"[porous-hydrate] eq_steps  : {args.eq_max_steps}")
    print(f"[porous-hydrate] flow_steps: {args.flow_max_steps}")
    print(f"{'='*60}")

    if args.skip_run:
        print("[porous-hydrate] --skip-run: geometry and params ready, not launching binary.")
        return 0

    ret, elapsed = run_one(
        params,
        case_name=case_name,
        geometry_src=None,          # no Tecplot file; binary uses build_circle_array
        out_root=str(out_root),
        app=str(HYDRATE_BINARY),
        resume=False,
        strict_geometry=False,
    )

    _write_case_manifest(
        case_dir=case_dir,
        morph_str=morph_str,
        morph_id=morph_id,
        init_temp=args.init_temp,
        inlet_temp=args.inlet_temp,
        sw=args.sw,
        gx=args.gx,
        eq_max_steps=args.eq_max_steps,
        flow_max_steps=args.flow_max_steps,
        output_every=args.output_every,
        return_code=ret,
        elapsed_s=elapsed,
    )
    return ret


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Porous-media hydrate dissociation benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--morph",
        choices=["coating", "porefill", "both"],
        default="coating",
        help=(
            "Hydrate morphology: 'coating' (film on grain, morph=2), "
            "'porefill' (sphere in pore, morph=1), or 'both' to run sequentially."
        ),
    )
    parser.add_argument("--out-root", default=RESULTS_DIR, help="Root output directory.")
    parser.add_argument("--sw",        type=float, default=1.0,    help="Initial water saturation.")
    parser.add_argument(
        "--gx",
        type=float,
        default=0.0,
        help="Body-force driving (Gx) in lattice units. 0=no flow; 1e-5=gentle driven flow.",
    )
    parser.add_argument(
        "--thermal-init-mode",
        type=int,
        default=1,
        choices=[0, 1],
        help=(
            "Temperature field initialization: 0=uniform T0_init, "
            "1=linear gradient T0_init(left)→T0_inlet(right). "
            "Default 1 for porous case to immediately establish spatial T contrast."
        ),
    )
    parser.add_argument("--init-temp", type=float, default=274.15, help="Initial temperature (K).")
    parser.add_argument(
        "--inlet-temp",
        type=float,
        default=298.15,
        help="Right-boundary heating temperature (K).",
    )
    parser.add_argument("--eq-max-steps",   type=int, default=100000,  help="Max eq-stage steps.")
    parser.add_argument("--flow-max-steps",  type=int, default=500000,  help="Max hydrate-stage steps.")
    parser.add_argument("--output-every",    type=int, default=5000,    help="VTK output interval.")
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help=(
            "Disable ALL early termination: flow steady-state monitor AND "
            "VOP Vh_frac cutoff. Simulation runs the full flow_max_steps."
        ),
    )
    parser.add_argument(
        "--vop-stop-frac",
        type=float,
        default=0.01,
        help=(
            "Stop when hydrate volume fraction drops below this value (0=never stop). "
            "Overridden to 0 by --no-early-stop."
        ),
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Print plan and prepare params but do not launch the binary.",
    )
    args = parser.parse_args()

    out_root = Path(args.out_root)
    morph_map = {"coating": MORPH_COATING, "porefill": MORPH_POREFILL}

    to_run: list[tuple[str, int]] = []
    if args.morph == "both":
        to_run = [("coating", MORPH_COATING), ("porefill", MORPH_POREFILL)]
    else:
        to_run = [(args.morph, morph_map[args.morph])]

    last_ret = 0
    for morph_str, morph_id in to_run:
        ret = _run_single_morph(
            morph_str=morph_str,
            morph_id=morph_id,
            args=args,
            out_root=out_root,
        )
        last_ret = ret
        if ret != 0:
            print(f"[porous-hydrate] morph={morph_str} exited with code {ret}, stopping.")
            break

    return last_ret


if __name__ == "__main__":
    raise SystemExit(main())
