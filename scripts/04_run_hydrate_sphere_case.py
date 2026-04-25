#!/usr/bin/env python3
"""Run a standard hydrate-sphere verification case.

Case definition:
- 300x300 structured domain
- center hydrate sphere with radius 50
- the rest of the domain is water
- top boundary is heated to trigger hydrate dissociation

The script generates a Tecplot geometry file, runs the hydrate-enabled binary,
and writes a manifest into the case result directory so the output format is
stable and easy to inspect.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from lbm_mrt.core.config import load_config, override
from lbm_mrt.core.paths import HYDRATE_BINARY, RESULTS_DIR
from lbm_mrt.runners.single_run import run_one


REPO_ROOT = Path(__file__).resolve().parents[1]
HYDRATE_CONFIG = REPO_ROOT / "configs" / "hydrate.yaml"

DEFAULT_NX = 300
DEFAULT_NY = 300
DEFAULT_RADIUS = 50
DEFAULT_CASE_NAME = "hydrate_sphere_300x300_r50_hot"


def write_hydrate_sphere_tecplot(
    path: Path,
    *,
    nx: int = DEFAULT_NX,
    ny: int = DEFAULT_NY,
    radius: int = DEFAULT_RADIUS,
) -> None:
    """Write a Tecplot ASCII geometry file understood by read_tecplot_to_flag()."""
    cx = nx // 2
    cy = ny // 2
    r2 = radius * radius

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write('TITLE = "Hydrate sphere verification case"\n')
        f.write('VARIABLES = "X", "Y", "value"\n')
        f.write(f'ZONE t = "hydrate_sphere", I = {nx}, J = {ny} F = point\n')

        for y in range(ny):
            for x in range(nx):
                dx = x - cx
                dy = y - cy
                inside = dx * dx + dy * dy <= r2
                phase = 0.5 if inside else 0.0
                f.write(f"{x},{y},{phase:.1f}\n")


def build_case_manifest(
    *,
    case_dir: Path,
    source_geometry: Path,
    nx: int,
    ny: int,
    radius: int,
    init_temp: float,
    inlet_temp: float,
    sw: float,
    eq_max_steps: int,
    flow_max_steps: int,
    output_every: int,
    return_code: int,
    elapsed_s: float,
) -> None:
    """Write a compact manifest describing the run and produced files."""
    vtk_files = sorted(glob.glob(str(case_dir / "outputdata_*.vtk")))
    manifest = {
        "case_name": case_dir.name,
        "geometry_source": str(source_geometry),
        "geometry_copy": str(case_dir / "geometry_case.plt"),
        "geometry": {
            "nx": nx,
            "ny": ny,
            "hydrate_radius": radius,
            "center": [nx // 2, ny // 2],
            "hydrate_phase": 0.5,
            "water_phase": 0.0,
        },
        "physics": {
            "hydrate_enable": True,
            "hydrate_start_step": 0,
            "Sw": sw,
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
            "geometry": str(case_dir / "geometry_case.plt"),
            "checkpoints": str(case_dir / "ckpt"),
            "vtk_outputs": vtk_files,
        },
        "expected_file_pattern": [
            "geometry_case.plt",
            "params.txt",
            "log.txt",
            "ckpt/",
            "outputdata_eq*.vtk",
            "outputdata_flow*.vtk",
        ],
    }

    manifest_path = case_dir / "case_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate and run a hydrate-sphere case with standardized outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--case-name",
        default=DEFAULT_CASE_NAME,
        help="Output case directory name under results/.",
    )
    parser.add_argument(
        "--out-root", default=RESULTS_DIR, help="Root output directory."
    )
    parser.add_argument("--nx", type=int, default=DEFAULT_NX, help="Grid size in x.")
    parser.add_argument("--ny", type=int, default=DEFAULT_NY, help="Grid size in y.")
    parser.add_argument(
        "--radius", type=int, default=DEFAULT_RADIUS, help="Hydrate sphere radius."
    )
    parser.add_argument(
        "--sw", type=float, default=1.0, help="Initial water saturation."
    )
    parser.add_argument(
        "--init-temp", type=float, default=278.15, help="Initial temperature in K."
    )
    parser.add_argument(
        "--inlet-temp",
        type=float,
        default=323.15,
        help="Heated inlet temperature in K.",
    )
    parser.add_argument(
        "--eq-max-steps", type=int, default=100000, help="Maximum eq-stage steps."
    )
    parser.add_argument(
        "--flow-max-steps",
        type=int,
        default=500000,
        help="Maximum hydrate-stage steps.",
    )
    parser.add_argument(
        "--output-every", type=int, default=5000, help="VTK output interval."
    )
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Disable steady-based early stop by setting very large consecutive-hit thresholds.",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Only generate geometry and print the run plan.",
    )
    args = parser.parse_args()

    out_root = Path(args.out_root)
    case_dir = out_root / args.case_name
    input_root = out_root / "_inputs" / args.case_name
    geometry_path = input_root / f"geometry_case_{args.nx}x{args.ny}_r{args.radius}.plt"

    write_hydrate_sphere_tecplot(
        geometry_path,
        nx=args.nx,
        ny=args.ny,
        radius=args.radius,
    )

    eq_need_consec = 2
    flow_need_consec = 3
    if args.no_early_stop:
        # Keep runtime bounded by max_steps instead of steady-monitor convergence.
        eq_need_consec = max(args.eq_max_steps + 1, 10**9)
        flow_need_consec = max(args.flow_max_steps + 1, 10**9)

    params = load_config(HYDRATE_CONFIG)
    params = override(
        params,
        Sw=args.sw,
        T0_init=args.init_temp,
        T0_inlet=args.inlet_temp,
        eq_max_steps=args.eq_max_steps,
        eq_need_consec=eq_need_consec,
        flow_max_steps=args.flow_max_steps,
        flow_need_consec=flow_need_consec,
        OUTPUT_EVERY=args.output_every,
        Gx=0.0,
        Gy=0.0,
        hydrate_enable=1,
        hydrate_start_step=0,
    )

    print(f"[hydrate-sphere] geometry: {geometry_path}")
    print(f"[hydrate-sphere] case dir : {case_dir}")
    print(
        f"[hydrate-sphere] run plan : Sw={args.sw}, T0_init={args.init_temp}, "
        f"T0_inlet={args.inlet_temp}, eq_max_steps={args.eq_max_steps}, "
        f"flow_max_steps={args.flow_max_steps}, OUTPUT_EVERY={args.output_every}, "
        f"eq_need_consec={eq_need_consec}, flow_need_consec={flow_need_consec}, "
        f"hydrate_enable=1"
    )

    if args.skip_run:
        return 0

    ret, elapsed = run_one(
        params,
        case_name=args.case_name,
        geometry_src=str(geometry_path),
        out_root=str(out_root),
        app=str(HYDRATE_BINARY),
        resume=False,
        strict_geometry=True,
    )

    build_case_manifest(
        case_dir=case_dir,
        source_geometry=geometry_path,
        nx=args.nx,
        ny=args.ny,
        radius=args.radius,
        init_temp=args.init_temp,
        inlet_temp=args.inlet_temp,
        sw=args.sw,
        eq_max_steps=args.eq_max_steps,
        flow_max_steps=args.flow_max_steps,
        output_every=args.output_every,
        return_code=ret,
        elapsed_s=elapsed,
    )

    print(f"[hydrate-sphere] manifest: {case_dir / 'case_manifest.json'}")
    return ret


if __name__ == "__main__":
    raise SystemExit(main())
