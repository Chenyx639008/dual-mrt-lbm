"""Build system for the CUDA LBM solver.

Replaces the legacy compile.sh. Compiles the MRT-LBM CUDA binary using nvcc.

Usage::

    uv run lbm-build                      # flow-only binary (mcmp_sim)
    uv run lbm-build --hydrate            # hydrate-enabled binary (mcmp_sim_hydrate)
    uv run lbm-build --arch sm_89         # different GPU arch (e.g. RTX 4090)
    uv run lbm-build --debug              # add -g -G for device debugging
    uv run lbm-build --dry-run            # print the nvcc command without running it
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from lbm_mrt.core.paths import SOLVER_DIR, SOLVER_INC, SOLVER_SRC

# Source files for each build variant
_BASE_SOURCES = ["main.cu", "LBM.cu", "steady_monitor.cu", "sim_utils.cu"]
_HYDRATE_SOURCES = ["hydrate.cu", "hydrate_vop.cu"]


def _detect_plt_dimensions(plt_path: str) -> tuple[int, int]:
    """Parse I= and J= from .plt ZONE header line."""
    import re
    with open(plt_path) as f:
        for line in f:
            m = re.search(r'I\s*=\s*(\d+).*J\s*=\s*(\d+)', line)
            if m:
                return int(m.group(1)), int(m.group(2))
    raise ValueError(f"Cannot detect I,J from {plt_path}")


def build(
    hydrate: bool = False,
    huang: bool = False,
    porous: bool = False,
    plt_path: str | None = None,
    grid: tuple[int, int] | None = None,
    steps: int | None = None,
    output_every: int | None = None,
    arch: str = "sm_120",
    debug: bool = False,
    output_dir: str | None = None,
    dry_run: bool = False,
) -> int:
    """Compile the CUDA LBM solver.

    Args:
        hydrate:    If True, compile the hydrate-enabled variant.
        huang:      If True, compile the Huang & Wu (2016) SCMP variant (mcmp_huang_256).
        porous:     If True, compile the porous-media SCMP variant (HUANG_POROUS_BUILD).
        plt_path:   Path to .plt geometry file for auto-detecting IxJ grid dimensions.
        grid:       Optional (NX, NY) override for the Huang/porous build.
        steps:      Optional HUANG_NSTEPS override for short validation runs.
        output_every: Optional HUANG_NOUTPUT override for short validation runs.
        arch:       GPU SM architecture string, e.g. "sm_120" for RTX 5090 / H100.
        debug:      If True, add -g -G flags for device-side debugging.
        output_dir: Directory to place the compiled binary. Defaults to lbm_mrt/solver/.
        dry_run:    If True, print the nvcc command but do not execute it.

    Returns:
        Return code of nvcc (0 = success). Returns 0 for dry_run.
    """
    out_dir = output_dir or SOLVER_DIR
    # Derive compute capability from arch: "sm_120" → "compute_120"
    compute = arch.replace("sm_", "compute_")

    source_names = _BASE_SOURCES + (_HYDRATE_SOURCES if hydrate else [])
    sources = [os.path.join(SOLVER_SRC, name) for name in source_names]

    # Auto-detect grid from .plt if porous mode
    if porous and plt_path and grid is None:
        grid = _detect_plt_dimensions(plt_path)
        print(f"[lbm-build] auto-detected grid {grid[0]}x{grid[1]} from {plt_path}")

    if porous:
        if grid is not None:
            nx, ny = grid
            binary_name = f"mcmp_huang_porous_{nx}x{ny}"
        else:
            binary_name = "mcmp_huang_porous"
    elif huang:
        if grid is not None:
            nx, ny = grid
            grid_suffix = f"{nx}x{ny}" if nx != ny else f"{ny}"
            binary_name = f"mcmp_huang_{grid_suffix}"
        else:
            binary_name = "mcmp_huang_256"
        if steps is not None:
            binary_name += f"_s{steps}"
    elif hydrate:
        binary_name = "mcmp_sim_hydrate"
    else:
        binary_name = "mcmp_sim"
    output_path = os.path.join(out_dir, binary_name)

    cmd: list[str] = [
        "nvcc",
        "-std=c++17",
        "-O3",
        "-rdc=true",
        "-lineinfo",
        f"-I{SOLVER_INC}",
        # Each -gencode flag must be two separate list elements for subprocess
        "-gencode",
        f"arch={compute},code={arch}",
        "-gencode",
        f"arch={compute},code={compute}",
    ]
    if porous:
        cmd.append("-DHUANG_POROUS_BUILD")
        if grid is not None:
            nx, ny = grid
            cmd.append(f"-DHUANG_NX={nx}")
            cmd.append(f"-DHUANG_NY={ny}")
        if steps is not None:
            cmd.append(f"-DHUANG_NSTEPS={steps}")
        if output_every is not None:
            cmd.append(f"-DHUANG_NOUTPUT={output_every}")
    if huang:
        cmd.append("-DHUANG_256_BUILD")
        if grid is not None:
            nx, ny = grid
            cmd.append(f"-DHUANG_NX={nx}")
            cmd.append(f"-DHUANG_NY={ny}")
        if steps is not None:
            cmd.append(f"-DHUANG_NSTEPS={steps}")
        if output_every is not None:
            cmd.append(f"-DHUANG_NOUTPUT={output_every}")
    if hydrate:
        cmd.append("-DHYDRATE_ENABLE")
    if debug:
        cmd += ["-g", "-G"]
    cmd += sources
    cmd += ["-o", output_path]

    print(f"[lbm-build] command: {' '.join(cmd)}")

    if dry_run:
        print("[lbm-build] dry-run: not executing.")
        return 0

    result = subprocess.run(cmd, cwd=SOLVER_SRC)
    if result.returncode == 0:
        size = os.path.getsize(output_path) // (1024 * 1024)
        print(f"[lbm-build] success: {output_path} ({size} MB)")
    else:
        print(
            f"[lbm-build] FAILED with exit code {result.returncode}",
            file=sys.stderr,
        )
    return result.returncode


def main() -> None:
    """CLI entry point for the lbm-build command."""
    p = argparse.ArgumentParser(
        description="Build the CUDA MRT-LBM solver binary.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--hydrate",
        action="store_true",
        help="Build the hydrate-enabled variant (mcmp_sim_hydrate).",
    )
    p.add_argument(
        "--huang",
        action="store_true",
        help="Build the Huang & Wu (2016) SCMP variant (mcmp_huang_256).",
    )
    p.add_argument(
        "--porous",
        action="store_true",
        help="Build the porous-media SCMP variant (HUANG_POROUS_BUILD). Auto-detects grid from --plt.",
    )
    p.add_argument(
        "--plt",
        default=None,
        metavar="PATH",
        help="Path to .plt geometry file for auto-detecting IxJ grid dimensions.",
    )
    p.add_argument(
        "--grid",
        default=None,
        metavar="NX[,NY]",
        help="Optional Huang grid override. Use a single value for square grids or NX,NY for rectangular grids.",
    )
    p.add_argument(
        "--steps",
        default=None,
        type=int,
        metavar="N",
        help="Optional Huang step-count override for short runs.",
    )
    p.add_argument(
        "--output-every",
        default=None,
        type=int,
        metavar="N",
        help="Optional Huang output interval override.",
    )
    p.add_argument(
        "--arch",
        default="sm_120",
        metavar="SM",
        help="GPU SM architecture string (e.g. sm_120, sm_89, sm_86).",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Add -g -G flags for device-side debugging (disables -O3).",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Directory to place the compiled binary. Defaults to lbm_mrt/solver/.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the nvcc command without executing it.",
    )
    args = p.parse_args()

    grid: tuple[int, int] | None = None
    if args.grid is not None:
        grid_text = str(args.grid).replace("x", ",").replace("X", ",")
        parts = [part.strip() for part in grid_text.split(",") if part.strip()]
        if len(parts) == 1:
            ny = int(parts[0])
            grid = (ny, ny)
        elif len(parts) == 2:
            grid = (int(parts[0]), int(parts[1]))
        else:
            raise ValueError(f"Invalid --grid value: {args.grid!r}")
    sys.exit(
        build(
            hydrate=args.hydrate,
            huang=args.huang,
            porous=args.porous,
            plt_path=args.plt,
            grid=grid,
            steps=args.steps,
            output_every=args.output_every,
            arch=args.arch,
            debug=args.debug,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
