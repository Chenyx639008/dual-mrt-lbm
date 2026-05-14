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


def build(
    hydrate: bool = False,
    huang: bool = False,
    arch: str = "sm_120",
    debug: bool = False,
    output_dir: str | None = None,
    dry_run: bool = False,
) -> int:
    """Compile the CUDA LBM solver.

    Args:
        hydrate:    If True, compile the hydrate-enabled variant.
        huang:      If True, compile the Huang & Wu (2016) SCMP variant (mcmp_huang_256).
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

    if huang:
        binary_name = "mcmp_huang_256"
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
    if huang:
        cmd.append("-DHUANG_256_BUILD")
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
    sys.exit(
        build(
            hydrate=args.hydrate,
            huang=args.huang,
            arch=args.arch,
            debug=args.debug,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
