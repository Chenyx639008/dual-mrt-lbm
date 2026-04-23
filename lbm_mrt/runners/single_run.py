"""Run a single LBM simulation from a params dict.

This module is the programmatic core of the lbm-run CLI entry point.
It handles directory setup, geometry file copying, params.txt writing,
and binary invocation via subprocess.

Usage as CLI::

    uv run lbm-run \\
        --geom data/geometry/case0001_pf/geometry_case0001.plt \\
        --case-name smoke_test \\
        --Sw 0.5

Usage as library::

    from lbm_mrt.core.config import load_config, override
    from lbm_mrt.runners.single_run import run_one

    params = override(load_config(), Sw=0.5)
    ret, elapsed = run_one(
        params,
        case_name="my_case",
        geometry_src="data/geometry/case0001_pf/geometry_case0001.plt",
    )
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import timedelta
from typing import Any

from lbm_mrt.core.config import load_config, override
from lbm_mrt.core.paths import DEFAULT_BINARY, RESULTS_DIR
from lbm_mrt.io.params_writer import write_params_txt

# The geometry file is always copied to this name inside the case directory,
# and the CUDA binary reads it under this name via the geom_file param.
_GEOM_COPY_NAME = "geometry_case.plt"


def resolve_geometry_path(geom_src: str | None, hint_dir: str | None = None) -> str | None:
    """Resolve a geometry source path to an absolute path.

    Tries the path as-is first; if it is relative and does not exist,
    tries resolving it relative to hint_dir (e.g. the CSV directory).

    Args:
        geom_src:  Geometry file path (may be relative or absolute).
        hint_dir:  Directory to try if the direct path doesn't exist.

    Returns:
        Absolute path if found, else None.
    """
    if not geom_src:
        return None
    if os.path.isfile(geom_src):
        return os.path.abspath(geom_src)
    if hint_dir:
        alt = os.path.join(hint_dir, geom_src)
        if os.path.isfile(alt):
            return os.path.abspath(alt)
    return None


def run_one(
    params: dict[str, Any],
    case_name: str,
    geometry_src: str | None = None,
    out_root: str | None = None,
    app: str | None = None,
    resume: bool = True,
    strict_geometry: bool = False,
    hint_dir: str | None = None,
) -> tuple[int, float]:
    """Execute a single simulation case.

    Sets up the output directory, copies the geometry file, writes params.txt,
    and runs the compiled CUDA binary via subprocess.

    The binary receives the output directory via the LBM_FILE_DIR environment
    variable and the checkpoint directory via LBM_CKPT_DIR.

    Args:
        params:          Flat params dict (from config.load_config / override).
        case_name:       Unique identifier; becomes the output subdirectory name.
        geometry_src:    Path to the .plt Tecplot geometry file (optional).
        out_root:        Root results directory; defaults to paths.RESULTS_DIR.
        app:             Path to the compiled CUDA binary; defaults to DEFAULT_BINARY.
        resume:          If False, removes existing checkpoints before running.
        strict_geometry: If True and the geometry file is missing, fail immediately.
        hint_dir:        Directory to try for relative geometry paths.

    Returns:
        Tuple of (return_code, elapsed_seconds).
    """
    _out_root = out_root or RESULTS_DIR
    _app = app or DEFAULT_BINARY

    exp_dir = os.path.join(_out_root, case_name)
    ckpt_dir = os.path.join(exp_dir, "ckpt")
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Optionally clear checkpoints for a fresh run
    if not resume:
        for fn in os.listdir(ckpt_dir):
            if fn.startswith("ckpt_"):
                try:
                    os.remove(os.path.join(ckpt_dir, fn))
                except OSError:
                    pass

    # Copy geometry file into case directory and register its path in params
    run_params = dict(params)
    geom_abs = resolve_geometry_path(geometry_src, hint_dir)
    if geom_abs:
        geom_dst = os.path.join(exp_dir, _GEOM_COPY_NAME)
        shutil.copy(geom_abs, geom_dst)
        run_params["geom_file"] = os.path.abspath(geom_dst)
    else:
        if geometry_src:
            print(f"[WARN] Geometry file not found: {geometry_src}")
        if strict_geometry:
            print(f"[FAIL] strict_geometry=True; aborting case {case_name!r}")
            return 2, 0.0

    # Write params.txt for this case
    params_path = os.path.join(exp_dir, "params.txt")
    write_params_txt(run_params, params_path)

    print(f">>> Running case: {case_name!r}  →  {exp_dir}")
    t0 = time.perf_counter()

    # Pass output and checkpoint directories to the binary via environment variables
    env = os.environ.copy()
    env["LBM_FILE_DIR"] = exp_dir
    env["LBM_CKPT_DIR"] = ckpt_dir

    log_path = os.path.join(exp_dir, "log.txt")
    with open(log_path, "w") as logf:
        ret = subprocess.run(
            [_app, params_path],
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=False,
            env=env,
        ).returncode

    elapsed = time.perf_counter() - t0
    status = "ok" if ret == 0 else f"FAILED(exit={ret})"
    print(f"[{status}] {case_name} | elapsed={timedelta(seconds=int(elapsed))} ({elapsed:.1f}s)")
    return ret, elapsed


def main() -> None:
    """CLI entry point for the lbm-run command."""
    p = argparse.ArgumentParser(
        description="Run a single LBM simulation case.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=None, metavar="YAML",
                   help="Path to YAML config file (defaults to configs/default.yaml).")
    p.add_argument("--case-name", required=True, metavar="NAME",
                   help="Case identifier; output will be written to results/{NAME}/.")
    p.add_argument("--geom", default=None, metavar="PLT",
                   help="Path to the Tecplot .plt geometry file.")
    p.add_argument("--app", default=None, metavar="BIN",
                   help="Path to the compiled CUDA binary (defaults to lbm_mrt/solver/mcmp_sim).")
    p.add_argument("--out-root", default=None, metavar="DIR",
                   help="Root output directory (defaults to results/).")
    p.add_argument("--resume", choices=["0", "1"],
                   default=os.environ.get("RESUME", "1"),
                   help="1=continue from checkpoint; 0=start fresh.")
    p.add_argument("--strict-geometry", action="store_true",
                   help="Fail immediately if the geometry file is not found.")
    # Allow arbitrary key=value param overrides
    p.add_argument("overrides", nargs="*", metavar="KEY=VALUE",
                   help="Additional param overrides, e.g. Sw=0.5 Gx=2e-8.")
    args = p.parse_args()

    params = load_config(args.config)

    # Parse KEY=VALUE overrides from positional args
    kv_overrides: dict[str, Any] = {}
    for token in args.overrides:
        if "=" not in token:
            print(f"[ERROR] Invalid override {token!r}; expected KEY=VALUE", file=sys.stderr)
            sys.exit(2)
        k, _, v = token.partition("=")
        # Try numeric conversion
        try:
            kv_overrides[k] = int(v)
        except ValueError:
            try:
                kv_overrides[k] = float(v)
            except ValueError:
                kv_overrides[k] = v

    if kv_overrides:
        params = override(params, **kv_overrides)

    ret, _ = run_one(
        params,
        case_name=args.case_name,
        geometry_src=args.geom,
        out_root=args.out_root,
        app=args.app,
        resume=(args.resume == "1"),
        strict_geometry=args.strict_geometry,
    )
    sys.exit(ret)


if __name__ == "__main__":
    main()
