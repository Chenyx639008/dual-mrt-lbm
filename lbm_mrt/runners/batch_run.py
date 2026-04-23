"""Batch runner: reads a design CSV and calls single_run.run_one() for each row.

This module is the programmatic core of the lbm-batch CLI entry point.

Usage as CLI::

    uv run lbm-batch \\
        --csv data/design_from_geometry.csv \\
        --strict-geometry

Usage as library::

    from lbm_mrt.runners.batch_run import run_batch

    n_ok, n_fail = run_batch(
        design_csv="data/design_from_geometry.csv",
        resume=True,
    )
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import timedelta
from typing import Any

import pandas as pd

from lbm_mrt.core import config as cfg
from lbm_mrt.core.paths import DATA_DIR, DEFAULT_BINARY, DEFAULT_CONFIG, RESULTS_DIR
from lbm_mrt.runners.single_run import run_one

# Columns in the design CSV that override params.txt keys.
# All other CSV columns are ignored (e.g. exp_id, case_name, geometry_src).
_META_COLS = {"exp_id", "exp_id_str", "case_name", "geometry_src"}


def run_batch(
    design_csv: str | None = None,
    config_path: str | None = None,
    out_root: str | None = None,
    app: str | None = None,
    resume: bool = True,
    strict_geometry: bool = False,
) -> tuple[int, int]:
    """Run all cases defined in a design CSV file.

    Each row in the CSV maps to one simulation case. Columns whose names match
    params.txt keys override the corresponding base config values.

    Args:
        design_csv:      Path to the design CSV. Defaults to data/design_from_geometry.csv.
        config_path:     Path to base YAML config. Defaults to configs/default.yaml.
        out_root:        Root output directory. Defaults to results/.
        app:             Path to CUDA binary. Defaults to lbm_mrt/solver/mcmp_sim.
        resume:          If False, clears checkpoints for each case before running.
        strict_geometry: If True, fail a case immediately if its geometry file is missing.

    Returns:
        Tuple of (n_ok, n_fail).
    """
    _csv = design_csv or os.path.join(DATA_DIR, "design_from_geometry.csv")
    _out_root = out_root or RESULTS_DIR
    _app = app or DEFAULT_BINARY

    base_params = cfg.load_config(config_path or DEFAULT_CONFIG)
    df = pd.read_csv(_csv)
    csv_dir = os.path.dirname(os.path.abspath(_csv))
    os.makedirs(_out_root, exist_ok=True)

    n_ok = n_fail = 0
    t0 = time.perf_counter()

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        row_dict: dict[str, Any] = row.to_dict()
        case_name: str = str(row_dict.get("case_name", f"case_{i:04d}"))
        geometry_src: str | None = row_dict.get("geometry_src", None)

        # Build per-case overrides from CSV columns that match known param keys
        overrides = {
            k: v for k, v in row_dict.items()
            if k not in _META_COLS
            and k in base_params
            and pd.notna(v)
        }
        run_params = cfg.override(base_params, **overrides)

        print(f"--- [{i}/{len(df)}] {case_name} ---")
        ret, _ = run_one(
            run_params,
            case_name=case_name,
            geometry_src=str(geometry_src) if pd.notna(geometry_src) else None,
            out_root=_out_root,
            app=_app,
            resume=resume,
            strict_geometry=strict_geometry,
            hint_dir=csv_dir,
        )
        if ret == 0:
            n_ok += 1
        else:
            n_fail += 1

    elapsed = time.perf_counter() - t0
    print(
        f"\n=== Batch done: total={len(df)}, ok={n_ok}, fail={n_fail}, "
        f"elapsed={timedelta(seconds=int(elapsed))} ==="
    )
    return n_ok, n_fail


def main() -> None:
    """CLI entry point for the lbm-batch command."""
    p = argparse.ArgumentParser(
        description="Batch-run LBM simulations from a design CSV file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--csv",
        default=os.path.join(DATA_DIR, "design_from_geometry.csv"),
        metavar="CSV",
        help="Path to the design CSV file.",
    )
    p.add_argument(
        "--config",
        default=None,
        metavar="YAML",
        help="Path to the base YAML config (defaults to configs/default.yaml).",
    )
    p.add_argument(
        "--out-root",
        default=None,
        metavar="DIR",
        help="Root output directory (defaults to results/).",
    )
    p.add_argument(
        "--app",
        default=None,
        metavar="BIN",
        help="Path to the compiled CUDA binary (defaults to lbm_mrt/solver/mcmp_sim).",
    )
    p.add_argument(
        "--resume",
        choices=["0", "1"],
        default=os.environ.get("RESUME", "1"),
        help="1=continue from checkpoint; 0=start fresh.",
    )
    p.add_argument(
        "--strict-geometry",
        action="store_true",
        help="Fail a case immediately if its geometry file is not found.",
    )
    args = p.parse_args()

    n_ok, n_fail = run_batch(
        design_csv=args.csv,
        config_path=args.config,
        out_root=args.out_root,
        app=args.app,
        resume=(args.resume == "1"),
        strict_geometry=args.strict_geometry,
    )
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
