"""Unified CLI entry point for lbm_mrt.

Provides::

    uv run lbm models              List all registered models
    uv run lbm info <name>         Show model details
    uv run lbm run <name> [...]    Run a model
    uv run lbm validate <name>     Pre-flight validation
    uv run lbm build --huang-unified  Build unified SCMP binary (Phase 2)

This wraps the existing single_run.py and build.py infrastructure.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any


def cmd_models() -> int:
    """List all registered models (SCMP + MCMP)."""
    from lbm_mrt.unified import ModelRegistry

    scmp = ModelRegistry.list_scmp()
    mcmp = ModelRegistry.list_mcmp()

    print("=== SCMP Models ===")
    for name in scmp:
        m = ModelRegistry.get(name)
        dim_str = f"{m.n_dimensions}D"
        grid_str = "×".join(str(g) for g in m.grid) if m.grid else "N/A"
        status_icon = {
            "stable": "✅",
            "experimental": "🔬",
            "pending_verification": "⏳",
        }.get(m.status, "❓")
        print(
            f"  {status_icon} {name:35s}  {dim_str} {grid_str:10s}  {m.description[:55]}"
        )

    if mcmp:
        print("\n=== MCMP Models ===")
        for name in mcmp:
            m = ModelRegistry.get(name)
            dim_str = f"{m.n_dimensions}D"
            binary = m.cuda_binary or "auto-detect"
            status_icon = {
                "stable": "✅",
                "experimental": "🔬",
                "pending_verification": "⏳",
            }.get(m.status, "❓")
            print(
                f"  {status_icon} {name:35s}  {dim_str} bin={binary:20s}  {m.description[:55]}"
            )
    else:
        print("\n(No MCMP models registered)")

    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Show detailed info for one model."""
    from lbm_mrt.unified import ModelRegistry

    name = args.model_name
    try:
        print(ModelRegistry.info(name))
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Also show generated params
    model = ModelRegistry.get(name)
    params = model.to_params_dict()
    print(f"\n--- Generated params.txt ({len(params)} keys) ---")
    for k, v in sorted(params.items()):
        print(f"  {k:30s} = {v}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run a registered model."""
    from lbm_mrt.unified import run_model

    # Collect KEY=VALUE overrides
    overrides: dict[str, Any] = {}
    for item in args.overrides or []:
        if "=" in item:
            k, v = item.split("=", 1)
            try:
                overrides[k] = float(v) if ("." in v or "e" in v.lower()) else int(v)
            except ValueError:
                overrides[k] = v

    result = run_model(
        model_name=args.model_name,
        geometry_src=args.geom,
        n_steps=args.steps,
        case_name=args.case_name,
        output_dir=args.output,
        resume=not args.fresh,
        **overrides,
    )

    if result["return_code"] != 0:
        print(
            f"\nSimulation FAILED (exit code {result['return_code']})", file=sys.stderr
        )
        return result["return_code"]

    print(f"\n✅ Simulation completed: {result['output_dir']}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Run pre-flight validation on a model."""
    from lbm_mrt.unified.runner import print_validation_report

    print_validation_report(args.model_name)
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    """Build a CUDA binary."""
    from lbm_mrt.solver.build import build

    grid = None
    if args.grid:
        parts = str(args.grid).replace("x", ",").split(",")
        if len(parts) == 1:
            grid = (int(parts[0]), int(parts[0]))
        else:
            grid = (int(parts[0]), int(parts[1]))

    return build(
        hydrate=args.hydrate,
        huang=args.huang,
        porous=args.porous,
        huang_unified=args.huang_unified,
        grid=grid,
        steps=args.steps,
        arch=args.arch,
        debug=args.debug,
        dry_run=args.dry_run,
    )


def main() -> int:
    """Main CLI dispatcher."""
    parser = argparse.ArgumentParser(
        prog="lbm",
        description="Unified LBM framework CLI (huang_mrt_2d)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run lbm models                        List all models
  uv run lbm info scmp_cs_huang_256        Show model details
  uv run lbm run scmp_cs_huang_256 --geom data/geometry/droplet.plt
  uv run lbm validate scmp_cs_huang_256    Pre-flight check
  uv run lbm build --huang-unified         Build unified SCMP binary
        """,
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # lbm models
    sub.add_parser("models", help="List registered models")

    # lbm info <name>
    p = sub.add_parser("info", help="Show model details and generated params")
    p.add_argument("model_name", help="Model identifier")

    # lbm run <name> [...]
    p = sub.add_parser("run", help="Run a simulation")
    p.add_argument("model_name", help="Model identifier")
    p.add_argument("--geom", default=None, help="Path to .plt geometry file")
    p.add_argument("--steps", type=int, default=None, help="Number of time steps")
    p.add_argument("--case-name", default=None, help="Output directory name")
    p.add_argument("--output", default=None, help="Root output directory")
    p.add_argument(
        "--fresh", action="store_true", help="Start fresh (no checkpoint resume)"
    )
    p.add_argument("overrides", nargs="*", help="KEY=VALUE overrides, e.g. cs_T=0.8")

    # lbm validate <name>
    p = sub.add_parser("validate", help="Pre-flight validation of a model")
    p.add_argument("model_name", help="Model identifier")

    # lbm build [...]
    p = sub.add_parser("build", help="Build CUDA binary")
    p.add_argument(
        "--hydrate", action="store_true", help="Build hydrate-enabled variant"
    )
    p.add_argument("--huang", action="store_true", help="Build Huang SCMP variant")
    p.add_argument(
        "--porous", action="store_true", help="Build porous-media SCMP variant"
    )
    p.add_argument(
        "--huang-unified",
        action="store_true",
        help="Build unified SCMP binary (runtime grid)",
    )
    p.add_argument("--grid", default=None, help="Grid override (e.g. 256 or 300,200)")
    p.add_argument("--steps", type=int, default=None, help="Step count override")
    p.add_argument("--arch", default="sm_120", help="GPU SM architecture")
    p.add_argument("--debug", action="store_true", help="Debug build")
    p.add_argument("--dry-run", action="store_true", help="Print nvcc command only")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    dispatch = {
        "models": lambda: cmd_models(),
        "info": lambda: cmd_info(args),
        "run": lambda: cmd_run(args),
        "validate": lambda: cmd_validate(args),
        "build": lambda: cmd_build(args),
    }

    handler = dispatch.get(args.command)
    if handler is None:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1
    return handler()


if __name__ == "__main__":
    sys.exit(main())
