"""Unified model runner for lbm_mrt.

Translates a ModelDefinition into a params.txt file and invokes the
appropriate CUDA binary via the existing single_run.py infrastructure.

Design principle: wraps, does not replace. All heavy lifting is delegated
to single_run.py → subprocess → CUDA binary. This module only adds the
model-abstraction layer on top.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from datetime import timedelta
from typing import Any

from lbm_mrt.core.paths import RESULTS_DIR, SOLVER_DIR
from lbm_mrt.io.params_writer import write_params_txt

from .models import ModelDefinition, ModelRegistry


# ═══════════════════════════════════════════════════════════════════════════
# Core runner
# ═══════════════════════════════════════════════════════════════════════════


def run_model(
    model_name: str,
    geometry_src: str | None = None,
    n_steps: int | None = None,
    case_name: str | None = None,
    output_dir: str | None = None,
    resume: bool = False,
    **overrides: Any,
) -> dict[str, Any]:
    """Run a registered model with optional parameter overrides.

    This is the primary entry point. It:
    1. Looks up the model definition from ModelRegistry
    2. Generates the params.txt dict (with optional overrides)
    3. Copies the geometry file into the case directory
    4. Invokes the CUDA binary via subprocess

    Parameters
    ----------
    model_name : str
        Registered model identifier (e.g. "scmp_cs_huang_256").
    geometry_src : str or None
        Path to the Tecplot .plt geometry file.
    n_steps : int or None
        Override the default number of time steps.
    case_name : str or None
        Output subdirectory name under results/. Defaults to model_name.
    output_dir : str or None
        Root results directory. Defaults to results/.
    resume : bool
        If True, continue from checkpoint; if False, start fresh.
    **overrides : Any
        Additional params.txt key=value overrides applied after model defaults.

    Returns
    -------
    dict
        {
            "case_name": str,
            "return_code": int,
            "elapsed_seconds": float,
            "output_dir": str,
            "params": dict,     # final params sent to CUDA
            "model": str,       # model name used
        }
    """
    # ── Lookup model ──
    model = ModelRegistry.get(model_name)

    # ── Build params dict ──
    params = model.to_params_dict()
    params.update(overrides)

    # ── Setup directories ──
    _output_dir = output_dir or RESULTS_DIR
    _case_name = case_name or model_name
    exp_dir = os.path.join(_output_dir, _case_name)
    ckpt_dir = os.path.join(exp_dir, "ckpt")
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Fresh start: clear checkpoints ──
    if not resume:
        for fn in os.listdir(ckpt_dir):
            if fn.startswith("ckpt_"):
                try:
                    os.remove(os.path.join(ckpt_dir, fn))
                except OSError:
                    pass

    # ── Handle geometry file ──
    geom_dst = os.path.join(exp_dir, "geometry_case.plt")
    if geometry_src and os.path.isfile(geometry_src):
        shutil.copy(geometry_src, geom_dst)
        params["geom_file"] = os.path.abspath(geom_dst)
    elif geometry_src:
        print(f"[WARN] Geometry file not found: {geometry_src}")

    # ── Apply n_steps override as flow_max_steps ──
    if n_steps is not None and n_steps > 0:
        params["flow_max_steps"] = n_steps

    # ── Write params.txt ──
    params_path = os.path.join(exp_dir, "params.txt")
    write_params_txt(params, params_path)

    # ── Select binary ──
    binary = model.resolve_binary()
    if not os.path.isfile(binary):
        build_hint = "uv run lbm-build --huang"
        if model.model_family == "mcmp":
            build_hint = (
                "uv run lbm-build"
                if "hydrate" not in binary
                else "uv run lbm-build --hydrate"
            )
        raise FileNotFoundError(
            f"CUDA binary not found: {binary}\nBuild it with: {build_hint}"
        )

    # ── Run ──
    print(f">>> [{model_name}] Running → {exp_dir}")
    print(f"    Binary: {binary}")
    print(f"    Steps:  {params.get('flow_max_steps', 'compile-time default')}")
    t0 = time.perf_counter()

    env = os.environ.copy()
    env["LBM_FILE_DIR"] = exp_dir
    env["LBM_CKPT_DIR"] = ckpt_dir

    log_path = os.path.join(exp_dir, "log.txt")
    with open(log_path, "w") as logf:
        ret = subprocess.run(
            [binary, params_path],
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=False,
            env=env,
        ).returncode

    elapsed = time.perf_counter() - t0
    status = "OK" if ret == 0 else f"FAILED (exit={ret})"
    print(f"    [{status}] elapsed={timedelta(seconds=int(elapsed))} ({elapsed:.1f}s)")

    return {
        "case_name": _case_name,
        "return_code": ret,
        "elapsed_seconds": elapsed,
        "output_dir": exp_dir,
        "params": params,
        "model": model_name,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Convenience wrappers
# ═══════════════════════════════════════════════════════════════════════════


def run_scmp(
    model_name: str = "scmp_cs_huang_256",
    geom: str | None = None,
    steps: int | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Convenience wrapper for SCMP model runs.

    Parameters
    ----------
    model_name : str
        SCMP model name (must be registered with model_family="scmp").
    geom : str or None
        Path to geometry .plt file.
    steps : int or None
        Number of time steps.
    **overrides
        Additional param overrides.

    Returns
    -------
    dict
        Same as run_model().
    """
    model = ModelRegistry.get(model_name)
    if model.model_family != "scmp":
        raise ValueError(
            f"run_scmp requires an SCMP model, but '{model_name}' "
            f"is a '{model.model_family}' model."
        )
    return run_model(model_name, geometry_src=geom, n_steps=steps, **overrides)


def validate_model_definition(model: ModelDefinition) -> list[str]:
    """Validate a ModelDefinition and return a list of warnings/issues.

    This performs pre-flight checks that catch common configuration errors
    before the CUDA binary runs. Call this during development of new models.

    Parameters
    ----------
    model : ModelDefinition
        The model definition to validate.

    Returns
    -------
    list[str]
        List of warning/error messages. Empty list means all checks passed.
    """
    issues: list[str] = []

    # Check binary exists
    try:
        binary = model.resolve_binary()
        if not os.path.isfile(binary):
            issues.append(f"CUDA binary not found: {binary}")
    except Exception as e:
        issues.append(f"Failed to resolve binary path: {e}")

    # Check EOS consistency
    params = model.to_params_dict()
    if model.model_family == "scmp":
        if params.get("cs_T", 0) >= 1.0:
            issues.append(
                f"cs_T={params['cs_T']} >= 1.0 — above critical point, "
                f"no two-phase region."
            )
        if params.get("cs_T", 0) < 0.4:
            issues.append(
                f"cs_T={params['cs_T']} < 0.4 — very low temperature, "
                f"may cause numerical instability."
            )

    # Check grid vs binary compatibility
    if model.grid:
        nx, ny = model.grid
        if nx > 1024 or ny > 1024:
            issues.append(f"Grid {nx}×{ny} may exceed GPU memory for SCMP.")

    return issues


def print_validation_report(model_name: str) -> None:
    """Print a validation report for a registered model.

    Parameters
    ----------
    model_name : str
        Registered model identifier.
    """
    model = ModelRegistry.get(model_name)
    print(f"=== Validation Report: {model_name} ===")
    print(ModelRegistry.info(model_name))
    print()

    issues = validate_model_definition(model)
    if issues:
        print("⚠️  Issues found:")
        for issue in issues:
            print(f"   • {issue}")
    else:
        print("✅ All pre-flight checks passed.")
    print()
    print("Params that would be sent to CUDA:")
    params = model.to_params_dict()
    for k, v in sorted(params.items()):
        print(f"   {k:30s} = {v}")
