"""Centralized path conventions for lbm_mrt.

All runtime I/O is rooted under the project root in standard subdirectories.
Import from here rather than constructing paths inline to keep path logic in one place.
"""

from __future__ import annotations

import os

# Project root: two levels up from this file (lbm_mrt/core/paths.py → project root)
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Standard top-level directories
CONFIGS_DIR = os.path.join(PROJ_ROOT, "configs")
DATA_DIR = os.path.join(PROJ_ROOT, "data")
GEOMETRY_DIR = os.path.join(DATA_DIR, "geometry")
RESULTS_DIR = os.path.join(PROJ_ROOT, "results")
LOGS_DIR = os.path.join(PROJ_ROOT, "logs")

# Solver directories
SOLVER_DIR = os.path.join(PROJ_ROOT, "lbm_mrt", "solver")
SOLVER_SRC = os.path.join(SOLVER_DIR, "src")
SOLVER_INC = os.path.join(SOLVER_DIR, "include")

# Default config and binary paths
DEFAULT_CONFIG = os.path.join(CONFIGS_DIR, "default.yaml")
DEFAULT_BINARY = os.path.join(SOLVER_DIR, "mcmp_sim")
HYDRATE_BINARY = os.path.join(SOLVER_DIR, "mcmp_sim_hydrate")


def case_results_dir(case_name: str) -> str:
    """Return the canonical results directory for a single simulation case.

    Args:
        case_name: Unique identifier for the case (becomes the subdirectory name).

    Returns:
        Absolute path to the case output directory under RESULTS_DIR.
    """
    return os.path.join(RESULTS_DIR, case_name)


def ensure_dirs() -> None:
    """Create all standard runtime directories if they do not already exist."""
    for d in [DATA_DIR, GEOMETRY_DIR, RESULTS_DIR, LOGS_DIR]:
        os.makedirs(d, exist_ok=True)
