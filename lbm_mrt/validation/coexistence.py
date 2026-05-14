"""Flat-interface coexistence validation for SCMP.

Implements:
1. Flat-interface initialization (stratified layer)
2. Density profile extraction at equilibrium
3. Coexistence point identification (ρ_l, ρ_g)
4. Comparison to Maxwell construction (analytical)
5. Pressure tensor forms (discrete vs continuum)

Reference: Huang & Wu (2016) Fig. 3-4
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class FlatInterfaceResult:
    """Equilibrium state of a flat-interface SCMP simulation."""

    T_reduced: float
    """Reduced temperature T/Tc"""

    tau: float
    """Relaxation time τ"""

    rho_g: float
    """Gas density (plateau value)"""

    rho_l: float
    """Liquid density (plateau value)"""

    interface_pos: float
    """Detected interface position (in lattice units)"""

    max_grad_rho: float
    """Maximum density gradient magnitude"""

    equilibration_steps: int
    """Number of time steps to reach equilibrium"""

    config: dict
    """Full simulation config used"""


def extract_flat_interface_profile(
    vtk_file: str, direction: str = "y"
) -> tuple[np.ndarray, np.ndarray, Optional[float]]:
    """Extract lineout of density perpendicular to interface.

    Args:
        vtk_file: Path to VTK output from flat-interface SCMP run
        direction: 'x' or 'y' — direction perpendicular to interface

    Returns:
        (line_coord, rho_profile, interface_pos_detected)
        where interface_pos is None if not clearly bimodal
    """
    from lbm_mrt.io.vtk_reader import read_vtk_scalars

    fields, nx, ny = read_vtk_scalars(vtk_file)
    rho = fields.get("rho")
    if rho is None:
        raise ValueError(f"'rho' scalar not found in {vtk_file}")

    # Reshape to 2D grid
    rho_2d = rho.reshape((ny, nx))

    if direction == "y":
        # Average over x, lineout in y
        profile = np.mean(rho_2d, axis=1)
        coord = np.arange(len(profile))
    elif direction == "x":
        # Average over y, lineout in x
        profile = np.mean(rho_2d, axis=0)
        coord = np.arange(len(profile))
    else:
        raise ValueError("direction must be 'x' or 'y'")

    # Detect interface: bimodal distribution with minimum in middle
    # Smooth to avoid noise
    profile_smooth = np.convolve(profile, np.ones(5) / 5, mode="same")
    interface_idx = np.argmin(profile_smooth)

    # Extract plateau values
    n_plateau = len(profile) // 8  # Use first/last 1/8
    rho_left = np.median(profile[:n_plateau])
    rho_right = np.median(profile[-n_plateau:])
    rho_g = min(rho_left, rho_right)
    rho_l = max(rho_left, rho_right)

    # Verify bimodal (gap between phases)
    if rho_l - rho_g < 0.001 * max(rho_l, rho_g):
        # Not phase-separated
        return coord, profile, None

    return coord, profile, float(interface_idx)


def identify_coexistence_point(
    rho_profile: np.ndarray,
) -> tuple[float, float]:
    """Extract (ρ_g, ρ_l) from equilibrium profile.

    Assumes:
    - Gas on left, liquid on right (or vice versa)
    - Plateau regions exist at both ends
    - Interface is smooth transition in middle

    Returns:
        (rho_g, rho_l) — gas and liquid densities
    """
    n = len(rho_profile)
    n_plateau = max(n // 8, 5)

    # Median of plateau regions (robust to noise)
    left_half = np.median(rho_profile[:n_plateau])
    right_half = np.median(rho_profile[-n_plateau:])

    rho_g = min(left_half, right_half)
    rho_l = max(left_half, right_half)

    return rho_g, rho_l


def compute_density_interface_tension_from_profile(
    coord: np.ndarray,
    rho_profile: np.ndarray,
    T_reduced: float,
    cs_a: float = 1.0,
    cs_b: float = 4.0,
    cs_R: float = 1.0,
    cs_G: float = -1.0,
) -> tuple[float, float]:
    """Compute surface tension from density profile via pressure integration.

    σ = ∫ (p_nn - p_tt) dz

    where p_nn (normal stress) and p_tt (tangential stress) are computed
    from the pressure tensor.

    This is a placeholder; full implementation requires:
    - Pressure tensor computation from discrete form (P_discrete)
    - Or from continuum thermodynamic pressure

    Returns:
        (sigma, interface_thickness) — approximate values
    """
    # For now, return placeholder
    # Full implementation deferred to Phase 5 (Huang paper Section 5.3)
    return 0.0, 0.0


def pressure_tensor_discrete_form(
    rho_profile: np.ndarray,
    ux_profile: np.ndarray,
    uy_profile: np.ndarray,
    cs_a: float = 1.0,
    cs_b: float = 4.0,
    cs_R: float = 1.0,
    cs_G: float = -1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute pressure tensor from discrete distribution (LBM moments).

    P_discrete_αβ = Σ_k e_k_α e_k_β f_k

    In equilibrium + mechanical stability condition:
    Δp_normal = ∂P_nn/∂ρ = 0

    Returns:
        (p_normal, p_tangential) — stress tensor diagonal elements
    """
    # Placeholder for Phase 4
    # Requires velocity and distribution field data from VTK
    raise NotImplementedError(
        "Pressure tensor discrete form computation deferred to Phase 5"
    )


def pressure_tensor_continuum_form(
    rho_profile: np.ndarray,
    T_reduced: float,
    cs_a: float = 1.0,
    cs_b: float = 4.0,
    cs_R: float = 1.0,
    cs_G: float = -1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute pressure tensor from continuum thermodynamics.

    P_continuum_αβ = p(ρ)δ_αβ + surface_tension·κ·n_α·n_β

    where:
    - p(ρ) from CS-EOS
    - κ is interfacial curvature (1D profile → κ=0)
    - n is interface normal

    Returns:
        (p_normal, p_tangential) — stress tensor diagonal elements
    """
    from lbm_mrt.validation.cs_eos import cs_pressure

    # In 1D, only normal direction (z) exists
    # p_normal = p_thermodynamic + surface_energy_contribution
    # p_tangential = p_thermodynamic

    rho_c = 0.0326  # Critical density for a=1, b=4, R=1
    T_c = 0.09433  # Critical temperature

    T = T_reduced * T_c

    p_profile = np.array([cs_pressure(rho, cs_a, cs_b, cs_R, T) for rho in rho_profile])

    # Approximate: tangential = hydrostatic pressure
    p_tangential = p_profile

    # Normal stress includes capillary pressure (interface curvature)
    # For 1D flat interface: Δp_capillary ≈ -σ·∇²ρ / ρ (interface tension)
    # Simplified: normal = tangential + capillary correction
    p_normal = p_profile  # Placeholder (full form requires interface curvature)

    return p_normal, p_tangential


def compare_coexistence_curves(
    maxwell_data: dict,
    numerical_data: list[FlatInterfaceResult],
    out_dir: str = "results",
) -> None:
    """Generate comparison plot: Maxwell vs analytical pressure tensors vs numerical.

    Args:
        maxwell_data: Output from cs_eos.coexistence_curve() (dict with T, rho_l, rho_g)
        numerical_data: List of FlatInterfaceResult from simulations
        out_dir: Output directory for plots
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from lbm_mrt.viz.viz_template import init_style, create_figure_ax, save_figure

    init_style()
    fig, ax = create_figure_ax(figsize=(8, 6))

    # Plot 1: Maxwell construction (analytical)
    if maxwell_data:
        T_vals = np.array(maxwell_data["T"])
        Tc = maxwell_data.get("Tc", 0.09433)
        Tr_vals = T_vals / Tc

        rho_l = np.array(maxwell_data["rho_l"])
        rho_g = np.array(maxwell_data["rho_g"])

        ax.plot(
            rho_l,
            Tr_vals,
            "b-",
            lw=2.0,
            label="Maxwell (thermodynamic)",
            alpha=0.7,
        )
        ax.plot(
            rho_g,
            Tr_vals,
            "b-",
            lw=2.0,
            alpha=0.7,
        )

    # Plot 2: Numerical simulation points
    if numerical_data:
        tau_1_data = [d for d in numerical_data if abs(d.tau - 1.0) < 0.01]
        tau_15_data = [d for d in numerical_data if abs(d.tau - 1.5) < 0.01]

        if tau_1_data:
            Tr_1 = np.array([d.T_reduced for d in tau_1_data])
            rho_g_1 = np.array([d.rho_g for d in tau_1_data])
            rho_l_1 = np.array([d.rho_l for d in tau_1_data])
            ax.plot(
                rho_g_1,
                Tr_1,
                "go",
                ms=6,
                label="Numerical τ=1 (gas)",
                alpha=0.6,
            )
            ax.plot(
                rho_l_1,
                Tr_1,
                "go",
                ms=6,
                label="Numerical τ=1 (liquid)",
                alpha=0.6,
            )

        if tau_15_data:
            Tr_15 = np.array([d.T_reduced for d in tau_15_data])
            rho_g_15 = np.array([d.rho_g for d in tau_15_data])
            rho_l_15 = np.array([d.rho_l for d in tau_15_data])
            ax.plot(
                rho_g_15,
                Tr_15,
                "m^",
                ms=6,
                label="Numerical τ=1.5 (gas)",
                alpha=0.6,
            )
            ax.plot(
                rho_l_15,
                Tr_15,
                "m^",
                ms=6,
                label="Numerical τ=1.5 (liquid)",
                alpha=0.6,
            )

    # Format
    ax.set_xlabel(r"Density $\rho$ (log scale)")
    ax.set_ylabel(r"Reduced temperature $T/T_c$")
    ax.set_title(
        "Coexistence Curve Comparison — Huang & Wu (2016) Fig. 3-4\n"
        "Maxwell vs. Pressure Tensor Forms vs. Numerical Simulation"
    )
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 1e-1)
    ax.set_ylim(0.55, 1.02)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3, which="both")

    os.makedirs(out_dir, exist_ok=True)
    save_figure(fig, os.path.join(out_dir, "coexistence_comparison"), dpi=150)
    plt.close(fig)

    print(f"Saved: {out_dir}/coexistence_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: Placeholder for analytical pressure tensor computation
# ─────────────────────────────────────────────────────────────────────────────


def load_coexistence_reference() -> dict:
    """Load Maxwell coexistence reference data from JSON."""
    ref_path = Path("data/validation_reference/huang_2016/cs_coexistence.json")
    if not ref_path.exists():
        raise FileNotFoundError(f"Missing reference: {ref_path}")

    with open(ref_path) as f:
        return json.load(f)
