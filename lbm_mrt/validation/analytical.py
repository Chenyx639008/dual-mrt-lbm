"""Shared post-processing for Huang & Wu (2016) SCMP validation.

Public API:
- detect_interface_radius(rho, nx, ny) → center, radius
- fit_pressure_inside_outside(rho, pressure, center, radius) → (p_in, p_out)
- laplace_sigma(R_list, ΔP_list) → sigma, R², intercept
- centerline_profile(rho, axis='y') → y, rho_profile
- extract_rho_l_g(rho, ny) → (rho_l, rho_g)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def detect_interface_radius(
    rho: NDArray[np.float64], nx: int, ny: int
) -> tuple[tuple[float, float], float]:
    """Detect droplet center and radius from density field.

    Uses the mid-density isosurface to find the interface.

    Returns ((cx, cy), radius).
    """
    rho_min = np.min(rho)
    rho_max = np.max(rho)
    rho_mid = 0.5 * (rho_min + rho_max)

    # Center of mass of dense region
    y_idx, x_idx = np.mgrid[0:ny, 0:nx]
    mask = rho > rho_mid
    if np.sum(mask) < 10:
        return ((float(nx) / 2, float(ny) / 2), 0.0)

    total = np.sum(rho[mask])
    cx = float(np.sum(x_idx[mask] * rho[mask]) / total)
    cy = float(np.sum(y_idx[mask] * rho[mask]) / total)

    # Radial density profile from center
    r_vals = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
    # Sort by radius for interface detection
    sorted_idx = np.argsort(r_vals.flatten())
    r_sorted = r_vals.flatten()[sorted_idx]
    rho_sorted = rho.flatten()[sorted_idx]

    # Find where rho crosses rho_mid
    cross_idx = np.where(np.diff(np.sign(rho_sorted - rho_mid)))[0]
    if len(cross_idx) > 0:
        radius = float(np.mean(r_sorted[cross_idx[: min(3, len(cross_idx))]]))
    else:
        radius = 0.0

    return ((cx, cy), radius)


def fit_pressure_inside_outside(
    rho: NDArray[np.float64],
    pressure: NDArray[np.float64],
    center: tuple[float, float],
    radius: float,
) -> tuple[float, float]:
    """Estimate pressure inside and outside the droplet.

    Returns (p_inside, p_outside).
    """
    ny, nx = rho.shape
    y_idx, x_idx = np.mgrid[0:ny, 0:nx]
    r_vals = np.sqrt((x_idx - center[0]) ** 2 + (y_idx - center[1]) ** 2)

    # Inside: r < 0.6*R, outside: r > 1.5*R
    inner_mask = r_vals < 0.6 * radius
    outer_mask = (r_vals > 1.5 * radius) & (r_vals < min(nx, ny) * 0.4)

    p_in = float(np.mean(pressure[inner_mask])) if np.any(inner_mask) else 0.0
    p_out = float(np.mean(pressure[outer_mask])) if np.any(outer_mask) else 0.0
    return p_in, p_out


def laplace_sigma(
    R_list: NDArray[np.float64],
    delta_P_list: NDArray[np.float64],
) -> tuple[float, float, float]:
    """Fit Laplace law: ΔP = σ / R.

    Returns (sigma, r_squared, intercept).
    """
    inv_R = 1.0 / np.array(R_list)
    coeffs = np.polyfit(inv_R, delta_P_list, 1)
    sigma = float(coeffs[0])
    intercept = float(coeffs[1])

    # R²
    fit = np.polyval(coeffs, inv_R)
    ss_res = np.sum((delta_P_list - fit) ** 2)
    ss_tot = np.sum((delta_P_list - np.mean(delta_P_list)) ** 2)
    r_squared = float(1.0 - ss_res / max(ss_tot, 1e-12))

    return sigma, r_squared, intercept


def centerline_profile(
    rho: NDArray[np.float64],
    axis: str = "y",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Extract density profile along centerline.

    Returns (position, density).
    """
    ny, nx = rho.shape
    if axis == "y":
        pos = np.arange(ny)
        profile = rho[:, nx // 2]
    else:
        pos = np.arange(nx)
        profile = rho[ny // 2, :]
    return pos, profile


def extract_rho_l_g(
    rho: NDArray[np.float64],
    ny: int,
    margin_frac: float = 0.10,
) -> tuple[float, float]:
    """Extract liquid/gas coexistence densities from a phase-separated field.

    Auto-detects interface orientation (horizontal or vertical) by comparing
    variance along each axis. Works with flat interfaces, slabs, and vertical
    interfaces that arise from periodic-BC slab evolution.

    Args:
        rho: 2D density field (ny, nx).
        ny: Grid size in y (unused; kept for backward compatibility).
        margin_frac: Fraction of domain to exclude from interface region.

    Returns:
        (rho_l, rho_g) — liquid (high density) and gas (low density).
    """
    ny_arr, nx_arr = rho.shape

    # Detect orientation: compare variance along x-averaged and y-averaged profiles
    y_profile = rho.mean(axis=1)  # avg over x → varies with y
    x_profile = rho.mean(axis=0)  # avg over y → varies with x
    y_var = float(np.std(y_profile))
    x_var = float(np.std(x_profile))

    # Choose the axis with larger variance (where the interface is)
    if x_var >= y_var:
        # Interface is vertical (varies in x)
        profile = x_profile
        n = nx_arr
    else:
        # Interface is horizontal (varies in y)
        profile = y_profile
        n = ny_arr

    # If profile is essentially flat (no phase separation), return density extremes
    profile_range = float(np.max(profile) - np.min(profile))
    if profile_range < 1e-6:
        # Uniform field — use 2D extremes as best guess
        rho_max = float(np.max(rho))
        rho_min = float(np.min(rho))
        if rho_max > rho_min:
            return rho_max, rho_min
        return rho_max, rho_max

    # Find interface position via maximum gradient
    grad = np.abs(np.gradient(profile))
    interface_idx = int(np.argmax(grad))

    # Exclude interface region (± margin_frac * n)
    margin = max(int(margin_frac * n), 5)
    lo_region = slice(0, max(interface_idx - margin, 1))
    hi_region = slice(min(interface_idx + margin, n - 1), n)

    # Determine which side is liquid (higher density)
    rho_lo = float(np.mean(profile[lo_region]))
    rho_hi = float(np.mean(profile[hi_region]))
    if rho_lo > rho_hi:
        rho_l, rho_g = rho_lo, rho_hi
    else:
        rho_l, rho_g = rho_hi, rho_lo
    return rho_l, rho_g


def compute_psi_from_rho(
    rho: NDArray[np.float64],
    cs_a: float = 1.0,
    cs_b: float = 4.0,
    cs_R: float = 1.0,
    cs_T: float = 0.70,
    cs_G: float = -1.0,
    cs2: float = 1.0 / 3.0,
) -> NDArray[np.float64]:
    """Compute pseudopotential ψ from density using Carnahan-Starling EOS.

    ψ = sqrt(2(p_eos − ρ·cs²) / (|G|·dx²))  with dx = 1 (lattice units).

    Args:
        rho: Density field.
        cs_a, cs_b, cs_R: CS-EOS parameters (default: 1.0, 4.0, 1.0).
        cs_T: Reduced temperature T/Tc (NOT absolute temperature).
        cs_G: Interaction strength (default: -1.0).
        cs2: Lattice sound speed squared (default: 1/3 for D2Q9).

    Returns:
        Pseudopotential ψ field, same shape as rho.
    """
    rho_safe = np.maximum(rho, 1e-8)
    Tc = 0.3773 * cs_a / (cs_b * cs_R + 1e-20)
    T_actual = cs_T * Tc

    # η = bρ/4
    eta = cs_b * rho_safe / 4.0
    eta2 = eta * eta
    eta3 = eta2 * eta
    denom = 1.0 - eta
    denom3 = np.maximum(denom * denom * denom, 1e-12)

    # p_eos = ρRT(1+η+η²−η³)/(1−η)³ − aρ²
    p_eos = (
        cs_R * T_actual * rho_safe * (1.0 + eta + eta2 - eta3) / denom3
        - cs_a * rho_safe * rho_safe
    )

    # ψ = sqrt(2(p − ρcs²) / (|G|·dx²))  with dx=1
    diff = p_eos - rho_safe * cs2
    numer = 2.0 * np.maximum(-diff, 0.0)
    gdenom = np.maximum(abs(cs_G), 1e-12)
    psi = np.sqrt(numer / gdenom)
    return psi


def compute_sigma_from_rho(
    rho: NDArray[np.float64],
    center: tuple[float, float],
    k1: float = 1.0 / 12.0,
    cs_a: float = 1.0,
    cs_b: float = 4.0,
    cs_R: float = 1.0,
    cs_T: float = 0.70,
    cs_G: float = -1.0,
    n_bins: int = 200,
) -> float:
    """Compute surface tension σ from density field via paper Eq. 62 integral.

    σ = −(G/6)(1−6k₁) ∫ (dψ/dr)² dr

    This method uses the pseudopotential ψ(r) computed from ρ(r) via CS-EOS,
    then integrates the squared radial gradient across the interface.
    It does NOT require the pressure tensor VTK fields (p_xx, p_yy).

    Args:
        rho: 2D density field (ny, nx).
        center: Droplet center (cx, cy) in lattice units.
        k1: Surface-tension knob k₁ (default: 1/12).
        cs_a, cs_b, cs_R, cs_T, cs_G: CS-EOS parameters.
        n_bins: Number of radial bins for profile extraction.

    Returns:
        Surface tension σ.
    """
    ny, nx = rho.shape
    y_idx, x_idx = np.mgrid[0:ny, 0:nx]
    r_vals = np.sqrt((x_idx - center[0]) ** 2 + (y_idx - center[1]) ** 2)

    # Radial binning: average ρ in each radial bin
    r_max = min(nx, ny) * 0.45
    r_edges = np.linspace(0, r_max, n_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    rho_radial = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)

    r_flat = r_vals.flatten()
    rho_flat = rho.flatten()
    for i in range(n_bins):
        mask = (r_flat >= r_edges[i]) & (r_flat < r_edges[i + 1])
        counts[i] = np.sum(mask)
        if counts[i] > 0:
            rho_radial[i] = np.mean(rho_flat[mask])

    # Only use bins with data
    valid = counts > 0
    if np.sum(valid) < 10:
        return 0.0
    r_rad = r_centers[valid]
    rho_rad = rho_radial[valid]

    # Compute ψ(r)
    psi_rad = compute_psi_from_rho(rho_rad, cs_a, cs_b, cs_R, cs_T, cs_G)

    # dψ/dr via central differences
    dpsi_dr = np.gradient(psi_rad, r_rad)

    # Integrate (dψ/dr)² dr  (NumPy ≥2.0: trapezoid replaces trapz)
    try:
        from numpy import trapezoid
    except ImportError:
        from scipy.integrate import trapezoid
    integral = trapezoid(dpsi_dr**2, r_rad)

    # σ = −(G/6)(1−6k₁) ∫ (dψ/dr)² dr
    sigma = -(cs_G / 6.0) * (1.0 - 6.0 * k1) * integral
    return float(sigma)
