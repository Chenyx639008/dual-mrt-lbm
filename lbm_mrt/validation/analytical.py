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
) -> tuple[float, float]:
    """Extract liquid and gas coexistence densities from a flat-interface profile.

    Uses plateau detection on centerline.
    """
    _, profile = centerline_profile(rho, axis="y")
    n = len(profile)

    # Top 1/4 = gas, bottom 1/4 = liquid
    rho_g = float(np.mean(profile[int(0.75 * n) :]))
    rho_l = float(np.mean(profile[: int(0.25 * n)]))
    return rho_l, rho_g
