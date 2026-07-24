"""Contact angle via ψ-based ghost boundary condition — pure JAX.

Implements Scheme IV from the CUDA update_ghost_psi_bc kernel:
    ψ_ghost = ψ_2 + |∇ψ|_1 · cot(θ)

This is the ψ-based geometric ghost BC that independently controls
contact angle without modifying the EOS or interaction parameters.

Also supports the "improved virtual density" scheme from JAX-LaB
for setting ghost node density directly.

Reference
---------
- CUDA: LBM.cu update_ghost_psi_bc (Scheme IV)
- JAX-LaB: src/jax_lab/multiphase.py apply_contact_angle()
- Li, Q. et al. (2019): curved boundary contact angle
"""

from functools import partial

import jax.numpy as jnp
from jax import jit

from jax_lbm.lattice import C, Q


# ═══════════════════════════════════════════════════════════════════════════
# Scheme IV: ψ-based ghost BC (matches CUDA kernel)
# ═══════════════════════════════════════════════════════════════════════════


@partial(jit, static_argnames=("wall",))
def apply_psi_ghost_bc(psi, theta_deg, wall="bottom"):
    """Apply ψ-based ghost boundary condition for contact angle.

    Algorithm:
    1. At ghost layer (y=0 for bottom wall):
       ψ_ghost = ψ(y=2) + |∇ψ(y=1)| · cot(θ)

    2. For θ=90°: cot→0 → ψ_ghost = ψ(y=2)  (neutral)
    3. For θ→0°:  cot→+∞ → ψ_ghost >> ψ(y=2) (superhydrophilic)
    4. For θ→180°: cot→-∞ → ψ_ghost << ψ(y=2) (superhydrophobic)

    Parameters
    ----------
    psi : (nx, ny, 1) array — pseudopotential field
    theta_deg : float — contact angle in degrees
    wall : str — 'bottom', 'top', 'left', 'right'

    Returns
    -------
    psi_updated : (nx, ny, 1) array — with ghost nodes updated
    """
    nx, ny = psi.shape[0], psi.shape[1]
    theta_rad = jnp.deg2rad(theta_deg)
    cot_theta = 1.0 / jnp.tan(theta_rad)

    if wall == "bottom":
        # Ghost layer: y=0, reference: y=2, gradient at y=1
        ghost_y = 0
        ref_y = 2
        grad_y = 1
    elif wall == "top":
        ghost_y = ny - 1
        ref_y = ny - 3
        grad_y = ny - 2
    elif wall == "left":
        ghost_y = 0  # using y coordinate for x-wall (less common)
        ref_y = 2
        grad_y = 1
        # TODO: implement properly for left/right walls
        return psi
    elif wall == "right":
        return psi  # TODO
    else:
        raise ValueError(f"Unknown wall: {wall}")

    psi_updated = psi.copy() if hasattr(psi, "copy") else psi

    # Get ψ at reference layer (y=2 or equivalent)
    psi_ref = psi[:, ref_y : ref_y + 1, :]  # (nx, 1, 1)

    # Compute gradient magnitude at y=1
    # ∂ψ/∂x ≈ (ψ(x+1,y=1) - ψ(x-1,y=1)) / 2
    psi_grad_layer = psi[:, grad_y : grad_y + 1, 0]  # (nx, 1)
    psi_grad_left = jnp.roll(psi_grad_layer, 1, axis=0)
    psi_grad_right = jnp.roll(psi_grad_layer, -1, axis=0)
    grad_x = (psi_grad_right - psi_grad_left) * 0.5

    # ∂ψ/∂y ≈ (ψ(x,y=2) - ψ(x,y=0)) / 2
    psi_ghost_old = psi[:, ghost_y : ghost_y + 1, 0]
    grad_y = (psi_ref[:, 0, 0:1] - psi_ghost_old) * 0.5

    grad_mag = jnp.sqrt(grad_x**2 + grad_y**2)

    # ψ_ghost_new = ψ(y=2) + |∇ψ| · cot(θ)
    psi_ghost_new = psi_ref[:, 0, 0:1] + grad_mag * cot_theta

    # Clamp to physical range (ψ ≥ 0)
    psi_ghost_new = jnp.maximum(psi_ghost_new, 0.0)

    # Update ghost layer (y=0)
    psi_updated = psi_updated.at[:, ghost_y : ghost_y + 1, :].set(
        psi_ghost_new[:, jnp.newaxis, :]
    )

    return psi_updated


# ═══════════════════════════════════════════════════════════════════════════
# Improved Virtual Density scheme (from JAX-LaB)
# ═══════════════════════════════════════════════════════════════════════════


@jit
def apply_improved_virtual_density(
    rho, mask_ghost, theta_deg, phi=1.0, delta_rho=None, rho_min=1e-6, rho_max=2.0
):
    """Set ghost node density based on contact angle.

    From JAX-LaB Multiphase.apply_contact_angle() with
    wetting_formulation="improved_virtual_density".

    - Hydrophilic (θ ≤ 90°): ρ_wall = φ · ρ_ave  (high density → liquid at wall)
    - Hydrophobic (θ > 90°): ρ_wall = ρ_ave - Δρ  (low density → gas at wall)

    Parameters
    ----------
    rho : (nx, ny, 1) array — density field
    mask_ghost : (nx, ny) bool — ghost nodes
    theta_deg : float — contact angle in degrees
    phi : float — density multiplier for hydrophilic case
    delta_rho : float or None — density reduction for hydrophobic case
    rho_min, rho_max : float — clipping range

    Returns
    -------
    rho_updated : (nx, ny, 1) array
    """
    nx, ny = rho.shape[0], rho.shape[1]
    theta_rad = jnp.deg2rad(theta_deg)
    pi_half = jnp.pi / 2.0

    if delta_rho is None:
        # Auto-compute delta_rho as ~10% of mean density
        delta_rho = 0.1 * jnp.mean(rho)

    # Average density over neighboring fluid nodes (simplified: use global mean)
    rho_ave = jnp.mean(rho)

    # Compute ghost density
    rho_ghost = jnp.where(
        theta_rad <= pi_half,
        phi * rho_ave,  # hydrophilic
        rho_ave - delta_rho,  # hydrophobic
    )
    rho_ghost = jnp.clip(rho_ghost, rho_min, rho_max)

    # Apply only at ghost nodes
    rho_updated = rho.copy() if hasattr(rho, "copy") else rho
    rho_updated = jnp.where(
        mask_ghost[..., jnp.newaxis],
        rho_ghost,
        rho_updated,
    )
    return rho_updated


# ═══════════════════════════════════════════════════════════════════════════
# Wetting Registry
# ═══════════════════════════════════════════════════════════════════════════

WETTING_REGISTRY = {
    "psi_ghost_bc": apply_psi_ghost_bc,  # Scheme IV (matches CUDA)
    "improved_virtual_density": apply_improved_virtual_density,  # JAX-LaB style
    "none": None,  # no wetting
}


def get_wetting(name: str):
    """Get wetting function by name."""
    if name not in WETTING_REGISTRY:
        raise KeyError(
            f"Unknown wetting '{name}'. Available: {list(WETTING_REGISTRY.keys())}"
        )
    return WETTING_REGISTRY[name]
