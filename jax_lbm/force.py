"""Force models — Shan-Chen + Adsorption, pure JAX.

Inspired by JAX-LaB `src/jax_lab/multiphase.py` (compute_fluid_fluid_force).

Supports:
- Shan-Chen interaction force: F = -G·ψ·Σ w_k·ψ(x+c_k)·c_k
- Adsorption force (Yang/Li style): F_ads = -G_ads·ψ·Σ w_k·s(x+c_k)·c_k
  where s=1 if neighbor is wall (-1 or 0), 0 otherwise
- Body force: F_body = (Gx, Gy)·ρ

Reference
---------
- JAX-LaB: src/jax_lab/multiphase.py
- Huang & Wu (2016): Phys. Rev. E 93, 043311
"""

from functools import partial

import jax.numpy as jnp
from jax import jit

from jax_lbm.lattice import C, W, Q, G_FF, CS2


# ═══════════════════════════════════════════════════════════════════════════
# Streaming helper (periodic)
# ═══════════════════════════════════════════════════════════════════════════

# ── Lattice velocities as Python ints for use inside jit ──
_CX = [0, 1, 0, -1, 0, 1, -1, -1, 1]
_CY = [0, 0, 1, 0, -1, 1, 1, -1, -1]


def _stream_field(field):
    """Stream a scalar field along all D2Q9 directions.

    Parameters
    ----------
    field : (nx, ny, 1) array

    Returns
    -------
    field_s : (nx, ny, 9) array — field at x+c_k for each k
    """
    field_s = jnp.zeros((field.shape[0], field.shape[1], Q), dtype=field.dtype)
    for k in range(Q):
        field_s = field_s.at[..., k].set(
            jnp.roll(jnp.roll(field[..., 0], -_CX[k], axis=0), -_CY[k], axis=1)
        )
    return field_s


# ═══════════════════════════════════════════════════════════════════════════
# Shan-Chen Interaction Force
# ═══════════════════════════════════════════════════════════════════════════


def shan_chen_force(psi, G=-1.0):
    """Shan-Chen interaction force.

    F(x) = -G·ψ(x)·Σ_i w_i·ψ(x+c_i)·c_i

    Parameters
    ----------
    psi : (nx, ny, 1) array — pseudopotential
    G : float — interaction strength

    Returns
    -------
    F : (nx, ny, 2) array
    """
    psi_s = _stream_field(jnp.repeat(psi, 1, axis=-1))
    # psi_s has shape (nx, ny, 9), we need (nx, ny, 9) with psi values
    F = -G * psi * jnp.dot(G_FF * psi_s, C.astype(jnp.float64))
    return F


# ═══════════════════════════════════════════════════════════════════════════
# Adsorption Force (Yang/Li style — for solid wall interaction)
# ═══════════════════════════════════════════════════════════════════════════


def adsorption_force(psi, mask_wall, G_ads=0.0):
    """Adsorption force from solid walls.

    F_ads(x) = -G_ads·ψ(x)·Σ_k w_F_k·s(x+c_k)·c_k
    where s(x+c_k) = 1 if neighbor is wall (ghost or boundary), 0 otherwise.

    This matches the CUDA compute_adsorption_force_scmp kernel.

    Parameters
    ----------
    psi : (nx, ny, 1) array — pseudopotential
    mask_wall : (nx, ny) bool — True for wall nodes (ghost + boundary)
    G_ads : float — adsorption strength

    Returns
    -------
    F_ads : (nx, ny, 2) array
    """
    if G_ads == 0.0:
        return jnp.zeros((psi.shape[0], psi.shape[1], 2), dtype=psi.dtype)

    # Build indicator field: s=1 for wall nodes, 0 for fluid
    s_field = jnp.where(mask_wall, 1.0, 0.0).astype(psi.dtype)
    s_field = s_field[..., jnp.newaxis]  # (nx, ny, 1)

    # Stream to get s(x+c_k) for each direction
    s_streamed = _stream_field(s_field)  # (nx, ny, 9)

    # Compute force: F_ads = -G_ads·ψ·Σ w_F_k·s(x+c_k)·c_k
    F_ads = -G_ads * psi * jnp.dot(G_FF * s_streamed, C.astype(jnp.float64))
    return F_ads


# ═══════════════════════════════════════════════════════════════════════════
# Body Force (gravity, pressure gradient)
# ═══════════════════════════════════════════════════════════════════════════


def body_force(rho, Gx=0.0, Gy=0.0):
    """Body force: F_body = (Gx, Gy) · ρ

    Parameters
    ----------
    rho : (nx, ny, 1) array
    Gx, Gy : float — body force components

    Returns
    -------
    F_body : (nx, ny, 2) array
    """
    Fx = Gx * rho[..., 0]
    Fy = Gy * rho[..., 0]
    return jnp.stack([Fx, Fy], axis=-1)


# ═══════════════════════════════════════════════════════════════════════════
# Combined force
# ═══════════════════════════════════════════════════════════════════════════


def total_force(psi, rho, mask_wall, G_sc=-1.0, G_ads=0.0, Gx=0.0, Gy=0.0):
    """Combine all force contributions.

    F_total = F_shan_chen + F_adsorption + F_body

    Parameters
    ----------
    psi : (nx, ny, 1) — pseudopotential
    rho : (nx, ny, 1) — density
    mask_wall : (nx, ny) bool — wall mask
    G_sc : float — Shan-Chen interaction strength
    G_ads : float — adsorption strength
    Gx, Gy : float — body force components

    Returns
    -------
    F_total : (nx, ny, 2)
    """
    F = shan_chen_force(psi, G=G_sc)
    if G_ads != 0.0:
        F = F + adsorption_force(psi, mask_wall, G_ads=G_ads)
    if Gx != 0.0 or Gy != 0.0:
        F = F + body_force(rho, Gx=Gx, Gy=Gy)
    return F
