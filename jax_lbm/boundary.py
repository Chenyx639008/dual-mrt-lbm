"""Boundary conditions — pure JAX, functional style.

Inspired by JAX-LaB `src/jax_lab/boundary_conditions.py`.
Each BC is a tuple (mask_fn, apply_fn) compatible with @jit and lax.scan.

Supported BC types:
- Periodic (default, no-op)
- BounceBack (fullway & halfway)
- EquilibriumBC (velocity/pressure inlet)
- ZouHe (pressure/velocity boundary, D2Q9)

All BCs operate on the full (nx, ny, 9) distribution array.
A mask array determines which nodes are affected.

Reference
---------
- JAX-LaB: src/jax_lab/boundary_conditions.py
- Zou & He (1997): Phys. Fluids 9, 1591-1598
"""

from functools import partial

import jax.numpy as jnp
from jax import jit

from jax_lbm.lattice import C, W, Q, CS2, INV_CS2, OPP


# ═══════════════════════════════════════════════════════════════════════════
# Mask builders
# ═══════════════════════════════════════════════════════════════════════════


def wall_mask(nx, ny, wall="bottom"):
    """Create a boolean mask for wall boundary nodes.

    Parameters
    ----------
    nx, ny : int
    wall : str — 'bottom', 'top', 'left', 'right'

    Returns
    -------
    mask : (nx, ny) bool array
    """
    x = jnp.arange(nx)
    y = jnp.arange(ny)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    if wall == "bottom":
        return Y == 1  # first fluid layer above ghost
    elif wall == "top":
        return Y == ny - 2
    elif wall == "left":
        return X == 1
    elif wall == "right":
        return X == nx - 2
    else:
        raise ValueError(f"Unknown wall: {wall}")


def ghost_mask(nx, ny, wall="bottom"):
    """Create a boolean mask for ghost nodes (y=0 or y=ny-1 etc.).

    Parameters
    ----------
    nx, ny : int
    wall : str

    Returns
    -------
    mask : (nx, ny) bool array
    """
    x = jnp.arange(nx)
    y = jnp.arange(ny)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    if wall == "bottom":
        return Y == 0
    elif wall == "top":
        return Y == ny - 1
    elif wall == "left":
        return X == 0
    elif wall == "right":
        return X == nx - 1
    else:
        raise ValueError(f"Unknown wall: {wall}")


# ═══════════════════════════════════════════════════════════════════════════
# Equilibrium helper
# ═══════════════════════════════════════════════════════════════════════════


@jit
def equilibrium(rho, u):
    """D2Q9 equilibrium distribution.

    Parameters
    ----------
    rho : (nx, ny, 1) array
    u : (nx, ny, 2) array

    Returns
    -------
    feq : (nx, ny, 9) array
    """
    cu = INV_CS2 * jnp.dot(u, C.astype(jnp.float64).T)
    usqr = 0.5 * INV_CS2 * jnp.sum(u**2, axis=-1, keepdims=True)
    return rho * W * (1.0 + cu * (1.0 + 0.5 * cu) - usqr)


# ═══════════════════════════════════════════════════════════════════════════
# Bounce-Back (Fullway)
# ═══════════════════════════════════════════════════════════════════════════


@jit
def bounce_back_fullway(f_postcollision, mask):
    """Fullway bounce-back: applies AFTER collision, BEFORE streaming.

    f_k(x, t+1) = f'_opp(k)(x, t)   where f' is post-collision

    Parameters
    ----------
    f_postcollision : (nx, ny, 9) — post-collision distributions
    mask : (nx, ny) bool — wall nodes

    Returns
    -------
    f_bounced : (nx, ny, 9)
    """
    f_bounced = jnp.zeros_like(f_postcollision)
    for k in range(Q):
        f_bounced = f_bounced.at[..., k].set(
            jnp.where(
                mask,
                f_postcollision[..., OPP[k]],
                f_postcollision[..., k],
            )
        )
    return f_bounced


# ═══════════════════════════════════════════════════════════════════════════
# Bounce-Back (Halfway) — solid wall, applied post-streaming
# ═══════════════════════════════════════════════════════════════════════════


@jit
def bounce_back_halfway(fin, fout, mask):
    """Halfway bounce-back: applies AFTER streaming.

    For wall nodes: fin_k = fout_opp(k)
    For non-wall nodes: fin_k = fout_k (already streamed)

    Parameters
    ----------
    fin : (nx, ny, 9) — post-streaming (incoming) distributions
    fout : (nx, ny, 9) — post-collision (outgoing) distributions
    mask : (nx, ny) bool — wall nodes

    Returns
    -------
    fin_corrected : (nx, ny, 9)
    """
    fin_corrected = fin.copy() if hasattr(fin, "copy") else fin
    for k in range(Q):
        fin_corrected = fin_corrected.at[..., k].set(
            jnp.where(mask, fout[..., OPP[k]], fin[..., k])
        )
    return fin_corrected


# ═══════════════════════════════════════════════════════════════════════════
# Equilibrium BC — prescribe velocity or density at inlet/outlet
# ═══════════════════════════════════════════════════════════════════════════


@jit
def equilibrium_bc(fin, mask, rho_target=None, u_target=None):
    """Equilibrium boundary condition.

    Sets distributions to equilibrium at specified density/velocity.

    Parameters
    ----------
    fin : (nx, ny, 9) — post-streaming distributions
    mask : (nx, ny) bool — boundary nodes
    rho_target : array or float — prescribed density (or None to use current)
    u_target : (2,) array — prescribed velocity (or None for u=0)

    Returns
    -------
    fin_corrected : (nx, ny, 9)
    """
    rho = jnp.sum(fin, axis=-1, keepdims=True) if rho_target is None else rho_target
    u = jnp.zeros((fin.shape[0], fin.shape[1], 2)) if u_target is None else u_target
    # u needs broadcast across spatial dims
    if u.ndim == 1:
        u = u[jnp.newaxis, jnp.newaxis, :]
    feq = equilibrium(rho, u)
    fin_corrected = fin.copy() if hasattr(fin, "copy") else fin
    for k in range(Q):
        fin_corrected = fin_corrected.at[..., k].set(
            jnp.where(mask, feq[..., k], fin[..., k])
        )
    return fin_corrected


# ═══════════════════════════════════════════════════════════════════════════
# Zou/He Boundary — D2Q9 velocity or pressure BC
# ═══════════════════════════════════════════════════════════════════════════


@jit
def zou_he_bottom_wall(fin, fout, Fx=None, Fy=None):
    """Zou/He no-slip bottom wall for D2Q9.

    Reconstructs unknown distributions f2, f5, f6 at y=1
    (post-streaming, incoming directions from ghost side).
    Based on bounce-back of non-equilibrium part normal to wall.

    Parameters
    ----------
    fin : (nx, ny, 9) — post-streaming distributions
    fout : (nx, ny, 9) — post-collision distributions (for non-eq bounce-back)
    Fx, Fy : (nx, ny) arrays or None — force fields at wall

    Returns
    -------
    fin_corrected : (nx, ny, 9)
    """
    nx, ny = fin.shape[0], fin.shape[1]
    fin_corrected = fin.copy() if hasattr(fin, "copy") else fin

    # Only modify y=1 (first fluid layer above bottom ghost y=0)
    y_mask = jnp.arange(ny) == 1

    # Known distributions (from streaming):
    #   f0 (rest), f1 (right), f3 (left), f4 (down), f7 (down-left), f8 (down-right)
    f0 = fin[..., 0]
    f1 = fin[..., 1]
    f3 = fin[..., 3]
    f4 = fin[..., 4]
    f7 = fin[..., 7]
    f8 = fin[..., 8]

    fx = jnp.zeros((nx, ny)) if Fx is None else Fx
    fy = jnp.zeros((nx, ny)) if Fy is None else Fy

    # Zou/He reconstruction for D2Q9 bottom wall (no-slip u=0):
    # Non-eq bounce-back normal to wall: f2 - f2^eq = f4 - f4^eq
    # With u=0: f2^eq = f4^eq = ρ/9  →  f2 = f4
    f2 = f4

    # x-momentum: (f1-f3)+(f5-f6)+(f8-f7) = -Fx/2
    # y-momentum: (f2+f5+f6)-(f4+f7+f8) = -Fy/2
    # Using f2=f4: f5+f6 = f7+f8 - Fy/2
    f5 = 0.5 * (f3 - f1) + f7 - 0.25 * (fx + fy)
    f6 = 0.5 * (f1 - f3) + f8 + 0.25 * (fx - fy)

    # Apply only at y=1
    mask_2d = y_mask[jnp.newaxis, :]  # broadcast across x
    for k_val, arr in [(2, f2), (5, f5), (6, f6)]:
        fin_corrected = fin_corrected.at[..., k_val].set(
            jnp.where(mask_2d.T, arr, fin_corrected[..., k_val])
        )

    return fin_corrected


# ═══════════════════════════════════════════════════════════════════════════
# BC Registry
# ═══════════════════════════════════════════════════════════════════════════

BC_REGISTRY = {
    "periodic": None,  # default, no-op
    "bounce_back_fullway": bounce_back_fullway,
    "bounce_back_halfway": bounce_back_halfway,
    "equilibrium": equilibrium_bc,
    "zou_he_bottom": zou_he_bottom_wall,
}


def get_bc(name: str):
    """Get boundary condition function by name."""
    if name not in BC_REGISTRY:
        raise KeyError(f"Unknown BC '{name}'. Available: {list(BC_REGISTRY.keys())}")
    return BC_REGISTRY[name]
