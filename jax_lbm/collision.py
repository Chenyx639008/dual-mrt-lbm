"""Collision operators — BGK and MRT, pure JAX.

Inspired by JAX-LaB `src/jax_lab/models.py` (BGKSim, MRTSim)
and `src/jax_lab/multiphase.py` (MultiphaseBGK, MultiphaseMRT).

Both BGK and MRT support Exact Difference Method (EDM) forcing.
MRT uses the standard D2Q9 MRT matrix from Lallemand & Luo (2000).

Reference
---------
- JAX-LaB: src/jax_lab/models.py (MRTSim), src/jax_lab/multiphase.py
- Lallemand & Luo (2000): Phys. Rev. E 61, 6546-6562
- Huang & Wu (2016): Phys. Rev. E 93, 043311
"""

from functools import partial

import jax.numpy as jnp
from jax import jit

from jax_lbm.lattice import C, W, Q, CS2, INV_CS2
from jax_lbm.boundary import equilibrium

# ═══════════════════════════════════════════════════════════════════════════
# Macroscopic fields
# ═══════════════════════════════════════════════════════════════════════════


@jit
def macroscopic(f):
    """Compute ρ and u from distribution functions.

    Parameters
    ----------
    f : (nx, ny, 9) array

    Returns
    -------
    rho : (nx, ny, 1) array
    u : (nx, ny, 2) array
    """
    rho = jnp.sum(f, axis=-1, keepdims=True)
    u = jnp.dot(f, C.astype(jnp.float64)) / rho
    return rho, u


# ═══════════════════════════════════════════════════════════════════════════
# BGK Collision
# ═══════════════════════════════════════════════════════════════════════════


@jit
def collision_bgk(f, omega, F):
    """BGK collision with Exact Difference Method (EDM) forcing.

    f_out = f - ω·(f - f^eq) + [f^eq(ρ, u+Δu) - f^eq(ρ, u)]
    where Δu = F/ρ

    Parameters
    ----------
    f : (nx, ny, 9) array — pre-collision distributions
    omega : float — relaxation frequency (1/τ)
    F : (nx, ny, 2) array — total force

    Returns
    -------
    fout : (nx, ny, 9) array — post-collision distributions
    """
    rho, u = macroscopic(f)
    feq = equilibrium(rho, u)
    # Exact Difference Method (Kupershtokh)
    delta_u = F / rho
    feq_force = equilibrium(rho, u + delta_u)
    fout = f - omega * (f - feq) + (feq_force - feq)
    return fout


# ═══════════════════════════════════════════════════════════════════════════
# D2Q9 MRT Matrix (Lallemand & Luo 2000, Eq. 34)
# ═══════════════════════════════════════════════════════════════════════════

# MRT transformation matrix M (9×9) — rows are moment basis vectors
MRT_M = jnp.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],  # ρ (density)
        [-4, -1, -1, -1, -1, 2, 2, 2, 2],  # e (energy)
        [4, -2, -2, -2, -2, 1, 1, 1, 1],  # ε (energy squared)
        [0, 1, 0, -1, 0, 1, -1, -1, 1],  # jx (x-momentum)
        [0, -2, 0, 2, 0, 1, -1, -1, 1],  # qx (x-energy flux)
        [0, 0, 1, 0, -1, 1, 1, -1, -1],  # jy (y-momentum)
        [0, 0, -2, 0, 2, 1, 1, -1, -1],  # qy (y-energy flux)
        [0, 1, -1, 1, -1, 0, 0, 0, 0],  # pxx (diagonal stress)
        [0, 0, 0, 0, 0, 1, -1, 1, -1],  # pxy (off-diagonal stress)
    ],
    dtype=jnp.float64,
)

# Inverse MRT matrix M⁻¹ (9×9) — computed as (1/M) normalized
MRT_MINV = jnp.array(
    [
        [1.0 / 9.0, -1.0 / 9.0, 1.0 / 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [
            1.0 / 9.0,
            -1.0 / 36.0,
            -1.0 / 18.0,
            1.0 / 6.0,
            -1.0 / 6.0,
            0.0,
            0.0,
            1.0 / 4.0,
            0.0,
        ],
        [
            1.0 / 9.0,
            -1.0 / 36.0,
            -1.0 / 18.0,
            0.0,
            0.0,
            1.0 / 6.0,
            -1.0 / 6.0,
            -1.0 / 4.0,
            0.0,
        ],
        [
            1.0 / 9.0,
            -1.0 / 36.0,
            -1.0 / 18.0,
            -1.0 / 6.0,
            1.0 / 6.0,
            0.0,
            0.0,
            1.0 / 4.0,
            0.0,
        ],
        [
            1.0 / 9.0,
            -1.0 / 36.0,
            -1.0 / 18.0,
            0.0,
            0.0,
            -1.0 / 6.0,
            1.0 / 6.0,
            -1.0 / 4.0,
            0.0,
        ],
        [
            1.0 / 9.0,
            1.0 / 18.0,
            1.0 / 36.0,
            1.0 / 6.0,
            1.0 / 12.0,
            1.0 / 6.0,
            1.0 / 12.0,
            0.0,
            1.0 / 4.0,
        ],
        [
            1.0 / 9.0,
            1.0 / 18.0,
            1.0 / 36.0,
            -1.0 / 6.0,
            -1.0 / 12.0,
            1.0 / 6.0,
            1.0 / 12.0,
            0.0,
            -1.0 / 4.0,
        ],
        [
            1.0 / 9.0,
            1.0 / 18.0,
            1.0 / 36.0,
            -1.0 / 6.0,
            -1.0 / 12.0,
            -1.0 / 6.0,
            -1.0 / 12.0,
            0.0,
            1.0 / 4.0,
        ],
        [
            1.0 / 9.0,
            1.0 / 18.0,
            1.0 / 36.0,
            1.0 / 6.0,
            1.0 / 12.0,
            -1.0 / 6.0,
            -1.0 / 12.0,
            0.0,
            -1.0 / 4.0,
        ],
    ],
    dtype=jnp.float64,
)


# Equilibrium moments m^eq in moment space
@jit
def meq_mrt(rho, u):
    """Compute equilibrium moments for MRT collision.

    Parameters
    ----------
    rho : (nx, ny, 1)
    u : (nx, ny, 2)

    Returns
    -------
    meq : (nx, ny, 9) — moments in MRT basis
    """
    ux = u[..., 0:1]
    uy = u[..., 1:2]
    u2 = ux * ux + uy * uy
    ux2 = ux * ux
    uy2 = uy * uy

    return jnp.concatenate(
        [
            rho,  # m0: ρ
            rho * (-2.0 + 3.0 * u2),  # m1: e
            rho * (1.0 - 3.0 * u2),  # m2: ε
            rho * ux,  # m3: jx
            rho * (-ux),  # m4: qx
            rho * uy,  # m5: jy
            rho * (-uy),  # m6: qy
            rho * (ux2 - uy2),  # m7: pxx
            rho * (ux * uy),  # m8: pxy
        ],
        axis=-1,
    )


# Default MRT relaxation times for D2Q9
# s_rho=0 (conserved), s_e, s_ε (bulk), s_j=0 (conserved), s_q, s_ν (shear)
MRT_S_DEFAULT = jnp.array(
    [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0], dtype=jnp.float64
)


@jit
def collision_mrt(f, S, F):
    """MRT collision with moment-space EDM forcing.

    Algorithm:
    1. Transform f → m = M·f
    2. Compute equilibrium moments m^eq
    3. Force in moment space: m^F = M·(f^eq(ρ,u+F/ρ) - f^eq(ρ,u))
    4. Collide: m' = m - S·(m - m^eq) + m^F
    5. Transform back: f' = M⁻¹·m'

    Parameters
    ----------
    f : (nx, ny, 9) — pre-collision distributions
    S : (9,) array — relaxation times (diagonal)
    F : (nx, ny, 2) — total force

    Returns
    -------
    fout : (nx, ny, 9) — post-collision distributions
    """
    rho, u = macroscopic(f)

    # Transform to moment space
    m = jnp.dot(f, MRT_M.T)

    # Equilibrium moments
    m_eq = meq_mrt(rho, u)

    # Force in moment space (EDM)
    delta_u = F / rho
    feq = equilibrium(rho, u)
    feq_force = equilibrium(rho, u + delta_u)
    m_force = jnp.dot((feq_force - feq), MRT_M.T)

    # Collision in moment space
    m_collided = m - S * (m - m_eq) + m_force

    # Transform back to distribution space
    fout = jnp.dot(m_collided, MRT_MINV.T)

    return fout


# ═══════════════════════════════════════════════════════════════════════════
# Collision Registry
# ═══════════════════════════════════════════════════════════════════════════

COLLISION_REGISTRY = {
    "bgk": collision_bgk,
    "mrt": collision_mrt,
}


def get_collision(name: str):
    """Get collision function by name.

    Parameters
    ----------
    name : str — 'bgk' or 'mrt'

    Returns
    -------
    collision_fn : callable
    """
    if name not in COLLISION_REGISTRY:
        raise KeyError(
            f"Unknown collision '{name}'. Available: {list(COLLISION_REGISTRY.keys())}"
        )
    return COLLISION_REGISTRY[name]
