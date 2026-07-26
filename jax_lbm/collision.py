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
def collision_mrt(f, s_relax, F=None, S_guo=None, C=None, alpha_meq=1.0):
    """MRT collision — EDM or Guo+C forcing (Huang & Wu 2016).

    Two modes (auto-selected by which arguments are provided):

    **EDM mode** (backward-compatible, ``F`` provided, ``S_guo=None``):
        m' = m − s·(m − m^eq) + M·(f^eq(ρ, u+F/ρ) − f^eq(ρ, u))

    **Guo+C mode** (CUDA-equivalent, ``S_guo`` and ``C`` provided):
        m' = m − s·(m − m^eq) + (1 − 0.5·s)·S_guo + C

    The second mode reproduces the CUDA ``mrt_collide_single_component_gpu``
    formula exactly (Δt = 1.0 in lattice units).

    Parameters
    ----------
    f : (nx, ny, 9) — pre-collision distributions
    s_relax : (9,) array — MRT relaxation rates (diagonal, s_k ∈ [0,2])
    F : (nx, ny, 2) or None — total force for EDM mode
    S_guo : (nx, ny, 9) or None — Guo source term in moment space
    C : (nx, ny, 9) or None — Q_m surface-tension correction in moment space
    alpha_meq : float — α coefficient for ε-moment equilibrium (Huang Eq. 5)

    Returns
    -------
    fout : (nx, ny, 9) — post-collision distributions
    """
    rho, u_raw = macroscopic(f)

    # Transform to moment space: m = M·f
    m = jnp.dot(f, MRT_M.T)

    # Equilibrium moments
    m_eq = meq_mrt(rho, u_raw)

    # Apply α to slot 2 (ε moment) — Huang & Wu (2016) Eq.(5)
    # When α=1.0, this is identity (standard m_eq[2] = (1−3u²)ρ)
    ux = u_raw[..., 0:1]
    uy = u_raw[..., 1:2]
    u2 = ux * ux + uy * uy
    m_eq_eps = (alpha_meq - 3.0 * u2) * rho  # (nx, ny, 1)
    m_eq = m_eq.at[..., 2].set(m_eq_eps[..., 0])

    # — Choose collision mode —
    if S_guo is not None:
        # ── Guo+C mode (CUDA-equivalent) ──
        m_collided = m - s_relax * (m - m_eq) + (1.0 - 0.5 * s_relax) * S_guo + C
    elif F is not None:
        # ── EDM mode (backward-compatible) ──
        delta_u = F / rho
        feq = equilibrium(rho, u_raw)
        feq_force = equilibrium(rho, u_raw + delta_u)
        m_force = jnp.dot((feq_force - feq), MRT_M.T)
        m_collided = m - s_relax * (m - m_eq) + m_force
    else:
        # ── No forcing ──
        m_collided = m - s_relax * (m - m_eq)

    # Transform back to distribution space: f' = M⁻¹·m'
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
