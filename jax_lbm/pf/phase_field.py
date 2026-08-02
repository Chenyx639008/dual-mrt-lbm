"""Phase-field LBM — JAX algorithm track (Stage 0 → 1).

Part of the phase-field development plan
(``research/phasefield_development_plan.md``). JAX-first: verify the
algorithm here before the CUDA production track.

Implemented so far
------------------
* **Stage 0** — single-phase incompressible D2Q9 NS (BGK + MRT), with
  body force (Guo). Benchmark: Poiseuille flow between no-slip walls.
* **Stage 1** — conservative Allen-Cahn order-parameter evolution (no flow),
  ``conservative_ac_step``: the interface-capturing scalar advection-diffusion
  used to build the full phase-field two-phase model (Stage 2+).

Not yet implemented
-------------------
* Stage 2 — AC + NS coupled with chemical-potential surface tension
  (``F_c = [4βφ(φ-1)(φ-0.5) − κ∇²φ]∇φ``, β=12σ/W, κ=3σW/2). See the plan.

References
----------
* Yang et al. (2024) Innov Energy — conservative Allen-Cahn phase-field LBM.
* Yang 2021b (PRE) — CST interface mass transfer (later hydrate coupling).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import jit, lax

from jax_lbm.lattice import C, W, Q, OPP
from jax_lbm.collision import macroscopic, equilibrium, collision_bgk
from jax_lbm.boundary import wall_mask, bounce_back_halfway
from jax_lbm.d2q9_bgk import streaming

# ═══════════════════════════════════════════════════════════════════════════
# Stage 0 — single-phase incompressible D2Q9 NS
# ═══════════════════════════════════════════════════════════════════════════


def run_poiseuille(
    nx: int = 64,
    ny: int = 32,
    omega: float = 1.0,
    gx: float = 1e-4,
    n_steps: int = 8000,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Stage-0 benchmark: plane Poiseuille flow between no-slip walls.

    Uses the framework ``(nx, ny, 9)`` layout, EDM body-force collision
    (``collision_bgk``) + half-way bounce-back walls (``wall_mask`` at the
    fluid layers y=1 and y=ny-2).

    Returns
    -------
    (u_profile, u_exact) :
        u_profile : (ny,) x-velocity across the channel (mid-column).
        u_exact   : (ny,) analytic parabolic profile (zero at walls).
    """
    rho0 = 1.0
    u0 = jnp.zeros((nx, ny, 2))
    f = equilibrium(jnp.full((nx, ny, 1), rho0), u0)

    # no-slip walls: first fluid layers adjacent to ghost rows y=0 / y=ny-1
    wall = wall_mask(nx, ny, "bottom") | wall_mask(nx, ny, "top")

    # constant body force along x (EDM forcing in collision_bgk)
    F = jnp.broadcast_to(
        jnp.array([gx, 0.0], dtype=jnp.float64), (nx, ny, 2)
    )

    def step(f, _):
        fout = collision_bgk(f, omega, F)
        fin = streaming(fout)
        fin = bounce_back_halfway(fin, fout, wall)
        return fin, None

    f_final, _ = lax.scan(step, f, None, length=n_steps)

    rho, u = macroscopic(f_final)
    ux = u[..., 0]
    u_profile = ux[nx // 2, :]  # velocity across y at mid-column

    # analytic: u(y) = (gx/(2ν))·(y−1)·(ny−2−y), ν = (1/ω−0.5)/3
    nu = (1.0 / omega - 0.5) / 3.0
    y = jnp.arange(ny, dtype=jnp.float64)
    u_exact = (gx / (2.0 * nu)) * (y - 1.0) * (ny - 2.0 - y)
    u_exact = jnp.maximum(u_exact, 0.0)
    return u_profile, u_exact


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1 — conservative Allen-Cahn order parameter (no flow)
# ═══════════════════════════════════════════════════════════════════════════


def tanh_interface_profile(x: jnp.ndarray, x0: float, W: float) -> jnp.ndarray:
    """Dual-hyperbolic-tangent order-parameter profile for a flat interface.

    φ = 0.5*(1 + tanh(2*(x0 − x)/W))  →  φ=1 liquid (x<x0), φ=0 gas (x>x0).
    """
    return 0.5 * (1.0 + jnp.tanh(2.0 * (x0 - x) / W))


@jit
def conservative_ac_step(
    phi: jnp.ndarray,
    ux: jnp.ndarray,
    uy: jnp.ndarray,
    M: float,
    W: float,
) -> jnp.ndarray:
    """One conservative Allen-Cahn update (no explicit surface-tension force).

    ∂φ/∂t + ∇·(φu) = ∇·[ M( ∇φ − (1/W)(1 − tanh²(½ ln(φ/(1−φ)))) n ) ]

    Divergence-form discretisation (conservative); Lax-Wendroff advection;
    central-difference diffusion of the reoriented flux (∇φ − anti·n).
    """
    # gradients (central differences)
    dphix = 0.5 * (jnp.roll(phi, -1, axis=1) - jnp.roll(phi, 1, axis=1))
    dphiy = 0.5 * (jnp.roll(phi, -1, axis=0) - jnp.roll(phi, 1, axis=0))
    grad2 = dphix * dphix + dphiy * dphiy
    inv_norm = 1.0 / (jnp.sqrt(grad2) + 1e-12)
    nx_ = dphix * inv_norm
    ny_ = dphiy * inv_norm

    # anti-diffusion coefficient (equilibrium interface profile); clip to keep
    # log() finite when φ runs to exactly 0 or 1
    phi_c = jnp.clip(phi, 1e-10, 1.0 - 1e-10)
    s = 0.5 * jnp.log(phi_c / (1.0 - phi_c))
    anti_coef = (1.0 / W) * (1.0 - jnp.tanh(s) ** 2)

    # diffusion flux: M*(∇φ − anti_coef·n)
    Jx = M * (dphix - anti_coef * nx_)
    Jy = M * (dphiy - anti_coef * ny_)
    divJ = 0.5 * (jnp.roll(Jx, -1, axis=1) - jnp.roll(Jx, 1, axis=1)) \
        + 0.5 * (jnp.roll(Jy, -1, axis=0) - jnp.roll(Jy, 1, axis=0))

    # advection ∇·(φu)
    fluxx = ux * phi
    fluxy = uy * phi
    div_adv = 0.5 * (jnp.roll(fluxx, -1, axis=1) - jnp.roll(fluxx, 1, axis=1)) \
        + 0.5 * (jnp.roll(fluxy, -1, axis=0) - jnp.roll(fluxy, 1, axis=0))

    return phi - div_adv + divJ
