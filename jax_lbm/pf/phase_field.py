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
from jax_lbm.boundary import wall_mask, bounce_back_halfway, equilibrium_bc
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
    F = jnp.broadcast_to(jnp.array([gx, 0.0], dtype=jnp.float64), (nx, ny, 2))

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


def run_lid_driven_cavity(
    nx: int = 64,
    ny: int = 64,
    u_lid: float = 0.05,
    omega: float = 1.0,
    n_steps: int = 20000,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Stage-0 benchmark: lid-driven cavity (Ghia et al. 1982).

    No-slip bottom/left/right walls (half-way bounce-back), moving top lid
    (equilibrium BC at u=(u_lid, 0)). Compares the vertical-centreline u(y)
    profile to the Ghia reference at the matching Reynolds number.

    Returns
    -------
    (u_centerline, v_centerline) :
        u_centerline : (ny,) x-velocity along the vertical centreline x=nx/2.
        v_centerline : (nx,) y-velocity along the horizontal centreline y=ny/2.
    """
    rho0 = 1.0
    u0 = jnp.zeros((nx, ny, 2))
    f = equilibrium(jnp.full((nx, ny, 1), rho0), u0)

    wall = (
        wall_mask(nx, ny, "bottom")
        | wall_mask(nx, ny, "left")
        | wall_mask(nx, ny, "right")
    )
    lid = wall_mask(nx, ny, "top")

    F0 = jnp.zeros((nx, ny, 2))

    def step(f, _):
        fout = collision_bgk(f, omega, F0)
        fin = streaming(fout)
        fin = bounce_back_halfway(fin, fout, wall)
        fin = equilibrium_bc(
            fin, lid, rho_target=rho0, u_target=jnp.array([u_lid, 0.0])
        )
        return fin, None

    f_final, _ = lax.scan(step, f, None, length=n_steps)
    _, u = macroscopic(f_final)
    u_centerline = u[nx // 2, :, 0]  # u(y) at x=nx/2
    v_centerline = u[:, ny // 2, 1]  # v(x) at y=ny/2
    return u_centerline, v_centerline


# Ghia, Ghia & Shin (1982) reference — vertical centreline u(y) at Re=100.
# y is the normalised coordinate (0=bottom wall, 1=lid); u is normalised by u_lid.
GHIA_RE100_UY = [
    (0.0000, 0.00000),
    (0.0547, -0.03717),
    (0.0625, -0.04192),
    (0.0703, -0.04775),
    (0.1016, -0.06434),
    (0.1719, -0.10150),
    (0.2813, -0.15662),
    (0.4531, -0.21090),
    (0.5000, -0.20581),
    (0.6172, -0.13641),
    (0.7344, 0.00332),
    (0.8516, 0.23151),
    (0.9531, 0.68717),
    (0.9609, 0.73722),
    (0.9688, 0.78871),
    (0.9766, 0.84123),
    (1.0000, 1.00000),
]


def ghia_uy_at(y_norm: float) -> float:
    """Linear interpolation of Ghia Re=100 u(y) reference."""
    for i in range(1, len(GHIA_RE100_UY)):
        y0, u0 = GHIA_RE100_UY[i - 1]
        y1, u1 = GHIA_RE100_UY[i]
        if y0 <= y_norm <= y1:
            t = (y_norm - y0) / (y1 - y0)
            return u0 + t * (u1 - u0)
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1 — conservative Allen-Cahn order parameter (no flow)
# ═══════════════════════════════════════════════════════════════════════════


def tanh_interface_profile(x: jnp.ndarray, x0: float, W: float) -> jnp.ndarray:
    """Dual-hyperbolic-tangent order-parameter profile for a flat interface.

    φ = 0.5*(1 + tanh(2*(x0 − x)/W))  →  φ=1 liquid (x<x0), φ=0 gas (x>x0).
    """
    return 0.5 * (1.0 + jnp.tanh(2.0 * (x0 - x) / W))


@jit
def gradient_isotropic(phi: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Isotropic D2Q9 gradient ∇φ = (∂φ/∂x, ∂φ/∂y).

    Framework axis convention: **axis 0 = x**, **axis 1 = y** (matches
    ``streaming`` / ``wall_mask``). Uses the D2Q9 weighted stencil
    (weights 1/3 cardinal, 1/12 diagonal) for interface-normal accuracy
    on curved interfaces (pitfall #13 in the 2D implementation plan).
    """
    # cardinal contributions
    dphix_card = jnp.roll(phi, -1, axis=0) - jnp.roll(phi, 1, axis=0)
    dphiy_card = jnp.roll(phi, -1, axis=1) - jnp.roll(phi, 1, axis=1)
    # diagonal contributions (x: (i±1, j±1); y: (i±1, j±1))
    dphix_diag = (
        jnp.roll(jnp.roll(phi, -1, axis=0), -1, axis=1)
        + jnp.roll(jnp.roll(phi, -1, axis=0), 1, axis=1)
        - jnp.roll(jnp.roll(phi, 1, axis=0), -1, axis=1)
        - jnp.roll(jnp.roll(phi, 1, axis=0), 1, axis=1)
    )
    dphiy_diag = (
        jnp.roll(jnp.roll(phi, -1, axis=1), -1, axis=0)
        + jnp.roll(jnp.roll(phi, -1, axis=1), 1, axis=0)
        - jnp.roll(jnp.roll(phi, 1, axis=1), -1, axis=0)
        - jnp.roll(jnp.roll(phi, 1, axis=1), 1, axis=0)
    )
    dphix = (1.0 / 3.0) * dphix_card + (1.0 / 12.0) * dphix_diag
    dphiy = (1.0 / 3.0) * dphiy_card + (1.0 / 12.0) * dphiy_diag
    return dphix, dphiy


@jit
def laplacian_isotropic(phi: jnp.ndarray) -> jnp.ndarray:
    """9-point isotropic Laplacian (D2Q9 stencil).

    ∇²φ = [4·Σ_cardinal + Σ_diagonal − 20·φ_center] / 6
    """
    cardinal = (
        jnp.roll(phi, -1, axis=0)
        + jnp.roll(phi, 1, axis=0)
        + jnp.roll(phi, -1, axis=1)
        + jnp.roll(phi, 1, axis=1)
    )
    diagonal = (
        jnp.roll(jnp.roll(phi, -1, axis=0), -1, axis=1)
        + jnp.roll(jnp.roll(phi, -1, axis=0), 1, axis=1)
        + jnp.roll(jnp.roll(phi, 1, axis=0), -1, axis=1)
        + jnp.roll(jnp.roll(phi, 1, axis=0), 1, axis=1)
    )
    return (4.0 * cardinal + diagonal - 20.0 * phi) / 6.0


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

    Direct divergence-form discretisation of Yang (2024) SI Eq. S10.
    Divergence-form flux (conservative); isotropic D2Q9 gradient for n;
    φ clipped to keep ln(φ/(1−φ)) finite.

    **Axis convention**: axis 0 = x (ux), axis 1 = y (uy) — matches the
    framework ``streaming``/NS convention. (Fixed 2026-08-02: the original
    implementation had x/y axes swapped, harmless for 1-D flat-interface
    conservation tests but wrong once coupled to the NS velocity field.)
    """
    # isotropic gradients (S10: n = ∇φ/|∇φ|)
    dphix, dphiy = gradient_isotropic(phi)
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
    divJ = 0.5 * (jnp.roll(Jx, -1, axis=0) - jnp.roll(Jx, 1, axis=0)) + 0.5 * (
        jnp.roll(Jy, -1, axis=1) - jnp.roll(Jy, 1, axis=1)
    )

    # advection ∇·(φu) — Lax–Wendroff (2nd order, stable for CFL < 1).
    # Central-difference advection alone is unconditionally unstable for
    # nonzero u (the pre-coupling Stage-1 tests only exercised u=0, so this
    # was latent); the LW second-order term adds the needed dissipation.
    #   φ^{n+1} = φ − [∂_x(u_xφ) + ∂_y(u_yφ)]
    #           + 0.5[∂_x(u_x ∂_xφ) + ∂_y(u_y ∂_yφ)]        (Δt = 1)
    fluxx = ux * phi
    fluxy = uy * phi
    div_adv = 0.5 * (jnp.roll(fluxx, -1, axis=0) - jnp.roll(fluxx, 1, axis=0)) + 0.5 * (
        jnp.roll(fluxy, -1, axis=1) - jnp.roll(fluxy, 1, axis=1)
    )
    # LW second-order correction 0.5·∇·(u·∇φ)
    dphix_c = 0.5 * (jnp.roll(phi, -1, axis=0) - jnp.roll(phi, 1, axis=0))
    dphiy_c = 0.5 * (jnp.roll(phi, -1, axis=1) - jnp.roll(phi, 1, axis=1))
    corr = 0.5 * (
        0.5 * (jnp.roll(ux * dphix_c, -1, axis=0) - jnp.roll(ux * dphix_c, 1, axis=0))
        + 0.5 * (jnp.roll(uy * dphiy_c, -1, axis=1) - jnp.roll(uy * dphiy_c, 1, axis=1))
    )

    return phi - div_adv + corr + divJ


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2 — AC + NS coupled two-phase (chemical-potential capillary force)
#
# Reference: Yang et al. (2024, Innov Energy) SI Eq. S10–S17, verified
# against research/literature/phase_field_2d_implementation_plan.md §3.
# NS solved with the pressure-based incompressible LBM (Guo et al. 2000,
# JCP 165:288 — the 2D parent of Yang's D3Q19 pressure formulation).
#
#   S13  ρ(φ)   = ρ_g + φ(ρ_w − ρ_g)
#   S14  F_c    = [4βφ(φ−1)(φ−0.5) − κ∇²φ]∇φ,   β = 12σ/W, κ = 3σW/2
#   p    = Σf   (Guo 2000 pressure form)
#   ρu   = Σfe  + Δt·F/2   (corrected velocity)
#   force: Guo forcing term; AC: conservative_ac_step (S10) with u_phys.
# ═══════════════════════════════════════════════════════════════════════════


@jit
def capillary_force(phi: jnp.ndarray, sigma: float, W: float) -> jnp.ndarray:
    """Chemical-potential capillary force (Yang S14).

    F_c = [4βφ(φ−1)(φ−0.5) − κ∇²φ]∇φ,   β = 12σ/W,  κ = 3σW/2

    Returns
    -------
    Fc : (nx, ny, 2) — capillary force field.
    """
    beta = 12.0 * sigma / W
    kappa = 1.5 * sigma * W  # 3σW/2
    dphix, dphiy = gradient_isotropic(phi)
    lap = laplacian_isotropic(phi)
    # chemical potential μ = 4βφ(φ−1)(φ−0.5) − κ∇²φ
    mu = 4.0 * beta * phi * (phi - 1.0) * (phi - 0.5) - kappa * lap
    fx = mu * dphix
    fy = mu * dphiy
    return jnp.stack([fx, fy], axis=-1)


@jit
def ns_equilibrium_pressure(
    p: jnp.ndarray, rho: jnp.ndarray, u: jnp.ndarray
) -> jnp.ndarray:
    """Pressure-based D2Q9 equilibrium (Guo 2000 incompressible).

    f_i^eq = w_i [ p + ρ( e_i·u/c_s² + (e_i·u)²/(2c_s⁴) − u²/(2c_s²) ) ]

    With c_s² = 1/3:  3(e_i·u) + 4.5(e_i·u)² − 1.5 u².
    Pressure is carried directly by the first moment: Σf^eq = p.
    """
    ux = u[..., 0]
    uy = u[..., 1]
    u2 = ux * ux + uy * uy
    cx = C[:, 0].astype(jnp.float64)
    cy = C[:, 1].astype(jnp.float64)
    e_dot_u = ux[..., None] * cx[None, ...] + uy[..., None] * cy[None, ...]  # (nx,ny,9)
    feq = W[None, None, :] * (
        p[..., None]
        + rho[..., None]
        * (3.0 * e_dot_u + 4.5 * e_dot_u * e_dot_u - 1.5 * u2[..., None])
    )
    return feq


@jit
def guo_force(F: jnp.ndarray, u: jnp.ndarray, omega: float) -> jnp.ndarray:
    """Guo et al. (2000) forcing term, D2Q9, c_s² = 1/3.

    F_i = (1 − ω/2) w_i [ (e_i − u)/c_s² + (e_i·u)e_i/c_s⁴ ] · F
        = (1 − ω/2) w_i [ 3(e_i·F − u·F) + 9(e_i·u)(e_i·F) ]

    Guarantees ρu = Σfe + Δt·F/2 (velocity correction consistent with
    the pressure-form macroscopic equation).
    """
    ux = u[..., 0]
    uy = u[..., 1]
    Fx = F[..., 0]
    Fy = F[..., 1]
    cx = C[:, 0].astype(jnp.float64)
    cy = C[:, 1].astype(jnp.float64)
    e_dot_u = ux[..., None] * cx[None, ...] + uy[..., None] * cy[None, ...]
    e_dot_F = Fx[..., None] * cx[None, ...] + Fy[..., None] * cy[None, ...]
    u_dot_F = ux * Fx + uy * Fy  # (nx,ny)
    Fi = (
        (1.0 - 0.5 * omega)
        * W[None, None, :]
        * (3.0 * (e_dot_F - u_dot_F[..., None]) + 9.0 * e_dot_u * e_dot_F)
    )
    return Fi


@jit
def thermodynamic_force(p: jnp.ndarray, rho: jnp.ndarray) -> jnp.ndarray:
    """Thermodynamic pressure correction (Yang S16, term −∇(p − ρc_s²)).

    For variable density two-phase flow, the pressure field carried by the
    distributions (p = Σf) is not the thermodynamic pressure ρc_s²; the
    gradient of their difference is added to the total force so that the
    Laplace pressure jump emerges correctly.

    F_p = −∇(p − ρ·c_s²),  c_s² = 1/3
    """
    s = p - rho * (1.0 / 3.0)
    dpx = 0.5 * (jnp.roll(s, -1, axis=0) - jnp.roll(s, 1, axis=0))
    dpy = 0.5 * (jnp.roll(s, -1, axis=1) - jnp.roll(s, 1, axis=1))
    return jnp.stack([-dpx, -dpy], axis=-1)


@jit
def coupled_ac_ns_step(
    f: jnp.ndarray,
    phi: jnp.ndarray,
    omega: float,
    M_ac: float,
    W: float,
    sigma: float,
    rho_g: float,
    rho_w: float,
    gx: float = 0.0,
    gy: float = 0.0,
    use_thermo: bool = False,
    wall: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """One coupled AC + NS step (Yang S10–S17; Guo 2000 pressure NS).

    Parameters
    ----------
    f : (nx, ny, 9) — NS pressure distributions (pre-collision)
    phi : (nx, ny) — order parameter
    omega : float — NS relaxation frequency (1/τ)
    M_ac : float — AC mobility M (lattice)
    W : float — interface width (lu)
    sigma : float — surface tension (lattice)
    rho_g, rho_w : float — gas / water densities
    gx, gy : float — body-force acceleration (lattice)
    use_thermo : bool — include Yang S16 thermodynamic pressure term
        −∇(p − ρc_s²). Off by default (calibration mode); the Laplace
        pressure calibration for the pure capillary-force form is measured
        empirically (see validation/phasefield/).
    wall : (nx, ny) bool or None — no-slip wall mask (half-way bounce-back,
        applied post-streaming). None = fully periodic (Stage 2 default).

    Returns
    -------
    (f_new, phi_new, p, u, rho) :
        f_new (nx,ny,9), phi_new (nx,ny), p (nx,ny), u (nx,ny,2), rho (nx,ny)
    """
    # S13: density interpolation
    rho = rho_g + phi * (rho_w - rho_g)  # (nx,ny)
    # pressure form: p = Σf
    p = jnp.sum(f, axis=-1)  # (nx,ny)
    # momentum (pre-force)
    mom = jnp.dot(f, C.astype(jnp.float64))  # (nx,ny,2)
    # S14: capillary force + (optional S16 thermodynamic term) + body force
    fc = capillary_force(phi, sigma, W)  # (nx,ny,2)
    if use_thermo:
        fc = fc + thermodynamic_force(p, rho)
    fb = jnp.stack([rho * gx, rho * gy], axis=-1)
    F = fc + fb  # total force (nx,ny,2)
    # corrected velocity: ρu = Σfe + F/2
    u = (mom + 0.5 * F) / rho[..., None]  # (nx,ny,2)
    # NS collision: BGK + Guo forcing (pressure form)
    feq = ns_equilibrium_pressure(p, rho, u)
    Fi = guo_force(F, u, omega)
    f_coll = f - omega * (f - feq) + Fi
    f_stream = streaming(f_coll)
    # optional no-slip wall (half-way bounce-back, post-streaming)
    if wall is not None:
        f_new = bounce_back_halfway(f_stream, f_coll, wall)
    else:
        f_new = f_stream
    # AC update (S10) with the physical (corrected) velocity
    phi_new = conservative_ac_step(phi, u[..., 0], u[..., 1], M_ac, W)
    return f_new, phi_new, p, u, rho


# ═══════════════════════════════════════════════════════════════════════════
# Stage 3 — surface-energy wetting (Yang S17)
# ═══════════════════════════════════════════════════════════════════════════


@jit
def surface_energy_phi_s(
    phi_f: jnp.ndarray, theta_deg: float, W: float, sigma: float
) -> jnp.ndarray:
    """Solid-node order parameter from the adjacent fluid node (Yang S17).

    φ_s = (2/a)(1 + a/2 − √((1+a/2)² − 2a·φ_f)) − φ_f,
    a   = −l_sf·√(2β/κ)·cosθ,   β = 12σ/W,  κ = 3σW/2,  l_sf = 1 (lattice)

    At θ=90° (neutral), a≈0 and φ_s = φ_f (recovered via the where-branch).
    The radical is clamped (pitfall #10: θ→0°/180° can make it negative).
    """
    theta = theta_deg * jnp.pi / 180.0
    beta = 12.0 * sigma / W
    kappa = 1.5 * sigma * W
    a = -1.0 * jnp.sqrt(2.0 * beta / kappa) * jnp.cos(theta)
    inner = (1.0 + 0.5 * a) ** 2 - 2.0 * a * phi_f
    inner = jnp.maximum(inner, 1e-12)
    term = (2.0 / a) * (1.0 + 0.5 * a - jnp.sqrt(inner))
    phi_s = jnp.where(jnp.abs(a) < 1e-10, phi_f, term - phi_f)
    return jnp.clip(phi_s, 0.0, 1.0)


@jit
def conservative_ac_step_walls(
    phi: jnp.ndarray,
    ux: jnp.ndarray,
    uy: jnp.ndarray,
    M: float,
    W: float,
    phi_wall_bottom: jnp.ndarray,
    phi_wall_top: jnp.ndarray,
) -> jnp.ndarray:
    """Conservative Allen-Cahn update with no-flux/wetting walls.

    Same physics as ``conservative_ac_step`` (Yang S10) but the y direction
    (axis 1) has ghost rows instead of a periodic wrap:
      * ghost row y=0    carries ``phi_wall_bottom`` (S17 wetting value)
      * ghost row y=ny+1 carries ``phi_wall_top``
    Left/right (axis 0) remain periodic. The flux through the walls is ~0
    when the ghosts sit at the wetting equilibrium, giving the no-flux BC.

    ⚠️ EXPERIMENTAL (Stage 3): the no-y-wrap wall stencil itself is correct
    (a forced fully-wetting ghost φ_s=1 correctly spreads a droplet), but the
    **direct finite-difference conservative AC cannot quantitatively reproduce
    the Yang S17 contact angle** — the S17 ghost interacts with the anti-
    diffusion/capillary balance and either over-sharpens (φ overshoot > 1,
    erratic contact angle) or freezes the interface. The literature-faithful
    fix is the **MRT-LB conservative AC (Yang SI S11, with the mass-conserving
    anti-diffusion source)** — tracked in research/phasefield_development_plan.md
    Stage 3 refinement.

    Returns the interior (nx, ny) updated field.
    """
    nx, ny = phi.shape
    # pad along axis 1 with ghost rows → (nx, ny+2)
    phi_pad = jnp.concatenate(
        [phi_wall_bottom[:, None], phi, phi_wall_top[:, None]], axis=1
    )
    ux_pad = jnp.concatenate([jnp.zeros((nx, 1)), ux, jnp.zeros((nx, 1))], axis=1)
    uy_pad = jnp.concatenate([jnp.zeros((nx, 1)), uy, jnp.zeros((nx, 1))], axis=1)

    # shifted views (x periodic via roll; y no-wrap via pad/slice)
    xp = lambda A: jnp.roll(A, -1, axis=0)
    xm = lambda A: jnp.roll(A, 1, axis=0)
    jp1 = lambda A: jnp.concatenate([A[:, 1:], A[:, -1:]], axis=1)
    jm1 = lambda A: jnp.concatenate([A[:, :1], A[:, :-1]], axis=1)

    P = phi_pad
    P_xp, P_xm = xp(P), xm(P)
    P_jp1, P_jm1 = jp1(P), jm1(P)
    P_xp_jp1 = xp(P_jp1)
    P_xp_jm1 = xp(P_jm1)
    P_xm_jp1 = xm(P_jp1)
    P_xm_jm1 = xm(P_jm1)

    # isotropic D2Q9 gradient
    dphix = (1.0 / 3.0) * (P_xp - P_xm) + (1.0 / 12.0) * (
        P_xp_jp1 + P_xp_jm1 - P_xm_jp1 - P_xm_jm1
    )
    dphiy = (1.0 / 3.0) * (P_jp1 - P_jm1) + (1.0 / 12.0) * (
        P_xp_jp1 + P_xm_jp1 - P_xp_jm1 - P_xm_jm1
    )
    lap = (
        4.0 * (P_xp + P_xm + P_jp1 + P_jm1)
        + (P_xp_jp1 + P_xp_jm1 + P_xm_jp1 + P_xm_jm1)
        - 20.0 * P
    ) / 6.0

    # interface normal + anti-diffusion
    gnorm = jnp.sqrt(dphix * dphix + dphiy * dphiy) + 1e-12
    nx_ = dphix / gnorm
    ny_ = dphiy / gnorm
    phc = jnp.clip(P, 1e-10, 1.0 - 1e-10)
    s = 0.5 * jnp.log(phc / (1.0 - phc))
    anti = (1.0 / W) * (1.0 - jnp.tanh(s) ** 2)

    # diffusion flux divergence
    Jx = M * (dphix - anti * nx_)
    Jy = M * (dphiy - anti * ny_)
    divJ = 0.5 * (xp(Jx) - xm(Jx)) + 0.5 * (jp1(Jy) - jm1(Jy))

    # advection ∇·(φu)
    fluxx = ux_pad * P
    fluxy = uy_pad * P
    div_adv = 0.5 * (xp(fluxx) - xm(fluxx)) + 0.5 * (jp1(fluxy) - jm1(fluxy))

    # Lax–Wendroff correction 0.5·∇·(u·∇φ)
    dphix_c = 0.5 * (P_xp - P_xm)
    dphiy_c = 0.5 * (P_jp1 - P_jm1)
    ax = ux_pad * dphix_c
    ay = uy_pad * dphiy_c
    corr = 0.5 * (0.5 * (xp(ax) - xm(ax)) + 0.5 * (jp1(ay) - jm1(ay)))

    phi_new_pad = P - div_adv + corr + divJ
    return phi_new_pad[:, 1 : ny + 1]  # interior (nx, ny)


def run_droplet_on_wall(
    nx: int,
    ny: int,
    R0: float,
    theta_deg: float,
    sigma: float,
    W: float,
    rho_g: float,
    rho_w: float,
    M_ac: float,
    omega: float,
    n_steps: int,
    theta_top_deg: float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Droplet resting on a bottom no-slip wall with surface-energy wetting (S17).

    - NS: half-way bounce-back no-slip at y=1 and y=ny−2 (bottom/top walls);
      left/right remain periodic.
    - AC: bottom ghost row y=0 carries the S17 wetting value from the fluid
      node y=1 (θ = theta_deg); top ghost y=ny−1 carries θ = theta_top_deg
      (default = neutral 90°). The wall-aware AC (no y-wrap) is used.

    ⚠️ EXPERIMENTAL (Stage 3): the no-slip NS wall works, but the direct-FD
    conservative AC cannot yet quantitatively reproduce the S17 contact angle
    (see conservative_ac_step_walls). Use for qualitative/mechanism studies;
    quantitative contact-angle calibration requires the MRT-LB AC.

    Returns
    -------
    (phi, u) : phi (nx, ny), u (nx, ny, 2) — final fields.
    """
    if theta_top_deg is None:
        theta_top_deg = 90.0
    xc = nx / 2.0
    yc = R0 + 2.0  # droplet rests on the wall (first fluid row y=1)
    f0, phi0 = init_phase_field_droplet(nx, ny, R0, xc, yc, W, sigma, rho_g, rho_w)

    # no-slip walls at the first fluid layers above/below the ghost rows
    wall = wall_mask(nx, ny, "bottom") | wall_mask(nx, ny, "top")

    def step(carry, _):
        f, phi = carry
        # wetting ghosts: S17 from the adjacent fluid node
        phi_wall_b = surface_energy_phi_s(phi[:, 1], theta_deg, W, sigma)
        phi_wall_t = surface_energy_phi_s(phi[:, ny - 2], theta_top_deg, W, sigma)
        # NS with no-slip wall
        f_new, phi_new, p, u, rho = coupled_ac_ns_step(
            f, phi, omega, M_ac, W, sigma, rho_g, rho_w, wall=wall
        )
        # AC with wall-aware stencil (uses the wetting ghosts)
        phi_new = conservative_ac_step_walls(
            phi, u[..., 0], u[..., 1], M_ac, W, phi_wall_b, phi_wall_t
        )
        return (f_new, phi_new), u

    (f, phi), u = lax.scan(step, (f0, phi0), None, length=n_steps)
    return phi, u[-1]


def measure_wall_contact_angle(
    phi: jnp.ndarray, wall_row: int = 1, level: float = 0.5
) -> tuple[float, float]:
    """Estimate the apparent contact angle of a droplet on the bottom wall.

    Fits the φ=level contour as a circular cap sitting on the wall:
      tan(θ) = 2·B·H/(B² − H²)
    where H = max contour height above the wall and B = half contact-line width.

    Returns (theta_deg, base_half_width).
    """
    nx, ny = phi.shape
    # φ=0.5 contour
    contour = (phi >= level) & (phi < level + 0.2)  # banded sampling
    ys, xs = jnp.where(contour)
    if xs.shape[0] == 0:
        return 0.0, 0.0
    xc = float(jnp.mean(xs.astype(jnp.float64)))
    H = float(jnp.max(ys.astype(jnp.float64)) - wall_row)  # height above wall
    # half base width: max |x − xc| where φ>level near the wall
    near_wall = phi[:, wall_row : wall_row + 3] > level
    cols = jnp.where(near_wall.any(axis=1))[0]
    if cols.shape[0] == 0:
        B = 0.0
    else:
        B = 0.5 * (float(cols.max()) - float(cols.min()))
    if B <= 0 or H <= 0:
        return 0.0, B
    tan_th = 2.0 * B * H / (B * B - H * H)
    theta = float(jnp.degrees(jnp.arctan(tan_th)))
    if theta < 0:
        theta = 180.0 + theta
    return theta, B


# ═══════════════════════════════════════════════════════════════════════════
# Initialisation & drivers
# ═══════════════════════════════════════════════════════════════════════════


def init_phi_droplet(
    nx: int, ny: int, R0: float, xc: float, yc: float, W: float
) -> jnp.ndarray:
    """Initialise a circular droplet order-parameter field.

    φ = 0.5(1 − tanh(2(r − R0)/W))  →  φ=1 (water) inside, φ=0 (gas) outside.
    """
    x = jnp.arange(nx, dtype=jnp.float64)
    y = jnp.arange(ny, dtype=jnp.float64)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    r = jnp.sqrt((X - xc) ** 2 + (Y - yc) ** 2)
    return 0.5 * (1.0 - jnp.tanh(2.0 * (r - R0) / W))


def init_phase_field_droplet(
    nx: int,
    ny: int,
    R0: float,
    xc: float,
    yc: float,
    W: float,
    sigma: float,
    rho_g: float,
    rho_w: float,
    p_ref: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Initialise (f, phi) for a static-droplet simulation.

    f is set to the pressure-form equilibrium at a uniform reference
    pressure p_ref and zero velocity; the capillary force then develops
    the Laplace pressure jump σ/R.
    """
    phi = init_phi_droplet(nx, ny, R0, xc, yc, W)
    rho = rho_g + phi * (rho_w - rho_g)
    u0 = jnp.zeros((nx, ny, 2))
    p0 = jnp.full((nx, ny), p_ref, dtype=jnp.float64)
    f = ns_equilibrium_pressure(p0, rho, u0)
    return f, phi


def run_static_droplet(
    nx: int,
    ny: int,
    R0: float,
    sigma: float,
    W: float,
    rho_g: float,
    rho_w: float,
    M_ac: float,
    omega: float,
    n_steps: int,
    gx: float = 0.0,
    gy: float = 0.0,
    xc: float | None = None,
    yc: float | None = None,
    p_ref: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run a static two-phase droplet to (near) equilibrium.

    Returns
    -------
    (f, phi, p, u, rho) — final fields (f (nx,ny,9), phi/p/rho (nx,ny), u (nx,ny,2)).
    """
    if xc is None:
        xc = nx / 2.0
    if yc is None:
        yc = ny / 2.0
    f0, phi0 = init_phase_field_droplet(
        nx, ny, R0, xc, yc, W, sigma, rho_g, rho_w, p_ref
    )

    def step(carry, _):
        f, phi = carry
        f_new, phi_new, p, u, rho = coupled_ac_ns_step(
            f, phi, omega, M_ac, W, sigma, rho_g, rho_w, gx, gy
        )
        return (f_new, phi_new), (p, u, rho)

    (f, phi), (p_stack, u_stack, rho_stack) = lax.scan(
        step, (f0, phi0), None, length=n_steps
    )
    return f, phi, p_stack[-1], u_stack[-1], rho_stack[-1]


def detect_droplet_radius(
    phi: jnp.ndarray, xc: float, yc: float, level: float = 0.5
) -> float:
    """Estimate droplet radius from the φ=level contour (mid-interface).

    Computes the mean distance from (xc, yc) of the **first** φ-level crossing
    along 8 radial directions (the interface nearest to the centre).
    """
    nx, ny = phi.shape
    g = phi - level  # + inside (liquid, φ>level), − outside (gas)
    maxr = max(nx, ny) * 0.6
    rr = jnp.arange(0.0, maxr, 1.0)
    n_r = rr.shape[0]
    radii = []
    for theta in jnp.linspace(0.0, 2.0 * jnp.pi, 8, endpoint=False):
        dx = jnp.cos(theta)
        dy = jnp.sin(theta)
        px = xc + rr * dx
        py = yc + rr * dy
        ix = jnp.clip(jnp.round(px).astype(jnp.int32), 0, nx - 1)
        iy = jnp.clip(jnp.round(py).astype(jnp.int32), 0, ny - 1)
        g_ray = g[ix, iy]
        sign = jnp.sign(g_ray)
        # first sign flip (liquid → gas) along the ray
        cross = (sign[:-1] != sign[1:]) & (sign[:-1] != 0)
        idxs = jnp.where(cross, jnp.arange(1, n_r), n_r - 1)
        idx = int(jnp.min(idxs))  # first crossing
        radii.append(float(rr[idx] + 0.5))
    return float(jnp.mean(jnp.array(radii)))


def measure_droplet_pressure(
    p: jnp.ndarray, phi: jnp.ndarray, xc: float, yc: float, R: float
) -> tuple[float, float, float]:
    """Measure (p_in, p_out, ΔP) around a droplet (Huang-guide style).

    p_in  = mean p where r < 0.6·R
    p_out = mean p where r > 1.5·R
    """
    nx, ny = p.shape
    x = jnp.arange(nx, dtype=jnp.float64)
    y = jnp.arange(ny, dtype=jnp.float64)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    r = jnp.sqrt((X - xc) ** 2 + (Y - yc) ** 2)
    inside = r < 0.6 * R
    outside = r > 1.5 * R
    p_in = float(jnp.mean(p[inside]))
    p_out = float(jnp.mean(p[outside]))
    return p_in, p_out, p_in - p_out
