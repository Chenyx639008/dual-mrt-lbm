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
    # Always compute the force (JIT-safe, no early return).
    # When G_ads=0 or mask_wall is all False, result is naturally zero.

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


# ═══════════════════════════════════════════════════════════════════════════
# Huang & Wu (2016) 三阶力修正 — Q_m 表面张力修正 + Guo 力源项
# ═══════════════════════════════════════════════════════════════════════════


def compute_Q_huang(
    psi,
    Fx_mol,
    Fy_mol,
    k1=1.0 / 12.0,
    k2=0.0,
    psi_cut=1e-3,
    G=-1.0,
    c=1.0,
    se=1.0,
    st=1.0,
    sp=1.0,
):
    """Q_m surface-tension correction in moment space (Huang & Wu 2016 Eq. 57-62).

    Translates CUDA ``compute_Q_huang_gpu`` line-for-line into JAX.

    Uses the identity ∇ψ = −F_mol/(G·ψ) to avoid explicit finite-difference
    gradients, then constructs the third-order isotropic correction in the
    MRT moment basis.

    Parameters
    ----------
    psi : (nx, ny, 1) array — pseudopotential
    Fx_mol, Fy_mol : (nx, ny, 1) arrays — molecular force (pure SC, no adsorption)
    k1 : float — first Huang coefficient (derived from ε: k1 = −ε/8 − k2)
    k2 : float — second Huang coefficient (anisotropy, default 0)
    psi_cut : float — ψ² floor to avoid division by zero (default 1e-3)
    G : float — interaction strength (default −1.0)
    c : float — lattice speed (default 1.0 for D2Q9)
    se : float — energy relaxation rate s_e (A_a[1])
    st : float — trace relaxation rate s_t (A_a[2])
    sp : float — stress relaxation rate s_p (A_a[7])

    Returns
    -------
    C : (nx, ny, 9) array — moment-space correction
        Only slots 1 (e), 2 (ε), 7 (pxx), 8 (pxy) are non-zero.
    """
    # — |F_mol|² —
    F2 = Fx_mol * Fx_mol + Fy_mol * Fy_mol  # (nx, ny, 1)

    # — ψ² + cut² floor —
    psi2 = psi * psi + psi_cut * psi_cut  # (nx, ny, 1)

    # — denominator G·ψ²·c² with sign-preserving clamp —
    denom = G * psi2 * c * c
    denom_safe = jnp.where(
        jnp.abs(denom) < 1e-12,
        jnp.sign(denom) * 1e-12,
        denom,
    )

    # — Moment-space Q_m components (paper Eq. 59-62) —
    # Qm1: isotropic trace (affects surface tension magnitude)
    Qm1 = 3.0 * (k1 + 2.0 * k2) * F2 / denom_safe
    # Qm7: diagonal stress anisotropy
    Qm7 = k1 * (Fx_mol * Fx_mol - Fy_mol * Fy_mol) / denom_safe
    # Qm8: off-diagonal stress
    Qm8 = k1 * Fx_mol * Fy_mol / denom_safe

    # — Bake relaxation rates into C (CUDA convention: +C·Δt in collision) —
    nx, ny = psi.shape[0], psi.shape[1]
    C = jnp.zeros((nx, ny, 9), dtype=jnp.float64)
    C = C.at[..., 1].set(se * Qm1[..., 0])  # s_e · Qm1
    C = C.at[..., 2].set(-st * Qm1[..., 0])  # −s_t · Qm1
    C = C.at[..., 7].set(sp * Qm7[..., 0])  # s_p · Qm7
    C = C.at[..., 8].set(sp * Qm8[..., 0])  # s_p · Qm8
    # slots 0,3,4,5,6 remain zero
    return C


def compute_S_guo(ux, uy, Fx, Fy):
    """Guo forcing source term in moment space (single-component).

    Translates CUDA ``compute_S_huang_gpu`` line-for-line into JAX.

    Parameters
    ----------
    ux, uy : (nx, ny, 1) arrays — macroscopic velocity (half-force corrected)
    Fx, Fy : (nx, ny, 1) arrays — total force (molecular + adsorption + body)

    Returns
    -------
    S : (nx, ny, 9) array — Guo source term in moment space
    """
    uF = ux * Fx + uy * Fy  # (nx, ny, 1)

    nx, ny = ux.shape[0], ux.shape[1]
    S = jnp.zeros((nx, ny, 9), dtype=jnp.float64)
    S = S.at[..., 0].set(0.0)
    S = S.at[..., 1].set(6.0 * uF[..., 0])
    S = S.at[..., 2].set(-6.0 * uF[..., 0])
    S = S.at[..., 3].set(Fx[..., 0])
    S = S.at[..., 4].set(-Fx[..., 0])
    S = S.at[..., 5].set(Fy[..., 0])
    S = S.at[..., 6].set(-Fy[..., 0])
    S = S.at[..., 7].set(2.0 * (ux * Fx - uy * Fy)[..., 0])
    S = S.at[..., 8].set((ux * Fy + uy * Fx)[..., 0])
    return S


def huang_zhang_force_and_correction(
    psi, k1=1.0 / 12.0, k2=0.0, psi_cut=1e-3, G=-1.0, c=1.0, se=1.0, st=1.0, sp=1.0
):
    """Full Huang-Zhang force pipeline: F_mol + Q_m(C) + S_guo.

    Convenience wrapper that computes the SC molecular force, the Q_m
    surface-tension correction, and the Guo source term in one call.

    Parameters
    ----------
    psi : (nx, ny, 1) — pseudopotential
    k1, k2 : float — Huang coefficients
    psi_cut : float — ψ² floor
    G : float — interaction strength
    c : float — lattice speed
    se, st, sp : float — relaxation rates

    Returns
    -------
    F_mol : (nx, ny, 2) — pure SC molecular force
    C : (nx, ny, 9) — Q_m moment-space correction
    S_guo : (nx, ny, 9) — placeholder (needs velocity for Guo term)
    """
    # Step 1: pure SC molecular force
    F_mol = shan_chen_force(psi, G=G)

    # Step 2: Q_m surface-tension correction
    Fx = F_mol[..., 0:1]
    Fy = F_mol[..., 1:2]
    C = compute_Q_huang(
        psi, Fx, Fy, k1=k1, k2=k2, psi_cut=psi_cut, G=G, c=c, se=se, st=st, sp=sp
    )

    # Step 3: placeholder S_guo (needs velocity, call compute_S_guo separately)
    S_guo = jnp.zeros_like(C)

    return F_mol, C, S_guo
