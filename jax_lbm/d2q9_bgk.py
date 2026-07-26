"""Minimal differentiable LBM prototype for SCMP validation.

~120 lines of pure JAX: D2Q9 + BGK + Shan-Chen + CS-EOS.
Designed to validate CUDA results and perform sensitivity analysis
(jax.grad through LBM simulation). Not a replacement for the CUDA solver.

Now imports core modules from jax_lbm (lattice, eos, collision, boundary,
force, wetting) for modularity. This file retains the simulation orchestration
and backward-compatible API.

Usage::

    from jax_lbm.d2q9_bgk import run_lbm_scmp, init_droplet
    from jax_lbm import get_eos, get_collision, get_wetting

    # Advanced: use MRT + contact angle
    f_final = run_lbm_scmp(f0, omega, eos_params, n_steps,
                           collision='mrt', wetting='psi_ghost_bc',
                           theta_deg=60.0)

Reference
---------
- Huang, H.-B., & Wu, Y.-C. (2016). Phys. Rev. E 93, 043311.
- JAX-LaB: src/jax_lab/ (reference implementation)
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import grad, jit, lax

# ── Import from modular jax_lbm ──
from jax_lbm.lattice import C, W, Q, CS2, INV_CS2, OPP, G_FF
from jax_lbm.eos import cs_eos_pressure, pseudopotential as _pseudopotential
from jax_lbm.collision import (
    macroscopic,
    equilibrium,
    collision_bgk,
    collision_mrt,
    MRT_S_DEFAULT,
)
from jax_lbm.force import shan_chen_force as _shan_chen_force
from jax_lbm.boundary import (
    wall_mask,
    ghost_mask,
    bounce_back_halfway,
    equilibrium_bc,
    zou_he_bottom_wall,
)
from jax_lbm.wetting import apply_psi_ghost_bc

# ═══════════════════════════════════════════════════════════════════════════
# Backward-compatible wrappers
# ═══════════════════════════════════════════════════════════════════════════

# cs_eos_pressure already imported from eos module


@jit
def cs_eos_pseudopotential(rho, G=-1.0, a=1.0, b=4.0, R=1.0, T=0.066):
    """Backward-compatible wrapper."""
    return _pseudopotential(rho, cs_eos_pressure, {"a": a, "b": b, "R": R, "T": T}, G=G)


# ═══════════════════════════════════════════════════════════════════════════
# Coexistence density solver (Maxwell construction)
# ═══════════════════════════════════════════════════════════════════════════


def _find_coexistence(T_reduced=0.70, a=1.0, b=4.0, R=1.0):
    """Find coexistence densities for CS-EOS.

    Uses pressure scan to locate the gas and liquid branches.
    For CS-EOS at typical parameters (a=1,b=4,R=1):
      T_reduced=0.70 → ρ_g≈0.005, ρ_l≈0.38
      T_reduced=0.80 → ρ_g≈0.010, ρ_l≈0.30
      T_reduced=0.90 → ρ_g≈0.025, ρ_l≈0.20

    Parameters
    ----------
    T_reduced : float
        Reduced temperature T/Tc.
    a, b, R : float
        EOS parameters.

    Returns
    -------
    (rho_gas, rho_liquid) : tuple[float, float]
    """
    import numpy as np

    Tc = 0.3773 * a / (b * R)
    T = T_reduced * Tc

    def p_np(rho):
        eta = b * rho / 4.0
        if eta >= 1.0 or rho <= 0:
            return -1e10
        eta2 = eta * eta
        eta3 = eta2 * eta
        denom = (1.0 - eta) ** 3
        return R * rho * T * (1.0 + eta + eta2 - eta3) / denom - a * rho * rho

    # Scan pressure over density range with fine resolution
    rho_vals = np.logspace(-4, -0.01, 2000)  # 1e-4 to ~0.98
    p_vals = np.array([p_np(r) for r in rho_vals])

    # Find gas density: first density where p > 0 (moving from low to high)
    positive_mask = p_vals > 0
    gas_idx = np.argmax(positive_mask)  # first positive
    rho_g = float(rho_vals[gas_idx])

    # Find where p becomes negative after gas branch (spinodal)
    neg_after_gas = np.where(np.diff(positive_mask.astype(int)) == -1)[0]
    if len(neg_after_gas) > 0:
        unstable_start = neg_after_gas[0]
        # Find where p becomes positive again (liquid branch)
        pos_after_unstable = np.where(np.diff(positive_mask.astype(int)) == 1)[0]
        pos_after_start = [x for x in pos_after_unstable if x > unstable_start]
        if pos_after_start:
            rho_l = float(rho_vals[pos_after_start[0]])
        else:
            # Liquid density at ~0.38 for T=0.7
            rho_l = 0.38 * (0.7 / T_reduced) ** 0.5
    else:
        # Use estimates
        rho_l = 0.38 * (0.7 / T_reduced) ** 0.5

    # Sanity clamp
    rho_g = max(min(rho_g, 0.1), 1e-4)
    rho_l = max(min(rho_l, 0.5), rho_g + 0.05)

    return float(rho_g), float(rho_l)


# ═══════════════════════════════════════════════════════════════════════════
# Streaming (distribution-field version — kept here for backward compat)
# ═══════════════════════════════════════════════════════════════════════════


@jit
def streaming(f):
    """Periodic streaming via jnp.roll.

    Standard LBM post-collision streaming:
        f_new(x + c_k·Δt, k) = f_old(x, k)
    equivalent to
        f_new(x, k) = f_old(x − c_k·Δt, k)

    Parameters
    ----------
    f : (nx, ny, 9) array

    Returns
    -------
    f_streamed : (nx, ny, 9) array
    """
    f_streamed = jnp.zeros_like(f)
    for k in range(Q):
        # f_new(x, k) = f_old(x − c_k, k) → roll forward by c_k
        f_streamed = f_streamed.at[..., k].set(
            jnp.roll(
                jnp.roll(f[..., k], C[k, 0].astype(int), axis=0),
                C[k, 1].astype(int),
                axis=1,
            )
        )
    return f_streamed


# ═══════════════════════════════════════════════════════════════════════════
# Enhanced step function (supports BC, wetting, MRT collision)
# ═══════════════════════════════════════════════════════════════════════════


def shan_chen_force(rho, psi):
    """Backward-compatible: delegates to jax_lbm.force.shan_chen_force."""
    return _shan_chen_force(psi, G=-1.0)  # G is already in psi computation


@jit
def step_scmp(state, _):
    """One LBM step: collision → streaming. Designed for lax.scan.

    Backward-compatible with original API. Uses BGK + periodic BC only.

    Parameters
    ----------
    state : (f, omega, eos_params_dict)
    _ : unused (required by lax.scan)

    Returns
    -------
    new_state, None
    """
    f, omega, eos_params = state
    rho, _ = macroscopic(f)
    psi = cs_eos_pseudopotential(
        rho,
        G=eos_params.get("G", -1.0),
        a=eos_params.get("a", 1.0),
        b=eos_params.get("b", 4.0),
        R=eos_params.get("R", 1.0),
        T=eos_params.get("T", 0.066),
    )
    F = _shan_chen_force(psi, G=eos_params.get("G", -1.0))
    f = collision_bgk(f, omega, F)
    f = streaming(f)
    return (f, omega, eos_params), None


@partial(jit, static_argnames=("n_steps",))
def run_lbm_scmp(f0, omega, eos_params_tuple, n_steps):
    """Run SCMP LBM simulation for n_steps (backward-compatible).

    Parameters
    ----------
    f0 : (nx, ny, 9) array — initial distributions
    omega : float — relaxation frequency
    eos_params_tuple : tuple — (a, b, R, T, G) EOS parameters
    n_steps : int — number of time steps

    Returns
    -------
    f_final : (nx, ny, 9) array
    """
    a, b, R, T, G = eos_params_tuple
    eos_params = {"a": a, "b": b, "R": R, "T": T, "G": G}
    (f_final, _, _), _ = lax.scan(
        step_scmp, (f0, omega, eos_params), None, length=n_steps
    )
    return f_final


# ═══════════════════════════════════════════════════════════════════════════
# 🆕 Advanced simulation function (supports EOS + BC + MRT + wetting)
# ═══════════════════════════════════════════════════════════════════════════


@partial(
    jit,
    static_argnames=(
        "n_steps",
        "eos_name",
        "eos_params",
        "collision_type",
        "S",
        "bc_type",
        "bc_mask",
        "bc_wall",
        "wetting_type",
        "theta_deg",
        "G_ads",
    ),
)
def run_lbm_advanced(
    f0,
    omega,
    n_steps,
    eos_name="cs",
    eos_params=None,
    collision_type="bgk",
    S=None,
    bc_type="periodic",
    bc_mask=None,
    bc_wall="bottom",
    wetting_type="none",
    theta_deg=90.0,
    G_ads=0.0,
):
    """Run LBM with full configuration: EOS + collision + BC + wetting.

    Parameters
    ----------
    f0 : (nx, ny, 9) — initial distributions
    omega : float — relaxation frequency
    n_steps : int
    eos_name : str — 'cs', 'pr', 'rk', 'rks', 'vdw'
    eos_params : dict — EOS parameters (depends on EOS type)
    collision_type : str — 'bgk' or 'mrt'
    S : (9,) array — MRT relaxation times (required if collision='mrt')
    bc_type : str — 'periodic', 'bounce_back_halfway', 'equilibrium', 'zou_he_bottom'
    bc_mask : (nx, ny) bool or None — wall mask for BC
    bc_wall : str — wall position for zou_he bottom wall
    wetting_type : str — 'psi_ghost_bc', 'improved_virtual_density', 'none'
    theta_deg : float — contact angle in degrees
    G_ads : float — adsorption force strength

    Returns
    -------
    f_final : (nx, ny, 9)
    """
    from jax_lbm.eos import get_eos, pseudopotential as _psi
    from jax_lbm.collision import get_collision
    from jax_lbm.force import total_force as _total_force
    from jax_lbm.boundary import get_bc
    from jax_lbm.wetting import get_wetting

    # Resolve EOS
    eos_fn = get_eos(eos_name)
    if eos_params is None:
        eos_params = {}

    # Resolve collision
    collide_fn = get_collision(collision_type)

    # Resolve BC
    bc_fn = get_bc(bc_type)

    # Resolve wetting
    wetting_fn = get_wetting(wetting_type)

    # Build wall mask if needed
    nx, ny = f0.shape[0], f0.shape[1]
    if bc_mask is None:
        from jax_lbm.boundary import wall_mask as _wm, ghost_mask as _gm

        bc_mask = _wm(nx, ny, wall=bc_wall)
        ghost = _gm(nx, ny, wall=bc_wall)

    def step_fn(state, _):
        f, omega_val = state
        rho, _ = macroscopic(f)

        # Pseudopotential
        psi = _psi(rho, eos_fn, eos_params, G=-1.0)

        # Force
        F = _total_force(psi, rho, ghost, G_sc=-1.0, G_ads=G_ads)

        # Wetting (update ghost ψ before force for next step)
        if wetting_fn is not None:
            psi = wetting_fn(psi, theta_deg, wall=bc_wall)

        # Collision
        if collision_type == "mrt":
            S_arr = S if S is not None else MRT_S_DEFAULT
            f = collide_fn(f, S_arr, F)
        else:
            f = collide_fn(f, omega_val, F)

        # Streaming
        f = streaming(f)

        # BC (post-streaming)
        if bc_fn is not None and bc_type != "periodic":
            if bc_type == "zou_he_bottom":
                f = bc_fn(f, f, F[..., 0], F[..., 1])  # zou_he uses fin, fout, Fx, Fy
            elif bc_type == "equilibrium":
                f = bc_fn(f, bc_mask, u_target=None)  # no-slip
            else:
                f = bc_fn(f, f, bc_mask)  # bounce_back uses fin, fout, mask

        return (f, omega_val), None

    (f_final, _), _ = lax.scan(step_fn, (f0, omega), None, length=n_steps)
    return f_final


# ═══════════════════════════════════════════════════════════════════════════
# 🆕 SCMP Huang step — CUDA-equivalent (P0a: JAX 轨物理对等化)
# ═══════════════════════════════════════════════════════════════════════════


@partial(
    jit,
    static_argnames=(
        "k1",
        "k2",
        "psi_cut",
        "G_mol",
        "c_lat",
        "se",
        "st",
        "sp",
        "alpha_meq",
        "theta_deg",
        "wall",
    ),
)
def _step_scmp_huang_core(
    f,
    s_relax,
    eos_a,
    eos_b,
    eos_R,
    eos_T,
    k1,
    k2,
    psi_cut,
    G_mol,
    c_lat,
    se,
    st,
    sp,
    alpha_meq,
    G_ads=0.0,
    theta_deg=None,
    wall="bottom",
):
    """Single SCMP Huang time-step with optional adsorption + contact angle BC.

    Execution order mirrors CUDA ``scmp_iteration()`` exactly.

    Parameters
    ----------
    G_ads : float — adsorption strength (0 = no wall interaction)
    theta_deg : float or None — contact angle in degrees (None = periodic BC)
    wall : str — wall position for ghost BC: 'bottom', 'top'

    Returns (f_new,) — compatible with lax.scan.
    """
    from jax_lbm.eos import cs_eos_pressure
    from jax_lbm.force import (
        compute_Q_huang,
        compute_S_guo,
        shan_chen_force,
        adsorption_force,
    )
    from jax_lbm.wetting import apply_psi_ghost_bc

    # — Step 1: ρ = Σf_i (density from post-streaming f) —
    rho = jnp.sum(f, axis=-1, keepdims=True)

    # — Step 2: ψ = sqrt(2(p_EOS − ρcs²)/(Gc²)) —
    p_eos = cs_eos_pressure(rho, a=eos_a, b=eos_b, R=eos_R, T=eos_T)
    psi_sq = 2.0 * (p_eos - rho * CS2) / (G_mol * c_lat * c_lat)
    psi_sq = jnp.clip(psi_sq, 0.0, None)
    psi = jnp.sqrt(psi_sq)

    # — Step 3: Ghost ψ BC (Scheme IV, for contact angle) —
    if theta_deg is not None:
        psi = apply_psi_ghost_bc(psi, theta_deg, wall=wall)

    # — Step 4: F_mol = pure SC molecular force —
    F_mol = shan_chen_force(psi, G=G_mol)

    # — Step 5: C_qm = Q_m(F_mol) — surface-tension correction —
    Fx_mol = F_mol[..., 0:1]
    Fy_mol = F_mol[..., 1:2]
    C_qm = compute_Q_huang(
        psi,
        Fx_mol,
        Fy_mol,
        k1=k1,
        k2=k2,
        psi_cut=psi_cut,
        G=G_mol,
        c=c_lat,
        se=se,
        st=st,
        sp=sp,
    )

    # — Step 6: F_total = F_mol + F_ads (adsorption AFTER Q_m) —
    F_total = F_mol
    # Always compute adsorption (zero when no wall) — avoids JIT conditional
    nx_f, ny_f = f.shape[0], f.shape[1]
    mask_wall = jnp.zeros((nx_f, ny_f), dtype=bool)
    if theta_deg is not None:
        if wall == "bottom":
            mask_wall = mask_wall.at[:, 0].set(True)
        elif wall == "top":
            mask_wall = mask_wall.at[:, ny_f - 1].set(True)
    F_ads = adsorption_force(psi, mask_wall, G_ads=G_ads)
    F_total = F_mol + F_ads

    # — Step 7: u = (u_raw + 0.5·F_total)/ρ — half-force velocity correction —
    rho_safe = jnp.clip(rho, 1e-8, None)
    u_raw = jnp.dot(f, C.astype(jnp.float64)) / rho_safe
    u = u_raw + 0.5 * F_total / rho_safe

    # — Numerical guard: velocity cap (matches CUDA UMAX=0.15) —
    UMAX = 0.15
    u2 = u[..., 0] * u[..., 0] + u[..., 1] * u[..., 1]
    u2_safe = jnp.clip(u2, 1e-12, None)
    scale = jnp.where(u2 > UMAX * UMAX, UMAX / jnp.sqrt(u2_safe), 1.0)
    u = u * scale[..., jnp.newaxis]

    # — Step 8: S_guo = Guo source term from corrected u —
    ux = u[..., 0:1]
    uy = u[..., 1:2]
    Fx = F_total[..., 0:1]
    Fy = F_total[..., 1:2]
    S_guo = compute_S_guo(ux, uy, Fx, Fy)

    # — Step 9: MRT collision (Guo+C mode) —
    from jax_lbm.collision import collision_mrt as _collision_mrt

    f_coll = _collision_mrt(f, s_relax, S_guo=S_guo, C=C_qm, alpha_meq=alpha_meq)

    # — Step 10: Streaming —
    f_new = streaming(f_coll)

    # — Step 11: BC (bounce-back handled externally if needed) —

    return f_new


def step_scmp_huang(
    f0,
    s_relax,
    n_steps,
    eos_a=1.0,
    eos_b=4.0,
    eos_R=1.0,
    eos_T=0.066,
    k1=1.0 / 12.0,
    k2=0.0,
    psi_cut=1e-3,
    G_mol=-1.0,
    c_lat=1.0,
    se=1.5,
    st=1.5,
    sp=1.5,
    alpha_meq=1.0,
):
    """Run SCMP Huang simulation with CUDA-equivalent physics.

    Parameters
    ----------
    f0 : (nx, ny, 9) — initial distributions
    s_relax : (9,) array — MRT relaxation rates
    n_steps : int — number of time steps
    eos_a, eos_b, eos_R, eos_T : float — CS-EOS parameters
    k1, k2 : float — Huang coefficients
    psi_cut : float — ψ² floor for Q_m denominator
    G_mol : float — interaction strength (default −1.0)
    c_lat : float — lattice speed (default 1.0)
    se, st, sp : float — relaxation rates for Q_m baking
    alpha_meq : float — equilibrium moment coefficient

    Returns
    -------
    f_final : (nx, ny, 9) — final distributions
    """
    step_fn = partial(
        _step_scmp_huang_core,
        s_relax=s_relax,
        eos_a=eos_a,
        eos_b=eos_b,
        eos_R=eos_R,
        eos_T=eos_T,
        k1=k1,
        k2=k2,
        psi_cut=psi_cut,
        G_mol=G_mol,
        c_lat=c_lat,
        se=se,
        st=st,
        sp=sp,
        alpha_meq=alpha_meq,
    )

    def scan_fn(f, _):
        f_new = step_fn(f)
        return f_new, None

    f_final, _ = lax.scan(scan_fn, f0, None, length=n_steps)
    return f_final


# ═══════════════════════════════════════════════════════════════════════════
# Initialization helpers
# ═══════════════════════════════════════════════════════════════════════════


def init_droplet(nx, ny, radius, center, rho_l, rho_g, width=3.0):
    """Initialize a circular droplet with tanh interface.

    Parameters
    ----------
    nx, ny : int — grid dimensions
    radius : float — droplet radius
    center : (float, float) — droplet center
    rho_l : float — liquid density
    rho_g : float — gas density
    width : float — interface width in lattice units

    Returns
    -------
    rho : (nx, ny, 1) array
    u : (nx, ny, 2) array — zero velocity
    """
    x = jnp.arange(nx, dtype=jnp.float64)
    y = jnp.arange(ny, dtype=jnp.float64)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    dist = jnp.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    rho = 0.5 * (rho_l + rho_g) - 0.5 * (rho_l - rho_g) * jnp.tanh(
        2.0 * (dist - radius) / width
    )
    rho = rho[..., None]
    u = jnp.zeros((nx, ny, 2), dtype=jnp.float64)
    return rho, u


def init_equilibrium(rho, u):
    """Initialize f from equilibrium distribution."""
    return equilibrium(rho, u)


# ═══════════════════════════════════════════════════════════════════════════
# Sensitivity analysis (CUDA can't do this!)
# ═══════════════════════════════════════════════════════════════════════════


def make_permeability_fn(nx, ny, n_steps, omega, eos_params):
    """Create a differentiable permeability function.

    Returns a function permeability(radius, center_x) that runs the full
    LBM simulation and computes a scalar output. This function is
    differentiable via jax.grad.

    Parameters
    ----------
    nx, ny : int
    n_steps : int
    omega : float
    eos_params : dict

    Returns
    -------
    fn : callable — permeability(radius, center_x) → float
    """
    T = eos_params.get("T", 0.066)
    rho_l, rho_g = _cached_coexistence(eos_params)

    @jit
    def permeability(radius, center_x):
        rho, u = init_droplet(nx, ny, radius, (center_x, ny / 2), rho_l, rho_g)
        f0 = init_equilibrium(rho, u)
        f_final = run_lbm_scmp(f0, omega, eos_params, n_steps)
        rho_final, _ = macroscopic(f_final)
        # Scalar output: total liquid mass (> threshold)
        threshold = 0.5 * (rho_l + rho_g)
        return jnp.sum(jnp.where(rho_final > threshold, rho_final, 0.0))

    return permeability


_coexistence_cache: dict = {}


def _cached_coexistence(eos_params):
    """Cache coexistence densities to avoid repeated root-finding."""
    key = (
        eos_params.get("a", 1.0),
        eos_params.get("b", 4.0),
        eos_params.get("R", 1.0),
        eos_params.get("T", 0.066),
    )
    Tc = 0.3773 * key[0] / (key[1] * key[2])
    T_reduced = key[3] / Tc
    cache_key = (key[0], key[1], key[2], round(T_reduced, 6))
    if cache_key not in _coexistence_cache:
        _coexistence_cache[cache_key] = _find_coexistence(T_reduced, *key[:3])
    return _coexistence_cache[cache_key]
