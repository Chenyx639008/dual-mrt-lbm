"""Batch parameter sweep via jax.vmap (P1a).

Enables one-line LBM parameter space exploration::

    from jax_lbm.parameter_sweep import sweep_surface_tension
    sigma_grid = sweep_surface_tension(T_range=(0.7, 0.9, 5), eps_range=(1.0, 2.5, 5))

Reference
---------
- Phase 6 §5.1 (P1a: JAX vmap 批量参数扫描)
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax

from jax_lbm.lattice import CS2
from jax_lbm.collision import collision_bgk, equilibrium
from jax_lbm.eos import cs_eos_pressure
from jax_lbm.force import shan_chen_force
from jax_lbm.d2q9_bgk import init_droplet, init_equilibrium, streaming


@partial(jit, static_argnames=("nx", "ny", "n_steps"))
def _single_droplet_run(T_reduced, k1, nx=64, ny=64, n_steps=200):
    """Run one droplet simulation, return final mass. (vmap-compatible)"""
    Tc = 0.3773 / 4.0
    T = T_reduced * Tc
    omega = 1.0 / 1.5

    rho_g = 0.045410  # T/Tc=0.90 coexistence
    rho_l = 0.248108

    rho_init, u_init = init_droplet(
        nx,
        ny,
        radius=nx // 4,
        center=(nx // 2, ny // 2),
        rho_l=rho_l,
        rho_g=rho_g,
        width=3.0,
    )
    f = init_equilibrium(rho_init, u_init)

    def step_fn(f, _):
        rho = jnp.sum(f, axis=-1, keepdims=True)
        p = cs_eos_pressure(rho, T=T)
        psi_sq = 2.0 * (p - rho * CS2) / (-1.0)
        psi = jnp.sqrt(jnp.clip(psi_sq, 0.0, None))
        F = shan_chen_force(psi, G=-1.0)
        return streaming(collision_bgk(f, omega, F)), None

    f_final, _ = lax.scan(step_fn, f, None, length=n_steps)
    rho_final = jnp.sum(f_final, axis=-1)
    return jnp.sum(rho_final)


# ── Vectorized versions ──
batch_droplet_run = vmap(_single_droplet_run, in_axes=(0, 0, None, None, None))


def sweep_surface_tension(
    T_range=(0.70, 0.95, 4), eps_range=(1.0, 2.5, 4), nx=64, ny=64, n_steps=200
):
    """Sweep (T_reduced, epsilon) parameter space.

    Parameters
    ----------
    T_range : tuple (min, max, n_points)
    eps_range : tuple (min, max, n_points)
    nx, ny : grid dimensions
    n_steps : steps per simulation

    Returns
    -------
    mass_grid : (n_T, n_eps) array — final mass for each (T, ε) pair
    T_vals, eps_vals : 1D arrays — parameter values
    """
    T_vals = jnp.linspace(T_range[0], T_range[1], T_range[2])
    eps_vals = jnp.linspace(eps_range[0], eps_range[1], eps_range[2])
    TT, EE = jnp.meshgrid(T_vals, eps_vals, indexing="ij")

    T_flat = TT.ravel()
    eps_flat = EE.ravel()
    k1_flat = -eps_flat / 8.0

    masses = batch_droplet_run(T_flat, k1_flat, nx, ny, n_steps)
    return masses.reshape(TT.shape), T_vals, eps_vals


def sweep_contact_angle(
    theta_range=(30, 150, 7), G_ads=0.1, nx=256, ny=128, n_steps=500
):
    """Sweep contact angle parameter space.

    Parameters
    ----------
    theta_range : tuple (min_deg, max_deg, n_points)
    G_ads : adsorption strength
    nx, ny : grid dimensions
    n_steps : steps per simulation

    Returns
    -------
    theta_vals : 1D array — contact angles in degrees
    """
    theta_vals = jnp.linspace(theta_range[0], theta_range[1], theta_range[2])
    # For now returns parameter grid; full wetting simulation TBD
    return theta_vals
