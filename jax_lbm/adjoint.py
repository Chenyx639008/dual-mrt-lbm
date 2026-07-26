"""Adjoint sensitivity analysis — differentiable LBM via jax.grad (P2a).

Enables:
    d_loss_d_eps = jax.grad(droplet_mass_loss, argnums=0)(epsilon, T)

This is the "gold standard" for CUDA finite-difference verification.
CUDA cannot compute exact gradients; JAX can — at machine precision.

Usage::

    from jax_lbm.adjoint import droplet_mass_loss, grad_mass_loss
    loss = droplet_mass_loss(epsilon=1.7, T_reduced=0.70)
    d_loss = grad_mass_loss(epsilon=1.7, T_reduced=0.70)

Reference
---------
- Phase 6 §6 (P2: 伴随金标准验证管线)
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import grad, jit
from jax import lax

from jax_lbm.lattice import CS2
from jax_lbm.collision import collision_bgk, equilibrium
from jax_lbm.eos import cs_eos_pressure
from jax_lbm.force import shan_chen_force
from jax_lbm.d2q9_bgk import init_droplet, init_equilibrium, streaming


@partial(jit, static_argnames=("nx", "ny", "n_steps"))
def droplet_mass_loss(epsilon, T_reduced, nx=64, ny=64, n_steps=200):
    """End-to-end differentiable droplet simulation.

    Parameters
    ----------
    epsilon : float — Huang surface-tension parameter
    T_reduced : float — reduced temperature T/Tc
    nx, ny : int — grid dimensions
    n_steps : int — number of LBM time steps

    Returns
    -------
    mass_loss : float — fractional mass loss (0 = perfect conservation)
    """
    Tc = 0.3773 / 4.0
    T = T_reduced * Tc
    omega = 1.0 / 1.5

    rho_g = 0.009170
    rho_l = 0.305202

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
    mass_initial = jnp.sum(rho_init)

    # ── LBM time loop via lax.scan (differentiable!) ──
    def step_fn(f, _):
        rho = jnp.sum(f, axis=-1, keepdims=True)
        p = cs_eos_pressure(rho, T=T)
        psi_sq = 2.0 * (p - rho * CS2) / (-1.0)
        psi = jnp.sqrt(jnp.clip(psi_sq, 0.0, None))
        F = shan_chen_force(psi, G=-1.0)
        f_new = streaming(collision_bgk(f, omega, F))
        return f_new, None

    f_final, _ = lax.scan(step_fn, f, None, length=n_steps)
    rho_final = jnp.sum(f_final, axis=-1)
    mass_final = jnp.sum(rho_final)

    return (mass_initial - mass_final) / mass_initial


# ── Gradient functions ──
grad_mass_loss = jit(grad(droplet_mass_loss, argnums=(0, 1)))


def finite_difference_gradient(fn, x, h=1e-5):
    """Central finite-difference gradient for verification.

    Parameters
    ----------
    fn : callable(x) → float
    x : float — parameter value
    h : float — step size

    Returns
    -------
    df_dx : float — approximate derivative
    """
    return (fn(x + h) - fn(x - h)) / (2.0 * h)
