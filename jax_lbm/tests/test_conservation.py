"""JAX LBM long-term conservation tests.

Run: JAX_ENABLE_X64=1 uv run pytest jax_lbm/tests/test_conservation.py -v
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from jax_lbm.lattice import C, CS2
from jax_lbm.collision import collision_bgk, collision_mrt, equilibrium
from jax_lbm.eos import cs_eos_pressure
from jax_lbm.force import shan_chen_force
from jax_lbm.d2q9_bgk import init_droplet, init_equilibrium, streaming


class TestLongTermConservation:
    """Mass and momentum conservation over extended runs."""

    def test_mass_conservation_10000_steps_uniform(self):
        """Uniform field must conserve mass exactly over 10000 steps."""
        nx, ny = 32, 32
        rho = jnp.full((nx, ny, 1), 0.30)
        u = jnp.zeros((nx, ny, 2))
        f = equilibrium(rho, u)
        mass0 = float(jnp.sum(rho))
        omega = 1.0 / 1.5

        for _ in range(10000):
            f = streaming(collision_bgk(f, omega, jnp.zeros_like(u)))

        mass_final = float(jnp.sum(jnp.sum(f, axis=-1)))
        assert abs(mass_final - mass0) / mass0 < 1e-14, (
            f"Mass drift: {abs(mass_final - mass0) / mass0:.2e}"
        )

    def test_mass_conservation_droplet_1000_steps(self):
        """Droplet must conserve mass over 1000 steps with SC force."""
        nx, ny = 64, 64
        Tc = 0.3773 / 4.0
        eos_T = 0.70 * Tc
        rho, u = init_droplet(
            nx,
            ny,
            radius=18,
            center=(32, 32),
            rho_l=0.305202,
            rho_g=0.009170,
            width=3.0,
        )
        f = init_equilibrium(rho, u)
        mass0 = float(jnp.sum(rho))
        omega = 1.0 / 1.5

        for _ in range(1000):
            rho_c = jnp.sum(f, axis=-1, keepdims=True)
            p = cs_eos_pressure(rho_c, T=eos_T)
            psi = jnp.sqrt(jnp.clip(2.0 * (p - rho_c * CS2) / (-1.0), 0, None))
            F = shan_chen_force(psi, G=-1.0)
            f = streaming(collision_bgk(f, omega, F))

        mass_final = float(jnp.sum(jnp.sum(f, axis=-1)))
        assert abs(mass_final - mass0) / mass0 < 1e-10, (
            f"Droplet mass drift: {abs(mass_final - mass0) / mass0:.2e}"
        )
